from enum import Enum
from typing import Annotated
from io import BytesIO
from zipfile import ZipFile, BadZipFile
import re

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.ner_redaction.pipeline import RedactionPipeline


class OperatorType(str, Enum):
    replace = "replace"
    mask = "mask"
    pseudonymize = "pseudonymize"


SUPPORTED_ENTITIES = ["PERSON", "EMAIL", "PHONE", "ADDRESS", "ID"]
SUPPORTED_OPERATORS = [op.value for op in OperatorType]
PIPELINE_NAME = "hybrid"

MAX_TEXT_LENGTH = 200_000
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_SIZE_BYTES = 10 * 1024 * 1024
MAX_FILES = 10


class DeidentifyTextRequest(BaseModel):
    text: str = Field(
        ...,
        examples=["Иван Петров написал на ivan@mail.ru"],
        description="Input Russian text for PII detection and anonymization",
    )
    mode: OperatorType = Field(
        default=OperatorType.replace,
        description="Anonymization mode: replace, mask, pseudonymize",
    )
    entities: list[str] | None = Field(
        default=None,
        examples=[["PERSON", "EMAIL", "PHONE"]],
        description="Entity types to detect. If null, all supported entities are used.",
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Acceptance threshold for detected entities",
    )
    allowlist: list[str] = Field(
        default_factory=list,
        description="Words that must not be treated as personal data",
    )
    denylist: list[str] = Field(
        default_factory=list,
        description="Words that must be treated as personal data even if not detected automatically",
    )


app = FastAPI(
    title="NER Redaction API",
    description="PII detection and anonymization service for Russian texts",
    version="0.1.0",
    openapi_version="3.0.3",
)

pipeline = RedactionPipeline(model_path="models/final_model")


def count_entities_by_type(entities: list[dict]) -> dict[str, int]:
    counts = {}
    for entity in entities:
        label = entity["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def parse_entities_param(entities: str | None) -> list[str] | None:
    if entities is None or entities.strip() == "":
        return None

    parsed = [item.strip().upper() for item in entities.split(",") if item.strip()]
    unsupported = [e for e in parsed if e not in SUPPORTED_ENTITIES]

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported entity type",
                "unsupported_entities": unsupported,
                "supported_entities": SUPPORTED_ENTITIES,
                "example": "PERSON,EMAIL,PHONE",
            },
        )

    return parsed


def validate_entities_list(entities: list[str] | None) -> list[str] | None:
    if entities is None:
        return None

    parsed = [entity.strip().upper() for entity in entities if entity.strip()]
    unsupported = [e for e in parsed if e not in SUPPORTED_ENTITIES]

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported entity type",
                "unsupported_entities": unsupported,
                "supported_entities": SUPPORTED_ENTITIES,
            },
        )

    return parsed


def parse_word_list_param(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def validate_text_size(text: str) -> None:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "Text is too large",
                "text_length": len(text),
                "max_text_length": MAX_TEXT_LENGTH,
            },
        )


def decode_uploaded_file(content: bytes, filename: str) -> str:
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "message": "File is too large",
                "filename": filename,
                "file_size_bytes": len(content),
                "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
            },
        )

    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Only UTF-8 text files are supported",
                "filename": filename,
            },
        )


def apply_allowlist(entities: list[dict], allowlist: list[str]) -> list[dict]:
    if not allowlist:
        return entities

    allowlist_lower = {word.lower() for word in allowlist}
    return [
        entity
        for entity in entities
        if entity.get("text", "").lower() not in allowlist_lower
    ]


def apply_denylist(text: str, entities: list[dict], denylist: list[str]) -> list[dict]:
    if not denylist:
        return entities

    result = list(entities)

    for word in denylist:
        pattern = re.escape(word)

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()

            has_overlap = any(
                not (end <= entity["start"] or start >= entity["end"])
                for entity in result
            )

            if has_overlap:
                continue

            result.append(
                {
                    "start": start,
                    "end": end,
                    "label": "DENYLIST",
                    "score": 1.0,
                    "source": "denylist",
                    "source_component": "rule",
                    "source_detector": "denylist",
                    "text": text[start:end],
                    "replacement": "[DENYLIST]",
                    "anonymization_mode": None,
                }
            )

    return sorted(result, key=lambda entity: (entity["start"], entity["end"]))


def build_highlight_segments(text: str, entities: list[dict]) -> list[dict]:
    segments = []
    cursor = 0

    sorted_entities = sorted(
        [e for e in entities if int(e.get("start", -1)) < int(e.get("end", -1))],
        key=lambda e: (int(e["start"]), int(e["end"])),
    )

    for idx, entity in enumerate(sorted_entities):
        start = int(entity["start"])
        end = int(entity["end"])

        if start < cursor:
            continue

        if cursor < start:
            segments.append({"type": "text", "text": text[cursor:start]})

        segments.append(
            {
                "type": "entity",
                "ui_id": f"entity-{idx}",
                "label": entity.get("label"),
                "text": entity.get("text", text[start:end]),
                "replacement": entity.get(
                    "replacement",
                    f"[{entity.get('label', 'ENTITY')}]",
                ) or "",
                "start": start,
                "end": end,
                "score": entity.get("score"),
                "source": entity.get("source"),
            }
        )

        cursor = end

    if cursor < len(text):
        segments.append({"type": "text", "text": text[cursor:]})

    return segments


def process_text_item(
    text: str,
    mode: str,
    entities_filter: list[str] | None = None,
    threshold: float = 0.0,
    allowlist: list[str] | None = None,
    denylist: list[str] | None = None,
    filename: str | None = None,
    index: int | None = None,
) -> dict:
    validate_text_size(text)

    detected_entities, metadata = pipeline.analyze(
        text=text,
        entities=entities_filter,
        threshold=threshold,
    )

    filtered_entities = apply_allowlist(
        entities=detected_entities,
        allowlist=allowlist or [],
    )

    filtered_entities = apply_denylist(
        text=text,
        entities=filtered_entities,
        denylist=denylist or [],
    )

    anonymized_text, processed_entities = pipeline.anonymize(
        text=text,
        entities=filtered_entities,
        mode=mode,
    )

    result = {
        "anonymized_text": anonymized_text,
        "highlight_segments": build_highlight_segments(text, processed_entities),
        "entities_count": len(processed_entities),
        "entities_count_by_type": count_entities_by_type(processed_entities),
        "entities": processed_entities,
        "text_length": metadata["text_length"],
        "truncated": metadata["truncated"],
        "mode": mode,
        "entities_filter": entities_filter,
        "threshold": threshold,
        "allowlist": allowlist or [],
        "denylist": denylist or [],
        "pipeline": PIPELINE_NAME,
    }

    if filename is not None:
        result["filename"] = filename

    if index is not None:
        result["file_index"] = index

    return result


@app.get(
    "/health",
    summary="Health check",
    description="Check that the API service is running.",
)
def health():
    return {
        "status": "ok",
        "pipeline": PIPELINE_NAME,
        "version": "0.1.0",
    }


@app.get(
    "/supported-entities",
    summary="Supported entities",
    description="Return entity types supported by the hybrid PII detection pipeline.",
)
def supported_entities():
    return {
        "supported_entities": SUPPORTED_ENTITIES,
        "count": len(SUPPORTED_ENTITIES),
        "pipeline": PIPELINE_NAME,
    }


@app.get(
    "/operators",
    summary="Supported anonymization operators",
    description="Return available anonymization modes.",
)
def operators():
    return {
        "operators": SUPPORTED_OPERATORS,
        "count": len(SUPPORTED_OPERATORS),
    }


@app.post(
    "/deidentify-text",
    summary="Deidentify text",
    description="Detect and anonymize PII in a single Russian text using the hybrid pipeline.",
)
def deidentify_text(request: DeidentifyTextRequest):
    entities_filter = validate_entities_list(request.entities)

    return process_text_item(
        text=request.text,
        mode=request.mode.value,
        entities_filter=entities_filter,
        threshold=request.threshold,
        allowlist=request.allowlist,
        denylist=request.denylist,
    )


@app.post(
    "/deidentify-ui",
    summary="Deidentify text for UI",
    description="UI-compatible endpoint. Uses the same logic as /deidentify-text.",
)
def deidentify_ui(request: DeidentifyTextRequest):
    return deidentify_text(request)


@app.post(
    "/deidentify-file",
    summary="Deidentify file",
    description="Upload one UTF-8 .txt file and anonymize detected PII.",
)
async def deidentify_file(
    file: Annotated[UploadFile, File(description="Upload one UTF-8 .txt file")],
    mode: Annotated[OperatorType, Query(description="Anonymization mode")] = OperatorType.replace,
    entities: Annotated[str | None, Query(description="Choose entities to detect, e.g. PERSON,EMAIL,PHONE")] = None,
    threshold: Annotated[float, Query(ge=0.0, le=1.0, description="Acceptance threshold")] = 0.0,
    allowlist: Annotated[str | None, Query(description="Comma-separated words to exclude from PII")] = None,
    denylist: Annotated[str | None, Query(description="Comma-separated words to force as PII")] = None,
):
    entities_filter = parse_entities_param(entities)
    allowlist_items = parse_word_list_param(allowlist)
    denylist_items = parse_word_list_param(denylist)

    content = await file.read()
    filename = file.filename or "uploaded_file"
    text = decode_uploaded_file(content, filename)

    result = process_text_item(
        text=text,
        mode=mode.value,
        entities_filter=entities_filter,
        threshold=threshold,
        allowlist=allowlist_items,
        denylist=denylist_items,
        filename=filename,
    )

    return JSONResponse(result)


@app.post(
    "/deidentify-archive",
    summary="Deidentify ZIP archive",
    description="Upload ZIP archive with UTF-8 .txt files and anonymize each file.",
)
async def deidentify_archive(
    file: Annotated[UploadFile, File(description="Upload ZIP archive with UTF-8 .txt files")],
    mode: Annotated[OperatorType, Query(description="Anonymization mode")] = OperatorType.replace,
    entities: Annotated[str | None, Query(description="Choose entities to detect, e.g. PERSON,EMAIL,PHONE")] = None,
    threshold: Annotated[float, Query(ge=0.0, le=1.0, description="Acceptance threshold")] = 0.0,
    allowlist: Annotated[str | None, Query(description="Comma-separated words to exclude from PII")] = None,
    denylist: Annotated[str | None, Query(description="Comma-separated words to force as PII")] = None,
):
    entities_filter = parse_entities_param(entities)
    allowlist_items = parse_word_list_param(allowlist)
    denylist_items = parse_word_list_param(denylist)

    content = await file.read()
    archive_name = file.filename or "uploaded_archive.zip"

    if len(content) > MAX_ARCHIVE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Archive is too large")

    if not archive_name.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archives are supported")

    try:
        zip_file = ZipFile(BytesIO(content))
    except BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP archive")

    txt_files = [
        info
        for info in zip_file.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and info.filename.lower().endswith(".txt")
    ]

    if not txt_files:
        raise HTTPException(status_code=400, detail="ZIP archive does not contain .txt files")

    if len(txt_files) > MAX_FILES:
        raise HTTPException(status_code=413, detail="Too many txt files in archive")

    results = []

    for index, info in enumerate(txt_files):
        if ".." in info.filename or info.filename.startswith("/"):
            raise HTTPException(status_code=400, detail=f"Unsafe filename in archive: {info.filename}")

        if info.file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail=f"File is too large: {info.filename}")

        text = decode_uploaded_file(zip_file.read(info), info.filename)

        results.append(
            process_text_item(
                text=text,
                mode=mode.value,
                entities_filter=entities_filter,
                threshold=threshold,
                allowlist=allowlist_items,
                denylist=denylist_items,
                filename=info.filename,
                index=index,
            )
        )

    return JSONResponse(
        {
            "archive_filename": archive_name,
            "results": results,
            "files_count": len(results),
            "mode": mode.value,
            "entities_filter": entities_filter,
            "threshold": threshold,
            "allowlist": allowlist_items,
            "denylist": denylist_items,
            "pipeline": PIPELINE_NAME,
        }
    )




@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>NER Redaction Service</title>

<style>
body { font-family: Arial, sans-serif; background:#f4f6f8; color:#1f2937; }
.container { max-width:1200px; margin:30px auto; background:white; padding:28px; border-radius:14px; box-shadow:0 8px 24px rgba(0,0,0,.08); }
.subtitle, .method { color:#6b7280; line-height:1.5; }
.tabs { display:flex; gap:10px; margin:24px 0; }
.tab-button { padding:12px 20px; border:0; border-radius:10px; background:#e5e7eb; cursor:pointer; font-size:15px; }
.tab-button.active { background:#2563eb; color:white; }
.tab { display:none; }
.tab.active { display:block; }
.controls { display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; align-items:stretch; margin:22px 0; }
.card { background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:18px; }
.card h3 { margin-top:0; margin-bottom:14px; }
textarea { width:100%; min-height:180px; padding:14px; font-size:15px; border:1px solid #d1d5db; border-radius:10px; box-sizing:border-box; }
select, input[type=file] { width:100%; padding:10px; box-sizing:border-box; }
.entities { display:flex; flex-wrap:wrap; gap:10px; }
.entity-item { background:#eef2ff; padding:8px 12px; border-radius:999px; font-size:14px; }
.button { background:#16a34a; color:white; border:0; padding:13px 22px; border-radius:10px; cursor:pointer; font-size:16px; margin-top:10px; }
.button:hover { background:#15803d; }
.threshold-card { overflow:hidden; }
.threshold-subtitle { display:block; color:#6b7280; font-size:14px; font-weight:600; margin-top:4px; }
.threshold-wrapper { position:relative; padding:34px 0 8px 0; }
#threshold { width:100%; margin:0; padding:0; display:block; box-sizing:border-box; accent-color:#d63384; }
.threshold-bubble { position:absolute; top:0; transform:translateX(-50%); background:#2563eb; color:white; padding:4px 9px; border-radius:999px; font-size:13px; font-weight:700; white-space:nowrap; }
.threshold-scale { display:flex; justify-content:space-between; margin-top:8px; color:#6b7280; font-size:13px; font-weight:600; }
.lists-card { margin:22px 0; }
.lists-grid { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
.list-block h3 { margin:0 0 10px 0; }
.list-input { width:100%; padding:13px; border:1px solid #d1d5db; border-radius:10px; font-size:15px; box-sizing:border-box; }
.list-description { color:#6b7280; line-height:1.5; margin-top:12px; margin-bottom:0; }
.word-tags { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.word-tag { background:#eef2ff; border-radius:999px; padding:6px 10px; font-size:13px; }
.word-tag button { border:none; background:transparent; cursor:pointer; margin-left:6px; color:#6b7280; }
.anonymized {
    background:#f3f4f6;
    padding:16px;
    border-radius:10px;
    white-space:pre-wrap;
    border:1px solid #e5e7eb;
    min-height:80px;
    line-height:1.7;
    font-size:15px;
}

.entity-highlight {
    display:inline-flex;
    align-items:center;
    gap:5px;
    padding:2px 7px;
    margin:0 2px;
    border-radius:7px;
    font-weight:700;
    cursor:pointer;
    transition:all .15s ease;
    border:1px solid transparent;
    white-space:nowrap;
    vertical-align:middle;
}

.entity-label {
    font-size:10px;
    line-height:1;
    opacity:.65;
}
.entity-highlight:hover, .entity-highlight.active { box-shadow:0 0 0 3px rgba(37,99,235,.18); border-color:#2563eb; }
.entity-PERSON {
    background:#fef3c7;  
    color:#92400e;
}

.entity-EMAIL {
    background:#e0ecff; 
    color:#1e40af;
}

.entity-PHONE {
    background:#dcfce7; 
    color:#166534;
}

.entity-ADDRESS {
    background:#fee2e2;   
    color:#7f1d1d;
}

.entity-ID {
    background:#ede9fe;  
    color:#5b21b6;
}

.entity-DENYLIST {
    background:#fce7f3;   
    color:#9d174d;
}

.meta { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0; }
.badge { background:#e0f2fe; color:#075985; padding:7px 10px; border-radius:999px; font-size:13px; }
.error { display:none; color:#b91c1c; background:#fee2e2; padding:12px; border-radius:10px; margin-top:16px; }
table { width:100%; border-collapse:collapse; margin-top:18px; font-size:14px; }
th, td { border-bottom:1px solid #e5e7eb; padding:10px; text-align:left; vertical-align:top; }
th { background:#f9fafb; }
.entity-row { cursor:pointer; transition:background .15s ease, transform .15s ease; }
.entity-row:hover, .entity-row.active { background:#eff6ff; }
.entity-type-pill { display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; font-weight:700; }
pre { background:#111827; color:#f9fafb; padding:16px; border-radius:10px; overflow:auto; max-height:350px; }
.file-list { margin-top:10px; font-size:14px; }
.file-item { background:#f3f4f6; padding:7px 10px; border-radius:8px; margin-top:6px; display:flex; justify-content:space-between; }
</style>
</head>

<body>
<div class="container">
<h1>NER Redaction Service</h1>
<div class="subtitle">Инструмент обнаружения и обезличивания персональных данных в русскоязычных текстах.</div>
<div class="method">Используется гибридный пайплайн: ML-модель применяется для распознавания сущностей PERSON и ADDRESS, а rule-based детекторы используются для EMAIL, PHONE и ID.</div>

<div class="tabs">
<button class="tab-button active" onclick="openTab('textTab', this)">Текст</button>
<button class="tab-button" onclick="openTab('fileTab', this)">Файл</button>
<button class="tab-button" onclick="openTab('zipTab', this)">ZIP-архив</button>
</div>

<div class="controls">
    <div class="card">
        <h3>Режим анонимизации</h3>
        <select id="mode" onchange="rerunLast()">
            <option value="replace">replace — замена на [TYPE]</option>
            <option value="mask">mask — маскирование</option>
            <option value="pseudonymize">pseudonymize — псевдонимизация</option>
        </select>
    </div>
    <div class="card">
        <h3>Сущности для поиска</h3>
        <div class="entities">
            <label class="entity-item"><input type="checkbox" value="PERSON" checked onchange="rerunLast()"> PERSON</label>
            <label class="entity-item"><input type="checkbox" value="EMAIL" checked onchange="rerunLast()"> EMAIL</label>
            <label class="entity-item"><input type="checkbox" value="PHONE" checked onchange="rerunLast()"> PHONE</label>
            <label class="entity-item"><input type="checkbox" value="ADDRESS" checked onchange="rerunLast()"> ADDRESS</label>
            <label class="entity-item"><input type="checkbox" value="ID" checked onchange="rerunLast()"> ID</label>
        </div>
    </div>
    <div class="card threshold-card">
        <h3>Порог уверенности <span class="threshold-subtitle">(Acceptance threshold)</span></h3>
        <div class="threshold-wrapper">
            <div id="thresholdBubble" class="threshold-bubble">0.00</div>
            <input type="range" id="threshold" min="0" max="1" step="0.01" value="0" oninput="updateThreshold()">
            <div class="threshold-scale"><span>0.0</span><span>1.0</span></div>
        </div>
    </div>
</div>

<div class="card lists-card">
    <h2>Запрещенные и разрешенные сущности</h2>
    <div class="lists-grid">
        <div class="list-block">
            <h3>Добавить слова в список запрещенных сущностей</h3>
            <input id="allowlistInput" class="list-input" placeholder="Введите слово и нажмите Enter" onkeydown="handleWordInput(event, 'allowlist')">
            <div id="allowlistTags" class="word-tags"></div>
            <p class="list-description">Список содержит слова, которые не считаются персональными данными, даже если модель или правила распознали их как сущность.</p>
        </div>
        <div class="list-block">
            <h3>Добавить слова в список разрешенных сущностей</h3>
            <input id="denylistInput" class="list-input" placeholder="Введите слово и нажмите Enter" onkeydown="handleWordInput(event, 'denylist')">
            <div id="denylistTags" class="word-tags"></div>
            <p class="list-description">Список содержит слова, которые следует считать персональными данными, даже если они не были найдены автоматически.</p>
        </div>
    </div>
</div>

<div id="textTab" class="tab active">
<h2>Анонимизация текста</h2>
<textarea id="textInput" placeholder="Введите текст для анализа..."></textarea>
<button class="button" onclick="sendText()">Анонимизировать текст</button>
</div>

<div id="fileTab" class="tab">
<h2>Анонимизация одного или нескольких файлов</h2>
<input type="file" id="fileInput" accept=".txt" multiple>
<div id="fileList" class="file-list">Файлы не выбраны</div>
<button class="button" onclick="sendFile()">Анонимизировать файл(ы)</button>
</div>

<div id="zipTab" class="tab">
<h2>Пакетная обработка ZIP-архива</h2>
<input type="file" id="zipInput" accept=".zip">
<button class="button" onclick="sendZip()">Анонимизировать архив</button>
</div>

<div id="error" class="error"></div>

<div class="result-box">
<h2>Результат</h2>
<h3>Анонимизированный текст</h3>
<div id="anonymizedText" class="anonymized">Пока нет результата</div>
<div id="meta" class="meta"></div>

<h3>Отчет о найденных сущностях</h3>
<table>
<thead>
<tr>
<th>№</th><th>Файл</th><th>Тип</th><th>Текст</th><th>Замена</th><th>start</th><th>end</th><th>score</th><th>source</th>
</tr>
</thead>
<tbody id="entitiesTable"><tr><td colspan="10">Нет данных</td></tr></tbody>
</table>

<h3>JSON-ответ</h3>
<pre id="rawJson">{}</pre>
</div>
</div>

<script>
let selectedFiles = [];
let lastAction = null;
let thresholdTimer = null;
let allowlist = [];
let denylist = [];

// Нужны для связи подсвеченного текста и таблицы.
let currentEntityCounter = 0;

document.addEventListener("DOMContentLoaded", () => {
    updateThresholdBubble();
    document.getElementById("fileInput").addEventListener("change", function(event) {
        selectedFiles = selectedFiles.concat(Array.from(event.target.files));
        event.target.value = "";
        renderFileList();
    });
});

window.addEventListener("resize", updateThresholdBubble);

function handleWordInput(event, listName) {
    if (event.key !== "Enter") return;
    event.preventDefault();
    const inputId = listName === "allowlist" ? "allowlistInput" : "denylistInput";
    const input = document.getElementById(inputId);
    const value = input.value.trim();
    if (!value) return;
    if (listName === "allowlist" && !allowlist.includes(value)) {
        allowlist.push(value);
        renderWordTags("allowlist");
    }
    if (listName === "denylist" && !denylist.includes(value)) {
        denylist.push(value);
        renderWordTags("denylist");
    }
    input.value = "";
    rerunLast();
}

function renderWordTags(listName) {
    const list = listName === "allowlist" ? allowlist : denylist;
    const containerId = listName === "allowlist" ? "allowlistTags" : "denylistTags";
    document.getElementById(containerId).innerHTML = list.map((word, index) => `
        <span class="word-tag">${escapeHtml(word)}<button onclick="removeWord('${listName}', ${index})">×</button></span>
    `).join("");
}

function removeWord(listName, index) {
    if (listName === "allowlist") {
        allowlist.splice(index, 1);
        renderWordTags("allowlist");
    } else {
        denylist.splice(index, 1);
        renderWordTags("denylist");
    }
    rerunLast();
}

function updateThresholdBubble() {
    const slider = document.getElementById("threshold");
    const bubble = document.getElementById("thresholdBubble");
    const min = Number(slider.min);
    const max = Number(slider.max);
    const value = Number(slider.value);
    const percent = (value - min) / (max - min);
    bubble.innerText = value.toFixed(2);
    const sliderWidth = slider.offsetWidth;
    const bubbleWidth = bubble.offsetWidth;
    const left = percent * (sliderWidth - bubbleWidth) + bubbleWidth / 2;
    bubble.style.left = left + "px";
}

function updateThreshold() {
    updateThresholdBubble();
    clearTimeout(thresholdTimer);
    thresholdTimer = setTimeout(() => rerunLast(), 350);
}

function openTab(tabId, button) {
    document.querySelectorAll(".tab").forEach(tab => tab.classList.remove("active"));
    document.querySelectorAll(".tab-button").forEach(btn => btn.classList.remove("active"));
    document.getElementById(tabId).classList.add("active");
    button.classList.add("active");
    clearResult();
}

function getMode() { return document.getElementById("mode").value; }
function getThreshold() { return Number(document.getElementById("threshold").value); }
function rerunLast() {
    if (lastAction === "text") sendText(false);
    if (lastAction === "file") sendFile(false);
    if (lastAction === "zip") sendZip(false);
}
function getEntities() {
    return Array.from(document.querySelectorAll(".entity-item input:checked")).map(cb => cb.value);
}
function entitiesParam() {
    const entities = getEntities();
    return entities.length > 0 ? entities.join(",") : "";
}
function showError(message) {
    const error = document.getElementById("error");
    error.style.display = "block";
    error.innerText = message;
}
function hideError() {
    const error = document.getElementById("error");
    error.style.display = "none";
    error.innerText = "";
}
function clearResult() {
    hideError();
    currentEntityCounter = 0;
    document.getElementById("anonymizedText").innerText = "Пока нет результата";
    document.getElementById("meta").innerHTML = "";
    document.getElementById("rawJson").innerText = "{}";
    document.getElementById("entitiesTable").innerHTML = "<tr><td colspan='9'>Нет данных</td></tr>";
}

function renderFileList() {
    const container = document.getElementById("fileList");
    if (selectedFiles.length === 0) {
        container.innerHTML = "Файлы не выбраны";
        return;
    }
    container.innerHTML = selectedFiles.map((file, index) => `
        <div class="file-item"><span>${escapeHtml(file.name)}</span><button onclick="removeFile(${index})">Удалить</button></div>
    `).join("");
}
function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
}

function entityClass(label) {
    return "entity-" + String(label || "UNKNOWN").replaceAll("_", "-");
}

function enrichSegmentsAndEntities(item, filename = "-") {
    const idMap = new Map();
    const segments = (item.highlight_segments || []).map(segment => {
        if (segment.type !== "entity") return segment;
        const uiId = "entity-" + currentEntityCounter++;
        idMap.set(`${segment.start}:${segment.end}:${segment.label}:${segment.text}`, uiId);
        return {...segment, ui_id: uiId, filename};
    });

    const entities = (item.entities || []).map(entity => {
        const key = `${entity.start}:${entity.end}:${entity.label}:${entity.text}`;
        const uiId = idMap.get(key) || "entity-" + currentEntityCounter++;
        return {...entity, ui_id: uiId, filename};
    });

    return {segments, entities};
}

function renderHighlightedSegments(segments) {
    if (!segments || segments.length === 0) {
        return "Нет анонимизированного текста";
    }

    return segments.map(segment => {
        if (segment.type === "text") {
            return escapeHtml(segment.text || "");
        }

        const label = escapeHtml(segment.label || "");
        const replacement = escapeHtml(segment.replacement || "");
        const title = escapeHtml(
            `${segment.label || ""} | ${segment.text || ""} → ${segment.replacement || ""} | score: ${segment.score ?? "-"}`
        );

        return `<span id="${segment.ui_id}" class="entity-highlight ${entityClass(segment.label)}" title="${title}" onmouseenter="highlightEntity('${segment.ui_id}')" onmouseleave="unhighlightEntity('${segment.ui_id}')" onclick="scrollToEntityRow('${segment.ui_id}')">${replacement}<span class="entity-label">${label}</span></span>`;
    }).join("");
}

function highlightEntity(uiId) {
    document.querySelectorAll(`[data-entity-id="${uiId}"], #${uiId}`).forEach(el => el.classList.add("active"));
}
function unhighlightEntity(uiId) {
    document.querySelectorAll(`[data-entity-id="${uiId}"], #${uiId}`).forEach(el => el.classList.remove("active"));
}
function scrollToEntityRow(uiId) {
    const row = document.querySelector(`tr[data-entity-id="${uiId}"]`);
    if (!row) return;
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    highlightEntity(uiId);
    setTimeout(() => unhighlightEntity(uiId), 1200);
}
function scrollToHighlightedEntity(uiId) {
    const entity = document.getElementById(uiId);
    if (!entity) return;
    entity.scrollIntoView({ behavior: "smooth", block: "center" });
    highlightEntity(uiId);
    setTimeout(() => unhighlightEntity(uiId), 1200);
}

function renderResult(data) {
    hideError();
    document.getElementById("rawJson").innerText = JSON.stringify(data, null, 2);
    currentEntityCounter = 0;
    if (data.results && Array.isArray(data.results)) {
        renderMultiResult(data);
    } else {
        renderSingleResult(data);
    }
}

function renderSingleResult(data) {
    const enriched = enrichSegmentsAndEntities(data, data.filename ?? "-");
    document.getElementById("anonymizedText").innerHTML = renderHighlightedSegments(enriched.segments);
    document.getElementById("meta").innerHTML = `
        <span class="badge">Файлов: 1</span>
        <span class="badge">Сущностей всего: ${data.entities_count ?? 0}</span>
        <span class="badge">Длина текста: ${data.text_length ?? 0}</span>
        <span class="badge">Режим: ${data.mode ?? "-"}</span>
        <span class="badge">Pipeline: ${data.pipeline ?? "-"}</span>
        <span class="badge">Порог: ${Number(data.threshold ?? getThreshold()).toFixed(2)}</span>
    `;
    renderEntities(enriched.entities);
}

function renderMultiResult(data) {
    let html = "";
    let allEntities = [];

    for (const item of data.results) {
        const filename = item.filename ?? "-";
        const enriched = enrichSegmentsAndEntities(item, filename);
        allEntities = allEntities.concat(enriched.entities);
        html += `<strong>Файл: ${escapeHtml(filename)}</strong><br>`;
        html += renderHighlightedSegments(enriched.segments);
        html += `<br><br>-------------------------<br><br>`;
    }

    const perFileStats = data.results.map(item => `
        <span class="badge">${escapeHtml(item.filename)}: сущностей ${item.entities_count ?? 0}, длина ${item.text_length ?? 0}</span>
    `).join("");

    document.getElementById("anonymizedText").innerHTML = html;
    document.getElementById("meta").innerHTML = `
        <span class="badge">Обработано файлов: ${data.files_count}</span>
        <span class="badge">Сущностей всего: ${allEntities.length}</span>
        <span class="badge">Режим: ${data.mode}</span>
        <span class="badge">Pipeline: ${data.pipeline}</span>
        <span class="badge">Порог: ${Number(data.threshold ?? getThreshold()).toFixed(2)}</span>
        ${perFileStats}
    `;
    renderEntities(allEntities);
}

function renderEntities(entities) {
    const table = document.getElementById("entitiesTable");
    if (!entities || entities.length === 0) {
        table.innerHTML = "<tr><td colspan='9'>Сущности не найдены</td></tr>";
        return;
    }

    table.innerHTML = entities.map((entity, index) => `
        <tr
            class="entity-row"
            data-entity-id="${entity.ui_id}"
            onmouseenter="highlightEntity('${entity.ui_id}')"
            onmouseleave="unhighlightEntity('${entity.ui_id}')"
            onclick="scrollToHighlightedEntity('${entity.ui_id}')"
        >
            <td>${index + 1}</td>
            <td>${escapeHtml(entity.filename ?? "-")}</td>
            <td><span class="entity-type-pill ${entityClass(entity.label)}">${escapeHtml(entity.label ?? "")}</span></td>
            <td>${escapeHtml(entity.text ?? "")}</td>
            <td>${escapeHtml(entity.replacement ?? "")}</td>
            <td>${entity.start ?? ""}</td>
            <td>${entity.end ?? ""}</td>
            <td>${entity.score !== undefined && entity.score !== null ? Number(entity.score).toFixed(3) : "-"}</td>
            <td>${escapeHtml(entity.source ?? "")}</td>
        </tr>
    `).join("");
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function sendText(setAction = true) {
    if (setAction) lastAction = "text";
    clearResult();

    const text = document.getElementById("textInput").value.trim();

    if (!text) {
        showError("Введите текст.");
        return;
    }

    document.getElementById("anonymizedText").innerText = "Обработка... подождите";
    document.getElementById("rawJson").innerText = "Запрос отправлен...";

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000);

    try {
        console.log("Sending request to /deidentify-text");

        const response = await fetch("/deidentify-text", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            signal: controller.signal,
            body: JSON.stringify({
                text: text,
                mode: getMode(),
                entities: getEntities(),
                threshold: getThreshold(),
                allowlist: allowlist,
                denylist: denylist
            })
        });

        clearTimeout(timeoutId);

        console.log("Response status:", response.status);

        const data = await response.json();
        console.log("Response data:", data);

        if (!response.ok) {
            showError(JSON.stringify(data, null, 2));
            document.getElementById("rawJson").innerText = JSON.stringify(data, null, 2);
            return;
        }

        renderResult(data);
    } catch (error) {
        clearTimeout(timeoutId);
        console.error("Request failed:", error);
        showError("Ошибка запроса: " + error.message);
        document.getElementById("rawJson").innerText = String(error.stack || error);
    }
}

async function sendFile(setAction = true) {
    if (setAction) lastAction = "file";
    clearResult();

    if (!selectedFiles || selectedFiles.length === 0) {
        showError("Выберите один или несколько .txt файлов.");
        return;
    }

    try {
        const allResults = [];

        for (const file of selectedFiles) {
            const params = new URLSearchParams();
            params.append("mode", getMode());
            params.append("threshold", getThreshold().toString());

            const ep = entitiesParam();
            if (ep) params.append("entities", ep);

            if (allowlist.length > 0) params.append("allowlist", allowlist.join(","));
            if (denylist.length > 0) params.append("denylist", denylist.join(","));

            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("/deidentify-file?" + params.toString(), {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                showError(JSON.stringify(data, null, 2));
                return;
            }

            allResults.push(data);
        }

        renderResult({
            results: allResults,
            files_count: allResults.length,
            mode: getMode(),
            threshold: getThreshold(),
            pipeline: "hybrid"
        });
    } catch (error) {
        showError("Ошибка обработки файла: " + error.message);
        console.error(error);
    }
}

async function sendZip(setAction = true) {
    if (setAction) lastAction = "zip";
    clearResult();

    const file = document.getElementById("zipInput").files[0];

    if (!file) {
        showError("Выберите .zip архив.");
        return;
    }

    try {
        const params = new URLSearchParams();
        params.append("mode", getMode());
        params.append("threshold", getThreshold().toString());

        const ep = entitiesParam();
        if (ep) params.append("entities", ep);

        if (allowlist.length > 0) params.append("allowlist", allowlist.join(","));
        if (denylist.length > 0) params.append("denylist", denylist.join(","));

        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/deidentify-archive?" + params.toString(), {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            showError(JSON.stringify(data, null, 2));
            return;
        }

        renderResult(data);
    } catch (error) {
        showError("Ошибка обработки архива: " + error.message);
        console.error(error);
    }
}
</script>
</body>
</html>
"""
