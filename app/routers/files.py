from io import BytesIO
from typing import Annotated
from zipfile import ZipFile, BadZipFile

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import (
    MAX_FILE_SIZE_BYTES,
    MAX_ARCHIVE_SIZE_BYTES,
    MAX_FILES,
    PIPELINE_NAME,
)
from app.dependencies import get_pipeline
from app.models import OperatorType
from app.services.processing import (
    decode_uploaded_file,
    parse_entities_param,
    parse_word_list_param,
    process_text_item,
)
from src.ner_redaction.pipeline import RedactionPipeline

router = APIRouter(tags=["Files"])


@router.post("/deidentify-file", summary="Deidentify single .txt file")
async def deidentify_file(
    file: Annotated[UploadFile, File(description="UTF-8 .txt file")],
    mode: Annotated[OperatorType, Query()] = OperatorType.replace,
    entities: Annotated[str | None, Query(description="PERSON,EMAIL,...")] = None,
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    allowlist: Annotated[str | None, Query()] = None,
    denylist: Annotated[str | None, Query()] = None,
    pipeline: RedactionPipeline = Depends(get_pipeline),
):

    entities_filter = parse_entities_param(entities)
    allowlist_items = parse_word_list_param(allowlist)
    denylist_items = parse_word_list_param(denylist)

    content = await file.read()
    filename = file.filename or "uploaded_file"
    text = decode_uploaded_file(content, filename, MAX_FILE_SIZE_BYTES)

    result = process_text_item(
        pipeline=pipeline,
        text=text,
        mode=mode.value,
        entities_filter=entities_filter,
        threshold=threshold,
        allowlist=allowlist_items,
        denylist=denylist_items,
        filename=filename,
    )
    return JSONResponse(result)


@router.post("/deidentify-archive", summary="Deidentify ZIP archive of .txt files")
async def deidentify_archive(
    file: Annotated[UploadFile, File(description="ZIP archive with UTF-8 .txt files")],
    mode: Annotated[OperatorType, Query()] = OperatorType.replace,
    entities: Annotated[str | None, Query(description="PERSON,EMAIL,...")] = None,
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    allowlist: Annotated[str | None, Query()] = None,
    denylist: Annotated[str | None, Query()] = None,
    pipeline: RedactionPipeline = Depends(get_pipeline),
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
        info for info in zip_file.infolist()
        if not info.is_dir()
        and not info.filename.startswith("__MACOSX/")
        and info.filename.lower().endswith(".txt")
    ]

    if not txt_files:
        raise HTTPException(status_code=400, detail="ZIP archive does not contain .txt files")

    if len(txt_files) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"Too many files (max {MAX_FILES})")

    results = []
    for index, info in enumerate(txt_files):
        if ".." in info.filename or info.filename.startswith("/"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsafe filename in archive: {info.filename}",
            )

        text = decode_uploaded_file(
            zip_file.read(info), info.filename, MAX_FILE_SIZE_BYTES
        )
        results.append(process_text_item(
            pipeline=pipeline,
            text=text,
            mode=mode.value,
            entities_filter=entities_filter,
            threshold=threshold,
            allowlist=allowlist_items,
            denylist=denylist_items,
            filename=info.filename,
            index=index,
        ))

    return JSONResponse({
        "archive_filename": archive_name,
        "results": results,
        "files_count": len(results),
        "mode": mode.value,
        "entities_filter": entities_filter,
        "threshold": threshold,
        "allowlist": allowlist_items,
        "denylist": denylist_items,
        "pipeline": PIPELINE_NAME,
    })
