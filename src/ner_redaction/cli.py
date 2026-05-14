from pathlib import Path
from typing import Optional
import json

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from src.ner_redaction.pipeline import RedactionPipeline


console = Console()

app = typer.Typer(
    name="ner-redaction",
    help="CLI-утилита для обнаружения и обезличивания персональных данных в русскоязычных текстах.",
)

SUPPORTED_MODES = {"replace", "mask", "pseudonymize"}
SUPPORTED_ENTITIES = {"PERSON", "EMAIL", "PHONE", "ADDRESS", "ID"}


def parse_entities(entities: Optional[str]) -> list[str] | None:
    if not entities:
        return None

    parsed = [item.strip().upper() for item in entities.split(",") if item.strip()]
    unsupported = [item for item in parsed if item not in SUPPORTED_ENTITIES]

    if unsupported:
        raise typer.BadParameter(
            f"Неподдерживаемые сущности: {unsupported}. "
            f"Доступны: {sorted(SUPPORTED_ENTITIES)}"
        )

    return parsed


def count_entities_by_type(entities: list[dict]) -> dict[str, int]:
    counts = {}

    for entity in entities:
        label = entity["label"]
        counts[label] = counts.get(label, 0) + 1

    return counts

def clean_entities_for_report(entities: list[dict]) -> list[dict]:
    hidden_fields = {
        "source_detector",
        "source_component",
    }

    cleaned = []

    for entity in entities:
        cleaned_entity = {
            key: value
            for key, value in entity.items()
            if key not in hidden_fields
        }
        cleaned.append(cleaned_entity)

    return cleaned


def build_report(
    text: str,
    anonymized_text: str,
    entities: list[dict],
    metadata: dict,
    mode: str,
    input_path: str | None = None,
) -> dict:
    return {
        "input_path": input_path,
        "mode": mode,
        "pipeline": "hybrid",
        "text_length": len(text),
        "chunks_count": metadata.get("chunks_count", 1),
        "truncated": metadata.get("truncated", False),
        "entities_count": len(entities),
        "entities_count_by_type": count_entities_by_type(entities),
        "entities": clean_entities_for_report(entities),
        "anonymized_text": anonymized_text,
    }


def save_outputs(
    output_dir: Path,
    input_name: str,
    anonymized_text: str,
    report: dict,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_name).stem

    anonymized_path = output_dir / f"{stem}.anonymized.txt"
    report_path = output_dir / f"{stem}.report.json"

    anonymized_path.write_text(anonymized_text, encoding="utf-8")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return anonymized_path, report_path




def print_entities_table(entities: list[dict]) -> None:
    if not entities:
        console.print("[bold yellow]Сущности не найдены[/bold yellow]")
        return

    table = Table(title="Найденные сущности", box=box.ROUNDED)

    table.add_column("№", style="dim", justify="right")
    table.add_column("Тип", style="cyan", no_wrap=True)
    table.add_column("Текст", style="white")
    table.add_column("Замена", style="green")
    table.add_column("start", justify="right")
    table.add_column("end", justify="right")
    table.add_column("score", style="magenta", justify="right")
    table.add_column("source", style="blue")

    for i, ent in enumerate(entities, 1):
        score = ent.get("score")

        table.add_row(
            str(i),
            str(ent.get("label", "")),
            str(ent.get("text", "")),
            str(ent.get("replacement", "")),
            str(ent.get("start", "")),
            str(ent.get("end", "")),
            f"{float(score):.3f}" if score is not None else "",
            str(ent.get("source", "")),
        )

    console.print(table)


def print_counts_table(counts: dict[str, int]) -> None:
    if not counts:
        return

    table = Table(title="Статистика по классам", box=box.SIMPLE)

    table.add_column("Тип", style="cyan")
    table.add_column("Количество", justify="right", style="green")

    for label, count in counts.items():
        table.add_row(label, str(count))

    console.print(table)


def print_batch_summary(summary: list[dict]) -> None:
    table = Table(title="Batch Summary", box=box.ROUNDED)

    table.add_column("№", justify="right", style="dim")
    table.add_column("Файл", style="cyan")
    table.add_column("Сущностей", justify="right", style="green")
    table.add_column("Типы", style="magenta")
    table.add_column("Результат", style="blue")

    for i, item in enumerate(summary, 1):
        types = ", ".join(
            f"{label}:{count}"
            for label, count in item["entities_count_by_type"].items()
        )

        table.add_row(
            str(i),
            item["input_path"],
            str(item["entities_count"]),
            types,
            item["output_text"],
        )

    console.print(table)


@app.command()
def text(
    text: str = typer.Argument(..., help="Текст для анонимизации."),
    mode: str = typer.Option(
        "replace",
        "--mode",
        "-m",
        help="Режим: replace, mask, pseudonymize.",
    ),
    entities: Optional[str] = typer.Option(
        None,
        "--entities",
        "-e",
        help="Сущности через запятую, например: PERSON,EMAIL,PHONE.",
    ),
    model_path: str = typer.Option(
        "models/final_model",
        "--model-path",
        help="Путь к обученной модели.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Вывести полный JSON-отчёт вместо красивого табличного вывода.",
    ),
):

    if mode not in SUPPORTED_MODES:
        raise typer.BadParameter(f"mode должен быть одним из: {sorted(SUPPORTED_MODES)}")

    entity_filter = parse_entities(entities)
    pipeline = RedactionPipeline(model_path=model_path)

    anonymized_text, processed_entities, metadata = pipeline.deidentify(
        text=text,
        mode=mode,
        entities=entity_filter,
    )

    result = build_report(
        text=text,
        anonymized_text=anonymized_text,
        entities=processed_entities,
        metadata=metadata,
        mode=mode,
    )

    if json_output:
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    console.print("\n[bold green]Анонимизированный текст:[/bold green]")
    console.print(anonymized_text)

    console.print()
    print_entities_table(processed_entities)
    print_counts_table(result["entities_count_by_type"])


@app.command()
def file(
    input_path: Path = typer.Argument(..., help="Путь к .txt файлу."),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        "-o",
        help="Папка для сохранения результата.",
    ),
    mode: str = typer.Option(
        "replace",
        "--mode",
        "-m",
        help="Режим: replace, mask, pseudonymize.",
    ),
    entities: Optional[str] = typer.Option(
        None,
        "--entities",
        "-e",
        help="Сущности через запятую, например: PERSON,EMAIL,PHONE.",
    ),
    model_path: str = typer.Option(
        "models/final_model",
        "--model-path",
        help="Путь к обученной модели.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Дополнительно вывести полный JSON-отчёт в терминал.",
    ),
):

    if mode not in SUPPORTED_MODES:
        raise typer.BadParameter(f"mode должен быть одним из: {sorted(SUPPORTED_MODES)}")

    if not input_path.exists():
        raise typer.BadParameter(f"Файл не найден: {input_path}")

    if not input_path.is_file():
        raise typer.BadParameter(f"Ожидался файл, получено: {input_path}")

    entity_filter = parse_entities(entities)
    text_value = input_path.read_text(encoding="utf-8-sig")

    pipeline = RedactionPipeline(model_path=model_path)

    anonymized_text, processed_entities, metadata = pipeline.deidentify(
        text=text_value,
        mode=mode,
        entities=entity_filter,
    )

    result = build_report(
        text=text_value,
        anonymized_text=anonymized_text,
        entities=processed_entities,
        metadata=metadata,
        mode=mode,
        input_path=str(input_path),
    )

    anonymized_path, report_path = save_outputs(
        output_dir=output_dir,
        input_name=input_path.name,
        anonymized_text=anonymized_text,
        report=result,
    )

    console.print(f"\n[bold green]Готово:[/bold green] {input_path.name}")
    console.print(f"[cyan]Анонимизированный текст:[/cyan] {anonymized_path}")
    console.print(f"[cyan]Отчёт:[/cyan] {report_path}\n")

    print_entities_table(processed_entities)
    print_counts_table(result["entities_count_by_type"])

    if json_output:
        console.print("\n[bold]JSON-отчёт:[/bold]")
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Папка с .txt файлами."),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir",
        "-o",
        help="Папка для сохранения результатов.",
    ),
    mode: str = typer.Option(
        "replace",
        "--mode",
        "-m",
        help="Режим: replace, mask, pseudonymize.",
    ),
    entities: Optional[str] = typer.Option(
        None,
        "--entities",
        "-e",
        help="Сущности через запятую, например: PERSON,EMAIL,PHONE.",
    ),
    model_path: str = typer.Option(
        "models/final_model",
        "--model-path",
        help="Путь к обученной модели.",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Искать .txt файлы во вложенных папках.",
    ),
):

    if mode not in SUPPORTED_MODES:
        raise typer.BadParameter(f"mode должен быть одним из: {sorted(SUPPORTED_MODES)}")

    if not input_dir.exists():
        raise typer.BadParameter(f"Папка не найдена: {input_dir}")

    if not input_dir.is_dir():
        raise typer.BadParameter(f"Ожидалась папка, получено: {input_dir}")

    entity_filter = parse_entities(entities)
    files = sorted(input_dir.rglob("*.txt") if recursive else input_dir.glob("*.txt"))

    if not files:
        console.print("[bold yellow]Файлы .txt не найдены.[/bold yellow]")
        raise typer.Exit(code=0)

    pipeline = RedactionPipeline(model_path=model_path)
    summary = []

    for input_path in files:
        text_value = input_path.read_text(encoding="utf-8-sig")

        anonymized_text, processed_entities, metadata = pipeline.deidentify(
            text=text_value,
            mode=mode,
            entities=entity_filter,
        )

        result = build_report(
            text=text_value,
            anonymized_text=anonymized_text,
            entities=processed_entities,
            metadata=metadata,
            mode=mode,
            input_path=str(input_path),
        )

        relative_name = input_path.relative_to(input_dir)
        safe_name = str(relative_name).replace("/", "__").replace("\\", "__")

        anonymized_path, report_path = save_outputs(
            output_dir=output_dir,
            input_name=safe_name,
            anonymized_text=anonymized_text,
            report=result,
        )

        summary.append(
            {
                "input_path": str(input_path),
                "entities_count": len(processed_entities),
                "entities_count_by_type": count_entities_by_type(processed_entities),
                "output_text": str(anonymized_path),
                "output_report": str(report_path),
            }
        )

        console.print(f"[green]Обработан файл:[/green] {input_path}")

    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "files_count": len(files),
                "mode": mode,
                "pipeline": "hybrid",
                "results": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    console.print()
    print_batch_summary(summary)

    console.print(f"\n[bold green]Готово. Обработано файлов:[/bold green] {len(files)}")
    console.print(f"[cyan]Сводный отчёт:[/cyan] {summary_path}")


if __name__ == "__main__":
    app()
