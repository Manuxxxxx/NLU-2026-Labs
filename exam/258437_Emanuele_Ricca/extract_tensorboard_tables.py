from __future__ import annotations

"""Usage:
1) Run training so each experiment writes a TensorBoard folder under runs/<family>_<timestamp>_<name>/tb.
2) If available, save run_config.json next to the tb folder; this script will use it automatically.
3) Execute:
    python extract_tensorboard_tables.py --workspace exam/258437_Emanuele_Ricca --family 1A
    python extract_tensorboard_tables.py --workspace exam/258437_Emanuele_Ricca --family 2A --include-history
4) The script writes CSV and Markdown tables to report/tables/.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError as exc:  # pragma: no cover - handled at runtime
    event_accumulator = None
    _TENSORBOARD_IMPORT_ERROR = exc
else:
    _TENSORBOARD_IMPORT_ERROR = None


PREFIX_RE = re.compile(
    r"^(?P<family>[0-9][AB])_(?P<timestamp>\d{8}_\d{6})_(?P<label>.+)$"
)


@dataclass
class SeriesSummary:
    tag: str
    first_step: int | None
    first_value: float | None
    best_step: int | None
    best_value: float | None
    last_step: int | None
    last_value: float | None


@dataclass
class RunSummary:
    family: str
    run_dir: str
    run_name: str
    display_name: str
    config_source: str
    config_text: str
    config_json: str
    primary_metric: str
    primary_best_step: int | None
    primary_best_value: float | None
    test_metrics: str
    results_json: str
    tags: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tables from TensorBoard runs and create report-ready CSV/Markdown files."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root containing the runs/ folder.",
    )
    parser.add_argument(
        "--family",
        default="all",
        help="Run family prefix to include, e.g. 1A or 2A. Use 'all' for every family.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where tables will be written. Defaults to report/tables under the workspace.",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also write a long-format CSV with all scalar series values.",
    )
    return parser.parse_args()


def ensure_tensorboard_available() -> None:
    if event_accumulator is None:
        raise SystemExit(
            "tensorboard is not installed in the active environment. "
            "Install it and run the script again."
        ) from _TENSORBOARD_IMPORT_ERROR


def discover_run_dirs(runs_root: Path, family: str) -> list[Path]:
    run_dirs: list[Path] = []
    if not runs_root.exists():
        return run_dirs

    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if family != "all" and not child.name.startswith(f"{family}_"):
            continue
        if (child / "tb").exists():
            run_dirs.append(child)
    return run_dirs


def humanize_run_name(run_name: str) -> str:
    match = PREFIX_RE.match(run_name)
    label = match.group("label") if match else run_name
    label = label.replace("_+_", " + ")
    label = label.replace("+", " + ")
    label = label.replace("_", " ")
    label = label.replace("( ", "(").replace(" )", ")")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def load_config(run_dir: Path) -> tuple[str, dict[str, Any]]:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            return "run_config.json", json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            pass

    return "run_name", {}


def format_config(config: dict[str, Any], fallback_name: str) -> str:
    if not config:
        return humanize_run_name(fallback_name)

    items = []
    for key in sorted(config):
        value = config[key]
        if isinstance(value, float):
            items.append(f"{key}={value:g}")
        else:
            items.append(f"{key}={value}")
    return "; ".join(items)


def format_config_json(config: dict[str, Any]) -> str:
    if not config:
        return ""
    return json.dumps(config, sort_keys=True)


def _best_by_direction(
    values: list[tuple[int, float]], higher_is_better: bool
) -> tuple[int | None, float | None]:
    if not values:
        return None, None
    best_step, best_value = values[0]
    for step, value in values[1:]:
        if higher_is_better and value > best_value:
            best_step, best_value = step, value
        elif not higher_is_better and value < best_value:
            best_step, best_value = step, value
    return best_step, best_value


def summarize_series(tag: str, values: list[tuple[int, float]]) -> SeriesSummary:
    if not values:
        return SeriesSummary(tag, None, None, None, None, None, None)

    higher_is_better = bool(
        re.search(r"(?:f1|acc|accuracy)$", tag, flags=re.IGNORECASE)
    )
    best_step, best_value = _best_by_direction(
        values, higher_is_better=higher_is_better
    )
    first_step, first_value = values[0]
    last_step, last_value = values[-1]
    return SeriesSummary(
        tag, first_step, first_value, best_step, best_value, last_step, last_value
    )


def extract_scalars(tb_dir: Path) -> dict[str, list[tuple[int, float]]]:
    accumulator = event_accumulator.EventAccumulator(
        str(tb_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()

    series: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        series[tag] = [
            (event.step, float(event.value)) for event in accumulator.Scalars(tag)
        ]
    return series


def select_primary_metric(series: dict[str, list[tuple[int, float]]]) -> str:
    priority = ["slot_f1/dev", "ppl/dev", "loss/dev", "intent_acc/dev"]
    for tag in priority:
        if tag in series:
            return tag

    dev_tags = [tag for tag in series if tag.endswith("/dev")]
    if dev_tags:
        return sorted(dev_tags)[0]

    test_tags = [tag for tag in series if tag.endswith("/test")]
    if test_tags:
        return sorted(test_tags)[0]

    if series:
        return sorted(series)[0]
    return ""


def build_test_metrics(series: dict[str, list[tuple[int, float]]]) -> str:
    entries = []
    for tag in sorted(tag for tag in series if tag.endswith("/test")):
        series_values = series[tag]
        if series_values:
            entries.append(f"{tag}={series_values[-1][1]:.6g}")
    return "; ".join(entries)


def run_summary(
    run_dir: Path, workspace: Path
) -> tuple[RunSummary, list[SeriesSummary]]:
    family_match = PREFIX_RE.match(run_dir.name)
    family = family_match.group("family") if family_match else "unknown"

    config_source, config = load_config(run_dir)
    display_name = format_config(config, run_dir.name)
    series = extract_scalars(run_dir / "tb")

    primary_metric = select_primary_metric(series)
    primary_series = series.get(primary_metric, [])
    primary_summary = summarize_series(primary_metric, primary_series)
    test_metrics = build_test_metrics(series)
    results_json = json.dumps(
        {
            "primary_metric": primary_metric,
            "primary_best_step": primary_summary.best_step,
            "primary_best_value": primary_summary.best_value,
            "test_metrics": {
                tag: values[-1][1]
                for tag, values in series.items()
                if tag.endswith("/test") and values
            },
        },
        sort_keys=True,
    )

    summary = RunSummary(
        family=family,
        run_dir=run_dir.relative_to(workspace).as_posix(),
        run_name=run_dir.name,
        display_name=display_name,
        config_source=config_source,
        config_text=format_config(config, run_dir.name),
        config_json=format_config_json(config),
        primary_metric=primary_metric,
        primary_best_step=primary_summary.best_step,
        primary_best_value=primary_summary.best_value,
        test_metrics=test_metrics,
        results_json=results_json,
        tags="; ".join(sorted(series)),
    )

    histories = [
        summarize_series(tag, values) for tag, values in sorted(series.items())
    ]
    return summary, histories


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| " + " | ".join(fmt(row.get(name, "")) for name in fieldnames) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_tensorboard_available()

    workspace = args.workspace.resolve()
    runs_root = workspace / "runs"
    output_dir = args.output_dir or (workspace / "report" / "tables")
    output_dir = output_dir.resolve()

    run_dirs = discover_run_dirs(runs_root, args.family)
    if not run_dirs:
        raise SystemExit(f"No runs found under {runs_root} for family={args.family!r}.")

    summaries: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        summary, series_summaries = run_summary(run_dir, workspace)
        summaries.append(asdict(summary))
        if args.include_history:
            for series in series_summaries:
                history_rows.append(
                    asdict(series)
                    | {"run_name": run_dir.name, "family": summary.family}
                )

    summary_fieldnames = [
        "family",
        "run_dir",
        "run_name",
        "display_name",
        "config_source",
        "config_text",
        "config_json",
        "primary_metric",
        "primary_best_step",
        "primary_best_value",
        "test_metrics",
        "results_json",
        "tags",
    ]

    summary_csv = output_dir / f"{args.family}_summary.csv"
    summary_md = output_dir / f"{args.family}_summary.md"
    write_csv(summary_csv, summaries, summary_fieldnames)
    summary_md.write_text(
        markdown_table(summaries, summary_fieldnames) + "\n", encoding="utf-8"
    )

    if args.include_history:
        history_fieldnames = [
            "family",
            "run_name",
            "tag",
            "first_step",
            "first_value",
            "best_step",
            "best_value",
            "last_step",
            "last_value",
        ]
        history_csv = output_dir / f"{args.family}_history.csv"
        write_csv(history_csv, history_rows, history_fieldnames)

    print(f"Saved summary to {summary_csv}")
    print(f"Saved markdown to {summary_md}")
    if args.include_history:
        print(f"Saved history to {output_dir / f'{args.family}_history.csv'}")
    print(f"Processed {len(run_dirs)} runs.")


if __name__ == "__main__":
    main()
