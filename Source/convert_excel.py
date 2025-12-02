"""
Utility to convert OCR JSON exports into tabular CSV/Excel files.

Usage:
    python convert_excel.py path/to/output.json --book-code LSE_001 --volume 4 --page 1

The script produces <book_code>_<volume>.csv (and optionally .xlsx) containing
the columns required by the downstream cataloging workflow.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_BASE_BOOK_CODE = "BOOK_BASE"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OCR JSON to CSV/Excel with prescribed columns."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        help="Path to the OCR JSON file (e.g. output/giam_cuong_7_res.json).",
    )
    parser.add_argument(
        "book_code_arg",
        nargs="?",
        help="Optional positional book code (e.g. LSE_001) when you want to supply only two arguments.",
    )
    parser.add_argument(
        "--book-code",
        type=str,
        help="Book code prefix (e.g. LSE_001). Defaults to the inferred name before the trailing number.",
    )
    parser.add_argument(
        "--base-book-code",
        type=str,
        default=DEFAULT_BASE_BOOK_CODE,
        help="Base book code used when neither --book-code nor inference provide a value.",
    )
    parser.add_argument(
        "--volume",
        type=int,
        help="Volume/episode number. Defaults to the trailing number inferred from the input filename.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number within the volume. Defaults to 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to store the generated files. Defaults to ./output.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Keep OCR entries even if the recognized text is empty.",
    )
    parser.add_argument(
        "--write-excel",
        action="store_true",
        help="Also emit an .xlsx file (requires pandas + openpyxl).",
    )
    parser.add_argument(
        "--volume-pad",
        type=int,
        default=3,
        help="Zero-padding width for the volume component. Defaults to 3 (e.g. 004).",
    )
    parser.add_argument(
        "--page-pad",
        type=int,
        default=3,
        help="Zero-padding width for the page component. Defaults to 3 (e.g. 001).",
    )
    parser.add_argument(
        "--column-pad",
        type=int,
        default=2,
        help="Zero-padding width for the column counter. Defaults to 2 (e.g. 01).",
    )
    return parser.parse_args(argv)


def ensure_json_path(args: argparse.Namespace) -> Path:
    """
    Allow running the script without CLI args (e.g. VSCode 'Run Python File'),
    by interactively requesting the JSON path if it wasn't provided.
    """
    if args.json_path is not None:
        return args.json_path

    try:
        user_input = input(
            "Enter path to OCR JSON (e.g. output/giam_cuong_7_res.json): "
        ).strip()
    except EOFError:  # pragma: no cover
        user_input = ""

    if not user_input:
        sys.exit("JSON path is required to proceed.")
    resolved = Path(user_input)
    args.json_path = resolved
    return resolved


def infer_book_and_volume(
    json_data: Dict[str, Any], json_path: Path
) -> Tuple[str, int]:
    """
    Attempt to infer the book code and volume/episode number from the input path.
    Falls back to the json filename if input_path is missing.
    """
    candidate_path = json_data.get("input_path") or json_path.stem
    candidate_stem = Path(candidate_path).stem
    parts = candidate_stem.split("_")
    volume = 1
    if parts and parts[-1].isdigit():
        volume = int(parts[-1])
        book_code = "_".join(parts[:-1]) or candidate_stem
    else:
        book_code = candidate_stem
    return book_code, volume


def build_id(
    book_code: str,
    volume: int,
    page: int,
    column_idx: int,
    volume_pad: int,
    page_pad: int,
    column_pad: int,
) -> str:
    volume_part = str(volume).zfill(volume_pad)
    page_part = str(page).zfill(page_pad)
    column_part = str(column_idx).zfill(column_pad)
    return f"{book_code}.{volume_part}.{page_part}.{column_part}"


def load_rows(
    json_data: Dict[str, Any],
    book_code: str,
    volume: int,
    page: int,
    include_empty: bool,
    volume_pad: int,
    page_pad: int,
    column_pad: int,
) -> List[Dict[str, Any]]:
    rec_texts: Sequence[str] = json_data.get("rec_texts", [])
    rec_polys: Sequence[Any] = json_data.get("rec_polys", [])
    rows: List[Dict[str, Any]] = []
    image_name = f"{book_code}_page_{str(page).zfill(page_pad)}.png"

    for idx, text in enumerate(rec_texts, start=1):
        cleaned = text.strip() if isinstance(text, str) else ""
        if not cleaned and not include_empty:
            continue
        box = rec_polys[idx - 1] if idx - 1 < len(rec_polys) else None
        row = {
            "ID": build_id(
                book_code, volume, page, idx, volume_pad, page_pad, column_pad
            ),
            "Image_name": image_name,
            "Han Char": cleaned,
            "Image Box": json.dumps(box, ensure_ascii=False) if box else "",
        }
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], columns: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(rows: List[Dict[str, Any]], columns: Sequence[str], output_path: Path) -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        print(
            f"[WARN] pandas is required to write Excel files ({exc}). Skipping .xlsx export.",
            file=sys.stderr,
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_excel(output_path, index=False)


def main() -> None:
    args = parse_args()
    json_path: Path = ensure_json_path(args)
    if not json_path.exists():
        sys.exit(f"Input JSON not found: {json_path}")

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    inferred_book, inferred_volume = infer_book_and_volume(json_data, json_path)

    book_code = (
        args.book_code
        or args.book_code_arg
        or inferred_book
        or args.base_book_code
    )
    volume = args.volume if args.volume is not None else inferred_volume
    page = args.page

    rows = load_rows(
        json_data,
        book_code=book_code,
        volume=volume,
        page=page,
        include_empty=args.include_empty,
        volume_pad=args.volume_pad,
        page_pad=args.page_pad,
        column_pad=args.column_pad,
    )

    if not rows:
        print("No rows generated (all OCR entries were empty).", file=sys.stderr)
        return

    base_name = f"{book_code}_{str(volume).zfill(args.volume_pad)}"
    output_dir = args.output_dir
    csv_path = output_dir / f"{base_name}.csv"
    write_csv(rows, ["ID", "Image_name", "Han Char", "Image Box"], csv_path)

    if args.write_excel:
        excel_path = output_dir / f"{base_name}.xlsx"
        write_excel(rows, ["ID", "Image_name", "Han Char", "Image Box"], excel_path)

    print(f"CSV written to {csv_path}")
    if args.write_excel:
        print(f"Excel written to {excel_path}")


if __name__ == "__main__":
    main()

#python convert_excel.py output\giam_cuong_7_res.json --book-code LSE_001 --volume 4 --page 1 --write-excel
#python convert_excel.py output\giam_cuong_7_res.json --volume 4 --page 1 --write-excel --base-book-code LSE_001