"""
CSV/XLSX as *text*. This is deliberately the cheap version: convert rows
to a readable text table and let it flow through the normal chunk+embed
pipeline like any other document. That's fine for semantic questions
("what does this sheet say about X") but it CANNOT answer computational
questions ("what's the average of column Y", "how many rows where Z>10")
correctly — an LLM reading a text dump of rows will guess, not compute.

That's the Phase 5 problem from the roadmap: real spreadsheet Q&A needs
a query engine (pandas/SQL agent) that runs actual code against the data,
routed to separately from vector search. This loader intentionally does
NOT try to solve that — it just makes spreadsheets retrievable as text,
which is enough for "what columns does this have" / "what's in row 12"
style questions, not aggregate/computed ones.
"""
from __future__ import annotations

from pathlib import Path

SPREADSHEET_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _dataframe_to_text(df, sheet_name: str | None) -> str:
    header = f"Sheet: {sheet_name}\n" if sheet_name else ""
    return header + df.to_string(index=False)


def load_spreadsheet(path: Path) -> tuple[str, dict]:
    import pandas as pd

    metadata = {"source_path": str(path), "filename": path.name, "format": "spreadsheet"}
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        text = _dataframe_to_text(df, sheet_name=None)
        metadata.update(num_rows=len(df), num_columns=len(df.columns), columns=list(df.columns))
        return text, metadata

    # .xlsx / .xls — may have multiple sheets
    sheets = pd.read_excel(path, sheet_name=None)
    blocks, total_rows = [], 0
    for name, df in sheets.items():
        blocks.append(_dataframe_to_text(df, sheet_name=name))
        total_rows += len(df)
    text = "\n\n".join(blocks)
    metadata.update(num_sheets=len(sheets), sheet_names=list(sheets.keys()), num_rows=total_rows)
    return text, metadata
