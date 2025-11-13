# utils/excel_parser.py
import pandas as pd
from unstructured.partition.xlsx import partition_xlsx
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def extract_tables_from_elements(elements) -> List[Dict]:
    """Extract tables from unstructured elements."""
    tables = []
    for el in elements:
        if el.category == "Table":
            # Try to convert HTML table to DataFrame
            try:
                df = pd.read_html(el.metadata.text_as_html)[0]
                tables.append({
                    "start_row": el.metadata.coordinates[0] if el.metadata.coordinates else None,
                    "data": df.fillna("").values.tolist(),
                    "columns": df.columns.tolist(),
                    "html": el.metadata.text_as_html
                })
            except Exception as e:
                logger.warning(f"Failed to parse table HTML: {e}")
                tables.append({
                    "data": [["[Failed to parse table]"]],
                    "columns": [],
                    "error": str(e)
                })
    return tables

def extract_text_regions(elements) -> List[Dict]:
    text_chunks = []
    current = []
    for el in elements:
        if el.category in ["NarrativeText", "Title", "ListItem"]:
            current.append(el.text.strip())
        else:
            if current:
                text_chunks.append({
                    "type": "text",
                    "content": "\n".join(current)
                })
                current = []
    if current:
        text_chunks.append({
            "type": "text",
            "content": "\n".join(current)
        })
    return text_chunks

async def parse_excel_file(file_path: str, workbook_id: str) -> Dict[str, Any]:
    try:
        logger.info(f"Loading file: {file_path}")  # ← NOW LOGS REAL PATH
        elements = partition_xlsx(filename=file_path)

        # Group elements by sheet
        sheet_map = {}
        for el in elements:
            sheet_name = el.metadata.page_name or "Unknown"
            if sheet_name not in sheet_map:
                sheet_map[sheet_name] = []
            sheet_map[sheet_name].append(el)

        parsed_sheets = []
        for sheet_name, sheet_elements in sheet_map.items():
            tables = extract_tables_from_elements(sheet_elements)
            text_regions = extract_text_regions(sheet_elements)

            parsed_sheets.append({
                "sheet_name": sheet_name,
                "tables": tables,
                "text_regions": text_regions,
                "metadata": {
                    "element_count": len(sheet_elements),
                    "table_count": len(tables),
                    "text_count": len(text_regions)
                }
            })

        return {
            "workbook_id": workbook_id,
            "file_name": file_path.split("/")[-1].split("\\")[-1],  # Handle Windows path
            "sheet_count": len(parsed_sheets),
            "sheets": parsed_sheets,
            "status": "parsed",
            "parsed_at": pd.Timestamp.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        return {
            "workbook_id": workbook_id,
            "file_name": file_path.split("/")[-1].split("\\")[-1],
            "status": "failed",
            "error": str(e),
            "sheets": []
        }