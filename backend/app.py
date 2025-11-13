from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import asyncio
from models import process_excel_file, init_qdrant_collection
import os

app = FastAPI(title="Excel to MongoDB + Qdrant Vector Store")

@app.on_event("startup")
async def startup_event():
    await init_qdrant_collection()

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload multiple Excel files (.xlsx).
    Each file is parsed, stored in MongoDB, embedded, and upserted to Qdrant.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    for file in files:
        if not file.filename.lower().endswith(".xlsx"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not .xlsx")

        content = await file.read()
        try:
            result = await process_excel_file(content, file.filename)
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

    return {
        "status": "success",
        "processed_files": len(results),
        "details": results
    }