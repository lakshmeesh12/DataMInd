# scripts/create_indexes.py
from db.mongo.config import mongo_session

async def create_indexes():
    async with mongo_session("excel_sheets") as col:
        await col.create_index([("upload_id", 1), ("sheet_name", 1)], unique=True)
    async with mongo_session("excel_uploads") as col:
        await col.create_index("upload_id", unique=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_indexes())