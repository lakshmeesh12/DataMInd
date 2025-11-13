# reset_all.py
import asyncio
import logging
import os
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "boq_db")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test12345")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "boq_chunks")

# === CLIENTS ===
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[MONGO_DB]

neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


async def clear_mongodb():
    logger.info("Clearing MongoDB...")
    collections = await db.list_collection_names()
    for coll in collections:
        await db[coll].delete_many({})
        logger.info(f"  → Cleared collection: {coll}")
    logger.info("MongoDB cleared.")


async def clear_neo4j():
    logger.info("Clearing Neo4j...")
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
        await session.run("DROP CONSTRAINT document_id IF EXISTS")
        await session.run("DROP CONSTRAINT item_name IF EXISTS")
        await session.run("DROP CONSTRAINT sheet_id IF EXISTS")
        await session.run("DROP CONSTRAINT table_id IF EXISTS")
        await session.run("DROP INDEX item_type IF EXISTS")
        await session.run("DROP INDEX currency_index IF EXISTS")
    logger.info("Neo4j cleared (nodes, relationships, constraints).")


async def clear_qdrant():
    logger.info("Clearing Qdrant...")
    try:
        qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION)
        logger.info(f"  → Deleted collection: {QDRANT_COLLECTION}")
    except Exception as e:
        logger.warning(f"  → Collection may not exist: {e}")

    # Recreate collection (optional)
    try:
        from db.qdrant.config import init_qdrant_collection
        await init_qdrant_collection()
        logger.info(f"  → Recreated collection: {QDRANT_COLLECTION}")
    except Exception as e:
        logger.error(f"  → Failed to recreate collection: {e}")


async def main():
    logger.info("STARTING FULL DATA WIPE...")
    await clear_mongodb()
    await clear_neo4j()
    await clear_qdrant()

    mongo_client.close()  # fixed
    await neo4j_driver.close()

    logger.info("ALL DATA WIPED SUCCESSFULLY. READY FOR FRESH INGEST.")



if __name__ == "__main__":
    asyncio.run(main())