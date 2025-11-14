# visual.py
import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from neo4j import AsyncGraphDatabase
import asyncio

logger = logging.getLogger(__name__)

# === CONFIG ===
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_1 = os.getenv("QDRANT_COLLECTION_1", "excel_documents")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test12345")

qdrant_client = QdrantClient(url=QDRANT_URL)
neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ====================== 1. FIND DOCUMENT IN QDRANT ======================
async def get_document_from_qdrant(doc_id: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
    if not doc_id and not filename:
        raise HTTPException(status_code=400, detail="Either doc_id or filename is required")

    filters = []
    if doc_id:
        filters.append(FieldCondition(key="_id", match=MatchValue(value=doc_id)))
    if filename:
        filters.append(FieldCondition(key="filename", match=MatchValue(value=filename)))

    try:
        results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_1,
            scroll_filter=Filter(must=filters) if filters else None,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0]
        if not points:
            raise HTTPException(status_code=404, detail="Document not found in vector DB")
        return points[0].payload
    except Exception as e:
        logger.error(f"Qdrant lookup failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


# ====================== 2. NEO4J VISUALIZATION QUERIES ======================
async def get_project_cost_breakdown(doc_id: str) -> Dict:
    query = """
    MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(s)-[:CONTAINS_TABLE]->(t)-[:HAS_ITEM]->(i)
    OPTIONAL MATCH (i)-[:HAS_AMOUNT]->(a:Number)
    WITH i, COALESCE(a.value, 0) AS amount
    WITH i.description AS item, SUM(amount) AS total
    RETURN item, total
    ORDER BY total DESC
    LIMIT 20
    """
    return await _run_neo4j_query(query, doc_id=doc_id, label="Top 20 Cost Items")


async def get_cost_by_section(doc_id: str) -> Dict:
    query = """
    MATCH (d:Document {id: $doc_id})
    OPTIONAL MATCH (d)-[:CONTAINS_SHEET]->(s)-[:CONTAINS_TABLE]->(t)-[:HAS_ITEM]->(i)-[:HAS_AMOUNT]->(a:Number)
    WITH i, a, s.name AS sheet
    WITH 
      CASE 
        WHEN i.description =~ '(?i).*flooring.*' THEN 'Flooring'
        WHEN i.description =~ '(?i).*civil.*|brick|concrete' THEN 'Civil Works'
        WHEN i.description =~ '(?i).*electrical.*' THEN 'Electrical'
        WHEN i.description =~ '(?i).*plumbing.*' THEN 'Plumbing'
        ELSE 'Others'
      END AS section,
      COALESCE(a.value, 0) AS cost
    WITH section, SUM(cost) AS total
    RETURN section, total
    ORDER BY total DESC
    """
    return await _run_neo4j_query(query, doc_id=doc_id, label="Cost by Section")


async def get_item_type_distribution(doc_id: str) -> Dict:
    query = """
    MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(s)-[:CONTAINS_TABLE]->(t)-[:HAS_ITEM]->(i)
    WHERE i.item_type IS NOT NULL
    RETURN i.item_type AS type, COUNT(*) AS count
    ORDER BY count DESC
    """
    return await _run_neo4j_query(query, doc_id=doc_id, label="Item Type Distribution")


async def get_top_expensive_items(doc_id: str) -> Dict:
    query = """
    MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(s)-[:CONTAINS_TABLE]->(t)-[:HAS_ITEM]->(i)-[:HAS_AMOUNT]->(a:Number)
    RETURN i.description AS item, a.value AS cost, s.name AS sheet
    ORDER BY cost DESC
    LIMIT 10
    """
    return await _run_neo4j_query(query, doc_id=doc_id, label="Top 10 Expensive Items")


async def get_knowledge_graph_structure(doc_id: str) -> Dict:
    """Return nodes & edges for Cytoscape/D3 graph"""
    nodes = []
    edges = []

    async with neo4j_driver.session() as session:
        # Document
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})
            RETURN d.id AS id, d.filename AS label, 'Document' AS type
        """, doc_id=doc_id)
        async for rec in result:
            nodes.append({"data": {"id": rec["id"], "label": rec["label"], "type": rec["type"]}})

        # Sheets
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(s:Sheet)
            RETURN s.id AS id, s.name AS label, 'Sheet' AS type, d.id AS source
        """, doc_id=doc_id)
        async for rec in result:
            nodes.append({"data": {"id": rec["id"], "label": rec["label"], "type": rec["type"]}})
            edges.append({"data": {"source": rec["source"], "target": rec["id"], "label": "contains"}})

        # Tables
        result = await session.run("""
            MATCH (s:Sheet)-[:CONTAINS_TABLE]->(t:Table)
            WHERE EXISTS((:Document {id: $doc_id})-[:CONTAINS_SHEET]->(s))
            RETURN t.id AS id, 'Table' AS label, 'Table' AS type, s.id AS source
        """, doc_id=doc_id)
        async for rec in result:
            nodes.append({"data": {"id": rec["id"], "label": rec["label"], "type": rec["type"]}})
            edges.append({"data": {"source": rec["source"], "target": rec["id"], "label": "contains"}})

        # Items (sample 50)
        result = await session.run("""
            MATCH (t:Table)-[:HAS_ITEM]->(i:ItemInstance)
            WHERE EXISTS((:Document {id: $doc_id})-[:CONTAINS_SHEET]->(:Sheet)-[:CONTAINS_TABLE]->(t))
            RETURN i.id AS id, 
                   COALESCE(i.description, 'Item') AS label, 
                   'Item' AS type, 
                   t.id AS source
            LIMIT 50
        """, doc_id=doc_id)
        async for rec in result:
            nodes.append({"data": {"id": rec["id"], "label": rec["label"][:50], "type": rec["type"]}})
            edges.append({"data": {"source": rec["source"], "target": rec["id"], "label": "has"}})

    return {"nodes": nodes, "edges": edges}


async def _run_neo4j_query(query: str, **params) -> Dict:
    async with neo4j_driver.session() as session:
        result = await session.run(query, **params)
        records = [dict(r) async for r in result]
        return {"data": records}


# ====================== 3. MAIN VISUALIZATION API ======================
async def visualize_document(
    doc_id: Optional[str] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    if not doc_id and not filename:
        raise HTTPException(status_code=400, detail="Provide doc_id or filename")

    # Step 1: Get document metadata
    doc_payload = await get_document_from_qdrant(doc_id=doc_id, filename=filename)
    doc_id = doc_payload["_id"]

    logger.info(f"[VISUAL] Generating visualization for document: {doc_payload['filename']} (ID: {doc_id})")

    # Step 2: Run all visualizations in parallel
    tasks = [
        get_project_cost_breakdown(doc_id),
        get_cost_by_section(doc_id),
        get_item_type_distribution(doc_id),
        get_top_expensive_items(doc_id),
        get_knowledge_graph_structure(doc_id),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any failed task
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.warning(f"Visualization task {i} failed: {res}")
            results[i] = {"data": [], "error": str(res)}

    # Step 3: Assemble response
    return {
        "document": {
            "id": doc_id,
            "filename": doc_payload["filename"],
            "uploaded_at": doc_payload["uploaded_at"],
            "sheet_count": doc_payload.get("sheet_count", 0),
        },
        "visualizations": {
            "cost_breakdown_pie": {
                "type": "pie",
                "title": "Top 20 Item Costs",
                "data": {
                    "labels": [r["item"] for r in results[0].get("data", [])],
                    "datasets": [{
                        "label": "Cost (₹)",
                        "data": [float(r["total"]) for r in results[0].get("data", [])]
                    }]
                }
            },
            "cost_by_section_bar": {
                "type": "bar",
                "title": "Cost Distribution by Section",
                "data": {
                    "labels": [r["section"] for r in results[1].get("data", [])],
                    "datasets": [{
                        "label": "Total Cost",
                        "data": [float(r["total"]) for r in results[1].get("data", [])]
                    }]
                }
            },
            "item_type_doughnut": {
                "type": "doughnut",
                "title": "Item Types in Document",
                "data": {
                    "labels": [r["type"] for r in results[2].get("data", [])],
                    "datasets": [{
                        "label": "Count",
                        "data": [r["count"] for r in results[2].get("data", [])]
                    }]
                }
            },
            "top_expensive_items_bar": {
                "type": "bar",
                "title": "Top 10 Most Expensive Items",
                "data": {
                    "labels": [f"{r['item'][:30]}... ({r['sheet']})" for r in results[3].get("data", [])],
                    "datasets": [{
                        "label": "Cost",
                        "data": [float(r["cost"]) for r in results[3].get("data", [])]
                    }]
                }
            },
            "knowledge_graph": {
                "type": "graph",
                "title": "Document Knowledge Graph",
                "data": results[4] if isinstance(results[4], dict) else {"nodes": [], "edges": []}
            }
        },
        "summary": {
            "total_items": len(results[0].get("data", [])),
            "total_sections": len(results[1].get("data", [])),
            "graph_nodes": len(results[4].get("nodes", []) if isinstance(results[4], dict) else []),
            "graph_edges": len(results[4].get("edges", []) if isinstance(results[4], dict) else []),
        }
    }