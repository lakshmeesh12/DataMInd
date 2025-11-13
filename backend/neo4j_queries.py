# neo4j_queries.py
"""
Knowledge Graph Query Utilities for Neo4j
Provides common Cypher queries for BOQ analysis and multi-hop reasoning
"""

from typing import Dict, Any, List, Optional
from neo4j import AsyncGraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test12345")

driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ==================== BASIC QUERIES ====================

async def get_document_overview(doc_id: str) -> Dict[str, Any]:
    """Get complete overview of a document's knowledge graph."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})
            OPTIONAL MATCH (d)-[:CONTAINS_SHEET]->(s:Sheet)
            OPTIONAL MATCH (s)-[:CONTAINS_TABLE]->(t:Table)
            OPTIONAL MATCH (t)-[:CONTAINS_ITEM]->(i:Item)
            RETURN d.filename as filename,
                   d.uploaded_at as uploaded_at,
                   count(DISTINCT s) as sheet_count,
                   count(DISTINCT t) as table_count,
                   count(DISTINCT i) as item_count
        """, doc_id=doc_id)
        
        record = await result.single()
        if not record:
            return None
        
        return {
            "filename": record["filename"],
            "uploaded_at": record["uploaded_at"],
            "sheet_count": record["sheet_count"],
            "table_count": record["table_count"],
            "item_count": record["item_count"]
        }


async def get_all_items_with_costs(doc_id: str) -> List[Dict[str, Any]]:
    """Get all items with their associated costs from a document."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(:Sheet)
                  -[:CONTAINS_TABLE]->(:Table)-[:CONTAINS_ITEM]->(i:Item)
            OPTIONAL MATCH (i)-[:HAS_COST]->(c:Cost)
            OPTIONAL MATCH (i)-[:HAS_QUANTITY]->(q:Quantity)
            RETURN i.name as item_name,
                   i.item_type as item_type,
                   i.keywords as keywords,
                   c.amount as cost_amount,
                   c.currency as currency,
                   q.amount as quantity,
                   q.unit as unit
            ORDER BY i.name
        """, doc_id=doc_id)
        
        items = []
        async for record in result:
            items.append({
                "item_name": record["item_name"],
                "item_type": record["item_type"],
                "keywords": record["keywords"],
                "cost": {
                    "amount": record["cost_amount"],
                    "currency": record["currency"]
                } if record["cost_amount"] else None,
                "quantity": {
                    "amount": record["quantity"],
                    "unit": record["unit"]
                } if record["quantity"] else None
            })
        
        return items


# ==================== AGGREGATION QUERIES ====================

async def aggregate_costs_by_item_type(doc_id: str) -> List[Dict[str, Any]]:
    """Aggregate total costs grouped by item type (e.g., concrete, steel, masonry)."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(:Sheet)
                  -[:CONTAINS_TABLE]->(:Table)-[:CONTAINS_ITEM]->(i:Item)
            MATCH (i)-[:HAS_COST]->(c:Cost)
            WHERE c.currency = 'INR'
            RETURN i.item_type as item_type,
                   count(i) as item_count,
                   sum(c.amount) as total_cost,
                   avg(c.amount) as avg_cost,
                   min(c.amount) as min_cost,
                   max(c.amount) as max_cost
            ORDER BY total_cost DESC
        """, doc_id=doc_id)
        
        aggregations = []
        async for record in result:
            aggregations.append({
                "item_type": record["item_type"],
                "item_count": record["item_count"],
                "total_cost": record["total_cost"],
                "avg_cost": record["avg_cost"],
                "min_cost": record["min_cost"],
                "max_cost": record["max_cost"],
                "currency": "INR"
            })
        
        return aggregations


async def get_total_project_cost(doc_id: str, currency: str = "INR") -> Dict[str, Any]:
    """Calculate total project cost from all items."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(:Sheet)
                  -[:CONTAINS_TABLE]->(:Table)-[:CONTAINS_ITEM]->(i:Item)
            MATCH (i)-[:HAS_COST]->(c:Cost {currency: $currency})
            RETURN sum(c.amount) as total_cost,
                   count(DISTINCT i) as item_count,
                   avg(c.amount) as avg_item_cost
        """, doc_id=doc_id, currency=currency)
        
        record = await result.single()
        
        return {
            "total_cost": record["total_cost"] or 0,
            "currency": currency,
            "item_count": record["item_count"],
            "avg_item_cost": record["avg_item_cost"] or 0
        }


# ==================== MULTI-HOP REASONING QUERIES ====================

async def find_similar_items_across_sheets(item_name: str, doc_id: str) -> List[Dict[str, Any]]:
    """Find similar items across different sheets (multi-hop reasoning)."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (i1:Item {name: $item_name, document_id: $doc_id})
            MATCH (i2:Item {document_id: $doc_id})
            WHERE i1.item_type = i2.item_type AND i1 <> i2
            OPTIONAL MATCH (i1)-[:HAS_COST]->(c1:Cost)
            OPTIONAL MATCH (i2)-[:HAS_COST]->(c2:Cost)
            RETURN i2.name as similar_item,
                   i2.item_type as item_type,
                   c1.amount as original_cost,
                   c2.amount as similar_cost,
                   c2.currency as currency,
                   abs(c1.amount - c2.amount) as cost_difference
            ORDER BY cost_difference
            LIMIT 10
        """, item_name=item_name, doc_id=doc_id)
        
        similar_items = []
        async for record in result:
            similar_items.append({
                "similar_item": record["similar_item"],
                "item_type": record["item_type"],
                "original_cost": record["original_cost"],
                "similar_cost": record["similar_cost"],
                "currency": record["currency"],
                "cost_difference": record["cost_difference"]
            })
        
        return similar_items


async def find_cost_outliers(doc_id: str, std_dev_threshold: float = 2.0) -> List[Dict[str, Any]]:
    """Find items with costs that are statistical outliers (multi-hop analysis)."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(:Sheet)
                  -[:CONTAINS_TABLE]->(:Table)-[:CONTAINS_ITEM]->(i:Item)
            MATCH (i)-[:HAS_COST]->(c:Cost {currency: 'INR'})
            WITH i.item_type as item_type,
                 avg(c.amount) as avg_cost,
                 stdev(c.amount) as std_dev,
                 collect({item: i.name, cost: c.amount}) as items
            UNWIND items as item_data
            WITH item_type, avg_cost, std_dev, item_data.item as item_name, item_data.cost as cost
            WHERE abs(cost - avg_cost) > ($threshold * std_dev)
            RETURN item_type,
                   item_name,
                   cost,
                   avg_cost,
                   std_dev,
                   (cost - avg_cost) / std_dev as z_score
            ORDER BY abs(z_score) DESC
        """, doc_id=doc_id, threshold=std_dev_threshold)
        
        outliers = []
        async for record in result:
            outliers.append({
                "item_type": record["item_type"],
                "item_name": record["item_name"],
                "cost": record["cost"],
                "avg_cost": record["avg_cost"],
                "std_dev": record["std_dev"],
                "z_score": record["z_score"],
                "currency": "INR"
            })
        
        return outliers


async def compare_documents(doc_id1: str, doc_id2: str) -> Dict[str, Any]:
    """Compare items and costs between two documents (cross-document reasoning)."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d1:Document {id: $doc_id1})-[:CONTAINS_SHEET]->(:Sheet)
                  -[:CONTAINS_TABLE]->(:Table)-[:CONTAINS_ITEM]->(i1:Item)
            MATCH (d2:Document {id: $doc_id2})-[:CONTAINS_SHEET]->(:Sheet)
                  -[:CONTAINS_TABLE]->(:Table)-[:CONTAINS_ITEM]->(i2:Item)
            WHERE i1.item_type = i2.item_type
            OPTIONAL MATCH (i1)-[:HAS_COST]->(c1:Cost)
            OPTIONAL MATCH (i2)-[:HAS_COST]->(c2:Cost)
            RETURN i1.item_type as item_type,
                   count(DISTINCT i1) as doc1_count,
                   count(DISTINCT i2) as doc2_count,
                   avg(c1.amount) as doc1_avg_cost,
                   avg(c2.amount) as doc2_avg_cost,
                   avg(c1.amount) - avg(c2.amount) as cost_difference
            ORDER BY abs(cost_difference) DESC
        """, doc_id1=doc_id1, doc_id2=doc_id2)
        
        comparisons = []
        async for record in result:
            comparisons.append({
                "item_type": record["item_type"],
                "doc1_item_count": record["doc1_count"],
                "doc2_item_count": record["doc2_count"],
                "doc1_avg_cost": record["doc1_avg_cost"],
                "doc2_avg_cost": record["doc2_avg_cost"],
                "cost_difference": record["cost_difference"],
                "currency": "INR"
            })
        
        return {
            "doc1_id": doc_id1,
            "doc2_id": doc_id2,
            "comparisons": comparisons
        }


# ==================== PATH FINDING QUERIES ====================

async def find_item_dependency_path(doc_id: str, item_type: str) -> List[Dict[str, Any]]:
    """Find relationships between items (e.g., items that appear in same tables)."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})-[:CONTAINS_SHEET]->(s:Sheet)
                  -[:CONTAINS_TABLE]->(t:Table)-[:CONTAINS_ITEM]->(i:Item {item_type: $item_type})
            MATCH (t)-[:CONTAINS_ITEM]->(related:Item)
            WHERE i <> related
            RETURN i.name as source_item,
                   related.name as related_item,
                   related.item_type as related_type,
                   s.name as sheet_name,
                   count(*) as co_occurrence_count
            ORDER BY co_occurrence_count DESC
            LIMIT 20
        """, doc_id=doc_id, item_type=item_type)
        
        paths = []
        async for record in result:
            paths.append({
                "source_item": record["source_item"],
                "related_item": record["related_item"],
                "related_type": record["related_type"],
                "sheet_name": record["sheet_name"],
                "co_occurrence_count": record["co_occurrence_count"]
            })
        
        return paths


# ==================== SEARCH QUERIES ====================

async def search_items_by_keyword(keyword: str, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search items by keyword in their description or keywords."""
    async with driver.session() as session:
        query = """
            MATCH (i:Item)
            WHERE toLower(i.name) CONTAINS toLower($keyword)
               OR toLower(i.keywords) CONTAINS toLower($keyword)
        """
        
        if doc_id:
            query += " AND i.document_id = $doc_id"
        
        query += """
            OPTIONAL MATCH (i)-[:HAS_COST]->(c:Cost)
            OPTIONAL MATCH (i)-[:HAS_QUANTITY]->(q:Quantity)
            RETURN i.name as item_name,
                   i.item_type as item_type,
                   i.document_id as document_id,
                   c.amount as cost,
                   c.currency as currency,
                   q.amount as quantity,
                   q.unit as unit
            LIMIT 50
        """
        
        params = {"keyword": keyword}
        if doc_id:
            params["doc_id"] = doc_id
        
        result = await session.run(query, **params)
        
        items = []
        async for record in result:
            items.append({
                "item_name": record["item_name"],
                "item_type": record["item_type"],
                "document_id": record["document_id"],
                "cost": {
                    "amount": record["cost"],
                    "currency": record["currency"]
                } if record["cost"] else None,
                "quantity": {
                    "amount": record["quantity"],
                    "unit": record["unit"]
                } if record["quantity"] else None
            })
        
        return items


# ==================== UTILITY FUNCTIONS ====================

async def delete_document_graph(doc_id: str) -> int:
    """Delete all nodes and relationships for a document."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {id: $doc_id})
            OPTIONAL MATCH (d)-[*]->(n)
            DETACH DELETE d, n
            RETURN count(*) as deleted_count
        """, doc_id=doc_id)
        
        record = await result.single()
        return record["deleted_count"]


async def get_graph_statistics() -> Dict[str, Any]:
    """Get overall Neo4j graph statistics."""
    async with driver.session() as session:
        result = await session.run("""
            MATCH (n)
            RETURN 
                count(DISTINCT CASE WHEN 'Document' IN labels(n) THEN n END) as documents,
                count(DISTINCT CASE WHEN 'Sheet' IN labels(n) THEN n END) as sheets,
                count(DISTINCT CASE WHEN 'Table' IN labels(n) THEN n END) as tables,
                count(DISTINCT CASE WHEN 'Item' IN labels(n) THEN n END) as items,
                count(DISTINCT CASE WHEN 'Cost' IN labels(n) THEN n END) as costs,
                count(DISTINCT CASE WHEN 'Quantity' IN labels(n) THEN n END) as quantities
        """)
        
        record = await result.single()
        
        return {
            "documents": record["documents"],
            "sheets": record["sheets"],
            "tables": record["tables"],
            "items": record["items"],
            "costs": record["costs"],
            "quantities": record["quantities"]
        }


async def close_driver():
    """Close Neo4j driver."""
    await driver.close()