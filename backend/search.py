# search.py - PRODUCTION BOQ RAG AGENT
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from fastapi import HTTPException
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from db.qdrant.config import qdrant_client, QDRANT_COLLECTION, EMBEDDING_MODEL
from db.neo4j.config import driver as neo4j_driver
from db.mongo.config import db as mongo_db

logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ========================================
# 1. QUERY INTENT CLASSIFICATION
# ========================================

class QueryIntent(Enum):
    """User's intent - determines retrieval strategy"""
    COST_TOTAL = "cost_total"           # "What is total cost of X?"
    COST_ITEM = "cost_item"             # "What is rate of Y?"
    QUANTITY = "quantity"               # "How many doors?"
    LIST_FILTER = "list_filter"         # "Show all items under 50k"
    COMPARISON = "comparison"           # "Compare X vs Y"
    STRUCTURE = "structure"             # "What sheets exist?"
    SUMMARY = "summary"                 # "Summarize document"
    OPEN_ENDED = "open_ended"           # General question

class QueryClassifier:
    """Classify user intent using lightweight rules + LLM fallback"""
    
    PATTERNS = {
        QueryIntent.COST_TOTAL: [
            r'total\s+cost',
            r'overall\s+cost',
            r'sum\s+of',
            r'how\s+much\s+for\s+all',
            r'complete\s+cost',
        ],
        QueryIntent.COST_ITEM: [
            r'cost\s+of\s+\w+',
            r'price\s+of\s+\w+',
            r'rate\s+of\s+\w+',
            r'how\s+much\s+is\s+\w+',
            r'what\s+is\s+the\s+(cost|price|rate)',
        ],
        QueryIntent.QUANTITY: [
            r'how\s+many',
            r'quantity\s+of',
            r'number\s+of',
            r'count\s+of',
            r'total\s+\w+\s+quantity',
        ],
        QueryIntent.LIST_FILTER: [
            r'(list|show|display|find)\s+all',
            r'items\s+(under|over|above|below)',
            r'filter\s+by',
            r'where\s+.+\s+(is|are)',
        ],
        QueryIntent.COMPARISON: [
            r'compare',
            r'difference\s+between',
            r'vs\.|versus',
            r'which\s+is\s+(cheaper|expensive|better)',
        ],
        QueryIntent.STRUCTURE: [
            r'what\s+sheets',
            r'list\s+(sheets|tables)',
            r'show\s+structure',
            r'document\s+contains',
        ],
        QueryIntent.SUMMARY: [
            r'summarize',
            r'overview',
            r'give\s+me\s+summary',
            r'what\s+is\s+this\s+document\s+about',
        ],
    }
    
    @classmethod
    async def classify(cls, query: str) -> Tuple[QueryIntent, Dict[str, Any]]:
        """
        Returns: (intent, extracted_entities)
        """
        q_lower = query.lower().strip()
        
        # Rule-based classification
        for intent, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    entities = cls._extract_entities(query, intent)
                    logger.info(f"Classified as {intent.value} (rule-based)")
                    return intent, entities
        
        # LLM fallback for complex queries
        intent, entities = await cls._llm_classify(query)
        logger.info(f"Classified as {intent.value} (LLM)")
        return intent, entities
    
    @classmethod
    def _extract_entities(cls, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """Extract keywords, numbers, comparisons from query"""
        entities = {
            "keywords": [],
            "numbers": [],
            "units": [],
            "operators": None
        }
        
        # Extract numbers (₹50,000 → 50000)
        num_pattern = r'[₹$]\s*[\d,]+(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?'
        entities["numbers"] = [
            float(re.sub(r'[₹$,]', '', n)) 
            for n in re.findall(num_pattern, query)
        ]
        
        # Extract comparison operators
        if intent == QueryIntent.LIST_FILTER:
            if re.search(r'under|below|less than|<', query.lower()):
                entities["operators"] = "lt"
            elif re.search(r'over|above|greater than|more than|>', query.lower()):
                entities["operators"] = "gt"
        
        # Extract keywords (nouns/adjectives)
        # Simple extraction - you can use spaCy for better NER
        words = re.findall(r'\b[a-z]{3,}\b', query.lower())
        stopwords = {'what', 'how', 'many', 'cost', 'total', 'show', 'list', 'the', 'and', 'for', 'are', 'is'}
        entities["keywords"] = [w for w in words if w not in stopwords]
        
        return entities
    
    @classmethod
    async def _llm_classify(cls, query: str) -> Tuple[QueryIntent, Dict[str, Any]]:
        """LLM-based classification for complex queries"""
        prompt = f"""Classify this BOQ query into ONE intent:

Query: "{query}"

Intents:
1. cost_total - Total cost of category/group
2. cost_item - Cost/rate of specific item
3. quantity - How many items/count
4. list_filter - Filter/list items by criteria
5. comparison - Compare two or more things
6. structure - Document structure questions
7. summary - Summarize document
8. open_ended - Other

Respond ONLY with JSON:
{{"intent": "...", "keywords": ["..."], "numbers": [...], "operators": null}}"""

        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",  # Fast classification
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            result = json.loads(resp.choices[0].message.content.strip())
            intent = QueryIntent(result["intent"])
            entities = {
                "keywords": result.get("keywords", []),
                "numbers": result.get("numbers", []),
                "operators": result.get("operators")
            }
            return intent, entities
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return QueryIntent.OPEN_ENDED, {"keywords": [], "numbers": [], "operators": None}

# ========================================
# 2. EMBEDDING & VECTOR RETRIEVAL
# ========================================

async def embed_query(text: str) -> List[float]:
    """Generate embedding for text"""
    try:
        resp = openai.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding failed")

async def retrieve_chunks(
    query: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = 15
) -> List[Dict[str, Any]]:
    """Semantic search in Qdrant"""
    vector = await embed_query(query)

    filter_conditions = None
    if document_ids:
        if len(document_ids) == 1:
            filter_conditions = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_ids[0]))]
            )
        else:
            filter_conditions = Filter(
                should=[
                    FieldCondition(key="document_id", match=MatchValue(value=doc_id))
                    for doc_id in document_ids
                ]
            )

    hits = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        query_filter=filter_conditions,
        with_payload=True
    )

    return [
        {
            "chunk_id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
            "content": _format_chunk_content(hit.payload)
        }
        for hit in hits
    ]

def _format_chunk_content(payload: Dict) -> str:
    """Format chunk for LLM context"""
    content = f"Sheet: {payload['sheet_name']}\n"
    content += f"Headers: {' | '.join(payload['headers'])}\n"
    content += "Data:\n"
    
    raw_data = payload.get("sample_data", {}).get("raw", [])
    for row in raw_data[:8]:  # Show more rows
        content += " | ".join(str(cell) for cell in row) + "\n"
    
    return content

# ========================================
# 3. NEO4J GRAPH QUERIES (INTENT-DRIVEN)
# ========================================

class GraphQueryEngine:
    """Adaptive Neo4j queries based on intent"""
    
    @staticmethod
    async def query_by_intent(
        intent: QueryIntent,
        entities: Dict[str, Any],
        doc_ids: List[str]
    ) -> Dict[str, Any]:
        """Route to appropriate graph query"""
        handlers = {
            QueryIntent.COST_TOTAL: GraphQueryEngine._cost_total,
            QueryIntent.COST_ITEM: GraphQueryEngine._cost_item,
            QueryIntent.QUANTITY: GraphQueryEngine._quantity,
            QueryIntent.LIST_FILTER: GraphQueryEngine._list_filter,
            QueryIntent.COMPARISON: GraphQueryEngine._comparison,
            QueryIntent.STRUCTURE: GraphQueryEngine._structure,
            QueryIntent.SUMMARY: GraphQueryEngine._summary,
        }
        
        handler = handlers.get(intent, GraphQueryEngine._fallback)
        return await handler(entities, doc_ids)
    
    @staticmethod
    async def _cost_total(entities: Dict, doc_ids: List[str]) -> Dict:
        """Sum costs for category (e.g., 'RESTROOM FINISHES')"""
        keywords = entities["keywords"]
        
        async with neo4j_driver.session() as session:
            result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})-[:CONTAINS_SHEET]->(s:Sheet)
                      -[:CONTAINS_TABLE]->(t:Table)-[:HAS_ITEM]->(ii:ItemInstance)
                      -[:HAS_AMOUNT]->(a:Number)
                WHERE any(kw IN $keywords WHERE toLower(ii.description) CONTAINS toLower(kw))
                   OR any(kw IN $keywords WHERE toLower(s.name) CONTAINS toLower(kw))
                RETURN 
                    ii.description AS item,
                    s.name AS sheet,
                    a.value AS amount,
                    a.currency AS currency,
                    SUM(a.value) OVER () AS total_cost
                ORDER BY a.value DESC
                LIMIT 100
            """, doc_ids=doc_ids, keywords=keywords)
            
            items = []
            total = 0
            async for rec in result:
                items.append({
                    "item": rec["item"],
                    "sheet": rec["sheet"],
                    "amount": rec["amount"],
                    "currency": rec["currency"]
                })
                total = rec["total_cost"]
            
            return {
                "type": "cost_total",
                "total": total,
                "items": items,
                "count": len(items)
            }
    
    @staticmethod
    async def _cost_item(entities: Dict, doc_ids: List[str]) -> Dict:
        """Get rate/cost of specific item"""
        keywords = entities["keywords"]
        
        async with neo4j_driver.session() as session:
            result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})-[:CONTAINS_SHEET]->(s:Sheet)
                      -[:CONTAINS_TABLE]->(t:Table)-[:HAS_ITEM]->(ii:ItemInstance)
                WHERE any(kw IN $keywords WHERE toLower(ii.description) CONTAINS toLower(kw))
                OPTIONAL MATCH (ii)-[:HAS_RATE]->(r:Number)
                OPTIONAL MATCH (ii)-[:HAS_AMOUNT]->(a:Number)
                OPTIONAL MATCH (ii)-[:HAS_QUANTITY]->(q:Number)
                RETURN DISTINCT
                    ii.description AS item,
                    s.name AS sheet,
                    r.value AS rate,
                    r.unit AS rate_unit,
                    a.value AS amount,
                    q.value AS quantity,
                    q.unit AS qty_unit
                ORDER BY r.value DESC
                LIMIT 20
            """, doc_ids=doc_ids, keywords=keywords)
            
            items = []
            async for rec in result:
                items.append({
                    "item": rec["item"],
                    "sheet": rec["sheet"],
                    "rate": rec["rate"],
                    "rate_unit": rec["rate_unit"],
                    "amount": rec["amount"],
                    "quantity": rec["quantity"],
                    "qty_unit": rec["qty_unit"]
                })
            
            return {"type": "cost_item", "items": items}
    
    @staticmethod
    async def _quantity(entities: Dict, doc_ids: List[str]) -> Dict:
        """Count items (e.g., 'How many doors?')"""
        keywords = entities["keywords"]
        
        async with neo4j_driver.session() as session:
            result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})-[:CONTAINS_SHEET]->(s:Sheet)
                      -[:CONTAINS_TABLE]->(t:Table)-[:HAS_ITEM]->(ii:ItemInstance)
                WHERE any(kw IN $keywords WHERE toLower(ii.description) CONTAINS toLower(kw))
                OPTIONAL MATCH (ii)-[:HAS_QUANTITY]->(q:Number)
                RETURN 
                    ii.description AS item,
                    s.name AS sheet,
                    q.value AS quantity,
                    q.unit AS unit,
                    COUNT(ii) AS count
                ORDER BY q.value DESC
                LIMIT 50
            """, doc_ids=doc_ids, keywords=keywords)
            
            items = []
            total_qty = 0
            async for rec in result:
                items.append({
                    "item": rec["item"],
                    "sheet": rec["sheet"],
                    "quantity": rec["quantity"],
                    "unit": rec["unit"],
                    "count": rec["count"]
                })
                if rec["quantity"]:
                    total_qty += rec["quantity"]
            
            return {
                "type": "quantity",
                "total_quantity": total_qty,
                "total_count": len(items),
                "items": items
            }
    
    @staticmethod
    async def _list_filter(entities: Dict, doc_ids: List[str]) -> Dict:
        """Filter items by criteria (e.g., 'under 50k')"""
        keywords = entities["keywords"]
        threshold = entities["numbers"][0] if entities["numbers"] else None
        operator = entities["operators"]
        
        if not threshold:
            # Fallback to keyword search
            return await GraphQueryEngine._cost_item(entities, doc_ids)
        
        # Build dynamic WHERE clause
        op_map = {"lt": "<", "gt": ">", "lte": "<=", "gte": ">="}
        cypher_op = op_map.get(operator, "<")
        
        async with neo4j_driver.session() as session:
            result = await session.run(f"""
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {{id: doc_id}})-[:CONTAINS_SHEET]->(s:Sheet)
                      -[:CONTAINS_TABLE]->(t:Table)-[:HAS_ITEM]->(ii:ItemInstance)
                      -[:HAS_AMOUNT]->(a:Number)
                WHERE a.value {cypher_op} $threshold
                  AND (SIZE($keywords) = 0 OR any(kw IN $keywords WHERE toLower(ii.description) CONTAINS toLower(kw)))
                OPTIONAL MATCH (ii)-[:HAS_QUANTITY]->(q:Number)
                RETURN 
                    ii.description AS item,
                    s.name AS sheet,
                    a.value AS amount,
                    q.value AS quantity,
                    q.unit AS unit
                ORDER BY a.value DESC
                LIMIT 100
            """, doc_ids=doc_ids, threshold=threshold, keywords=keywords)
            
            items = []
            async for rec in result:
                items.append({
                    "item": rec["item"],
                    "sheet": rec["sheet"],
                    "amount": rec["amount"],
                    "quantity": rec["quantity"],
                    "unit": rec["unit"]
                })
            
            return {
                "type": "list_filter",
                "filter": f"{operator} {threshold}",
                "items": items,
                "count": len(items)
            }
    
    @staticmethod
    async def _comparison(entities: Dict, doc_ids: List[str]) -> Dict:
        """Compare multiple items"""
        keywords = entities["keywords"]
        
        async with neo4j_driver.session() as session:
            result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})-[:CONTAINS_SHEET]->(s:Sheet)
                      -[:CONTAINS_TABLE]->(t:Table)-[:HAS_ITEM]->(ii:ItemInstance)
                WHERE any(kw IN $keywords WHERE toLower(ii.description) CONTAINS toLower(kw))
                OPTIONAL MATCH (ii)-[:HAS_RATE]->(r:Number)
                OPTIONAL MATCH (ii)-[:HAS_AMOUNT]->(a:Number)
                OPTIONAL MATCH (ii)-[:HAS_QUANTITY]->(q:Number)
                RETURN 
                    ii.description AS item,
                    s.name AS sheet,
                    r.value AS rate,
                    a.value AS amount,
                    q.value AS quantity,
                    q.unit AS unit
                ORDER BY a.value DESC
                LIMIT 20
            """, doc_ids=doc_ids, keywords=keywords)
            
            items = []
            async for rec in result:
                items.append({
                    "item": rec["item"],
                    "sheet": rec["sheet"],
                    "rate": rec["rate"],
                    "amount": rec["amount"],
                    "quantity": rec["quantity"],
                    "unit": rec["unit"]
                })
            
            return {"type": "comparison", "items": items}
    
    @staticmethod
    async def _structure(entities: Dict, doc_ids: List[str]) -> Dict:
        """Get document structure"""
        async with neo4j_driver.session() as session:
            result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})
                OPTIONAL MATCH (d)-[:CONTAINS_SHEET]->(s:Sheet)
                OPTIONAL MATCH (s)-[:CONTAINS_TABLE]->(t:Table)
                RETURN 
                    d.id AS doc_id,
                    d.filename AS filename,
                    collect(DISTINCT s.name) AS sheets,
                    count(DISTINCT t) AS table_count
            """, doc_ids=doc_ids)
            
            structure = []
            async for rec in result:
                structure.append({
                    "document": rec["filename"],
                    "sheets": rec["sheets"],
                    "table_count": rec["table_count"]
                })
            
            return {"type": "structure", "documents": structure}
    
    @staticmethod
    async def _summary(entities: Dict, doc_ids: List[str]) -> Dict:
        """Get document summary"""
        async with neo4j_driver.session() as session:
            result = await session.run("""
                UNWIND $doc_ids AS doc_id
                MATCH (d:Document {id: doc_id})
                OPTIONAL MATCH (d)-[:CONTAINS_SHEET]->(s:Sheet)
                      -[:CONTAINS_TABLE]->(t:Table)-[:HAS_ITEM]->(ii:ItemInstance)
                OPTIONAL MATCH (ii)-[:HAS_AMOUNT]->(a:Number)
                RETURN 
                    d.filename AS filename,
                    count(DISTINCT s) AS sheet_count,
                    count(DISTINCT t) AS table_count,
                    count(DISTINCT ii) AS item_count,
                    SUM(a.value) AS total_cost,
                    collect(DISTINCT s.name)[..5] AS sample_sheets
            """, doc_ids=doc_ids)
            
            summaries = []
            async for rec in result:
                summaries.append({
                    "filename": rec["filename"],
                    "sheet_count": rec["sheet_count"],
                    "table_count": rec["table_count"],
                    "item_count": rec["item_count"],
                    "total_cost": rec["total_cost"],
                    "sample_sheets": rec["sample_sheets"]
                })
            
            return {"type": "summary", "summaries": summaries}
    
    @staticmethod
    async def _fallback(entities: Dict, doc_ids: List[str]) -> Dict:
        """Fallback: keyword search"""
        return await GraphQueryEngine._cost_item(entities, doc_ids)

# ========================================
# 4. ADAPTIVE LLM SYNTHESIS
# ========================================

async def synthesize_answer(
    query: str,
    intent: QueryIntent,
    chunks: List[Dict[str, Any]],
    graph_data: Dict[str, Any]
) -> str:
    """Generate answer using intent-aware prompting"""
    
    # Build context
    vector_context = "\n\n".join([
        f"[Source {i+1}] {c['content']}" 
        for i, c in enumerate(chunks[:5])
    ])
    
    # Build graph context
    graph_context = _format_graph_data(graph_data)
    
    # Intent-specific prompts
    prompts = {
        QueryIntent.COST_TOTAL: f"""Calculate the TOTAL cost for the queried category.

STRUCTURED DATA (PRIMARY SOURCE):
{graph_context}

VECTOR SEARCH CONTEXT:
{vector_context}

Instructions:
- Sum all amounts in the structured data
- Show breakdown by item
- Format: "Total: ₹X,XXX,XXX.XX\nBreakdown:\n- Item1: ₹Y\n- Item2: ₹Z"
- If no data, say "No cost data found for [category]" """,

        QueryIntent.COST_ITEM: f"""Provide the rate/cost for the specific item.

STRUCTURED DATA:
{graph_context}

VECTOR CONTEXT:
{vector_context}

Instructions:
- Extract rate, unit, amount for the item
- If multiple variants, list all
- Format: "Item: X\nRate: ₹Y per [unit]\nAmount: ₹Z" """,

        QueryIntent.QUANTITY: f"""Count/quantify the items.

STRUCTURED DATA:
{graph_context}

VECTOR CONTEXT:
{vector_context}

Instructions:
- Sum quantities
- List items with their quantities
- Format: "Total: X [units]\nBreakdown:\n- Item1: Y [unit]\n- Item2: Z [unit]" """,

        QueryIntent.LIST_FILTER: f"""Filter and list items matching criteria.

FILTERED RESULTS:
{graph_context}

Instructions:
- List all matching items
- Show amount/quantity for each
- Sort by amount (highest first)
- Format as numbered list """,

        QueryIntent.COMPARISON: f"""Compare the items side-by-side.

COMPARISON DATA:
{graph_context}

Instructions:
- Create comparison table
- Show rate, amount, quantity for each
- Highlight differences
- Format as markdown table """,

        QueryIntent.STRUCTURE: f"""Describe document structure.

STRUCTURE DATA:
{graph_context}

Instructions:
- List all sheets
- Show table/item counts
- Format as bullet points """,

        QueryIntent.SUMMARY: f"""Provide a concise summary.

SUMMARY DATA:
{graph_context}

VECTOR CONTEXT:
{vector_context}

Instructions:
- Total cost, item count
- Major categories
- Key highlights
- 3-5 sentences max """,

        QueryIntent.OPEN_ENDED: f"""Answer the question using available data.

STRUCTURED DATA:
{graph_context}

VECTOR CONTEXT:
{vector_context}

Instructions:
- Use structured data first
- Fall back to vector context
- Be precise, cite sources
- If unsure, say "Based on available data..." """
    }
    
    prompt = prompts.get(intent, prompts[QueryIntent.OPEN_ENDED])
    full_prompt = f"USER QUESTION: {query}\n\n{prompt}\n\nANSWER:"
    
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",  # Use GPT-4 for complex reasoning
            messages=[
                {
                    "role": "system",
                    "content": "You are a BOQ analyst. Always prioritize structured graph data over vector context. Be precise with numbers. Never hallucinate."
                },
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        # Fallback to structured data
        if graph_data and graph_data.get("items"):
            return _format_fallback_answer(graph_data)
        return "Unable to generate answer. Please check the logs."

def _format_graph_data(graph_data: Dict) -> str:
    """Format graph results for LLM context"""
    if not graph_data:
        return "No structured data available."
    
    data_type = graph_data.get("type", "unknown")
    
    if data_type == "cost_total":
        output = f"TOTAL COST: {graph_data['total']:,.2f}\n\n"
        output += f"Items ({graph_data['count']}):\n"
        for item in graph_data["items"][:30]:
            output += f"- {item['item']}: {item['amount']:,.2f} {item.get('currency', 'INR')} (Sheet: {item['sheet']})\n"
    
    elif data_type in ["cost_item", "comparison"]:
        output = "ITEMS:\n"
        for item in graph_data["items"]:
            output += f"\n- {item['item']} (Sheet: {item['sheet']})\n"
            if item.get("rate"):
                output += f"  Rate: {item['rate']:,.2f} per {item.get('rate_unit', 'unit')}\n"
            if item.get("amount"):
                output += f"  Amount: {item['amount']:,.2f}\n"
            if item.get("quantity"):
                output += f"  Quantity: {item['quantity']} {item.get('qty_unit', '')}\n"
    
    elif data_type == "quantity":
        output = f"TOTAL QUANTITY: {graph_data['total_quantity']}\n"
        output += f"TOTAL COUNT: {graph_data['total_count']}\n\n"
        for item in graph_data["items"][:20]:
            output += f"- {item['item']}: {item['quantity']} {item.get('unit', '')} (Sheet: {item['sheet']})\n"
    
    elif data_type == "list_filter":
        output = f"FILTER: {graph_data['filter']}\n"
        output += f"FOUND: {graph_data['count']} items\n\n"
        for item in graph_data["items"][:50]:
            output += f"- {item['item']}: {item['amount']:,.2f} (Sheet: {item['sheet']})\n"
    
    elif data_type == "structure":
        output = "DOCUMENT STRUCTURE:\n"
        for doc in graph_data["documents"]:
            output += f"\n{doc['document']}:\n"
            output += f"- Sheets: {', '.join(doc['sheets'])}\n"
            output += f"- Tables: {doc['table_count']}\n"
    
    elif data_type == "summary":
        output = "DOCUMENT SUMMARY:\n"
        for summ in graph_data["summaries"]:
            output += f"\n{summ['filename']}:\n"
            output += f"- Sheets: {summ['sheet_count']}\n"
            output += f"- Tables: {summ['table_count']}\n"
            output += f"- Items: {summ['item_count']}\n"
            if summ['total_cost']:
                output += f"- Total Cost: {summ['total_cost']:,.2f}\n"
            output += f"- Sample Sheets: {', '.join(summ['sample_sheets'])}\n"
    
    else:
        output = json.dumps(graph_data, indent=2)
    
    return output

def _format_fallback_answer(graph_data: Dict) -> str:
    """Simple fallback if LLM fails"""
    items = graph_data.get("items", [])
    if not items:
        return "No data found."
    
    output = "Found the following:\n"
    for item in items[:10]:
        output += f"- {item.get('item', 'Unknown')}"
        if item.get("amount"):
            output += f": ₹{item['amount']:,.2f}"
        output += "\n"
    
    return output

# ========================================
# 5. MAIN QUERY ENGINE
# ========================================

async def query_boq(
    question: str,
    document_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main entry point - handles ANY BOQ query
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Query: {question} | Docs: {document_ids}")

    # STEP 1: Classify Intent
    intent, entities = await QueryClassifier.classify(question)
    logger.info(f"Intent: {intent.value} | Entities: {entities}")

    # STEP 2: Graph Query (Primary - most accurate)
    graph_data = {}
    if document_ids:
        try:
            graph_data = await GraphQueryEngine.query_by_intent(intent, entities, document_ids)
            logger.info(f"Graph results: {graph_data.get('type')} with {len(graph_data.get('items', []))} items")
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
    
    # STEP 3: Vector Retrieval (Secondary - for context/fallback)
    chunks = []
    try:
        # Adjust top_k based on intent
        top_k = 20 if intent in [QueryIntent.SUMMARY, QueryIntent.STRUCTURE] else 10
        chunks = await retrieve_chunks(question, document_ids, top_k=top_k)
        logger.info(f"Vector results: {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Vector retrieval failed: {e}")

    # STEP 4: Synthesize Answer
    answer = await synthesize_answer(question, intent, chunks, graph_data)

    # STEP 5: Build Response
    sources = [
        {
            "document_id": c["payload"]["document_id"],
            "filename": c["payload"]["filename"],
            "sheet": c["payload"]["sheet_name"],
            "score": round(c["score"], 4),
            "chunk_id": c["chunk_id"]
        }
        for c in chunks[:5]
    ]

    return {
        "answer": answer,
        "intent": intent.value,
        "entities": entities,
        "sources": sources,
        "retrieved_chunks": len(chunks),
        "graph_matches": len(graph_data.get("items", [])),
        "graph_data_type": graph_data.get("type"),
        "metadata": {
            "query": question,
            "document_ids": document_ids,
            "strategy": "intent_based_retrieval"
        }
    }


# ========================================
# 6. ADVANCED FEATURES (Optional)
# ========================================

async def multi_hop_reasoning(
    question: str,
    document_ids: List[str]
) -> Dict[str, Any]:
    """
    For complex queries requiring multiple steps
    Example: "Which is cheaper: steel or concrete per ton?"
    """
    # Decompose query into sub-questions
    decomposition_prompt = f"""Break this complex query into simple sub-queries:

Query: "{question}"

Return JSON array of sub-queries:
{{"sub_queries": ["query1", "query2", ...]}}"""

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": decomposition_prompt}],
            temperature=0.0
        )
        sub_queries = json.loads(resp.choices[0].message.content)["sub_queries"]
        
        # Execute each sub-query
        sub_results = []
        for sq in sub_queries:
            result = await query_boq(sq, document_ids)
            sub_results.append({
                "query": sq,
                "answer": result["answer"]
            })
        
        # Synthesize final answer
        synthesis_prompt = f"""Combine these sub-answers into a final answer:

Original Question: {question}

Sub-Results:
{json.dumps(sub_results, indent=2)}

Final Answer:"""

        final_resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.0
        )
        
        return {
            "answer": final_resp.choices[0].message.content.strip(),
            "sub_queries": sub_results,
            "strategy": "multi_hop_reasoning"
        }
    except Exception as e:
        logger.error(f"Multi-hop reasoning failed: {e}")
        # Fallback to standard query
        return await query_boq(question, document_ids)


async def conversational_query(
    question: str,
    document_ids: List[str],
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Handle follow-up questions with context
    Example:
    User: "What's the cost of cement?"
    Bot: "₹500,000"
    User: "And steel?" <- needs context from previous turn
    """
    # Resolve coreferences using conversation history
    if conversation_history:
        context = "\n".join([
            f"User: {turn['question']}\nBot: {turn['answer']}"
            for turn in conversation_history[-3:]  # Last 3 turns
        ])
        
        resolution_prompt = f"""Given this conversation, resolve the current query:

Conversation:
{context}

Current Query: "{question}"

Resolved Query (standalone):"""

        try:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": resolution_prompt}],
                temperature=0.0
            )
            resolved_query = resp.choices[0].message.content.strip()
            logger.info(f"Resolved query: {resolved_query}")
            
            # Execute resolved query
            return await query_boq(resolved_query, document_ids)
        except Exception as e:
            logger.error(f"Query resolution failed: {e}")
    
    # Fallback to standard query
    return await query_boq(question, document_ids)


async def batch_query(
    questions: List[str],
    document_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Process multiple queries in parallel
    Useful for dashboards/reports
    """
    import asyncio
    
    tasks = [query_boq(q, document_ids) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [
        result if not isinstance(result, Exception) else {"error": str(result)}
        for result in results
    ]


async def explain_reasoning(
    question: str,
    document_ids: List[str]
) -> Dict[str, Any]:
    """
    Return detailed reasoning trace for transparency
    Useful for debugging/auditing
    """
    result = await query_boq(question, document_ids)
    
    # Add reasoning trace
    result["reasoning_trace"] = {
        "step_1_classification": {
            "intent": result["intent"],
            "entities": result["entities"],
            "method": "rule_based + LLM"
        },
        "step_2_graph_query": {
            "type": result["graph_data_type"],
            "matches": result["graph_matches"],
            "cypher_pattern": _get_cypher_pattern(result["intent"])
        },
        "step_3_vector_search": {
            "chunks_retrieved": result["retrieved_chunks"],
            "top_scores": [s["score"] for s in result["sources"][:3]]
        },
        "step_4_synthesis": {
            "model": "gpt-4o",
            "temperature": 0.0,
            "context_tokens": _estimate_tokens(result)
        }
    }
    
    return result

def _get_cypher_pattern(intent: str) -> str:
    """Return Cypher pattern used for given intent"""
    patterns = {
        "cost_total": "MATCH ()-[:HAS_ITEM]->()-[:HAS_AMOUNT]->()",
        "cost_item": "MATCH ()-[:HAS_ITEM]->()-[:HAS_RATE]->()",
        "quantity": "MATCH ()-[:HAS_ITEM]->()-[:HAS_QUANTITY]->()",
        "list_filter": "MATCH ()-[:HAS_ITEM]->()-[:HAS_AMOUNT]->() WHERE a.value < threshold",
    }
    return patterns.get(intent, "Dynamic query")

def _estimate_tokens(result: Dict) -> int:
    """Rough token count estimate"""
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    text = json.dumps(result["metadata"])
    return len(enc.encode(text))


# ========================================
# 7. QUERY VALIDATION & PREPROCESSING
# ========================================

class QueryValidator:
    """Validate and preprocess queries"""
    
    @staticmethod
    async def validate(query: str) -> Tuple[bool, Optional[str]]:
        """
        Returns: (is_valid, error_message)
        """
        # Check length
        if len(query) < 3:
            return False, "Query too short (min 3 characters)"
        
        if len(query) > 1000:
            return False, "Query too long (max 1000 characters)"
        
        # Check for malicious patterns
        malicious = ['DROP', 'DELETE', 'UPDATE', 'INSERT', '<script>', 'EXEC']
        if any(m.lower() in query.lower() for m in malicious):
            return False, "Query contains suspicious patterns"
        
        # Check if query is actually a question/statement
        if not any(c.isalpha() for c in query):
            return False, "Query must contain letters"
        
        return True, None
    
    @staticmethod
    def preprocess(query: str) -> str:
        """Normalize query"""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Fix common typos (optional - can expand)
        replacements = {
            'restrom': 'restroom',
            'electic': 'electric',
            'quantiy': 'quantity',
            'tmt': 'TMT',
            'rcc': 'RCC',
        }
        
        for wrong, right in replacements.items():
            query = re.sub(rf'\b{wrong}\b', right, query, flags=re.IGNORECASE)
        
        return query.strip()


# ========================================
# 8. CACHING LAYER (Optional)
# ========================================

class QueryCache:
    """Simple in-memory cache for repeated queries"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Dict]:
        import time
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache HIT: {key}")
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Dict):
        import time
        self.cache[key] = (value, time.time())
        logger.info(f"Cache SET: {key}")
    
    def _make_key(self, query: str, doc_ids: List[str]) -> str:
        import hashlib
        data = f"{query}:{sorted(doc_ids)}"
        return hashlib.md5(data.encode()).hexdigest()

# Global cache instance
_query_cache = QueryCache(ttl=1800)  # 30 min TTL


# ========================================
# 9. ENHANCED MAIN FUNCTION WITH ALL FEATURES
# ========================================

async def query_boq_enhanced(
    question: str,
    document_ids: Optional[List[str]] = None,
    use_cache: bool = True,
    explain: bool = False,
    conversation_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Enhanced query engine with all features
    
    Args:
        question: User's natural language query
        document_ids: List of document IDs to search (None = all docs)
        use_cache: Enable caching for repeated queries
        explain: Return detailed reasoning trace
        conversation_history: Previous conversation turns for context
    
    Returns:
        Complete answer with sources and metadata
    """
    # 1. Validate
    is_valid, error = await QueryValidator.validate(question)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # 2. Preprocess
    question = QueryValidator.preprocess(question)
    
    # 3. Check cache
    if use_cache and document_ids:
        cache_key = _query_cache._make_key(question, document_ids)
        cached = _query_cache.get(cache_key)
        if cached:
            cached["from_cache"] = True
            return cached
    
    # 4. Handle conversational context
    if conversation_history:
        result = await conversational_query(question, document_ids, conversation_history)
    else:
        result = await query_boq(question, document_ids)
    
    # 5. Add explanation if requested
    if explain:
        result = await explain_reasoning(question, document_ids)
    
    # 6. Cache result
    if use_cache and document_ids:
        _query_cache.set(cache_key, result)
    
    result["from_cache"] = False
    return result


# ========================================
# 10. UTILITY FUNCTIONS
# ========================================

async def get_document_stats(document_ids: List[str]) -> Dict[str, Any]:
    """Get quick stats about documents"""
    async with neo4j_driver.session() as session:
        result = await session.run("""
            UNWIND $doc_ids AS doc_id
            MATCH (d:Document {id: doc_id})
            OPTIONAL MATCH (d)-[:CONTAINS_SHEET]->(s:Sheet)
            OPTIONAL MATCH (s)-[:CONTAINS_TABLE]->(t:Table)
            OPTIONAL MATCH (t)-[:HAS_ITEM]->(ii:ItemInstance)
            OPTIONAL MATCH (ii)-[:HAS_AMOUNT]->(a:Number)
            RETURN 
                d.filename AS filename,
                count(DISTINCT s) AS sheets,
                count(DISTINCT t) AS tables,
                count(DISTINCT ii) AS items,
                SUM(a.value) AS total_cost,
                MIN(a.value) AS min_cost,
                MAX(a.value) AS max_cost,
                AVG(a.value) AS avg_cost
        """, doc_ids=document_ids)
        
        stats = []
        async for rec in result:
            stats.append({
                "filename": rec["filename"],
                "sheets": rec["sheets"],
                "tables": rec["tables"],
                "items": rec["items"],
                "total_cost": rec["total_cost"],
                "min_cost": rec["min_cost"],
                "max_cost": rec["max_cost"],
                "avg_cost": rec["avg_cost"]
            })
        
        return {"documents": stats}


async def suggest_queries(document_ids: List[str]) -> List[str]:
    """Suggest relevant queries based on document content"""
    stats = await get_document_stats(document_ids)
    
    suggestions = [
        "What is the total cost of this project?",
        "Show me the most expensive items",
        "How many items are there in total?",
        "What sheets are in this document?",
        "Summarize this BOQ"
    ]
    
    # Add dynamic suggestions based on content
    if stats["documents"]:
        doc = stats["documents"][0]
        if doc["items"] > 100:
            suggestions.append(f"Show me items under ₹{int(doc['avg_cost'])}")
        suggestions.append(f"What is the average cost per item?")
    
    return suggestions


# ========================================
# EXPORTS
# ========================================

__all__ = [
    'query_boq',
    'query_boq_enhanced',
    'multi_hop_reasoning',
    'conversational_query',
    'batch_query',
    'explain_reasoning',
    'get_document_stats',
    'suggest_queries',
    'QueryIntent',
    'QueryClassifier',
    'GraphQueryEngine'
]