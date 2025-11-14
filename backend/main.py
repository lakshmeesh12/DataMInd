# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from ingest import ingest_excel_files, close_neo4j_driver
from typing import List, Optional, AsyncGenerator
import uvicorn
import logging
from contextlib import asynccontextmanager
from db.qdrant.config import init_qdrant_collection, init_qdrant_excel_collection
from search import (
    query_boq_enhanced,
    multi_hop_reasoning,
    conversational_query,
    batch_query,
    explain_reasoning,
    get_document_stats,
    suggest_queries
)
from typing import List, Optional
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import logging
from fastapi import Query
from query import search_documents
from query import (
    ultimate_hierarchical_search,
    stream_hierarchical_search,        # <-- NEW
)
from fastapi.middleware.cors import CORSMiddleware
from visual import visualize_document
# Import Neo4j query utilities
from neo4j_queries import (
    get_document_overview,
    get_all_items_with_costs,
    aggregate_costs_by_item_type,
    get_total_project_cost,
    find_similar_items_across_sheets,
    find_cost_outliers,
    compare_documents,
    search_items_by_keyword,
    get_graph_statistics,
    delete_document_graph,
    close_driver
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- STARTUP ----------
    logger.info("Application starting up...")
    
    logger.info("Initializing Qdrant collections...")
    await init_qdrant_collection()        # boq_chunks
    await init_qdrant_excel_collection()  # excel_documents

    yield  
    
    logger.info("Application shutting down...")
    await close_neo4j_driver()
    await close_driver()

app = FastAPI(
    title="Excel BOQ Ingestion & Knowledge Graph API", 
    version="2.0",
    lifespan=lifespan
)

origins = [
    "http://localhost:8080",   # Vite dev server
    "http://127.0.0.1:8080",
    # Add your production URL later, e.g. "https://yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# In-memory conversation store (use Redis in production)
conversations = {}



# ==================== INGESTION ENDPOINTS ====================

@app.post("/ingest/excel")
async def ingest_excel_batch(
    files: List[UploadFile] = File(..., description="Multiple .xlsx files"),
    user_id: str = Form(None, description="Optional user identifier")
):
    """
    Upload multiple Excel files → parse in parallel → store in MongoDB → build Knowledge Graph.
    Returns list of results (success + failures) with KG statistics.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    invalid_files = [f for f in files if not f.filename.lower().endswith(('.xlsx', '.xls'))]
    if invalid_files:
        names = ", ".join(f.filename for f in invalid_files)
        raise HTTPException(status_code=400, detail=f"Invalid file types: {names}. Only .xlsx supported.")

    try:
        logger.info(f"Starting parallel ingestion of {len(files)} files (user: {user_id})")
        results = await ingest_excel_files(files, user_id=user_id)
        logger.info(f"Batch completed: {len([r for r in results if r.get('status') == 'success'])} success, {len(results)} total")
        return JSONResponse(status_code=200, content=results)
    except Exception as e:
        logger.error(f"Batch ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ----------------------------------------------------------------------
# LEGACY NON-STREAMING SEARCH (kept for backward compatibility)
# ----------------------------------------------------------------------
@app.get("/search")
async def search_endpoint(
    q: str = Query(..., description="Natural language query")
):
    result = await ultimate_hierarchical_search(q)
    return result

# ----------------------------------------------------------------------
# NEW STREAMING SEARCH ENDPOINT
# ----------------------------------------------------------------------
@app.get("/search-stream")
async def search_stream_endpoint(
    q: str = Query(..., description="Natural language query (streaming)")
):
    """
    Runs the full hierarchical pipeline **once** and then streams the
    final LLM synthesis token-by-token (NDJSON).
    """
    async def event_generator():
        async for line in stream_hierarchical_search(q):
            yield line

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/visualize")
async def visualize(
    doc_id: Optional[str] = Query(None, description="Document ID from ingestion"),
    filename: Optional[str] = Query(None, description="Exact filename to visualize")
):
    """
    Visualize uploaded document using Qdrant + Neo4j.
    Returns structured JSON for charts + knowledge graph.
    """
    if not doc_id and not filename:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'doc_id' or 'filename' as query parameter"
        )

    try:
        result = await visualize_document(doc_id=doc_id, filename=filename)
        return JSONResponse(status_code=200, content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail="Visualization failed")
    
class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    use_cache: bool = True
    explain: bool = False
    conversation_id: Optional[str] = None


class BatchQueryRequest(BaseModel):
    questions: List[str]
    document_ids: List[str]


class ConversationMessage(BaseModel):
    question: str
    answer: str


# ========================================
# MAIN ENDPOINTS
# ========================================

@app.post("/query")
async def search_boq(
    question: str = Form(..., description="Natural language question about BOQ"),
    document_ids: Optional[str] = Form(None, description="Comma-separated document IDs"),
    use_cache: bool = Form(True, description="Enable caching"),
    explain: bool = Form(False, description="Return reasoning trace"),
    conversation_id: Optional[str] = Form(None, description="Conversation ID for follow-ups")
):
    """
    Main query endpoint - handles ANY question about BOQ documents
    
    Examples:
    - "What is the total cost of RESTROOM FINISHES?"
    - "How many doors are in the building?"
    - "What is the rate of TMT steel per ton?"
    - "Show me all plumbing items under ₹50,000"
    - "Compare RCC vs PCC costs"
    - "Summarize this document"
    """
    # Parse document IDs
    doc_ids = None
    if document_ids:
        doc_ids = [d.strip() for d in document_ids.split(",") if d.strip()]
    
    try:
        # Get conversation history if available
        conversation_history = None
        if conversation_id and conversation_id in conversations:
            conversation_history = conversations[conversation_id]
        
        # Execute query
        result = await query_boq_enhanced(
            question=question,
            document_ids=doc_ids,
            use_cache=use_cache,
            explain=explain,
            conversation_history=conversation_history
        )
        
        # Store in conversation history
        if conversation_id:
            if conversation_id not in conversations:
                conversations[conversation_id] = []
            conversations[conversation_id].append({
                "question": question,
                "answer": result["answer"]
            })
            # Keep only last 10 turns
            conversations[conversation_id] = conversations[conversation_id][-10:]
        
        return JSONResponse(status_code=200, content=result)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/json")
async def search_boq_json(request: QueryRequest):
    """
    JSON version of /query endpoint
    Better for programmatic access
    """
    conversation_history = None
    if request.conversation_id and request.conversation_id in conversations:
        conversation_history = conversations[request.conversation_id]
    
    try:
        result = await query_boq_enhanced(
            question=request.question,
            document_ids=request.document_ids,
            use_cache=request.use_cache,
            explain=request.explain,
            conversation_history=conversation_history
        )
        
        # Store in conversation history
        if request.conversation_id:
            if request.conversation_id not in conversations:
                conversations[request.conversation_id] = []
            conversations[request.conversation_id].append({
                "question": request.question,
                "answer": result["answer"]
            })
            conversations[request.conversation_id] = conversations[request.conversation_id][-10:]
        
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/batch")
async def batch_search(request: BatchQueryRequest):
    """
    Process multiple queries in parallel
    Useful for generating reports/dashboards
    
    Example:
    {
      "questions": [
        "Total project cost?",
        "Most expensive item?",
        "How many line items?"
      ],
      "document_ids": ["doc_123"]
    }
    """
    try:
        results = await batch_query(request.questions, request.document_ids)
        return {
            "queries": request.questions,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Batch query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/multi-hop")
async def multi_hop_search(
    question: str = Form(..., description="Complex question requiring multiple steps"),
    document_ids: str = Form(..., description="Comma-separated document IDs")
):
    """
    Handle complex queries requiring multiple reasoning steps
    
    Example: "Which is more expensive per ton: TMT steel or cement?"
    
    This will:
    1. Find TMT steel rate
    2. Find cement rate
    3. Compare and answer
    """
    doc_ids = [d.strip() for d in document_ids.split(",") if d.strip()]
    
    try:
        result = await multi_hop_reasoning(question, doc_ids)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"Multi-hop query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/explain")
async def explain_query(
    question: str = Form(...),
    document_ids: Optional[str] = Form(None)
):
    """
    Get detailed reasoning trace for a query
    Useful for debugging and understanding how the system works
    """
    doc_ids = None
    if document_ids:
        doc_ids = [d.strip() for d in document_ids.split(",") if d.strip()]
    
    try:
        result = await explain_reasoning(question, doc_ids)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"Explain query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# DOCUMENT MANAGEMENT
# ========================================

@app.get("/documents/{document_id}/stats")
async def document_stats(document_id: str):
    """
    Get statistics about a document
    
    Returns:
    - Number of sheets, tables, items
    - Total/min/max/avg costs
    - Useful for quick overview
    """
    try:
        stats = await get_document_stats([document_id])
        if not stats["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        return stats["documents"][0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}/suggestions")
async def query_suggestions(document_id: str):
    """
    Get suggested queries for a document
    Helps users discover what they can ask
    """
    try:
        suggestions = await suggest_queries([document_id])
        return {
            "document_id": document_id,
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Suggestion generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# CONVERSATION MANAGEMENT
# ========================================

@app.post("/conversation/start")
async def start_conversation(document_ids: str = Form(...)):
    """
    Start a new conversation session
    Returns a conversation_id to use in subsequent queries
    """
    import uuid
    conversation_id = str(uuid.uuid4())
    conversations[conversation_id] = []
    
    doc_ids = [d.strip() for d in document_ids.split(",") if d.strip()]
    
    return {
        "conversation_id": conversation_id,
        "document_ids": doc_ids,
        "message": "Conversation started. You can now ask follow-up questions."
    }


@app.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "history": conversations[conversation_id],
        "turn_count": len(conversations[conversation_id])
    }


@app.delete("/conversation/{conversation_id}")
async def end_conversation(conversation_id: str):
    """
    End and clear a conversation
    """
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": "Conversation ended"}
    raise HTTPException(status_code=404, detail="Conversation not found")


# ========================================
# HEALTH & INFO
# ========================================

@app.get("/")
async def root():
    return {
        "service": "BOQ RAG Agent",
        "version": "2.0",
        "features": [
            "Intent-based query classification",
            "Hybrid retrieval (vector + graph)",
            "Multi-hop reasoning",
            "Conversational follow-ups",
            "Batch queries",
            "Query explanation",
            "Caching"
        ],
        "endpoints": {
            "query": "/query (main endpoint)",
            "batch": "/query/batch",
            "multi_hop": "/query/multi-hop",
            "explain": "/query/explain",
            "stats": "/documents/{id}/stats",
            "suggestions": "/documents/{id}/suggestions",
            "conversation": "/conversation/*"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_conversations": len(conversations)
    }


# ========================================
# EXAMPLE USAGE
# ========================================

"""
EXAMPLE QUERIES:

1. Cost queries:
   - "What is the total cost of RESTROOM FINISHES?"
   - "What is the rate of TMT steel per ton?"
   - "Show me the most expensive items"

2. Quantity queries:
   - "How many doors are in the building?"
   - "What is the total quantity of cement in bags?"
   - "Count all electrical items"

3. Filtering:
   - "Show me all plumbing items under ₹50,000"
   - "List items over ₹1 lakh"
   - "Find all items in the 10k-50k range"

4. Comparison:
   - "Compare RCC vs PCC costs"
   - "Which is cheaper: TMT steel or MS steel?"

5. Structure:
   - "What sheets are in this document?"
   - "Show me document structure"

6. Summary:
   - "Summarize this BOQ"
   - "Give me an overview"

7. Conversational:
   User: "What's the cost of cement?"
   Bot: "₹500,000"
   User: "And steel?" (system understands context)
   Bot: "₹2,500,000"

8. Complex multi-hop:
   - "Which is more expensive per kg: steel or aluminum?"
   - "What's the cost difference between interior and exterior painting?"
"""


# ==================== KNOWLEDGE GRAPH QUERY ENDPOINTS ====================

@app.get("/kg/document/{doc_id}/overview")
async def kg_document_overview(doc_id: str):
    """Get knowledge graph overview for a document."""
    try:
        overview = await get_document_overview(doc_id)
        if not overview:
            raise HTTPException(status_code=404, detail="Document not found in knowledge graph")
        return overview
    except Exception as e:
        logger.error(f"Failed to get document overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/document/{doc_id}/items")
async def kg_document_items(doc_id: str):
    """Get all items with costs and quantities from a document."""
    try:
        items = await get_all_items_with_costs(doc_id)
        return {"document_id": doc_id, "item_count": len(items), "items": items}
    except Exception as e:
        logger.error(f"Failed to get document items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/document/{doc_id}/aggregate-costs")
async def kg_aggregate_costs(doc_id: str):
    """Aggregate costs grouped by item type (e.g., concrete, steel, masonry)."""
    try:
        aggregations = await aggregate_costs_by_item_type(doc_id)
        return {
            "document_id": doc_id,
            "aggregations": aggregations,
            "total_categories": len(aggregations)
        }
    except Exception as e:
        logger.error(f"Failed to aggregate costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/document/{doc_id}/total-cost")
async def kg_total_cost(doc_id: str, currency: str = Query("INR", description="Currency code")):
    """Calculate total project cost."""
    try:
        total = await get_total_project_cost(doc_id, currency)
        return total
    except Exception as e:
        logger.error(f"Failed to calculate total cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/document/{doc_id}/similar-items")
async def kg_similar_items(doc_id: str, item_name: str = Query(..., description="Item name to find similar items for")):
    """Find similar items across sheets with cost comparisons."""
    try:
        similar = await find_similar_items_across_sheets(item_name, doc_id)
        return {
            "document_id": doc_id,
            "query_item": item_name,
            "similar_items": similar,
            "count": len(similar)
        }
    except Exception as e:
        logger.error(f"Failed to find similar items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/document/{doc_id}/cost-outliers")
async def kg_cost_outliers(
    doc_id: str, 
    threshold: float = Query(2.0, description="Standard deviation threshold (default: 2.0)")
):
    """Find items with costs that are statistical outliers."""
    try:
        outliers = await find_cost_outliers(doc_id, threshold)
        return {
            "document_id": doc_id,
            "threshold": threshold,
            "outliers": outliers,
            "count": len(outliers)
        }
    except Exception as e:
        logger.error(f"Failed to find cost outliers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/compare")
async def kg_compare_documents(
    doc_id1: str = Query(..., description="First document ID"),
    doc_id2: str = Query(..., description="Second document ID")
):
    """Compare items and costs between two documents."""
    try:
        comparison = await compare_documents(doc_id1, doc_id2)
        return comparison
    except Exception as e:
        logger.error(f"Failed to compare documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/search")
async def kg_search_items(
    keyword: str = Query(..., description="Search keyword"),
    doc_id: Optional[str] = Query(None, description="Optional: filter by document ID")
):
    """Search items by keyword across all documents or within a specific document."""
    try:
        items = await search_items_by_keyword(keyword, doc_id)
        return {
            "keyword": keyword,
            "document_id": doc_id,
            "results": items,
            "count": len(items)
        }
    except Exception as e:
        logger.error(f"Failed to search items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/stats")
async def kg_statistics():
    """Get overall knowledge graph statistics."""
    try:
        stats = await get_graph_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/kg/document/{doc_id}")
async def kg_delete_document(doc_id: str):
    """Delete document and all related nodes from knowledge graph."""
    try:
        deleted_count = await delete_document_graph(doc_id)
        return {
            "document_id": doc_id,
            "deleted_nodes": deleted_count,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Failed to delete document graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HEALTH CHECK ====================

@app.get("/")
async def root():
    return {
        "message": "Excel BOQ Ingestion & Knowledge Graph API Ready",
        "version": "2.0",
        "endpoints": {
            "ingestion": "/ingest/excel",
            "kg_overview": "/kg/document/{doc_id}/overview",
            "kg_items": "/kg/document/{doc_id}/items",
            "kg_aggregate": "/kg/document/{doc_id}/aggregate-costs",
            "kg_total": "/kg/document/{doc_id}/total-cost",
            "kg_similar": "/kg/document/{doc_id}/similar-items",
            "kg_outliers": "/kg/document/{doc_id}/cost-outliers",
            "kg_compare": "/kg/compare",
            "kg_search": "/kg/search",
            "kg_stats": "/kg/stats"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "BOQ Ingestion API"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)