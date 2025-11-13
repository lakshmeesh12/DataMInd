# search.py - COMPLETE REWRITE WITH HIERARCHICAL AWARENESS
import os
import json
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
import openai
from dotenv import load_dotenv
import asyncio

load_dotenv()
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                              CONFIGURATION                                 #
# --------------------------------------------------------------------------- #
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_1", "excel_documents")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL       = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
openai.api_key    = os.getenv("OPENAI_API_KEY")

qdrant_client = QdrantClient(url=QDRANT_URL)

MAX_TOKENS_PER_CHUNK = 80_000
SYSTEM_PROMPT_OVERHEAD = 2000
INITIAL_RETRIEVAL_LIMIT = 20

# --------------------------------------------------------------------------- #
#                         HIERARCHICAL TEXT BUILDERS                          #
# --------------------------------------------------------------------------- #
def build_hierarchical_context(point: ScoredPoint, query_entities: List[str] = None) -> str:
    """
    Build searchable text that preserves document hierarchy.
    Filters content based on query entities to reduce noise.
    """
    p = point.payload
    filename = p.get("filename", "Unknown")
    sheets = p.get("sheets", {})
    
    text_blocks = []
    text_blocks.append(f"FILE: {filename}")
    text_blocks.append("=" * 80)
    
    for sheet_name, sheet_data in sheets.items():
        text_blocks.append(f"\nSHEET: {sheet_name}")
        text_blocks.append("-" * 80)
        
        # Get document hierarchy
        hierarchy = sheet_data.get("hierarchy", {})
        
        # Document headers (project name, etc.)
        doc_metadata = hierarchy.get("document_metadata", {})
        if doc_metadata:
            text_blocks.append("\n📋 DOCUMENT INFO:")
            for key, value in doc_metadata.items():
                text_blocks.append(f"  {value}")
        
        # Process sections with hierarchy
        sections = hierarchy.get("sections", [])
        for section in sections:
            section_code = section.get("code", "")
            section_title = section.get("title", "")
            section_text = f"{section_code} {section_title}"
            
            # Check if this section is relevant to query
            if query_entities:
                section_relevant = any(
                    entity.lower() in section_text.lower() 
                    for entity in query_entities
                )
            else:
                section_relevant = True
            
            if not section_relevant:
                # Skip irrelevant sections for efficiency
                continue
            
            text_blocks.append(f"\n{'  ' * section.get('level', 1)}📁 {section_text}")
            
            # Subsections
            for subsection in section.get("subsections", []):
                subsec_title = subsection.get("title", "")
                text_blocks.append(f"{'  ' * (section.get('level', 1) + 1)}📂 {subsec_title}")
                
                # Items under subsection
                for item in subsection.get("items", [])[:30]:  # Limit items
                    item_text = item.get("text", "")
                    row_context = item.get("row_context", [])
                    if row_context:
                        context_str = " | ".join(row_context[:3])
                        text_blocks.append(f"{'  ' * (section.get('level', 1) + 2)}• {item_text} ({context_str})")
                    else:
                        text_blocks.append(f"{'  ' * (section.get('level', 1) + 2)}• {item_text}")
            
            # Items directly under section
            for item in section.get("items", [])[:30]:
                item_text = item.get("text", "")
                row_context = item.get("row_context", [])
                if row_context:
                    context_str = " | ".join(row_context[:3])
                    text_blocks.append(f"{'  ' * (section.get('level', 1) + 1)}• {item_text} ({context_str})")
                else:
                    text_blocks.append(f"{'  ' * (section.get('level', 1) + 1)}• {item_text}")
        
        # Also include traditional table data if hierarchy is empty
        if not sections:
            tables = sheet_data.get("tables", [])
            for table in tables:
                headers = table.get("headers", [])
                if headers:
                    text_blocks.append(f"\n  TABLE Headers: {' | '.join(headers)}")
                for row in table.get("rows", [])[:50]:
                    row_text = " | ".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        text_blocks.append(f"    {row_text}")
    
    return "\n".join(text_blocks)

def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 100

# --------------------------------------------------------------------------- #
#                    STEP 1 – HIERARCHICAL QUERY ANALYSIS                     #
# --------------------------------------------------------------------------- #
async def analyze_hierarchical_query(query: str) -> Dict[str, Any]:
    """
    Enhanced query analysis that understands document structure queries.
    """
    logger.info("[AGENT-1] Analyzing hierarchical query...")
    
    prompt = f"""You are analyzing a query about structured documents (BOQs, make lists, specifications).

**User Query**: {query}

**TASK**: Analyze this query to understand what the user is looking for.

1. **Query Type**:
   - "section_lookup" - Looking for a specific section (e.g., "flooring section")
   - "filtered_list" - Need items from a section filtered by criteria (e.g., "flooring items in building X")
   - "comparison" - Comparing sections/items across documents
   - "specification" - Looking for specific specs/details
   
2. **Document Context Keywords**: Extract document identifiers (project names, building codes, locations)
   Examples: "5F", "Satva", "Hyd", "4F", project names, building names
   
3. **Section/Category Keywords**: Extract section/category being queried
   Examples: "flooring", "civil works", "doors", "painting", "plumbing"
   
4. **Filter Keywords**: Any additional filters
   Examples: "make list", "brand", "specifications", "rate", "quantity"
   
5. **Expected Result Format**: What format does the user expect?
   - "list" - Simple list of items
   - "table" - Structured data
   - "detailed" - Full specifications

Return JSON:
{{
  "query_type": "...",
  "document_context_keywords": [...],
  "section_keywords": [...],
  "filter_keywords": [...],
  "expected_format": "...",
  "search_strategy": "Brief explanation of how to search",
  "estimated_scope": "narrow|medium|wide"
}}
"""

    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())
        logger.info(f"[AGENT-1] Query type: {result.get('query_type')}, Section: {result.get('section_keywords')}")
        return result
    except Exception as e:
        logger.error(f"[AGENT-1] Failed: {e}")
        return {
            "query_type": "section_lookup",
            "document_context_keywords": [],
            "section_keywords": query.split(),
            "filter_keywords": [],
            "expected_format": "list",
            "search_strategy": "Search for query terms",
            "estimated_scope": "medium"
        }

# --------------------------------------------------------------------------- #
#                    STEP 2 – CONTEXT-AWARE RETRIEVAL                         #
# --------------------------------------------------------------------------- #
async def hierarchical_retrieval(query: str, query_analysis: Dict) -> List[ScoredPoint]:
    """
    Retrieves documents using hierarchical understanding.
    """
    logger.info("[AGENT-2] Hierarchical retrieval...")
    
    all_hits = []
    seen_ids = set()
    
    # Stage 1: Main query
    hits_main = await _vector_search(query, limit=INITIAL_RETRIEVAL_LIMIT)
    for hit in hits_main:
        if hit.id not in seen_ids:
            all_hits.append(hit)
            seen_ids.add(hit.id)
    logger.info(f"[AGENT-2] Main query: {len(hits_main)} docs")
    
    # Stage 2: Document context (project names, building codes)
    for ctx in query_analysis.get("document_context_keywords", [])[:3]:
        if ctx and len(ctx) > 1:
            hits_ctx = await _vector_search(ctx, limit=10)
            for hit in hits_ctx:
                if hit.id not in seen_ids:
                    all_hits.append(hit)
                    seen_ids.add(hit.id)
    
    # Stage 3: Section keywords
    for section in query_analysis.get("section_keywords", [])[:3]:
        if section and len(section) > 2:
            hits_section = await _vector_search(section, limit=10)
            for hit in hits_section:
                if hit.id not in seen_ids:
                    all_hits.append(hit)
                    seen_ids.add(hit.id)
    
    logger.info(f"[AGENT-2] Total retrieved: {len(all_hits)} unique docs")
    return all_hits

async def _vector_search(query: str, limit: int = 10) -> List[ScoredPoint]:
    try:
        resp = openai.embeddings.create(input=[query], model=EMBED_MODEL)
        vec = resp.data[0].embedding
        hits = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vec,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return hits
    except Exception as e:
        logger.error(f"[VECTOR-SEARCH] Error: {e}")
        return []

# --------------------------------------------------------------------------- #
#                    STEP 3 – HIERARCHICAL FILTERING                          #
# --------------------------------------------------------------------------- #
async def filter_by_hierarchy(query: str, points: List[ScoredPoint], 
                              query_analysis: Dict) -> List[Dict[str, Any]]:
    """
    Filter and prepare documents with hierarchical awareness.
    """
    logger.info(f"[AGENT-3] Filtering {len(points)} documents...")
    
    # Extract all search terms
    all_keywords = (
        query_analysis.get("document_context_keywords", []) +
        query_analysis.get("section_keywords", []) +
        query_analysis.get("filter_keywords", [])
    )
    
    relevant_docs = []
    for point in points:
        # Build hierarchical context with keyword filtering
        text = build_hierarchical_context(point, all_keywords)
        
        # Quick relevance check
        text_lower = text.lower()
        relevance_score = sum(1 for kw in all_keywords if kw.lower() in text_lower)
        
        if relevance_score > 0 or not all_keywords:  # Include if has any keyword match
            relevant_docs.append({
                "filename": point.payload.get("filename"),
                "text": text,
                "tokens": estimate_tokens(text),
                "score": point.score,
                "relevance_score": relevance_score,
                "raw_payload": point.payload
            })
    
    # Sort by relevance
    relevant_docs.sort(key=lambda x: (x["relevance_score"], x["score"]), reverse=True)
    
    logger.info(f"[AGENT-3] Filtered to {len(relevant_docs)} relevant documents")
    return relevant_docs

# --------------------------------------------------------------------------- #
#                    STEP 4 – HIERARCHICAL EXTRACTION                         #
# --------------------------------------------------------------------------- #
async def extract_with_hierarchy(query: str, docs: List[Dict], 
                                query_analysis: Dict) -> List[Dict]:
    """
    Extract data while preserving hierarchical relationships.
    """
    logger.info(f"[AGENT-4] Extracting from {len(docs)} documents...")
    
    # Process in chunks
    chunks = _create_smart_chunks(docs, MAX_TOKENS_PER_CHUNK)
    
    all_extractions = []
    for i, chunk in enumerate(chunks):
        logger.info(f"[AGENT-4] Processing chunk {i+1}/{len(chunks)}")
        extractions = await _extract_hierarchical_chunk(query, chunk, query_analysis)
        all_extractions.extend(extractions)
    
    logger.info(f"[AGENT-4] Total extractions: {len(all_extractions)}")
    return all_extractions

def _create_smart_chunks(docs: List[Dict], max_tokens: int) -> List[List[Dict]]:
    chunks = []
    current = []
    current_tokens = SYSTEM_PROMPT_OVERHEAD
    
    for doc in docs:
        if current_tokens + doc["tokens"] > max_tokens:
            if current:
                chunks.append(current)
            current = [doc]
            current_tokens = SYSTEM_PROMPT_OVERHEAD + doc["tokens"]
        else:
            current.append(doc)
            current_tokens += doc["tokens"]
    
    if current:
        chunks.append(current)
    
    return chunks

async def _extract_hierarchical_chunk(query: str, chunk: List[Dict], 
                                     query_analysis: Dict) -> List[Dict]:
    """
    Extract data understanding document hierarchy.
    """
    docs_text = ""
    for doc in chunk:
        docs_text += f"\n{'='*100}\n{doc['text']}\n"
    
    # Build extraction instructions based on query type
    doc_ctx_str = ", ".join(query_analysis.get("document_context_keywords", []))
    section_str = ", ".join(query_analysis.get("section_keywords", []))
    
    prompt = f"""You are a **Hierarchical Data Extraction Agent**. Extract data while understanding document structure.

**User Query**: {query}

**What user wants**:
- Document context: {doc_ctx_str or 'Any document'}
- Section/Category: {section_str or 'Relevant sections'}
- Format: {query_analysis.get('expected_format', 'list')}

**Documents with Hierarchy**:
{docs_text}

**CRITICAL EXTRACTION RULES**:
1. **Understand Context Path**: Each item exists in a hierarchy (Document → Section → Subsection → Item)
2. **Match Document Context**: If query mentions project/building name, ONLY extract from documents with that context
3. **Find Correct Section**: Navigate to the right section (e.g., "FLOORING", "CIVIL WORKS")
4. **Extract Complete Data**: Get all items under that section
5. **Preserve Structure**: Note the hierarchical path for each extracted item

**Output Format** - Return JSON:
{{
  "extractions": [
    {{
      "filename": "...",
      "sheet": "...",
      "document_context": "Project name / building identifier from document header",
      "section_code": "2.03",
      "section_title": "FLOORING",
      "subsection": "Vinyl Flooring" (if applicable),
      "items": [
        {{"item": "Armstrong", "context": "Material/Brand name", "row_context": ["related", "values"]}},
        {{"item": "Jeoflor/Gerflor", "context": "Alternative brand"}},
        ...
      ],
      "hierarchy_path": ["Document", "Section", "Subsection"],
      "confidence": "high|medium|low",
      "relevance_reason": "Why this section matches the query"
    }}
  ]
}}

**IMPORTANT**: 
- If document context in query doesn't match document, return empty extractions
- Extract ALL items from matched sections
- Preserve the hierarchical relationship
"""

    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5000,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 1)[-1].rsplit("```", 1)[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        
        result = json.loads(raw)
        return result.get("extractions", [])
    except Exception as e:
        logger.error(f"[AGENT-4] Extraction failed: {e}")
        return []

# --------------------------------------------------------------------------- #
#                    STEP 5 – HIERARCHICAL SYNTHESIS                          #
# --------------------------------------------------------------------------- #
async def synthesize_hierarchical_answer(query: str, extractions: List[Dict],
                                          query_analysis: Dict) -> Dict[str, Any]:
    """
    Synthesize answer preserving hierarchical context.
    NOW includes logic to decide on and format data for visualizations.
    """
    logger.info(f"[AGENT-5] Synthesizing from {len(extractions)} extractions...")

    if not extractions:
        return {
            "answer": "No data found matching your query criteria.",
            "visualization": None,
            "citations": [],
            "confidence": "low",
            "completeness": "none"
        }

    # Group by document context and section
    context = ""
    for ext in extractions:
        context += f"\n{'=' * 80}\n"
        context += f"**File**: {ext.get('filename', 'Unknown')}\n"
        context += f"**Sheet**: {ext.get('sheet', 'Unknown')}\n"
        context += f"**Document Context**: {ext.get('document_context', 'N/A')}\n"
        context += f"**Section**: {ext.get('section_code', '')} {ext.get('section_title', '')}\n"
        if ext.get('subsection'):
            context += f"**Subsection**: {ext.get('subsection')}\n"
        context += f"**Hierarchy**: {' → '.join(ext.get('hierarchy_path', []))}\n"
        context += f"\n**Items**:\n"

        # --- MODIFIED CONTEXT BUILDER ---
        # This makes the data easier for the LLM to parse.
        for item in ext.get('items', []):
            item_text = item.get('item', '')
            item_ctx = item.get('context', '')
            row_ctx = item.get('row_context', [])

            # Join all row context as a string. This is what the LLM will read.
            # e.g., "1 | Supply of Vanguard approved cutlery sets | 1,11,720.00"
            row_ctx_str = " | ".join(str(val) for val in row_ctx if val is not None)

            # If item_text is *already* in the row_ctx, the row_ctx_str is all we need
            if item_text in row_ctx_str:
                context += f"  - Data Row: [{row_ctx_str}]\n"
            else:
                # Otherwise, combine them
                context += f"  - Item: {item_text} (Context: {item_ctx}) (Data: {row_ctx_str})\n"
        context += "\n"

    # --- MODIFIED PROMPT ---
    prompt = f"""You are the **Hierarchical Answer Synthesis Agent** with data visualization capabilities.

**User Query**: {query}

**Query Analysis**:
- Looking for: {', '.join(query_analysis.get('section_keywords', []))}
- In document: {', '.join(query_analysis.get('document_context_keywords', []))}
- Expected format: {query_analysis.get('expected_format', 'list')}

**Extracted Data with Context**:
{context}

**SYNTHESIS INSTRUCTIONS**:
1.  **Analyze Data**: Look at the extracted items in the context.
2.  **Decide Visualization**:
    * If the query is a **comparison** (e.g., "compare make lists"), choose `type: "table"`.
    * If the data is **numerical** (costs, quantities) for different categories (e.g., "C&I variation costs"), choose `type: "bar_chart"` or `type: "pie_chart"`.
    * If no visualization is appropriate (e.g., a simple text definition), set `visualization: null`.
3.  **Format Data**:
    * **For `table`**: `data: {{ "headers": ["Header 1", "Header 2"], "rows": [ ["r1c1", "r1c2"], ["r2c1", "r2c2"] ] }}`
    * **For `bar_chart` or `pie_chart`**: `data: {{ "labels": ["Item A", "Item B"], "datasets": [ {{ "label": "Cost", "data": [111720.00, 73623.00] }} ] }}`

4.  **CRITICAL DATA EXTRACTION RULE**:
    * You MUST extract the numerical values (e.g., `1,11,720.00`, `73,623.00`, `4,70,788.98`) from the **Extracted Data with Context** block.
    * You MUST **clean** these numbers: remove commas (`,`), currency symbols (like `₹`), and convert them to simple floats (e.g., `"1,11,720.00"` becomes `111720.00`).
    * The `data` array in your `visualization` object MUST contain these *real*, *cleaned* numbers from the context.
    * **DO NOT INVENT or HALLUCINATE numbers** like `1200`, `2500`. Use the *exact* numbers found in the context.

5.  **Write Answer**: Write a text summary (`answer`) that introduces the data. The visualization will be shown after.
6.  **Be Complete**: Ensure all data is used in *either* the text or the visualization.

**Return JSON**:
{{
  "answer": "Text summary of the findings.",
  "visualization": {{
    "type": "table | bar_chart | pie_chart",
    "title": "A descriptive title for the chart/table",
    "data": {{ ... formatted data ... }}
  }} | null,
  "citations": ["Detailed citations with hierarchy"],
  "confidence": "high|medium|low",
  "completeness": "full|partial|limited",
  "matched_context": "Which document/project context was matched",
  "sections_covered": ["List of sections included in answer"]
}}
"""

    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=6000,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content.strip())

        # Basic validation to ensure data is clean
        if result.get("visualization") and result["visualization"]["type"] in ["bar_chart", "pie_chart"]:
            logger.info("Validating chart data...")
            datasets = result["visualization"]["data"].get("datasets", [])
            for ds in datasets:
                # Ensure all data points are floats/ints, not strings
                ds["data"] = [float(v) for v in ds.get("data", [])]

        result["total_extractions"] = len(extractions)
        logger.info(f"[AGENT-5] Synthesis complete: {result.get('confidence')}, {result.get('completeness')}")
        return result

    except Exception as e:
        logger.error(f"[AGENT-5] Synthesis failed: {e}")
        logger.error(f"[AGENT-5] Raw OpenAI output (if any): {resp.choices[0].message.content.strip() if 'resp' in locals() else 'No response'}")
        return {
            "answer": f"Error during synthesis: {e}",
            "visualization": None,
            "citations": [],
            "confidence": "low",
            "completeness": "failed"
        }


# --------------------------------------------------------------------------- #
#                    MAIN ORCHESTRATOR                                        #
# --------------------------------------------------------------------------- #
async def ultimate_hierarchical_search(query: str) -> Dict[str, Any]:
    """
    Master orchestrator for hierarchical search.
    Coordinates multiple specialized agents:
    1️⃣ Query Analysis → 2️⃣ Retrieval → 3️⃣ Filtering → 4️⃣ Extraction → 5️⃣ Synthesis
    """
    logger.info("\n" + "=" * 100)
    logger.info("🏗️  HIERARCHICAL SEARCH STARTED")
    logger.info("=" * 100)

    # 🧩 AGENT 1: Hierarchical Query Analysis
    query_analysis = await analyze_hierarchical_query(query)

    # 🔍 AGENT 2: Hierarchical Retrieval
    points = await hierarchical_retrieval(query, query_analysis)
    if not points:
        return {
            "answer": "No documents found.",
            "visualization": None,
            "citations": [],
            "confidence": "low",
            "query_analysis": query_analysis,
        }

    # 🧱 AGENT 3: Hierarchical Filtering
    docs = await filter_by_hierarchy(query, points, query_analysis)
    if not docs:
        return {
            "answer": "No relevant sections found in retrieved documents.",
            "visualization": None,
            "citations": [],
            "confidence": "low",
            "query_analysis": query_analysis,
        }

    # 🧠 AGENT 4: Hierarchical Extraction
    extractions = await extract_with_hierarchy(query, docs, query_analysis)
    if not extractions:
        return {
            "answer": "Could not extract relevant data from matched sections.",
            "visualization": None,
            "citations": [],
            "confidence": "low",
            "query_analysis": query_analysis,
            "documents_searched": [d.get("filename", "Unknown") for d in docs],
        }

    # 🧾 AGENT 5: Hierarchical Synthesis
    result = await synthesize_hierarchical_answer(query, extractions, query_analysis)
    result["query_analysis"] = query_analysis

    logger.info("\n" + "=" * 100)
    logger.info(f"✅ HIERARCHICAL SEARCH COMPLETE: {result.get('completeness', 'unknown')} answer")
    logger.info("=" * 100 + "\n")

    return result



# --------------------------------------------------------------------------- #
#                             MAIN ENTRYPOINT                                #
# --------------------------------------------------------------------------- #
async def search_documents(query: str) -> Dict[str, Any]:
    if not query.strip():
        return {"error": "Query cannot be empty"}
    
    return await ultimate_hierarchical_search(query)