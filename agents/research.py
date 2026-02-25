"""
Research Agent - Performs web research and builds knowledge base.

This agent uses Tavily API to search for relevant content, summarizes the findings,
and stores them in a vector database for later retrieval (RAG).
"""

import os
import logging
from typing import Dict, Any, List
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research agent: Performs web search and stores embeddings.
    
    This agent searches the web using Tavily, summarizes the results,
    and stores them in a vector database for RAG during content generation.
    
    Input:
        state["topic"]: The blog topic
        state["plan"]: Blog plan with keywords
        
    Output:
        state["research_docs"]: List of summarized research documents
        state["_vector_store"]: Vector store instance for RAG
        
    Args:
        state: Current blog state
        
    Returns:
        Updated state with research documents and vector store
    """
    topic = state["topic"]
    plan = state.get("plan", {})
    keywords = plan.get("keywords", [])
    
    logger.info(f"Research: Starting research for topic - '{topic}'")
    
    # Initialize Tavily client
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY not found in environment")
        # Return fallback research
        return _create_fallback_research(state)
    
    try:
        tavily_client = TavilyClient(api_key=tavily_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Tavily client: {e}")
        return _create_fallback_research(state)
    
    # Initialize vector store
    vector_store = VectorStore()
    collection_name = vector_store.create_collection(topic)
    logger.info(f"Research: Created vector collection - '{collection_name}'")
    
    # Perform web search
    search_query = _build_search_query(topic, keywords)
    logger.info(f"Research: Searching for - '{search_query}'")
    
    try:
        results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=7
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return _create_fallback_research(state)
    
    # Process and summarize results
    research_docs = []
    documents = []
    metadatas = []
    ids = []
    
    # Initialize LLM for summarization
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=0.3
    )
    
    search_results = results.get('results', [])
    logger.info(f"Research: Processing {len(search_results)} search results")
    
    for idx, result in enumerate(search_results):
        try:
            # Summarize each result
            summary_prompt = f"""Summarize the following content in 2-3 concise bullet points.
Focus on key facts and insights relevant to the topic: {topic}

Title: {result.get('title', 'No title')}
Content: {result.get('content', 'No content')[:1000]}

Provide only the bullet points, no additional text."""
            
            response = llm.invoke(summary_prompt)
            summary = response.content if isinstance(response.content, str) else str(response.content)
            research_docs.append(summary)
            
            # Prepare for vector store
            full_content = f"Title: {result.get('title', '')}\n\n{result.get('content', '')}"
            documents.append(full_content)
            metadatas.append({
                "source": result.get('url', ''),
                "title": result.get('title', ''),
                "score": result.get('score', 0.0)
            })
            ids.append(f"doc_{idx}")
            
            logger.info(f"Research: Processed result {idx+1}/{len(search_results)}")
            
        except Exception as e:
            logger.warning(f"Failed to process result {idx}: {e}")
            continue
    
    # Store in vector database
    if documents:
        try:
            vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Research: Stored {len(documents)} documents in vector database")
        except Exception as e:
            logger.error(f"Failed to store documents in vector database: {e}")
    else:
        logger.warning("Research: No documents to store")
        return _create_fallback_research(state)
    
    # Update state
    state["research_docs"] = research_docs
    state["_vector_store"] = vector_store
    
    logger.info(f"Research: Completed with {len(research_docs)} summaries")
    return state


def _build_search_query(topic: str, keywords: List[str]) -> str:
    """
    Build an effective search query from topic and keywords.
    
    Args:
        topic: Blog topic
        keywords: List of relevant keywords
        
    Returns:
        Search query string
    """
    # Use topic plus top 3 keywords
    query_parts = [topic]
    if keywords:
        query_parts.extend(keywords[:3])
    
    return " ".join(query_parts)


def _create_fallback_research(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create fallback research when Tavily is unavailable.
    
    Args:
        state: Current blog state
        
    Returns:
        Updated state with minimal research
    """
    topic = state["topic"]
    logger.warning("Research: Using fallback research (Tavily unavailable)")
    
    # Create minimal research docs
    state["research_docs"] = [
        f"• {topic} is an important subject in modern technology and business",
        f"• Understanding {topic} requires knowledge of fundamental concepts",
        f"• {topic} has various applications across different industries"
    ]
    
    # Create empty vector store
    vector_store = VectorStore()
    vector_store.create_collection(topic)
    state["_vector_store"] = vector_store
    
    return state