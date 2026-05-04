import asyncio
from typing import List
from tavily import AsyncTavilyClient
from langsmith import traceable

def deduplicate_and_format_sources(search_response, max_tokens_per_source=500, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
    Args:
        search_response: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Collect all results
    images_list = []
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])
        if 'images' in response: images_list.extend(response['images'])
    
    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Content from sources:\n"
    for _, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*10}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*10}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()


@traceable
async def tavily_search_async(search_queries):
    """
    Performs concurrent web searches using the Tavily API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process

    Returns:
            List[dict]: List of search responses from Tavily API, one per query. Each response has format:
                {
                    'query': str, # The original search query
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """
    print("asdfasdf")
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,
                    include_raw_content=True,
                    include_images=False,
                    topic="general"
                )
            )

    # Execute all searches concurrently
    try:
        search_docs = await asyncio.gather(*search_tasks)
        return search_docs
    except Exception as e:
        print(f"Tavily search error: {e}")
        return []


async def search_web(state, config):
    """
    tavily 웹 검색을 수행하는 LangGraph Node
    """
    search_queries = state.get('search_queries', [])
    existing_documents = state.get('documents', [])
    current_search_count = state.get('number_of_search', 0)

    if not search_queries:
        return {
            "documents": existing_documents,
            "number_of_search": current_search_count + 1
        }

    # 웹 검색 수행
    search_response = await tavily_search_async(search_queries)
    
    # 검색 결과를 List[str] 형태로 포맷팅
    web_documents = deduplicate_and_format_sources(
        search_response, 
        max_tokens_per_source=500, 
        include_raw_content=True
    )

    #import logging
    #logging.error("web_documents: %s", web_documents)
    
    updated_documents = existing_documents + [web_documents]
    
    return {
        "documents": updated_documents,
        "number_of_search": current_search_count + 1
    }