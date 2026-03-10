from src.clients.async_tavily_client import async_tavily_client
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
import re
from urllib.parse import urlparse
from pathlib import Path 
from typing import Optional, Literal, Union, List, Tuple, Annotated, Dict, Any
from langgraph.types import Command
from langchain.messages import ToolMessage
from src.agents.tools.utils.tavily_functions import (
    tavily_search as _tavily_search_fn,
    format_tavily_search_response,
    tavily_extract as _tavily_extract_fn,
    tavily_map as _tavily_map_fn,
    format_tavily_extract_response,
    format_tavily_map_response, 
    tavily_crawl as _tavily_crawl_fn, 
    format_tavily_crawl_response, 
)    


from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from typing import Optional, Set

def _normalize_url_for_dedupe(
    url: str,
    *,
    drop_fragment: bool = True,
    drop_query: bool = False,
    keep_query_params: Optional[set[str]] = None,
) -> str:
    """
    Normalize URL for dedupe while preserving meaning.

    Defaults:
      - drop_fragment=True: removes #section anchors
      - drop_query=False: keep query by default (safer for Shopify/Woo/etc)
    Optional:
      - keep_query_params: if provided and drop_query=False, keep only these params.
    """
    s = url.strip()
    parts = urlsplit(s)

    fragment = "" if drop_fragment else parts.fragment

    if drop_query:
        query = ""
    elif keep_query_params is not None:
        # keep only whitelisted params (stable canonicalization)
        q = parse_qsl(parts.query, keep_blank_values=True)
        q2 = [(k, v) for (k, v) in q if k in keep_query_params]
        query = urlencode(q2, doseq=True)
    else:
        query = parts.query

    # Normalize scheme/host casing; keep path as-is
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()

    return urlunsplit((scheme, netloc, parts.path, query, fragment))


def dedupe_urls(
    urls: list[str],
    *,
    drop_fragment: bool = True,
    drop_query: bool = False,
    keep_query_params: Optional[set[str]] = None,
) -> list[str]:
    """
    Dedupe URLs preserving first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        key = _normalize_url_for_dedupe(
            u,
            drop_fragment=drop_fragment,
            drop_query=drop_query,
            keep_query_params=keep_query_params,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out






_URL_RE = re.compile(r"https?://[^\s\)\]\}<>\"']+")

def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    # normalize + dedupe while preserving order
    out: List[str] = []
    seen = set()
    for u in urls:
        u = u.rstrip(".,;:!?)\"]}")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    if d.startswith("http"):
        try:
            return urlparse(d).netloc
        except Exception:
            return d
    return d

def _merge_visited(runtime: ToolRuntime, new_urls: List[str]) -> List[str]:
    cur = list(runtime.state.get("visited_urls") or [])
    cur_set = set(cur)
    for u in new_urls:
        if u not in cur_set:
            cur.append(u)
            cur_set.add(u)
    # keep it bounded to avoid unbounded growth
    return cur[-2000:]




@tool(
    description=(
        "Search the web for information using Tavily. Best for: finding company info, "
        "product details, recent news, market research, and general web knowledge. "
        "Returns either a summarized digest with citations or raw formatted results."
    ),
    parse_docstring=False,
)
async def tavily_search(
    runtime: ToolRuntime,
    query: Annotated[
        str,
        "Search query. Keep short but specific (entity + required fields). Use site: filters and official-domain terms when helpful."
    ],
    max_results: Annotated[
        int,
        "Max results to return (1–20). Higher increases recall but adds noise + tokens."
    ] = 10,
    search_depth: Annotated[
        Literal["basic", "advanced"],
        "basic=faster/cheaper; advanced=better relevance + more evidence snippets per source."
    ] = "advanced",
    topic: Annotated[
        Optional[Literal["general", "news", "finance"]],
        "Retrieval category hint. Use 'news' for recent coverage, 'finance' for market/earnings context."
    ] = "general",
    include_images: Annotated[
        bool,
        "Include image URLs (usually unnecessary for validation)."
    ] = False,
    include_raw_content: Annotated[
        bool | Literal["markdown", "text"],
        "Include cleaned page content per result. Prefer False to avoid bloat; use 'markdown' or 'text' when you need page text."
    ] = False,
    include_domains: Annotated[
        Optional[List[str]],
        "Optional domain allowlist to lock onto official sources (e.g., ['example.com','shop.example.com'])."
    ] = None,
    exclude_domains: Annotated[
        Optional[List[str]],
        "Optional domain blocklist to avoid retailers/affiliates/low-quality sources."
    ] = None,
    start_date: Annotated[
        Optional[str],
        "Optional start date filter (YYYY-MM-DD). Most useful with topic='news'."
    ] = None,
    end_date: Annotated[
        Optional[str],
        "Optional end date filter (YYYY-MM-DD). Most useful with topic='news'."
    ] = None,
) -> str:
    search_results = await _tavily_search_fn(
        client=async_tavily_client,
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        include_images=include_images,
        include_raw_content=include_raw_content,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        start_date=start_date,
        end_date=end_date,
    )   

    formatted = format_tavily_search_response(
        search_results,
        max_results=max_results,
        max_content_chars=1200,  
    )

    

    return formatted  


@tool(
    description=(
        "Extract and analyze content from specific URLs using Tavily Extract. "
        "Best for: reading product pages, documentation, about pages, and company websites. "
        "Can filter content with a query to focus on specific information. "
        "Returns either a summarized digest with citations or raw extracted content."
    ),
    parse_docstring=False,
)
async def tavily_extract(
    runtime: ToolRuntime,
    urls: Annotated[
        Union[str, List[str]],
        "1 URL or a list of URLs to extract from (prefer official pages: /products, /shop, /collections, /ingredients, /about)."
    ],
    query: Annotated[
        Optional[str],
        "Optional extraction filter. Provide required fields/keywords to return only relevant chunks (recommended for long pages)."
    ] = None,
    chunks_per_source: Annotated[
        int,
        "How many relevant chunks to pull per URL (1–5). Higher = more coverage + more tokens."
    ] = 3,
    extract_depth: Annotated[
        Literal["basic", "advanced"],
        "basic=faster/cheaper; advanced=better for dense pages, tables, and storefront/JS-heavy pages."
    ] = "basic",
    include_images: Annotated[
        bool,
        "Include images found on the page (usually unnecessary)."
    ] = False,
    include_favicon: Annotated[
        bool,
        "Include site favicon URL in results (cosmetic/attribution)."
    ] = False,
    format: Annotated[
        Literal["markdown", "text"],
        "Output format for extracted content. markdown is usually best for structure; text can be simpler but may be slower."
    ] = "markdown",
) -> str:
    extract_results = await _tavily_extract_fn(
        client=async_tavily_client,
        urls=urls,
        query=query,
        chunks_per_source=chunks_per_source,
        extract_depth=extract_depth,
        include_images=include_images,
        include_favicon=include_favicon,
        format=format,
    )

   

    formatted = format_tavily_extract_response(
        extract_results,
        max_results=10,
        max_content_chars=5000,
    ) 

    return formatted 


@tool(
    description=(
        "Map and discover links within a website using Tavily Map. "
        "Best for: finding product catalogs, documentation structure, site navigation, "
        "and discovering all relevant pages on a domain. "
        "Returns a formatted list or raw URL-per-line output (deduped by default)."
    ),
    parse_docstring=False,
)
async def tavily_map(
    runtime: ToolRuntime,
    url: Annotated[
        str,
        "Root URL or domain to map (prefer official home/docs root)."
    ],
    instructions: Annotated[
        Optional[str],
        "Optional guidance to steer discovery (e.g., 'Find product pages, shop/collections, ingredients/supplement facts; ignore blog/careers')."
    ] = None,
    max_depth: Annotated[
        int,
        "How many link-hops from the root to explore (1–5). Depth increases cost quickly; 2 is typical for product discovery."
    ] = 2,
    max_breadth: Annotated[
        int,
        "Max links to follow per level/page (1–500). Higher increases recall on storefront navs but can explode."
    ] = 120,
    limit: Annotated[
        int,
        "Hard cap on total URLs processed/returned by the mapper. Raise when enumerating large catalogs."
    ] = 250,
    
    dedupe: Annotated[
        bool,
        "Deduplicate discovered URLs before returning them to the model (recommended True)."
    ] = True,
    max_return_urls: Annotated[
        Optional[int],
        "Cap URLs returned to the model after dedupe (artifact still stores full raw output). None = no cap."
    ] = 300,
    drop_fragment: Annotated[
        bool,
        "Drop #fragment anchors during dedupe (recommended True)."
    ] = True,
    drop_query: Annotated[
        bool,
        "Drop ?query strings during dedupe. NOT recommended for ecommerce (variants/SKUs may be encoded in query params)."
    ] = False,
) -> str:
        map_results = await _tavily_map_fn(
        client=async_tavily_client,
        url=url,
        instructions=instructions,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
    )

   


        urls: list[str] = (map_results.get("results", []) or []) 

    
    

   
        return format_tavily_map_response(map_results, urls_override=urls)




@tool(
    description=(
        "Crawl a website starting from a root URL using Tavily Crawl. "
        "Returns either a formatted raw crawl output, or a summarized output if your runtime supports summarization-with-citations."
    ),
    parse_docstring=False,
)
async def tavily_crawl(
    runtime: ToolRuntime,
    url: Annotated[
        str,
        "Root URL to begin the crawl (e.g., 'https://docs.tavily.com'). Prefer the most canonical entrypoint."
    ],
    instructions: Annotated[
        Optional[str],
        "Optional natural-language goal to focus the crawl (e.g., 'Find all pages about the Python SDK'). Enables chunks_per_source behavior."
    ] = None,
    chunks_per_source: Annotated[
        int,
        "Max relevant chunks per crawled page (1–5). Only applies when instructions are provided."
    ] = 3,
    max_depth: Annotated[
        int,
        "Max depth from the base URL (1–5). Start small; increase only when you need deeper traversal."
    ] = 1,
    max_breadth: Annotated[
        int,
        "Max links followed per page/level (1–500). Higher increases coverage but can add noise."
    ] = 20,
    limit: Annotated[
        int,
        "Total pages to process before stopping (>=1). Acts as the global crawl cap."
    ] = 50,
    select_paths: Annotated[
        Optional[List[str]],
        "Optional regex list to include only matching URL paths (e.g., ['/docs/.*','/sdk/.*'])."
    ] = None,
    select_domains: Annotated[
        Optional[List[str]],
        "Optional regex list to include only matching domains/subdomains (e.g., ['^docs\\.example\\.com$'])."
    ] = None,
    exclude_paths: Annotated[
        Optional[List[str]],
        "Optional regex list to exclude URL paths (e.g., ['/admin/.*','/private/.*'])."
    ] = None,
    exclude_domains: Annotated[
        Optional[List[str]],
        "Optional regex list to exclude domains/subdomains (e.g., ['^private\\.example\\.com$'])."
    ] = None,
    allow_external: Annotated[
        bool,
        "Whether to include external-domain links in final results."
    ] = True,
    include_images: Annotated[
        bool,
        "Whether to include images in crawl results. Usually False for research/validation."
    ] = False,
    extract_depth: Annotated[
        Literal["basic", "advanced"],
        "basic=faster/cheaper; advanced=more thorough extraction (tables/embedded content) but can increase latency."
    ] = "basic",
    format: Annotated[
        Literal["markdown", "text"],
        "Output content format. markdown is usually best for downstream chunking."
    ] = "markdown",
    include_favicon: Annotated[
        bool,
        "Include favicon URL for each result."
    ] = False,
    timeout: Annotated[
        float,
        "Max seconds to wait for the crawl (10–150)."
    ] = 150.0,
    include_usage: Annotated[
        bool,
        "Include credit usage info in the response if available."
    ] = False,
) -> str:
    response  = await _tavily_crawl_fn(
        client=async_tavily_client,
        url=url,
        instructions=instructions,
        chunks_per_source=chunks_per_source,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
        select_paths=select_paths,
        select_domains=select_domains,
        exclude_paths=exclude_paths,
        exclude_domains=exclude_domains,
        allow_external=allow_external,
        include_images=include_images,
        extract_depth=extract_depth,
        format=format,
        include_favicon=include_favicon,
        timeout=timeout,
        include_usage=include_usage,
     
    )  



       
        
    formatted_response = format_tavily_crawl_response(response) 
    return formatted_response 