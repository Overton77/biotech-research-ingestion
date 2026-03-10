from typing import Literal, Union, List, Dict, Any, Optional
from tavily import AsyncTavilyClient  
from datetime import datetime  
import asyncio   


async def tavily_crawl(
    client: AsyncTavilyClient,
    url: str,
    instructions: Optional[str] = None,
    chunks_per_source: int = 3,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    select_paths: Optional[List[str]] = None,
    select_domains: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    allow_external: bool = True,
    include_images: bool = False,
    extract_depth: Literal["basic", "advanced"] = "basic",
    format: Literal["markdown", "text"] = "markdown",
    include_favicon: bool = False,
    timeout: float = 150.0,
    include_usage: bool = False,
) -> Dict[str, Any]:
    """
    Async wrapper around Tavily's crawl method.

    Uses explicit keyword args (no **kwargs). Tries to avoid passing optional fields
    when not provided (notably instructions).
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("tavily_crawl: url must be a non-empty string.")

    # Tavily docs: chunks_per_source is only available when instructions are provided.
    if instructions and instructions.strip():
        # Keep chunks_per_source in its documented range when instructions are present.
        if not (1 <= int(chunks_per_source) <= 5):
            raise ValueError("tavily_crawl: chunks_per_source must be between 1 and 5 when instructions are provided.")

        return await client.crawl(
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

    # instructions is None/empty -> do not pass instructions nor chunks_per_source
    return await client.crawl(
        url=url,
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


async def tavily_crawl_multiple(
    client: AsyncTavilyClient,
    urls: List[str],
    instructions: Optional[str] = None,
    chunks_per_source: int = 3,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    select_paths: Optional[List[str]] = None,
    select_domains: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    allow_external: bool = True,
    include_images: bool = False,
    extract_depth: Literal["basic", "advanced"] = "basic",
    format: Literal["markdown", "text"] = "markdown",
    include_favicon: bool = False,
    timeout: float = 150.0,
    include_usage: bool = False,
    concurrency: int = 3,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def _one(u: str) -> Dict[str, Any]:
        async with sem:
            res = await tavily_crawl(
                client=client,
                url=u,
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
            res["__url"] = u
            return res

    cleaned = [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]
    if not cleaned:
        raise ValueError("tavily_crawl_multiple: urls must contain at least one valid URL.")
    return await asyncio.gather(*[_one(u) for u in cleaned])


def format_tavily_crawl_response(
    response: Dict[str, Any] | List[Dict[str, Any]],
    *,
    max_results: Optional[int] = None,
    max_content_chars: Optional[int] = 2000,
) -> str:
    """
    Format Tavily crawl results into a compact, LLM-friendly string.

    Handles:
      - single crawl response: { base_url, results: [{url, raw_content, ...}], ... }
      - list of crawl responses (from tavily_crawl_multiple)
    """
    def _format_one(resp: Dict[str, Any]) -> str:
        base_url = resp.get("base_url") or resp.get("__url") or "(unknown base_url)"
        results = resp.get("results") or []
        usage = resp.get("usage")
        response_time = resp.get("response_time")
        request_id = resp.get("request_id")

        if max_results is not None:
            results_local = results[:max_results]
        else:
            results_local = results

        lines: List[str] = []
        lines.append(f"=== Crawl Results: {base_url} ===")
        lines.append(f"Pages: {len(results)}")
        if response_time is not None:
            lines.append(f"Response time: {response_time}")
        if request_id:
            lines.append(f"Request id: {request_id}")
        if usage:
            # usage format depends on SDK; just render json-ish
            lines.append(f"Usage: {usage}")
        lines.append("")

        if not results_local:
            lines.append("(no pages)")
            return "\n".join(lines).strip()

        for idx, page in enumerate(results_local, start=1):
            purl = page.get("url") or "(no url)"
            raw = page.get("raw_content") or ""
            favicon = page.get("favicon")
            images = page.get("images") or page.get("image_urls") or None

            if max_content_chars is not None and raw and len(raw) > max_content_chars:
                raw = raw[:max_content_chars] + " …[truncated]"

            lines.append(f"[Page {idx}]")
            lines.append(f"URL: {purl}")
            if favicon:
                lines.append(f"Favicon: {favicon}")
            if images:
                try:
                    lines.append(f"Images: {len(images)}")
                except Exception:
                    lines.append("Images: (present)")
            lines.append("Content:")
            lines.append(raw.strip() if raw else "(empty)")
            lines.append("")

        return "\n".join(lines).strip()

    if isinstance(response, list):
        blocks = [_format_one(r) for r in response if isinstance(r, dict)]
        return "\n\n".join([b for b in blocks if b]).strip()

    if isinstance(response, dict):
        return _format_one(response)

    return f"(unexpected response type: {type(response).__name__})"





async def tavily_search(
    client: AsyncTavilyClient,
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    topic: Optional[Literal["general", "news", "finance"]] = None,
    include_images: bool = False,
    include_raw_content: bool | Literal["markdown", "text"] = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("tavily_search: query must be non-empty")

    # NOTE: explicit keyword args only; only pass domain args if not None
    if include_domains is not None or exclude_domains is not None:
        return await client.search(
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

    return await client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        include_images=include_images,
        include_raw_content=include_raw_content,
        start_date=start_date,
        end_date=end_date,
    )

async def tavily_search_multiple(
    client: AsyncTavilyClient,
    queries: List[str],
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    topic: Optional[Literal["general", "news", "finance"]] = None,
    include_images: bool = False,
    include_raw_content: bool | Literal["markdown", "text"] = False,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    concurrency: int = 5,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: str) -> Dict[str, Any]:
        async with sem:
            res = await tavily_search(
                client=client,
                query=q,
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
            # preserve query
            res["__query"] = q
            return res

    cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    return await asyncio.gather(*[_one(q) for q in cleaned])

def format_tavily_search_response(
    response: Dict[str, Any] | List[Dict[str, Any]],
    *,
    max_results: Optional[int] = None,
    max_content_chars: Optional[int] = None,
) -> str:
    """
    Format Tavily search results into a compact, LLM-friendly string.

    Works with:
      - The raw TavilyClient.search() response (dict with 'results' and 'images').
      - The langchain TavilySearch tool output (same shape).
      - Or directly with a bare list of result dicts.

    Fields included per result:
      - title
      - url
      - score (float)
      - optional published_date
      - content (falls back to raw_content if content is missing)

    Also includes image results (if present) with:
      - url
      - optional description

    Args:
        response: Tavily search response or list of result objects.
        max_results: Optional cap on number of results to format.
        max_content_chars: Optional cap on content length per result.

    Returns:
        A well-structured string ready to feed to a summarization model.
    """
    # Normalize to dict with "results" and "images"
    if isinstance(response, list):
        results = response
        images: List[Any] = []
        query_str = None
    elif isinstance(response, dict):
        results = response.get("results") or []
        images = response.get("images") or []
        query_str = response.get("query") 



    if max_results is not None:
        results = results[:max_results]

    lines: List[str] = []

    # Optional header with original query
    if query_str:
        lines.append(f"Search query: {query_str}")
        lines.append("")

    # --- Main results ---
    lines.append("=== Web Results ===")
    if not results:
        lines.append("(no results)")
    else:
        for idx, r in enumerate(results, start=1):
            title = r.get("title") or "(no title)"
            url = r.get("url") or "(no url)"
            score = r.get("score")
            published_date = r.get("published_date")

            # content fallback to raw_content
            content = r.get("content") or r.get("raw_content") or ""
            if max_content_chars is not None and content and len(content) > max_content_chars:
                content = content[:max_content_chars] + " …[truncated]"

            lines.append(f"[Result {idx}]")
            lines.append(f"Title: {title}")
            lines.append(f"URL: {url}")

            if score is not None:
                try:
                    lines.append(f"Score: {float(score):.4f}")
                except (TypeError, ValueError):
                    lines.append(f"Score: {score}")

            if published_date:
                lines.append(f"Published date: {published_date}")

            if content:
                lines.append("Content:")
                lines.append(content.strip())
            else:
                lines.append("Content: (empty)")

            lines.append("")  # blank line between results

    # --- Image results ---
    lines.append("=== Image Results ===")
    if not images:
        lines.append("(no images)")
    else:
        for idx, img in enumerate(images, start=1):
            # Tavily can return either a simple string URL or a dict with more fields
            if isinstance(img, str):
                img_url = img
                description = None
            elif isinstance(img, dict):
                img_url = img.get("url") or img.get("image_url") or "(no url)"
                description = img.get("description") or img.get("alt") or None
            else:
                img_url = f"(unexpected image type: {img!r})"
                description = None

            lines.append(f"[Image {idx}]")
            lines.append(f"URL: {img_url}")
            if description:
                lines.append(f"Description: {description}")
            lines.append("")

    return "\n".join(lines).strip() 


async def tavily_extract(
    client: AsyncTavilyClient,
    urls: Union[str, List[str]],
    query: Optional[str] = None,
    chunks_per_source: int = 3,
    extract_depth: Literal["basic", "advanced"] = "basic",
    include_images: bool = False,
    include_favicon: bool = False,
    format: Literal["markdown", "text"] = "markdown",
) -> Dict[str, Any]:
    urls_list = [urls] if isinstance(urls, str) else list(urls or [])
    urls_list = [u for u in urls_list if isinstance(u, str) and u.strip()]
    if not urls_list:
        raise ValueError("tavily_extract: urls must contain at least one valid URL.")

    if query and query.strip():
        return await client.extract(
            urls=urls_list,
            query=query,
            chunks_per_source=chunks_per_source,
            extract_depth=extract_depth,
            include_images=include_images,
            include_favicon=include_favicon,
            format=format,
        )

    return await client.extract(
        urls=urls_list,
        extract_depth=extract_depth,
        include_images=include_images,
        include_favicon=include_favicon,
        format=format,
    )

async def tavily_map(
    client: AsyncTavilyClient,
    url: str,
    instructions: Optional[str] = None,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 25,
) -> Dict[str, Any]:
    """
    Async wrapper around Tavily's map method.

    ✅ No **kwargs dict passed into the SDK (avoids Tavily SDK signature issues).
    ✅ Only includes 'instructions' if provided.
    ✅ Default limit reduced (fast research path).

    Works with tavily-python>=0.7.13
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("tavily_map: url must be a non-empty string.")

    # Explicit argument passing, like search:
    if instructions:
        return await client.map(
            url=url,
            instructions=instructions,
            max_depth=max_depth,
            max_breadth=max_breadth,
            limit=limit,
        )

    # instructions is None → do not pass it
    return await client.map(
        url=url,
        max_depth=max_depth,
        max_breadth=max_breadth,
        limit=limit,
    )


def format_tavily_extract_response(
    response: Dict[str, Any],
    *,
    max_results: Optional[int] = None,
    max_content_chars: Optional[int] = None,
) -> str:
    """
    Format Tavily extract results into a compact, LLM-friendly string.

    Args:
        response: Tavily extract response (dict with 'results' and 'failed_results').
        max_results: Optional cap on number of results to format.
        max_content_chars: Optional cap on content length per result.

    Returns:
        A well-structured string ready to feed to a summarization model.
    """
    results = response.get("results", [])
    failed_results = response.get("failed_results", [])

    if max_results is not None:
        results = results[:max_results]

    lines: List[str] = []

    # --- Main results ---
    lines.append("=== Extracted Web Content ===")
    if not results:
        lines.append("(no results)")
    else:
        for idx, r in enumerate(results, start=1):
            url = r.get("url") or "(no url)"
            raw_content = r.get("raw_content") or ""
            images = r.get("images") or []
            favicon = r.get("favicon")

            # Try to extract a title from the content if available
            title = None
            if raw_content:
                # Look for first heading or use first line
                content_lines = raw_content.split("\n")
                for line in content_lines[:10]:  # Check first 10 lines
                    line = line.strip()
                    if line and (line.startswith("#") or len(line) < 200):
                        title = line.replace("#", "").strip()
                        break
                if not title and content_lines:
                    title = content_lines[0][:100].strip()

            title = title or "(no title)"

            # Truncate content if needed
            if max_content_chars is not None and raw_content and len(raw_content) > max_content_chars:
                raw_content = raw_content[:max_content_chars] + " …[truncated]"

            lines.append(f"[Result {idx}]")
            lines.append(f"Title: {title}")
            lines.append(f"URL: {url}")
            
            if favicon:
                lines.append(f"Favicon: {favicon}")

            if raw_content:
                lines.append("Content:")
                lines.append(raw_content.strip())
            else:
                lines.append("Content: (empty)")

            if images:
                lines.append(f"Images: {len(images)} image(s) found")

            lines.append("")  # blank line between results

    # --- Failed results ---
    if failed_results:
        lines.append("=== Failed Extractions ===")
        for idx, failed in enumerate(failed_results, start=1):
            failed_url = failed.get("url") if isinstance(failed, dict) else str(failed)
            lines.append(f"[Failed {idx}] {failed_url}")
        lines.append("")

    return "\n".join(lines).strip()


def format_tavily_map_response(response: Dict[str, Any], urls_override: Optional[List[str]] = None) -> str:
    """
    Format Tavily map results into a simple list of URLs.

    Args:
        response: Tavily map response (dict with 'base_url' and 'results' list of URLs).
        urls_override: Optional list of URLs to use instead of response['results'].
                      Useful when URLs have been deduped/capped after the API call.

    Returns:
        A formatted string with base URL and list of discovered URLs.
    """
    base_url = response.get("base_url", "unknown")
    results = urls_override if urls_override is not None else response.get("results", [])

    lines: List[str] = []
    lines.append(f"=== Site Map: {base_url} ===")
    lines.append(f"Discovered {len(results)} URL(s):")
    lines.append("")

    if not results:
        lines.append("(no URLs discovered)")
    else:
        for idx, url in enumerate(results, start=1):
            lines.append(f"{idx}. {url}")

    return "\n".join(lines).strip()



