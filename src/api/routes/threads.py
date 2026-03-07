"""Thread and message REST routes."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from beanie.odm.fields import PydanticObjectId

from src.models import Thread, Message
from src.api.schemas.common import envelope, CursorPage
from src.api.schemas.threads import ThreadCreate, ThreadResponse, MessageResponse

router = APIRouter(prefix="/threads", tags=["threads"])


def _thread_to_response(t: Thread) -> ThreadResponse:
    return ThreadResponse(
        id=str(t.id),
        title=t.title,
        created_at=t.created_at,
        updated_at=t.updated_at,
        status=t.status,
        metadata=t.metadata or {},
    )


def _message_to_response(m: Message) -> MessageResponse:
    return MessageResponse(
        id=str(m.id),
        thread_id=str(m.thread_id),
        role=m.role,
        content=m.content,
        created_at=m.created_at,
        run_id=m.run_id,
        metadata=m.metadata or {},
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_thread(body: ThreadCreate) -> dict:
    """Create a new thread."""
    thread = Thread(
        title=body.title or "New research",
        status="active",
    )
    await thread.insert()
    return envelope(_thread_to_response(thread))


@router.get("")
async def list_threads(
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> dict:
    """List threads, most recently updated first, cursor pagination."""
    if cursor:
        try:
            cursor_thread = await Thread.get(PydanticObjectId(cursor))
            if cursor_thread:
                items = (
                    await Thread.find(Thread.updated_at < cursor_thread.updated_at)
                    .sort(-Thread.updated_at)
                    .limit(limit + 1)
                    .to_list()
                )
            else:
                items = await Thread.find_all().sort(-Thread.updated_at).limit(limit + 1).to_list()
        except Exception:
            items = await Thread.find_all().sort(-Thread.updated_at).limit(limit + 1).to_list()
    else:
        items = await Thread.find_all().sort(-Thread.updated_at).limit(limit + 1).to_list()
    has_more = len(items) > limit
    if has_more:
        items = items[:limit]
    next_cursor = str(items[-1].id) if items and has_more else None
    return envelope(
        CursorPage(
            items=[_thread_to_response(t) for t in items],
            next_cursor=next_cursor,
            has_more=has_more,
        )
    )


@router.get("/{thread_id}")
async def get_thread(thread_id: str) -> dict:
    """Get a single thread by ID."""
    try:
        oid = PydanticObjectId(thread_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    thread = await Thread.get(oid)
    if not thread:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    return envelope(_thread_to_response(thread))


@router.get("/{thread_id}/messages")
async def list_messages(
    thread_id: str,
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> dict:
    """List messages for a thread, oldest first, cursor pagination."""
    try:
        tid = PydanticObjectId(thread_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    thread = await Thread.get(tid)
    if not thread:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    q = Message.find(Message.thread_id == tid).sort(Message.created_at)
    if cursor:
        try:
            cursor_msg = await Message.get(PydanticObjectId(cursor))
            if cursor_msg and cursor_msg.thread_id == tid:
                q = q.find(Message.created_at > cursor_msg.created_at)
        except Exception:
            pass
    items = await q.limit(limit + 1).to_list()
    has_more = len(items) > limit
    if has_more:
        items = items[:limit]
    next_cursor = str(items[-1].id) if items and has_more else None
    return envelope(
        CursorPage(
            items=[_message_to_response(m) for m in items],
            next_cursor=next_cursor,
            has_more=has_more,
        )
    )
