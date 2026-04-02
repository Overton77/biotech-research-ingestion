"""Socket.IO ASGI app — /research namespace with CORS."""

import logging
from typing import Any

import socketio

from src.config import get_settings
from src.config.cors import build_allowed_origins
from src.api.socketio.handlers import (
    handle_send_message,
    handle_plan_approved,
    handle_plan_rejected,
)

logger = logging.getLogger(__name__)

_sio: socketio.AsyncServer | None = None


def get_sio() -> socketio.AsyncServer:
    global _sio
    if _sio is None:
        settings = get_settings()
        cors = "*" if settings.SOCKETIO_CORS_STAR else build_allowed_origins(settings)
        _sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=cors,
            logger=logger.isEnabledFor(logging.DEBUG),
            engineio_logger=False,
        )
        _sio.register_namespace(ResearchNamespace("/research"))
    return _sio


class ResearchNamespace(socketio.AsyncNamespace):
    namespace = "/research"

    async def on_connect(self, sid: str, environ: Any, auth: Any) -> bool:
        logger.info("Socket connected: sid=%s namespace=%s", sid, self.namespace)
        return True

    async def on_disconnect(self, sid: str) -> None:
        logger.info("Socket disconnected: sid=%s", sid)

    async def on_join_thread(self, sid: str, data: dict[str, Any]) -> None:
        thread_id = data.get("thread_id")
        if thread_id:
            room = f"thread:{thread_id}"
            await self.enter_room(sid, room)
            logger.debug("sid=%s joined room %s", sid, room)

    async def on_join_mission(self, sid: str, data: dict[str, Any]) -> None:
        mission_id = data.get("mission_id")
        if mission_id:
            room = f"mission:{mission_id}"
            await self.enter_room(sid, room)
            logger.debug("sid=%s joined mission room %s", sid, room)

    async def on_leave_mission(self, sid: str, data: dict[str, Any]) -> None:
        mission_id = data.get("mission_id")
        if mission_id:
            room = f"mission:{mission_id}"
            await self.leave_room(sid, room)
            logger.debug("sid=%s left mission room %s", sid, room)

    async def on_send_message(self, sid: str, data: dict[str, Any]) -> None:
        thread_id = data.get("thread_id")
        content = data.get("content") or ""
        if not thread_id:
            await self.emit("error", {"message": "thread_id required", "code": "bad_request"}, to=sid)
            return
        await handle_send_message(self.emit, thread_id, content)

    async def on_plan_approved(self, sid: str, data: dict[str, Any]) -> None:
        thread_id = data.get("thread_id")
        interrupt_id = data.get("interrupt_id")
        plan = data.get("plan") or {}
        if not thread_id or not interrupt_id:
            await self.emit(
                "error",
                {"message": "thread_id and interrupt_id required", "code": "bad_request"},
                to=sid,
            )
            return
        await handle_plan_approved(self.emit, thread_id, interrupt_id, plan)

    async def on_plan_rejected(self, sid: str, data: dict[str, Any]) -> None:
        thread_id = data.get("thread_id")
        interrupt_id = data.get("interrupt_id")
        notes = data.get("notes") or ""
        if not thread_id or not interrupt_id:
            await self.emit(
                "error",
                {"message": "thread_id and interrupt_id required", "code": "bad_request"},
                to=sid,
            )
            return
        await handle_plan_rejected(self.emit, thread_id, interrupt_id, notes)


def get_sio_mount_app() -> socketio.ASGIApp:
    """Return a Socket.IO ASGI app for mounting inside FastAPI.

    Mount at ``/socket.io`` with empty socketio_path so Starlette strips
    the prefix before handing off to Socket.IO.
    """
    return socketio.ASGIApp(get_sio(), socketio_path="")
