"""Socket.IO ASGI app — /research namespace, Redis adapter."""

import logging
from typing import Any

import socketio

from src.config import get_settings
from src.api.socketio.handlers import (
    handle_send_message,
    handle_plan_approved,
    handle_plan_rejected,
)

logger = logging.getLogger(__name__)

settings = get_settings()

# Redis manager for multi-process room broadcast
client_manager = socketio.AsyncRedisManager(settings.REDIS_URL)

# AsyncServer with ASGI mode
sio = socketio.AsyncServer(
    async_mode="asgi",
    client_manager=client_manager,
    logger=logger.isEnabledFor(logging.DEBUG),
    engineio_logger=False,
)


class ResearchNamespace(socketio.AsyncNamespace):
    """Namespace /research — thread and run rooms, send_message handled in handlers."""

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

    async def on_join_run(self, sid: str, data: dict[str, Any]) -> None:
        run_id = data.get("run_id")
        if run_id:
            room = f"run:{run_id}"
            await self.enter_room(sid, room)
            logger.debug("sid=%s joined room %s", sid, room)

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


sio.register_namespace(ResearchNamespace("/research"))


def get_sio_mount_app() -> socketio.ASGIApp:
    """Return a Socket.IO ASGI app suitable for mounting inside FastAPI.

    Mount at ``/socket.io`` with an empty socketio_path so that
    Starlette strips the prefix before handing off to Socket.IO::

        app.mount("/socket.io", get_sio_mount_app())

    Clients should connect with ``{ path: "/socket.io" }`` (the default).
    """
    return socketio.ASGIApp(sio, socketio_path="")
