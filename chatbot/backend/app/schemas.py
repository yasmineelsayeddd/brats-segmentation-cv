from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    session_id: str | None = None


class ToolCallInfo(BaseModel):
    tool_name: str
    tool_input: dict
    result_summary: str


class StreamEvent(BaseModel):
    type: str
    content: str | None = None
    tool_call: ToolCallInfo | None = None
    done: bool = False
