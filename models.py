"""Pydantic models for API request/response."""

from pydantic import BaseModel
from typing import Optional


class PrismRequest(BaseModel):
    query: str


class ToolCallInfo(BaseModel):
    tool_name: str
    args: dict
    result: str


class AgentResponse(BaseModel):
    agent_name: str
    color: str
    response: str
    tool_calls: list[ToolCallInfo]
    execution_time_ms: int
    error: str = ""


class PrismResponse(BaseModel):
    session_id: int
    user_request: str
    agents: dict[str, AgentResponse]
    merged_result: str
    contributions: list[dict]


class TaskOut(BaseModel):
    id: int
    title: str
    description: str
    status: str
    priority: str
    due_date: Optional[str] = None
    tags: list[str]
    created_by: str
    created_at: str
    updated_at: str


class NoteOut(BaseModel):
    id: int
    title: str
    content: str
    tags: list[str]
    linked_task_id: Optional[int] = None
    created_by: str
    created_at: str


class CreateTaskRequest(BaseModel):
    title: str
    description: str = ""
    priority: str = "medium"
    due_date: Optional[str] = None
    tags: str = ""


class CreateNoteRequest(BaseModel):
    title: str
    content: str = ""
    tags: str = ""
    linked_task_id: Optional[int] = None


class UpdateTaskRequest(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
