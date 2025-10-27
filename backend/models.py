"""
Pydantic models for API request/response validation.

These models define the contract between frontend and backend.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message (user or assistant)."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Request to send a message to the agent."""

    message: str = Field(..., min_length=1, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for history")
    model: Optional[str] = Field(None, description="Override default model")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "error"]
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """Available model information."""

    id: str
    name: str
    provider: Literal["anthropic", "openai", "google"]
    description: str


class ModelsResponse(BaseModel):
    """List of available models."""

    models: List[ModelInfo]
    default_model: str
