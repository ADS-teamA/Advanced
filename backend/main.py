"""
FastAPI Backend for SUJBOT2 Web Interface

Provides RESTful API and SSE streaming for the agent.
Strictly imports from src/ without modifications.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from agent_adapter import AgentAdapter
from models import ChatRequest, HealthResponse, ModelsResponse, ModelInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global agent adapter instance
agent_adapter: Optional[AgentAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global agent_adapter

    # Startup
    try:
        logger.info("Initializing agent adapter...")
        agent_adapter = AgentAdapter()
        logger.info("Agent adapter initialized successfully")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize agent: {e}", exc_info=True)
        logger.error("Server cannot start without agent. Fix configuration and restart.")
        # Fail fast - prevent server from starting in broken state
        raise RuntimeError("Cannot start server without initialized agent") from e

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SUJBOT2 Web API",
    description="Web interface for SUJBOT2 RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for localhost development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns agent status and readiness.
    """
    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    health_status = agent_adapter.get_health_status()

    if health_status["status"] == "error":
        raise HTTPException(
            status_code=503,
            detail=health_status["message"]
        )

    return HealthResponse(**health_status)


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """
    Get list of available models.

    Returns models with provider and description.
    """
    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    models = agent_adapter.get_available_models()
    default_model = agent_adapter.config.model

    return ModelsResponse(
        models=[ModelInfo(**m) for m in models],
        default_model=default_model
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response using Server-Sent Events (SSE).

    Events:
    - text_delta: Streaming text chunks from agent response
    - tool_call: Tool execution started (streamed immediately when detected)
    - tool_calls_summary: Summary of all tool calls with results (sent after completion)
    - cost_update: Token usage and cost update
    - done: Stream completed
    - error: Error occurred

    Example event format:
    ```
    event: text_delta
    data: {"content": "Hello"}

    event: tool_call
    data: {"tool_name": "search", "tool_input": {}, "call_id": "tool_search"}

    event: tool_calls_summary
    data: {"tool_calls": [...], "count": 2}

    event: cost_update
    data: {"summary": "...", "total_cost": 0.001, ...}

    event: done
    data: {}
    ```

    Note: tool_call events are sent IMMEDIATELY when Claude decides to use a tool,
    before the tool execution completes. This enables real-time UI updates showing
    which tools are being invoked.
    """
    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    # Switch model if specified
    if request.model and request.model != agent_adapter.config.model:
        logger.info(f"Switching model to: {request.model}")
        agent_adapter.switch_model(request.model)

    async def event_generator():
        """Generate SSE events from agent stream."""
        try:
            async for event in agent_adapter.stream_response(
                query=request.message,
                conversation_id=request.conversation_id
            ):
                # Format as SSE event
                event_type = event["event"]

                # Try to serialize with UTF-8, fall back to ASCII on error
                try:
                    event_data = json.dumps(event["data"], ensure_ascii=False)
                except (TypeError, ValueError, UnicodeDecodeError) as e:
                    logger.error(
                        f"Failed to serialize SSE event data with UTF-8: {e}. "
                        f"Event type: {event.get('event')}. Falling back to ASCII.",
                        exc_info=True
                    )
                    # Fall back to ASCII encoding (escapes non-ASCII as \uXXXX)
                    try:
                        event_data = json.dumps(event["data"], ensure_ascii=True)
                    except Exception as fallback_error:
                        logger.error(f"ASCII fallback also failed: {fallback_error}", exc_info=True)
                        # Send error event instead of crashing entire stream
                        yield {
                            "event": "error",
                            "data": json.dumps({
                                "error": f"Server failed to encode response data: {type(e).__name__}",
                                "type": "EncodingError",
                                "event_type": event.get("event")
                            }, ensure_ascii=True)
                        }
                        continue

                yield {
                    "event": event_type,
                    "data": event_data
                }

        except asyncio.CancelledError:
            # Client disconnected - this is normal, don't log as error
            logger.info("Stream cancelled by client")
            # Don't yield error event, just stop
            return
        except (KeyboardInterrupt, SystemExit):
            # Don't catch these - let them propagate for clean shutdown
            raise
        except MemoryError as e:
            logger.critical(f"OUT OF MEMORY during streaming: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "Server out of memory. Please contact administrator.",
                    "type": "MemoryError"
                }, ensure_ascii=True)
            }
        except Exception as e:
            logger.error(f"Error in event generator: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "type": type(e).__name__
                }, ensure_ascii=True)  # Use ASCII for error messages (defensive)
            }

    return EventSourceResponse(event_generator())


@app.post("/model/switch")
async def switch_model(model: str):
    """
    Switch to a different model.

    Args:
        model: Model identifier

    Returns:
        Success confirmation
    """
    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    try:
        agent_adapter.switch_model(model)
        return {"success": True, "model": model}
    except Exception as e:
        logger.error(f"Failed to switch model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch model: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SUJBOT2 Web API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
