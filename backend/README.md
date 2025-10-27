# SUJBOT2 Backend

FastAPI backend for SUJBOT2 web interface with SSE streaming support.

## Setup

```bash
# Install dependencies with uv (recommended for this project)
uv pip install -r requirements.txt

# Or with standard pip
pip install -r requirements.txt
```

## Running

### Option 1: Simple Script (Recommended)

```bash
# From backend/ directory
./start_backend.sh
```

### Option 2: Manual Start

```bash
# Development mode (auto-reload)
uv run python main.py

# Or with uvicorn directly
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Start Everything (Backend + Frontend)

```bash
# From project root
./start_web.sh
```

Server will start on: http://localhost:8000

## API Endpoints

### Health Check
```
GET /health
```

Returns agent status and readiness.

### Get Models
```
GET /models
```

Returns list of available models.

### Chat Stream (SSE)
```
POST /chat/stream
Content-Type: application/json

{
  "message": "What is RAG?",
  "conversation_id": "optional-id",
  "model": "claude-sonnet-4-5-20250929"
}
```

Returns Server-Sent Events stream with:
- `text_delta`: Streaming text chunks
- `tool_call`: Tool execution started
- `tool_result`: Tool execution completed
- `cost_update`: Token usage and cost
- `done`: Stream completed
- `error`: Error occurred

### Switch Model
```
POST /model/switch?model=claude-3-5-haiku-20241022
```

Switches to a different model.

## Architecture

- **main.py**: FastAPI app with SSE endpoints
- **agent_adapter.py**: Wrapper around `AgentCore` from `src/`
- **models.py**: Pydantic schemas for request/response validation

**Key Design Principle**: Zero modifications to `src/` directory - only imports.

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Get models
curl http://localhost:8000/models

# Stream chat (SSE)
curl -N -H "Content-Type: application/json" \
  -d '{"message": "What is RAG?"}' \
  http://localhost:8000/chat/stream
```
