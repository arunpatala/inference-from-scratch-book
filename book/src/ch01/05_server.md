# The Server

The engine from the previous section is a Python object. Anything that wants to use it must be Python code running in the same process. Wrapping the engine in an HTTP server removes that constraint: any client anywhere on the network — a curl command, a Python script, a browser, another microservice — can call the engine by making an ordinary HTTP POST request. This is how every production LLM deployment works. OpenAI's API, Anthropic's API, and self-hosted systems like vLLM and SGLang all expose the same basic shape: a POST endpoint that accepts a JSON body describing the request and returns a JSON body containing the generated text.

This section builds that server in roughly twenty lines of application code, reusing a factory from `shared/server.py` that every chapter server will use unchanged.

Run it:

```
uvicorn 05_server:app --port 8001
```

Test it (in a second terminal):

```bash
curl http://localhost:8001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## FastAPI and uvicorn

```python
from fastapi import FastAPI
from pydantic import BaseModel
```

FastAPI is a Python web framework built on two foundations. The networking layer is handled by uvicorn, an ASGI server. ASGI — Asynchronous Server Gateway Interface — is the protocol that connects a Python application to the outside world: uvicorn opens a TCP socket, waits for HTTP connections, reads each request from the wire, and hands control to FastAPI as a structured Python object. FastAPI then inspects the request, routes it to the right handler function, validates the request body with Pydantic, calls the handler, serialises the return value back to JSON, and instructs uvicorn to write the response bytes to the socket.

For our purposes the important practical consequence is that uvicorn and FastAPI are separate processes of concern. Running `uvicorn 05_server:app --port 8001` tells uvicorn to import the `app` object from `05_server.py` and start listening on port 8001. The application code never calls `accept()` or manages sockets; it only defines handlers.

## Request model

```python
class Message(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    model:       str   = "default"
    messages:    list[Message]
    max_tokens:  int   = 100
    temperature: float = 0.0
    top_p:       float = 1.0
```

`ChatRequest` is a Pydantic `BaseModel`. Pydantic models declare their fields as class-level annotations with optional defaults. When FastAPI receives an incoming POST body, it deserialises the raw JSON bytes and attempts to construct a `ChatRequest` from them. If any required field is missing, if a field's value cannot be coerced to the declared type, or if a list element does not match the inner type, Pydantic raises a `ValidationError` and FastAPI returns a `422 Unprocessable Entity` response to the client automatically. The handler function is never called. This means validation is essentially free from the handler's perspective: by the time the function body runs, every field is guaranteed to be present and correctly typed.

The field names follow the OpenAI chat completions API convention. `messages` is a list of `Message` objects, each with a `role` (`"user"`, `"assistant"`, `"system"`) and a `content` string. The convention exists because the ecosystem around LLM APIs — client libraries, proxies, benchmarking tools, evaluation harnesses — speaks this language. Matching it means our server is compatible with any tool that targets OpenAI's endpoint, which in Chapter 6 lets the benchmark script treat our server identically to a production one.

## The make_app factory

```python
def make_app(engine: BaseOfflineEngine) -> FastAPI:
    app  = FastAPI()
    _uid = [0]

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict:
        prompt = req.messages[-1].content
        params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
        )
        result: GenerateResult = engine.generate(prompt, req.max_tokens, params)
        _uid[0] += 1
        return _format_response(result, req, uid=_uid[0])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app
```

`make_app` is a factory function: it takes an engine, builds a FastAPI application wired to that engine, and returns the application object. The engine is captured in the closure over the `chat` handler. This is the same pattern used by `shared/server.py` for every chapter in the book — the factory is written once and never changes; only the engine passed to it changes.

The `chat` handler extracts the user's message from the last element of the `messages` list (`req.messages[-1].content`). Multi-turn conversation history — earlier messages in the list — is not passed to the engine in Chapter 1. Building a proper multi-turn engine requires concatenating or summarising context, which is a topic for later chapters. For now, the server behaves as a stateless single-turn responder.

`SamplingParams` is constructed from the request fields, translating the HTTP API's vocabulary into the internal contract. The `engine.generate()` call is synchronous and blocking: the handler thread calls into the engine, waits until the model produces all `max_tokens` output tokens, and only then returns the result. This works correctly for a single concurrent request. It becomes a bottleneck under load — a second request arriving while the first is generating must wait in a queue — and Chapter 9 addresses this by separating the inference loop from the HTTP handler using asynchronous scheduling.

`_uid` is a module-level list containing a single integer used as a request counter. A list rather than a bare integer is used because Python closures over a name, not a value: reassigning an integer with `_uid = _uid + 1` inside the closure would create a new local binding, leaving the outer name unchanged. Mutating the first element of a list (`_uid[0] += 1`) modifies the existing object, which the closure sees correctly. In production code a threading lock or an atomic counter would be appropriate; here a simple integer is sufficient because the server handles one request at a time.

The `/health` endpoint returns a trivial JSON object. `06_benchmark_online.py` polls this endpoint after starting the server process to confirm it is ready before sending benchmark requests. Any HTTP 200 response from `/health` signals readiness.

## The response format

```python
def _format_response(result: GenerateResult, req: ChatRequest, uid: int) -> dict:
    m = result.metrics
    return {
        "id":     f"cmpl-{uid}",
        "object": "chat.completion",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     m.input_tokens,
            "completion_tokens": m.output_tokens,
            "total_tokens":      m.input_tokens + m.output_tokens,
        },
        "metrics": m.as_dict(),
    }
```

The response follows the OpenAI chat completions object shape. `id` is a unique identifier per completion — benchmark scripts and clients use it to match responses to requests. `object` is a fixed string identifying the response type. `choices` is a list of completions; we always return exactly one, at `index: 0`. Each choice wraps the generated text in a `message` object with `role: "assistant"` and sets `finish_reason: "stop"` to indicate the model reached `max_new_tokens` without an error or a content filter triggering.

`usage` reports token counts using OpenAI's field names: `prompt_tokens`, `completion_tokens`, and their sum as `total_tokens`. These values come directly from `RequestMetrics.input_tokens` and `RequestMetrics.output_tokens`, which the engine measured during generation. Tools that consume the response — cost calculators, rate limiters, evaluation harnesses — depend on this field being accurate.

`metrics` is a non-standard extension, absent from the OpenAI specification. It surfaces the full `RequestMetrics.as_dict()` output — throughput and token counts for Chapter 1 — so that clients and benchmark scripts can read performance data from the HTTP response without needing a separate metrics endpoint. Future chapters add `ttft_ms` and `tpot_ms` to `RequestMetrics`; when they do, those fields appear in `metrics` automatically because `as_dict()` includes them conditionally.

## Wiring the engine to the server

```python
engine = BaselineEngine()
app    = make_app(engine)
```

These two lines are the entire chapter-specific portion of `05_server.py`. The engine is constructed at module import time, meaning the model is loaded when uvicorn imports the module, before the first request arrives. `make_app` wraps it. The resulting `app` object is what uvicorn holds a reference to and calls for every incoming HTTP request.

`BaselineEngine` appears in full in `05_server.py` rather than being imported from `04_engine.py` because this file is designed to be read as a standalone introduction to the engine-plus-server pattern: everything a reader needs to understand how the server works is visible in one file. The thin `engine.py` wrapper in the same directory uses `importlib` to import `BaselineEngine` from `04_engine.py` for the benchmark scripts, which need a clean import path. Both point to the same implementation; neither is a copy of different code.

A `curl` response from this server looks like:

```json
{
  "id": "cmpl-1",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "The bee flies anyway."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 40,
    "total_tokens": 58
  },
  "metrics": {
    "throughput": 12.3,
    "input_tokens": 18,
    "output_tokens": 40
  }
}
```

The next step is to measure how fast this server actually runs — both directly, by calling the engine in-process, and over the network, by sending HTTP requests to the running server.
