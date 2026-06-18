<div align="center">

# Ariadne

**The thread through any AI labyrinth.**

[![CI](https://github.com/Used4Work/ariadne-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Used4Work/ariadne-cpp/actions/workflows/ci.yml)
[![Eval](https://github.com/Used4Work/ariadne-cpp/actions/workflows/eval.yml/badge.svg)](https://github.com/Used4Work/ariadne-cpp/actions/workflows/eval.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](#build)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-160%20passed-brightgreen)](#tests)

</div>

C++17 LLM workflow orchestration library. Automatic DAG planning, ReACT agents, native function calling (OpenAI + Anthropic + Gemini), multimodal vision, dynamic multi-agent orchestration, plan caching, circuit breakers, streaming, and a visual workflow editor.

## Why C++?

| | Ariadne C++ | LangChain Python |
|---|---|---|
| Memory | **11 MB** | ~158 MB |
| Cold start | **12 ms** | ~1800 ms |
| Thread safety | Built-in (shared_mutex, atomics) | GIL-limited |
| Native tools | All 9 providers | OpenAI + Anthropic |
| Multimodal | Image input (base64 + URL) | Image input |
| Dynamic orchestration | parallel/pipeline/loop_until | LCEL chains |
| Auto strategy | AdaptiveOrchestrator (5 modes) | Manual selection |

## Features

### Core Orchestration
- **DAG workflows** -- single-call planning, automatic topological parallelization
- **ReACT agents** -- iterative reasoning with convergence detection (LoopDetector)
- **Native function calling** -- OpenAI/Anthropic/Gemini tools API, parallel tool calls, `tool_choice` control
- **Multi-agent handoffs** -- agents transfer control while sharing history
- **AdaptiveOrchestrator** -- LLM auto-selects optimal strategy per task
- **Multimodal vision** -- image input via base64 or URL (OpenAI + Anthropic + Gemini)

### Dynamic Workflow (Ultracode-level)
- `parallel()` -- fan-out N tasks, barrier wait
- `pipeline()` -- no-barrier item-by-item flow through stages
- `map()` -- parallel map over collection
- `loop_until()` -- dynamic loops with budget awareness
- `fan_out_agents()` -- parallel ReACT agent spawning
- `adversarial_verify()` -- multi-vote claim verification

### Reliability
- **9 providers** -- OpenAI, Anthropic, Gemini, Groq, GitHub Models, Cerebras, SambaNova, Mistral, LLM7
- **Circuit breakers** -- per-provider fault isolation (CLOSED/OPEN/HALF_OPEN)
- **Rate limiting** -- token bucket per provider, configurable RPS
- **429/5xx retry** -- exponential backoff with Retry-After header parsing
- **Idempotency keys** -- auto-generated per request, prevents double-billing
- **Token budget** -- enforcement with `TokenBudgetError` on exceed

### Observability
- **Structured logging** -- `ILogger` interface, zero stdout/stderr by default
- **Cost tracking** -- `ModelPricing` with auto-pricing per provider
- **Metrics** -- pluggable `IMetricsCollector`, 10 event kinds
- **Token estimation** -- `estimate_tokens()` heuristic

### Persistence
- **Plan caching** -- NeurIPS 2025 APC pattern, context-aware LRU
- **Response caching** -- FNV-1a hash, temperature-aware
- **Checkpointing** -- `ICheckpointStore` / `FileCheckpointStore`
- **Serialization** -- `WorkflowPlan::to_json()` / `from_json()`

### Safety
- **Guardrails** -- input/output/tool validation hooks
- **Human-in-the-loop** -- `InterruptError` + `set_interrupt_hook()`
- **Cancellation** -- `cancel()` + `set_deadline()`
- **Thread-safe** -- `shared_mutex` on ToolRegistry, atomic flags

### Integration
- **MCP client** -- Model Context Protocol (stdio transport, JSON-RPC 2.0)
- **Ariadne Studio** -- visual workflow editor (localhost web UI)
- **Streaming** -- SSE token delivery
- **CMake** -- FetchContent, find_package, pkg-config

## Quick Start

```cpp
#include "ariadne.hpp"
using namespace ariadne;

int main() {
    // One-line setup
    WorkflowEngine engine(ProviderConfig::github_models(token));
    engine.register_tool({"web_search","Search the web",
        {{"required",json::array({"query"})}},{}},
        [](const json& p)->json{ return {{"result","data"}}; });

    // 1. Adaptive orchestration (auto-selects best strategy)
    AdaptiveOrchestrator orch(engine);
    auto r = orch.run("Compare Tesla and BYD Q4 2025 sales");

    // 2. Native tool calling agent (97-99% accuracy)
    auto ar = engine.run_agent_native("Research Tesla revenue", 10);

    // 3. Dynamic workflow
    DynamicWorkflow dw(engine);
    auto results = dw.fan_out_agents({"Search Tesla","Search BYD"}, 8);

    // 4. Multimodal vision
    auto msg = ChatMessage::with_image_url("Describe this image",
        "https://example.com/photo.jpg");

    // 5. DAG workflow
    auto wr = engine.run("Search Tesla revenue and write a summary");

    // 6. Streaming
    engine.run_stream("Write a poem", [](auto& c){ std::cout << c; });
}
```

## Build

```bash
# Linux / macOS
sudo apt install libcurl4-openssl-dev nlohmann-json3-dev  # or brew install
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
./build/unit_tests          # 160 tests
./build/ariadne-studio      # visual editor at localhost:8080

# Windows (vcpkg)
vcpkg install --triplet x64-windows-static
cmake -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# As a dependency (FetchContent)
include(FetchContent)
FetchContent_Declare(ariadne GIT_REPOSITORY https://github.com/Used4Work/ariadne-cpp.git GIT_TAG v2.3.0)
FetchContent_MakeAvailable(ariadne)
target_link_libraries(myapp PRIVATE ariadne::ariadne)
```

## Ariadne Studio

Visual workflow editor -- drag-and-drop DAG builder with real-time execution.

```bash
./build/ariadne-studio              # default port 8080
./build/ariadne-studio 9090         # custom port
ARIADNE_PORT=3000 ./ariadne-studio  # via env var (auto-retry on conflict)
```

Features: Drawflow canvas, 4 node types (LLM/Tool/Transform/Condition), adaptive orchestration button, SSE execution streaming, save/load/export/import, keyboard shortcuts (Ctrl+S/R, Del).

## Provider Support

All 9 providers support native tool calling via `complete_chat()`.

| Factory | Default Model | Free? | Rate Limit | Native Tools |
|---|---|---|---|---|
| `github_models(token)` | openai/gpt-4o-mini | Yes | 6 RPM | Yes |
| `llm7()` | deepseek-v3-0324 | Yes (no signup) | 30 RPM | Yes |
| `gemini(key)` | gemini-2.0-flash | Yes (15 RPM) | 15 RPM | Yes |
| `cerebras(key)` | llama-3.3-70b | Yes (1M tok/day) | 30 RPM | Yes |
| `sambanova(key)` | Meta-Llama-3.3-70B | Yes | 30 RPM | Yes |
| `groq(key)` | llama-3.3-70b | Yes | 30 RPM | Yes |
| `mistral(key)` | mistral-small | Yes (1B tok/mo) | 60 RPM | Yes |
| `openai_chat(key)` | gpt-4o | Paid | Unlimited | Yes |
| `anthropic(key)` | claude-opus-4-8 | Paid | Unlimited | Yes |

## Exception Hierarchy

```
AriadneError
+-- ProviderError -> AllProvidersExhaustedError
+-- PlanningError
+-- WorkflowCancelledError
+-- GuardrailError
+-- TokenBudgetError
+-- InterruptError
+-- ToolError -> ToolNotFoundError, ToolInputError
+-- DAGValidationError
+-- StepExecutionError
```

## CI/CD

| Workflow | Trigger | What |
|---|---|---|
| `ci.yml` | every push | Build (Linux+Windows+macOS+ASan/UBSan) + 160 tests |
| `eval.yml` | push to main + weekly | 5 eval cases via GitHub Models |
| `release.yml` | tag `v*` | Cross-platform binaries -> GitHub Releases |

## License

MIT
