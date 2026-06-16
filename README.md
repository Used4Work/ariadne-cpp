<div align="center">

# Ariadne

**The thread through any AI labyrinth.**

[![CI](https://github.com/Used4Work/ariadne-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Used4Work/ariadne-cpp/actions/workflows/ci.yml)
[![Eval](https://github.com/Used4Work/ariadne-cpp/actions/workflows/eval.yml/badge.svg)](https://github.com/Used4Work/ariadne-cpp/actions/workflows/eval.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](#build)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

C++ LLM workflow orchestration. DAG planning, ReACT agent loops, auto-probe providers, circuit breakers, streaming, plan caching, guardrails, multi-agent handoffs.

## Why C++ over LangChain?

| | Ariadne C++ | LangChain Python |
|---|---|---|
| Process memory | **11 MB** | **~158 MB** |
| Cold start | **~12 ms** | **~1 800 ms** |
| DAG parallelism | ✓ automatic | requires LCEL |
| Agent loops | ✓ | ✓ LangGraph |
| Structured output | ✓ strict json_schema | ✓ |
| Plan caching | ✓ LRU template cache | partial |
| Token tracking | ✓ per-run | ✓ |
| Guardrails | ✓ input/output/tool | ✓ |
| Multi-agent handoffs | ✓ | ✓ LangGraph |
| Exception types | ✓ hierarchy | ✓ |

## Features

- **DAG workflows** — single-call planning, automatic topological parallelization
- **ReACT agents** — iterative reasoning with convergence detection (auto-stops stuck loops)
- **Multi-agent handoffs** — agents transfer control while sharing history
- **Plan caching** — normalized-key LRU cache skips redundant planning (−50% cost per research)
- **Strict structured output** — provider-level JSON schema enforcement (OpenAI json_schema)
- **Guardrails** — input/output/tool validation hooks that abort on violation
- **Token usage tracking** — per-run input/output/total exposed in results
- **Cancellation & timeout** — `cancel()` and `set_deadline()` for runaway protection
- **Circuit breakers + rate limiting** — per-provider fault tolerance with 429 handling
- **Streaming** — SSE token delivery
- **LLM response caching** — exact-match FNV-1a hash, LRU eviction, temperature-aware
- **MCP client** — Model Context Protocol stdio transport, auto-discover and call remote tools
- **Metrics** — pluggable `IMetricsCollector`, 7 event kinds emitted


## Windows

```powershell
# 安装依赖（vcpkg，推荐）
vcpkg install curl:x64-windows-static nlohmann-json:x64-windows-static

# 构建
cmake -B build ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows-static ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 4

set GITHUB_TOKEN=ghp_...
.\build\Release\example.exe dag
```

或直接下载 [Releases](https://github.com/Used4Work/ariadne-cpp/releases) 里的预编译 `.lib` + 头文件。

## Build

```bash
sudo apt install libcurl4-openssl-dev nlohmann-json3-dev cmake
cmake -B build && cmake --build build --parallel 4
export GITHUB_TOKEN=ghp_...
./build/example dag
./build/example agent
./build/example probe
```

## Quick Start

```cpp
#include "ariadne.hpp"
using namespace ariadne;

// Auto-probe: finds fastest alive provider
ProviderAutoPlanner planner;
planner.add_candidate("claude", ProviderConfig::anthropic(key), "strong", 1);
planner.add_candidate("groq",   ProviderConfig::openai_compatible(groq_key,
    "https://api.groq.com/openai", "llama-3.3-70b-versatile"), "fast", 1);
auto cfg = planner.probe_and_plan();

WorkflowEngine engine(cfg.config);
engine.register_tool({"web_search","Search",
    {{"required",json::array({"query"})}},{}},
    [](const json& p)->json{ /* impl */ return {}; });

// DAG workflow (single-call planning, parallel execution)
auto r1 = engine.run("Research Tesla Q4 revenue");

// ReACT agent with per-step callback
auto r2 = engine.run_agent("Compare Tesla and BYD", 10,
    [](const AgentStep& s){
        std::cout << "[" << s.iteration << "] " << s.thought << "\n"; });

// Structured output (json_mode)
Step s; s.json_mode = true;
s.output_schema = {{"type","object"},{"properties",{{"revenue",{{"type","number"}}}}}};

// Streaming
engine.run_stream("Write summary", [](const std::string& c){ std::cout << c; });
```

## Exception Types

```cpp
try { engine.run("task"); }
catch (const AllProvidersExhaustedError& e) { /* retry / re-probe */ }
catch (const ToolInputError& e)             { /* bad tool params */ }
catch (const PlanningError& e)             { /* DAG generation failed */ }
catch (const StepExecutionError& e)        { /* specific step failed */ }
catch (const AriadneError& e)              { /* any framework error */ }
```

## Provider Support

<!-- AUTO-UPDATED by model-check.yml -->
<!-- Last updated: 2026-06-08 -->

| Provider | Top Model | Tier | Key | Notes |
|---|---|---|---|---|
| Anthropic | `claude-opus-4-8` | ORCHESTRATOR | `ANTHROPIC_API_KEY` | Strongest |
| OpenAI | `gpt-5.5-2026-04-23` | ORCHESTRATOR | `OPENAI_API_KEY` | Latest flagship |
| Github Models | `openai/gpt-4.1` | SUBAGENT | `GITHUB_TOKEN` | **free** |
| Groq | `llama-3.3-70b-versatile` | SUBAGENT | `GROQ_API_KEY` | **free** ~394 tok/s |
| Ollama | `llama3.2` | SUBAGENT | *(none)* | local |

## CI/CD

| Workflow | Trigger | What |
|---|---|---|
| `ci.yml` | every push | build + 85 unit tests |
| `eval.yml` | push to main + weekly | 5 eval cases via GitHub Models |
| `model-check.yml` | weekly | update models.json |

## License
MIT
