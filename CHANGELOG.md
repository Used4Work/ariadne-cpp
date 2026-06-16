# Changelog

All notable changes to Ariadne are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [1.5.0] - 2026-06-16

### Added
- Multimodal image input: `ChatMessage::with_image()`, `with_image_url()`
- OpenAI + Anthropic multimodal content in `complete_chat()`
- Gemini native tool calling: `complete_chat()` with `functionDeclarations`
  + `functionCall`/`functionResponse` parts, multimodal support
- All 3 major providers (OpenAI/Anthropic/Gemini) now have native tools

### Fixed
- All `json::parse(resp.body)` calls now guarded with try-catch
- Gemini API key moved from URL query to `x-goog-api-key` header (security)

## [1.4.0] - 2026-06-16

### Added
- Google Gemini provider (`GeminiProvider`, `generativelanguage.googleapis.com`)
- CMake version generation (`cmake/version.hpp.in` -> single source of truth)
- Now 9 providers total

### Fixed
- Version sync: all locations (hpp/CMake/vcpkg/Doxyfile) derive from one source

## [1.3.1] - 2026-06-16

### Fixed
- Parallel tool calls: `run_agent_native()` now executes ALL `tool_calls`
  (was only executing first, silently dropping parallel calls)
- Native agent hardening: metrics, LoopDetector, guardrails added

## [1.3.0] - 2026-06-16

### Added
- pkg-config `.pc` file generation
- Doxyfile for API documentation
- 10-item performance benchmark suite
- `WorkflowPlan::from_json()` preserves id + metadata

## [1.2.1] - 2026-06-16

### Added
- CONTRIBUTING.md, SECURITY.md, CHANGELOG.md
- GitHub issue templates (bug report, feature request) + PR template

### Fixed
- Version sync: hpp/CMake/vcpkg all aligned
- LICENSE expanded to full MIT text

## [1.2.0] - 2026-06-16

### Added
- README rewrite with all v1.x features
- Token-based agent history eviction via `estimate_tokens()`
- CMake FetchContent/add_subdirectory support

## [1.1.0] - 2026-06-16

### Added
- Anthropic native tool calling (`complete_chat()`)
- Retry-After header parsing
- `estimate_tokens()` heuristic
- AdaptiveOrchestrator uses `run_agent_native()` by default

## [1.0.1] - 2026-06-16

### Fixed
- Thread safety: ToolRegistry shared_mutex, cache locking, atomic flags
- HTTP 500/502/504 as retryable, safe JSON parse guards
- CURLOPT_CONNECTTIMEOUT=10s, zero raw stdout/stderr

### Added
- CI: macOS ARM64 + Linux ASan/UBSan

## [1.0.0] - 2026-06-16

### Added
- Native function calling: OpenAI tools API + `run_agent_native()`
- ChatMessage/LLMToolCall/LLMResponse types

## [0.9.x] - 2026-06-16

### Added
- Structured logging (ILogger), cost tracking (ModelPricing)
- Version API, agent traces, Studio adaptive endpoints
- Anthropic structured output fix, idempotency keys, auto-pricing

## [0.8.0] - 2026-06-16

### Added
- AdaptiveOrchestrator: 5 auto strategies

## [0.7.0] - 2026-06-16

### Added
- DynamicWorkflow: parallel/pipeline/map/loop_until/fan_out/adversarial_verify

## [0.6.x] - 2026-06-16

### Added
- Token budget, checkpointing, human-in-the-loop, serialization
- Studio port auto-retry, provider factories (cerebras/sambanova/mistral)

### Fixed
- 429 retry with exponential backoff, agent metrics, PlanCache context

## [0.5.0] - 2026-06-15

### Added
- Ariadne Studio, MCP client, guardrails, multi-agent handoffs
- Response/plan caching, LoopDetector, token usage tracking

## [0.2.0] - 2026-06-08

### Added
- Initial release: DAG workflows, ReACT agents, multi-provider, circuit breakers
