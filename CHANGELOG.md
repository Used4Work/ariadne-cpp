# Changelog

All notable changes to Ariadne are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [2.2.0] - 2026-06-17

### Added
- **Observation masking** (D64): replaces simple truncation in agent history. Old tool results → `[masked, N chars]`, keeps last 6 verbatim. Based on JetBrains/NeurIPS research (+2.6% solve rate, -50% cost). Applied to `run_agent()`, `run_multi_agent()`, and `run_agent_native()`
- **Sorted tool definitions** (D65): tools sorted alphabetically before prompt construction for stable provider cache prefix. Applied to both `build_agent_prompt()` and `run_agent_native()` tool list
- 3 new tests (153 total)

### Changed
- `build_agent_prompt()` takes tools by value for in-place sorting
- Native agent history eviction upgraded from message deletion to content masking (preserves conversation structure)

## [2.1.1] - 2026-06-17

### Added
- MockProvider now supports native tool calling (`complete_chat()`, `supports_native_tools()`, `set_tool_calls()`)
- Native agent history eviction: `run_agent_native()` trims messages when token estimate > 8000 (D60)
- Explicit `= delete` for copy/move on WorkflowEngine and DynamicWorkflow
- 3 new tests (150 total)

### Fixed
- **StdioTransport Windows pipe leak**: 4 handles leaked if `CreateProcessA` failed — now cleaned up before throw
- `test_mock_supports_native` updated to match new MockProvider behavior

## [2.1.0] - 2026-06-17

### Added
- `tool_choice` parameter across all providers (OpenAI/Anthropic/Gemini) — auto/none/required/specific
- `McpError` exception class for MCP transport/protocol errors
- Real Gemini SSE streaming via `streamGenerateContent?alt=sse` (was fake word-split)
- Anthropic `json_schema` structured output (constrained decoding via `response_format`)
- Gemini `responseSchema` for JSON schema-constrained output
- 7 new unit tests (147 total)

### Fixed
- **Version drift**: all 5 version sources (CMake, vcpkg, Doxyfile, hpp, ports) synced to 2.1.0
- **Exception hierarchy**: 21 `std::runtime_error` → typed exceptions (ProviderError, StepExecutionError, McpError)
- **D41 violation**: `ConsoleMetrics::record()` used `std::cout` → now uses `log_msg()`
- **ResponseCache collision**: FNV-1a hash key → full canonical key (zero collision risk)
- **Silent SSE overflow**: 16MB buffer abort now logs `LOG_ERROR` instead of failing silently
- SseParser moved before provider implementations (fixes Gemini streaming compilation)

### Changed
- `ILLMProvider::complete_chat()` signature: added `tool_choice` parameter (default "auto")
- `AnthropicProvider::complete()`: `output_schema` now used for `json_schema` response format
- `GeminiProvider::complete()`: `output_schema` now used for `responseSchema` constrained decoding

## [2.0.1] - 2026-06-17

### Fixed
- MCP HTTP notification crash on empty 202 response body
- Dead code: unreachable `cosine_similarity()` removed after pre-normalization
- Vector store: dimension mismatch returns score=0 instead of silent truncation

## [2.0.0] - 2026-06-17

### Added
- `InMemoryVectorStore` — cosine similarity, pre-normalized, O(N log K) partial_sort query
- MCP HTTP transport (`HttpTransport`, Streamable HTTP)
- Doxygen → GitHub Pages CI workflow
- Convenience constructors: `WorkflowEngine(ProviderConfig)`, `WorkflowEngine(orc, sub)`

## [1.6.0] - 2026-06-16

### Added
- Full README rewrite, comprehensive Quick Start
- Integration tests (mock DAG, dynamic parallel, multimodal roundtrip)
- Version sync test

## [1.5.1] - 2026-06-16

### Fixed
- All C4244 int64→long truncation warnings (21 occurrences)
- Doc accuracy (Studio API, file line counts)

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
