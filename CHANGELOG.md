# Changelog

All notable changes to Ariadne are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [2.9.0] - 2026-06-24

Theme: **Composability & production ops.** Researched against live June 2026 docs (LangGraph 1.0/1.2 subgraphs + durable execution, eval-driven prompt development, MCP 2025-11-25 stable spec).

### Added
- **Sub-workflow nesting** (D93): `WorkflowRegistry` registers named `SubWorkflowFn` units (input json → output json) with thread-safety + recursion-depth guard. `WorkflowEngine::register_sub_workflow()`/`run_sub_workflow()`/`register_workflow_as_tool()` let a parent agent or DAG invoke a reusable workflow as a tool (LangGraph "subgraph as a node" pattern). Self-recursion past `max_depth` throws `StepExecutionError`.
- **Prompt versioning** (D94): `PromptVersionStore` keeps multiple named prompt versions with an active-version pointer + history (`add`/`get`/`render`/`set_active`/`versions`/`active_version`). Git-style prompt management; complements the template-rendering `PromptRegistry`.
- **Prompt eval gate** (D95): `PromptEvalGate` runs a golden set of `EvalCase`s through a `Runner` + `Scorer`, returns an `EvalReport` (per-case pass count, avg score) and gates on an avg-score threshold. Eval-driven development primitive — pluggable runner/scorer (LLM, local, or test stub), testable without HTTP.
- **MCP 2025-11-25 protocol version** (D96): `MCP_PROTOCOL_VERSION` constant; `McpClient::initialize()` advertises `2025-11-25` (was `2025-06-18`); `HttpTransport` sends the `MCP-Protocol-Version` header on every request. Aligns with the latest stable MCP spec (async Tasks, OAuth, extensions).
- 13 new tests (229 total)

### Changed
- MCP transport now advertises protocol version `2025-11-25` (was `2025-06-18`).

## [2.8.0] - 2026-06-23

Theme: **Retrieval, memory & context engineering + A2A interop.** Researched against live June 2026 docs (A2A v1.0 Linux Foundation, MCP 2025-11-25, hybrid-retrieval RAG gold standard, context-compaction research arXiv 2601.07190).

### Added
- **BM25 + RRF hybrid retrieval** (D89): `Bm25Index` (Okapi BM25, k1=1.2/b=0.75, Lucene恒正 IDF) for sparse lexical search; `reciprocal_rank_fusion()` (k=60) rank-based fusion; `HybridRetriever` combines vector cosine + BM25 via RRF. Now the industry RAG default — sparse handles exact/rare-term matches, dense handles paraphrase.
- **Memory scoping + temporal** (D90): `MemoryQuery` options (`scope_prefix`, `recency_half_life_sec`, `now_ts`); `InMemoryVectorStore::add_scoped()` + scoped `query()` overload. Filter by `user:`/`session:`/`agent:` scope, re-rank with exponential recency decay `0.5^(age/half_life)`.
- **Context compaction** (D91): `ContextCompactor` with pluggable `SummarizerFn` replaces the oldest conversation turns with one LLM summary message when a token threshold is exceeded (complements D64 observation masking). `CompactionConfig` (`trigger_tokens`, `keep_recent`) + `should_compact()`.
- **A2A (Agent2Agent) client** (D92): Linux Foundation A2A v1.0 interop. Canonical types `A2AAgentCard`/`A2AAgentSkill`/`A2AMessage`/`A2APart`/`A2ATask` (+states), JSON-RPC envelope builder, `A2AClient` (`fetch_agent_card()` via `/.well-known/agent-card.json` + `message/send` over HTTP, reusing the per-request CurlHandle pattern). New `A2AError` exception.
- 21 new tests (217 total)

### Changed
- Exception hierarchy gains `A2AError` (sibling of `McpError` under `AriadneError`).

### Security
- **A2A endpoint pivot is same-origin by default** (D92): `fetch_agent_card()` now only routes subsequent RPCs to the card-declared `url` when it shares scheme+host with the configured base (prevents Bearer-token leak / SSRF to an attacker-controlled host). New `origin_of()` / `endpoint_pivot_allowed()` + opt-in `set_allow_cross_origin_endpoint(true)`.
- **No credentialed redirects** (D92): A2A agent-card GET sets `CURLOPT_FOLLOWLOCATION=0`, so the `Authorization` header is never forwarded across a redirect (matches MCP `HttpTransport`).
- **Segment-safe memory scoping** (D90): `scope_prefix` now matches on `:`-delimited boundaries via `memory_scope_matches()`, so `"user:alice"` no longer leaks `"user:alice2"` entries (tenant isolation).

## [2.7.0] - 2026-06-23

Theme: **Latest provider APIs + OpenTelemetry observability.** Researched against live June 2026 docs (Gemini 3, GPT-5.5, Claude structured-output GA, OTel GenAI semconv 1.40).

### Added
- **Unified reasoning effort** (D83): `ProviderConfig::reasoning_effort` (`minimal`/`low`/`medium`/`high`/`xhigh`) + `verbosity`. Gemini maps to `generationConfig.thinking_level` (Gemini 3 reasoning depth) across `complete()`/`complete_chat()`/`complete_stream()`; OpenAI passes `reasoning_effort`/`verbosity` through.
- **GPT-5.x parameter compatibility** (D84): `gpt-5*` models use `max_completion_tokens` and omit `temperature` (reasoning models reject non-default temperature). New `is_gpt5_family()` helper. Opt-in `strict_tools` adds `strict:true` to OpenAI function defs.
- **Anthropic structured output GA migration** (D85): `response_format`→`output_config.format` in `complete()` + `complete_chat()` (no beta header). Opt-in `strict_tools` adds `strict:true` per tool. New `anthropic_output_config()` helper.
- **Gemini 3.5 default** (D86): default model `gemini-2.5-flash`→`gemini-3.5-flash` (2.5 deprecated/shutting down 2026). `ModelPricing::gemini_flash()`/`gemini_pro()` presets.
- **OpenTelemetry GenAI trace export** (D87): `GenAiSpan` emits `gen_ai.*` semconv attributes (operation.name, provider.name, request.model, usage.input/output_tokens, response.finish_reasons). `ISpanExporter`/`NullSpanExporter`/`OtelJsonSpanExporter` + `set_span_exporter()`/`emit_span()`. Auto-emitted on every `try_slots()` and `complete_chat()` success.
- **Secret redaction** (D88): `redact_secrets()` masks `sk-`/`ghp_`/`AIza`/`Bearer`/`x-api-key` etc. in trace/log output. Applied in `OtelJsonSpanExporter`.
- 9 new tests (196 total)

### Changed
- Gemini default model is now `gemini-3.5-flash` (was `gemini-2.5-flash`). Pass an explicit model to override.

## [2.6.0] - 2026-06-23

### Added
- **FileCheckpointStore robustness** (D77): constructor wraps `create_directories()` in try/catch, `save()` checks ofstream open + good(), `load()` catches parse_error — all throw `AriadneError` with descriptive messages
- **StdioTransport send() validation** (D78): Windows checks `WriteFile` return + written count, POSIX checks `::write` for negative/short write — both throw `McpError` on failure
- **InMemoryVectorStore serialization** (D80): `to_json()` serializes entries under shared lock, `load_json()` clears and repopulates from JSON array. Uses void method (not static factory) because `shared_mutex` is non-movable in MSVC
- **complete_chat() output_schema** (D81): optional `output_schema` parameter wired through all providers (OpenAI json_schema, Anthropic json_schema, Gemini responseSchema) and LLMClient — native agent structured output parity
- **Planner tool sorting** (D82): `WorkflowPlanner::plan()` sorts tools alphabetically, matching agent paths (D65) for consistent prefix cache hits
- 7 new tests (187 total)

### Fixed
- FileCheckpointStore silent failures on directory creation, file write, and corrupted JSON
- StdioTransport silently ignoring failed writes causing hung MCP sessions

## [2.5.0] - 2026-06-23

### Added
- **Hedged requests** (D76): `enable_hedging(bool)` races 2 providers, returns first success. Trades cost for latency (P50 = min of both)
- **ConsoleLogger thread safety** (D71): mutex on `std::cerr` output
- **Retry-After substr safety** (D72): guards against `npos` subtraction
- **from_json DAG validation** (D73): `WorkflowPlan::from_json()` calls `validate_dag()`
- **resolve_value array bounds** (D74): validates index before access
- **AllProvidersExhaustedError diagnostics** (D75): collects all slot errors into message
- 7 new tests (180 total)

## [2.4.0] - 2026-06-18

### Fixed (CI green)
- **cppcheck uninitvar**: `PlanResult.config` now value-initialized with `{}`
- **Member init order UB**: moved `metrics_` before `executor_` in WorkflowEngine to fix use-before-init in constructor init list
- **cppcheck CI**: suppressed C++20-only suggestions (stlIfStrFind, uselessCallsSubstr, virtualCallInConstructor) 
- **Docs CI**: `mkdir -p docs/html` before running Doxygen

### Added
- **complete_chat() full parity with try_slots()** (D69): token budget enforcement, 429/503 retry with exponential backoff, provider latency tracking. Previously native agents bypassed all three.
- **cppcheck static analysis in CI** (D70): `warning` + `performance` checks on Linux, fails build on findings
- 4 new tests — complete_chat budget, chat latency, health_check, engine_llm_complete (160 total)

### Fixed
- **Native agent budget bypass**: `run_agent_native()` → `complete_chat()` never checked token budget. Now enforced before every chat call.
- **Native agent no retry**: `complete_chat()` immediately fell through to next provider on 429/503. Now retries 3x with backoff (3/6/12s) + Retry-After honor, matching `try_slots()`.

## [2.3.0] - 2026-06-17

### Added
- **Provider latency tracking** (D66): `ProviderStats.avg_latency_ms` and `last_latency_ms`. Tracked per-slot in `try_slots()` on success path.
- **Adaptive timeout** (D67): HTTP timeout now scales with `max_tokens` — `max(base, max_tokens/30 + 10s)`. Prevents premature timeout on long generations.
- 3 new tests (156 total)

### Fixed
- **Gemini systemInstruction** (D68): `complete()` and `complete_stream()` now use proper `systemInstruction` API field instead of prepending system prompt to user content. Matches `complete_chat()` behavior.

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
