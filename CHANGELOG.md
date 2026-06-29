# Changelog

All notable changes to Ariadne are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [2.12.0] - 2026-06-29

Theme: **Correctness, structural security & orchestration guardrails.** Researched against live June 2026 sources (CaMeL / IFC agent security, OWASP LLM03 supply chain + Agentic T8 repudiation, ASCII-smuggling / EchoLeak exfiltration, MAST multi-agent failure taxonomy, MCP 2025-11-25 transport spec, Anthropic / Gemini / OpenAI provider docs), adversarially verified across four research passes.

### Fixed (correctness — found by research, verified against primary docs)
- **Gemini `thinking_level` invalid value** (D115): `gemini_thinking_level()` no longer emits `"minimal"` — Gemini 3.x only accepts `low`/`medium`/`high` and 400s on `minimal` (D83/D113 had mapped `none`/`minimal` → `minimal`, a latent bug). Now `none`/`minimal` → `low`.
- **Anthropic version header** (D115): `anthropic-version` was `2024-10-22` (a model-snapshot date, never a valid header value) at all three call sites → corrected to `2023-06-01` (the documented current version).
- **Anthropic refusal handling** (D115): a safety refusal returns HTTP 200 with `stop_reason:"refusal"` and `content:[]`; `complete()` / `complete_chat()` now surface this as a clear `ProviderError` instead of a generic "empty content" / out-of-bounds read.
- **MCP Streamable-HTTP `Accept` header** (D116): now sends `application/json, text/event-stream` (the spec MUST) and parses an SSE-framed JSON-RPC response via `mcp_extract_sse_message()` — previously only `application/json` was accepted, so a spec-conformant streaming server broke the client. Also: `initialize()` now records + validates the negotiated `protocolVersion` (disconnects on an unsupported one).

### Added
- **Invisible-Unicode + role-marker sanitizers** (D117): `strip_invisible_unicode()` removes Tags-block / variation-selector / zero-width / invisible-math code points (ASCII-smuggling injection + covert exfiltration channel); `sanitize_role_markers()` strips chat-template delimiters (`<|im_start|>`, `[INST]`, …) to block fake-turn injection.
- **Tool-manifest pinning** (D118): `tool_manifest_hash()` + `ToolPinStore` (pin / verify → Unknown·Unchanged·Drifted / diff) detect MCP rug-pull / tool-poisoning drift; fail-closed (first-seen and drifted both require re-approval). Generalizes D107's `approval_checksum` to the full tool manifest. OWASP LLM03.
- **Egress allowlist** (D119): `EgressAllowlist` (scheme + host allowlist, rejects IP-literals / `data:` / `file:`, blocks suffix confusion) + `extract_markdown_urls()` deterministically sever the exfiltration leg of the lethal trifecta (EchoLeak / CVE-2025-32711 class). Complements D106 taint: taint decides *if*, allowlist decides *where*.
- **Tamper-evident audit log** (D120): self-contained `sha256_hex()` + `AuditLog` — a SHA-256 hash chain where any edit/delete/reorder is located by `verify()`. OWASP Agentic T8 (Repudiation & Untraceability); genuinely cryptographic, unlike an FNV chain.
- **Plan linter** (D121): `PlanLinter` runs deterministic structural lint over a `WorkflowPlan` *before* the executor spends tokens — dangling/undefined deps, dup/empty ids, self-deps, empty actions, json-step-without-schema, fan-out width, dependency cycles; built-in rules + `add_rule` extensibility. MAST: ~42% of multi-agent failures are pre-execution-catchable spec/structure defects.
- **Rubric scoring** (D122): `rubric_score()` / `rubric_score_ensemble()` — multi-criteria weighted LLM-judge scoring (`clamp01(Σ vᵢ·wᵢ / Σ w⁺)`, negative weights = penalties) with Majority / Weighted / Unanimous / Any ensembles. Rubric decomposition is the documented fix for the style/verbosity judge bias that still dominates on frontier judges.
- **Note store** (D123): `NoteStore` — a structured, serializable scratchpad *outside* the context window (put / append / str_replace / erase / render-within-budget); the Anthropic memory-tool pattern, with incremental edits to avoid context collapse. Distinct from D91 compaction (which summarizes the live transcript).
- **Memory write-path** (D124): `FactUpsertPolicy` decides ADD / UPDATE / REMOVE / NOOP for a candidate fact against its top-s neighbors (cosine thresholds + optional injected LLM relation fn) — the Mem0 4-op dedup-merge that a bare vector store lacks.
- 11 new tests (272 total).

### Security
- This release is security-led: four of the ten additions are deterministic, structural defenses — D117 invisible-Unicode / role-marker stripping (LLM01), D118 tool-manifest pinning (LLM03), D119 egress allowlist (LLM02), D120 tamper-evident audit log (Agentic T8) — closing standards-mandated gaps with **no new dependencies**. They layer with the existing taint tracker (D106), risk-tiered authorization (D107), and spotlighting (D100). The new `sha256_hex()` is a real cryptographic hash (not FNV), so the audit chain is tamper-*evident against a forging adversary*, not merely accident-detecting.

## [2.11.0] - 2026-06-25

Theme: **Defense, rigor & provider correctness.** Researched against live June 2026 docs (CaMeL / lethal-trifecta agent security, OWASP LLM06 excessive agency, τ²-bench eval rigor, ADK/LangSmith trajectory evals, MCP 2025-11-25 resources/prompts, Gemini 3.x thought signatures, Anthropic `output_config.effort` GA + prompt caching).

### Added
- **Lethal-trifecta taint tracking** (D106): `TaintTracker` + `ToolCapability` implement the deterministic data-flow defense behind spotlighting (which is only probabilistic). Once untrusted content enters context, `blocks_external_send()` flags any externally-communicating tool — injected instructions can't weaponize an exfiltration tool. CaMeL is 77% provably-secure on AgentDojo; this is the structural complement to D100.
- **Risk-tiered tool authorization** (D107): `ToolAuthorizationPolicy` maps tools to `ToolRiskLevel` (Auto / Confirm / DualConfirm / Deny), **fail-closed** by default (unknown tools require approval, never silent Auto). `approval_checksum()` binds an approval to the exact `(tool, args)` so a token can't be replayed on a different action (OWASP LLM06 "complete mediation").
- **Deterministic JSON repair** (D108): `repair_json()` / `parse_json_lenient()` — single-pass char scanner (no ML, no regex) that strips markdown fences, leading/trailing prose, trailing commas, single quotes and `//`+`/* */` comments, and LIFO-closes unclosed strings/objects/arrays on truncation. Wired as the final fallback in `WorkflowPlanner::extract_json()`. Providers' "structured output" under-enforces complex schemas (OpenAI ~29% empirical coverage on hard schemas), so robust parsing matters.
- **Eval statistical rigor** (D109): `wilson_interval()` (small-sample binomial CI), `pass_at_k()` (the *unbiased* Codex/HumanEval estimator `1 − C(n−c,k)/C(n,k)`), and `paired_bootstrap()` regression gate (per-case paired deltas → 95% CI; flags a regression only when the interval is entirely below zero). Deterministic (fixed seed) for CI use.
- **Trajectory evaluation** (D110): `score_trajectory()` scores an agent's tool-call sequence against an expected one in four modes (Strict / Unordered / Superset / Subset) plus overlap (recall) and redundant-call count (efficiency) — ADK `tool_trajectory` / LangSmith `agentevals` pattern, complementing output-only scoring.
- **MCP resources** (D111): `McpClient::list_resources()` (cursor-paginated) + `read_resource()` / `read_resource_all()` returning text-or-blob `McpResourceContent`. Lets MCP servers supply readable context for RAG. (Resource content is server-supplied and untrusted — spotlight before feeding an LLM.)
- **MCP prompts** (D112): `McpClient::list_prompts()` (paginated) + `get_prompt(name, args)` → `std::vector<ChatMessage>`. Server-provided reusable prompt templates; complements `PromptVersionStore`. `tools/list` was refactored onto a shared `paginate()` helper (same hardened cursor loop).
- **Anthropic prompt caching** (D114): opt-in `ProviderConfig::prompt_caching` emits `cache_control: {type:"ephemeral"}` breakpoints on the stable prefix (last tool + system) — cache reads are 0.1× input price (90% off). Stacks with the existing sorted tool defs (D65) for maximum hit rate. Model registry refresh: `ModelPricing::claude_fable()` ($10/$50), `anthropic()` routes Fable/Mythos pricing.
- 10 new tests (261 total).

### Changed
- **Provider correctness** (D113):
  - **Gemini 3.x multi-turn native tools** — `gemini_build_contents()` / `gemini_parse_chat()` (now testable pure helpers) capture the per-turn encrypted `thoughtSignature` and re-emit it unchanged, and thread real function-call `id`s into `functionResponse`. Without this, Gemini 3.x multi-turn tool loops degrade or 400 (a latent regression). `LLMToolCall` gains a `thought_signature` field.
  - **Anthropic `output_config.effort`** — the unified `reasoning_effort` now maps to Anthropic's GA effort control (D83 previously skipped Anthropic); `anthropic_output_config()` carries both `format` (schema) and `effort`.
  - **Anthropic temperature lock** — Claude Opus 4.7/4.8 and Fable/Mythos 5 reject non-default `temperature` (like GPT-5); `anthropic_locks_temperature()` omits it for those models.
  - `none` is now a recognized reasoning-effort value (OpenAI GPT-5.1 default); Gemini maps `none`/`minimal` → `minimal`.
  - OpenAI `complete_chat()` now sanitizes assistant `tool_calls` to OpenAI-only fields (strips `thoughtSignature`).

### Security
- **Lethal-trifecta taint tracking + risk-tiered authorization** (D106/D107): structural, deterministic defenses against prompt-injection-driven data exfiltration and excessive agency — the by-design complement to probabilistic spotlighting (D100). Fail-closed authorization + checksum-bound approval implement OWASP LLM06 "complete mediation."
- **MCP resource content is untrusted** (D111): server-supplied resource bytes flow into prompts and must be spotlighted/guarded exactly like tool output (per MCP security best practices).

## [2.10.0] - 2026-06-24

Theme: **Durability, reliability & safety.** Researched against live June 2026 docs (LangGraph durable execution + time-travel, AG-UI event streaming, τ²-bench reliability science, GPTCache semantic caching, Microsoft/Google prompt-injection spotlighting, MCP 2025-11-25 tool annotations, A2A v1.0.1 Linux Foundation).

### Added
- **Durable auto-resume** (D97): the DAG executor now checkpoints `WorkflowState` after each topological batch and can resume skipping already-completed steps. `WorkflowEngine::run_durable(task, id)` persists `{plan,state}` per batch; `resume(id)` reloads and finishes the remaining DAG. Closes the roadmap's "durable auto-resume" item — every major framework (LangGraph/CrewAI/ADK) ships skip-completed resume.
- **Time-travel / state fork** (D98): `HistoryCheckpointStore` keeps every version of a checkpoint; `WorkflowState::fork(edits, invalidate)` branches a snapshot; `WorkflowEngine::fork_and_resume(from, new, edits, invalidate)` replays from a prior checkpoint with edits, auto-invalidating the transitive downstream so edited values propagate (LangGraph "time travel").
- **Structured event stream** (D99): typed `AgentEvent` taxonomy (RUN/STEP/TOOL/MESSAGE started·finished) + `AgentEventSink`; `WorkflowEngine::set_event_sink()` emits from `run_agent_native`. `to_ag_ui_json()` maps to AG-UI protocol event names (cross-framework UI wire format) without committing to the unstable spec.
- **Tool-output spotlighting** (D100): `spotlight_text()` (Delimit / Datamark / Encode modes) + `SPOTLIGHT_SYS` defend against indirect prompt injection by marking untrusted tool output as data-not-instructions (Microsoft spotlighting, deployed in Google Gemini). Opt-in via `enable_tool_output_spotlighting()` — applied to native-agent tool results.
- **pass^k reliability metric** (D101): `PromptEvalGate::run_reliability(cases, k, …)` returns a `ReliabilityReport` with `pass^k` (all-k-succeed, the τ²-bench production-reliability metric) alongside per-run pass rate — surfaces flakiness that pass@1 hides.
- **Semantic response cache** (D102): `SemanticCache` keys cached responses by embedding cosine similarity (pluggable `EmbedFn`, FIFO cap) instead of exact-match hashing — 30–70% production hit rates on free-form prompts (GPTCache pattern). Reuses `InMemoryVectorStore`.
- **CRITIC verify-and-retry** (D103): `DynamicWorkflow::verify_and_retry(produce, verify, max_attempts)` runs an external-signal correction loop (verifier returns nullopt=pass or an error fed back into the next attempt) — complements N-vote `adversarial_verify`. External signal only (ICLR'24: intrinsic self-reflection is unreliable).
- **MCP tool annotations + pagination** (D104): `tools/list` now follows `nextCursor` pagination (fixes silent truncation on servers with many tools) and parses `ToolAnnotations` (`readOnlyHint`/`destructiveHint`/`idempotentHint`/`openWorldHint`, treated as untrusted hints) onto `ToolDef`. First offline MCP-client tests via a mock transport.
- **A2A v1.0.1 alignment** (D105): task lifecycle `tasks/get`/`tasks/cancel` + `poll_until_terminal`; `a2a_parse_stream_frame()` parses `message/stream` SSE frames (task/message/status-update/artifact-update by `kind`, with the v1.0-removed `final` field now inferred from terminal state); `A2AAgentCard` parses v1.0 fields (`pushNotifications`, `extendedAgentCard`, `signature`, `securitySchemes`, `preferredTransport`).
- 22 new tests (251 total)

### Changed
- `ToolDef` gains an `annotations` field (default-constructed; existing aggregate initializers unaffected).
- `WorkflowExecutor::execute()` gains two trailing optional params (`resume_state`, `on_batch_done`) — backward compatible.

### Fixed
- `WorkflowPlan::from_json()` no longer throws when a serialized plan's `metadata` is null/absent (robust task extraction) — surfaced by durable checkpoint round-trips.

### Security
- **Tool-output spotlighting** (D100): defense-in-depth against indirect prompt injection — untrusted tool/web output marked as data, with a system-prompt clause instructing the model to never follow embedded instructions (probabilistic; layer with `GuardrailFn`).
- **MCP annotations are untrusted** (D104): `ToolAnnotations` are parsed for UX/routing hints only, never as a security boundary (per spec); destructive/read-only hints default conservatively.

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
