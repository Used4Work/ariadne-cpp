# Changelog

All notable changes to Ariadne are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [1.2.0] - 2026-06-16

### Added
- Comprehensive README rewrite with all v1.x features
- Token-based agent history eviction via `estimate_tokens()`
- CMake FetchContent/add_subdirectory support (extras auto-disable)
- CONTRIBUTING.md, SECURITY.md, issue/PR templates, CHANGELOG.md

### Fixed
- Version sync: ARIADNE_VERSION, CMakeLists, vcpkg.json all 1.2.0
- LICENSE expanded to full MIT text

## [1.1.0] - 2026-06-16

### Added
- Anthropic native tool calling (`complete_chat()` with `tool_use`/`tool_result`)
- Retry-After header parsing (CURLOPT_HEADERFUNCTION)
- `estimate_tokens()` lightweight token estimation
- AdaptiveOrchestrator/DynamicWorkflow use `run_agent_native()` by default

## [1.0.1] - 2026-06-16

### Fixed
- Thread safety: ToolRegistry shared_mutex, cache locking, atomic flags
- HTTP 500/502/504 as retryable errors
- Safe JSON parse guards on all provider responses
- CURLOPT_CONNECTTIMEOUT=10s for fast failover
- Zero raw stdout/stderr in library code (all via ILogger)

### Added
- CI: macOS ARM64 + Linux ASan/UBSan in test matrix

## [1.0.0] - 2026-06-16

### Added
- Native function calling: OpenAI tools API + `run_agent_native()`
- ChatMessage/LLMToolCall/LLMResponse types for multi-turn conversations
- `complete_chat()` on ILLMProvider with auto-fallback

## [0.9.0] - 2026-06-16

### Added
- Structured logging: ILogger/NullLogger/ConsoleLogger
- Cost tracking: ModelPricing with USD presets
- Version API: `ariadne::version()`
- Agent traces: AgentResult.traces
- Studio: /api/adaptive, /api/analyze endpoints, Adaptive button

## [0.8.0] - 2026-06-16

### Added
- AdaptiveOrchestrator: auto strategy selection (5 modes)

## [0.7.0] - 2026-06-16

### Added
- DynamicWorkflow: parallel/pipeline/map/loop_until/fan_out_agents/adversarial_verify

## [0.6.0] - 2026-06-16

### Added
- Token budget enforcement (TokenBudgetError)
- Checkpointing (ICheckpointStore, FileCheckpointStore)
- WorkflowPlan/State serialization (to_json/from_json)
- Human-in-the-loop (InterruptError, set_interrupt_hook)
- Provider factories: cerebras(), sambanova(), mistral()

### Fixed
- 429 retry: same provider with exponential backoff before fallback
- Agent loop: WORKFLOW_START/END metrics + guardrails
- PlanCache key includes conversation context hash

## [0.5.0] - 2026-06-15

### Added
- Ariadne Studio visual workflow editor
- MCP client (stdio transport, JSON-RPC 2.0)
- Guardrails (input/output/tool validation)
- Multi-agent handoffs (AgentDef, HANDOFF action)
- LLM response caching (FNV-1a, LRU)
- Plan template caching (NeurIPS 2025 APC)
- Agent convergence detection (LoopDetector)
- Token usage tracking

## [0.2.0] - 2026-06-08

### Added
- Initial release: DAG workflows, ReACT agents, multi-provider, circuit breakers, streaming
