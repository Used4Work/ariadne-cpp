/**
 * ariadne.hpp — The thread through any AI labyrinth.
 * ====================================================
 * C++ LLM workflow orchestration: automatic DAG planning,
 * parallel execution, circuit breakers, mixed-provider tiers.
 *
 * Requires: libcurl, nlohmann/json (header-only), C++17
 */
#ifdef _WIN32
#  define NOMINMAX  // prevent Windows.h min/max macros
#endif
#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>
#include <functional>
#include <memory>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <optional>
#include <nlohmann/json.hpp>
#include <curl/curl.h>

using json = nlohmann::json;

namespace ariadne {


// ════════════════════════════════════════════════════════════════
// Token 用量追踪
// ════════════════════════════════════════════════════════════════

struct TokenUsage {
    long input_tokens  = 0;
    long output_tokens = 0;
    long total_tokens  = 0;
    TokenUsage& operator+=(const TokenUsage& o) {
        input_tokens += o.input_tokens; output_tokens += o.output_tokens;
        total_tokens += o.total_tokens; return *this;
    }
};

inline thread_local TokenUsage g_last_token_usage{};

// ════════════════════════════════════════════════════════════════
// 异常层次 — 替代散落的 std::runtime_error
// ════════════════════════════════════════════════════════════════

/** 取消令牌 — 线程安全的共享取消信号 */
using CancelToken = std::shared_ptr<std::atomic<bool>>;

/** 所有 Ariadne 异常的基类 */
class AriadneError : public std::exception {
public:
    explicit AriadneError(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }
protected:
    std::string msg_;
};

/** Provider 层错误（网络、鉴权、限速等） */
class ProviderError : public AriadneError {
    using AriadneError::AriadneError;
};

/** 所有配置的 Provider 均已耗尽或熔断 */
class AllProvidersExhaustedError : public ProviderError {
    using ProviderError::ProviderError;
};

/** 规划阶段错误（JSON 解析、DAG 校验等） */
class PlanningError : public AriadneError {
    using AriadneError::AriadneError;
};

/** 工作流被主动取消或超时 */
class WorkflowCancelledError : public AriadneError {
    using AriadneError::AriadneError;
};

/** Guardrail 验证失败 */
class GuardrailError : public AriadneError {
    using AriadneError::AriadneError;
};

/** Guardrail 验证函数：返回 nullopt=通过，或 string=错误原因 */
using GuardrailFn = std::function<std::optional<std::string>(const json&)>;

/** 工具相关错误 */
class ToolError : public AriadneError {
public:
    ToolError(std::string tool_name, const std::string& msg)
        : AriadneError("Tool '" + tool_name + "': " + msg)
        , tool_name(std::move(tool_name)) {}
    std::string tool_name;
};

/** 工具未注册 */
class ToolNotFoundError : public ToolError {
    using ToolError::ToolError;
};

/** 工具输入不满足 schema 约束 */
class ToolInputError : public ToolError {
    using ToolError::ToolError;
};

/** 流式输出回调：每收到一个文本 chunk 调用一次 */
using StreamCallback = std::function<void(const std::string& chunk)>;

// ════════════════════════════════════════════════════════════════
// 1. CURL 生命周期管理 (B1, B2)
// ════════════════════════════════════════════════════════════════

/** 进程级单例——保证 curl_global_init / cleanup 各调用一次 */
class CurlGlobal {
    CurlGlobal()  { curl_global_init(CURL_GLOBAL_ALL); }
    ~CurlGlobal() { curl_global_cleanup(); }
public:
    static void ensure() { static CurlGlobal inst; (void)inst; }
};

/** 单个 easy handle 的 RAII 包装 */
class CurlHandle {
public:
    CurlHandle() {
        CurlGlobal::ensure();
        handle_ = curl_easy_init();
        if (!handle_) throw std::runtime_error("curl_easy_init() failed");
    }
    ~CurlHandle() { if (handle_) curl_easy_cleanup(handle_); }
    CurlHandle(const CurlHandle&)            = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
    CURL* get() const noexcept { return handle_; }
private:
    CURL* handle_;
};

// ════════════════════════════════════════════════════════════════
// 2. Provider 层
// ════════════════════════════════════════════════════════════════

enum class ProviderType { ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES };

struct ProviderConfig {
    ProviderType type;
    std::string  api_key;
    std::string  model;
    std::string  base_url;
    int          max_tokens       = 4096;
    double       timeout_sec      = 60.0;
    std::string  completions_path = "";   // 空 = /v1/chat/completions
    double       max_rps          = 0.0;  // 每秒最大请求数；0=不限速

    static ProviderConfig anthropic(const std::string& key,
                                     const std::string& model = "claude-opus-4-8") {
        return {ProviderType::ANTHROPIC, key, model, "", 4096, 60.0, ""};
    }
    static ProviderConfig openai_chat(const std::string& key,
                                       const std::string& model = "gpt-4o") {
        return {ProviderType::OPENAI_CHAT, key, model, "", 4096, 60.0, ""};
    }
    static ProviderConfig openai_responses(const std::string& key,
                                            const std::string& model = "gpt-4o") {
        return {ProviderType::OPENAI_RESPONSES, key, model, "", 4096, 60.0, ""};
    }
    static ProviderConfig openai_compatible(const std::string& key,
                                             const std::string& base_url,
                                             const std::string& model,
                                             double max_rps = 0.0) {
        ProviderConfig c{ProviderType::OPENAI_CHAT, key, model, base_url};
        c.max_rps = max_rps;
        return c;
    }
    static ProviderConfig github_models(const std::string& token,
                                         const std::string& model = "openai/gpt-4o-mini",
                                         double max_rps = 0.1) {  // 6 RPM per tier (safe under GH Models limits)  // 15 RPM default
        ProviderConfig c;
        c.type             = ProviderType::OPENAI_CHAT;
        c.api_key          = token;
        c.model            = model;
        c.base_url         = "https://models.github.ai/inference";
        c.completions_path = "/chat/completions";
        c.max_rps          = max_rps;
        return c;
    }
    /** Groq free tier: 30 RPM = 0.5 RPS */
    static ProviderConfig groq(const std::string& key,
                                const std::string& model = "llama-3.3-70b-versatile") {
        return openai_compatible(key, "https://api.groq.com/openai", model, 0.5);
    }
};

/** 所有 Provider 的公共接口 */
class ILLMProvider {
public:
    virtual ~ILLMProvider() = default;
    virtual std::string complete(const std::string& prompt,
                                  const std::string& system     = "",
                                  double             temperature = 0.0,
                                  bool               force_json  = false,
                                  const json&        output_schema = json()) const = 0;
    virtual void complete_stream(const std::string& prompt,
                                  const std::string& system,
                                  double             temperature,
                                  StreamCallback     on_chunk) const = 0;
    virtual std::string provider_name() const = 0;
    virtual std::string model_name()    const = 0;
};

/**
 * HttpProvider — 共享 curl 逻辑的基类 (R1)
 * 每次请求创建独立 CurlHandle，线程安全。
 */
class HttpProvider : public ILLMProvider {
public:
    explicit HttpProvider(const ProviderConfig& cfg) : cfg_(cfg) {}
protected:
    ProviderConfig cfg_;

    struct HttpResponse {
        std::string body;
        long        status_code = 0;
    };

    HttpResponse http_post(const std::string& url,
                            const std::vector<std::string>& headers,
                            const std::string& body) const;
private:
    static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata);
};

/** POST /v1/messages → content[0].text */
class AnthropicProvider : public HttpProvider {
public:
    explicit AnthropicProvider(const ProviderConfig& cfg) : HttpProvider(cfg) {}
    std::string complete(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          bool force_json = false,
                          const json& output_schema = json()) const override;
    void complete_stream(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          StreamCallback on_chunk) const override;
    std::string provider_name() const override { return "anthropic"; }
    std::string model_name()    const override { return cfg_.model;   }
};

/** POST /v1/chat/completions → choices[0].message.content */
class OpenAIChatProvider : public HttpProvider {
public:
    explicit OpenAIChatProvider(const ProviderConfig& cfg) : HttpProvider(cfg) {}
    std::string complete(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          bool force_json = false,
                          const json& output_schema = json()) const override;
    void complete_stream(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          StreamCallback on_chunk) const override;
    std::string provider_name() const override {
        return cfg_.base_url.empty() ? "openai_chat" : "openai_compatible";
    }
    std::string model_name() const override { return cfg_.model; }
};

/** POST /v1/responses → output[0].content[0].text */
class OpenAIResponsesProvider : public HttpProvider {
public:
    explicit OpenAIResponsesProvider(const ProviderConfig& cfg) : HttpProvider(cfg) {}
    std::string complete(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          bool force_json = false,
                          const json& output_schema = json()) const override;
    void complete_stream(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          StreamCallback on_chunk) const override;
    std::string provider_name() const override { return "openai_responses"; }
    std::string model_name()    const override { return cfg_.model;          }
};

std::unique_ptr<ILLMProvider> make_provider(const ProviderConfig& cfg);

/** 测试用 Mock Provider — 返回预设响应 */
class MockProvider : public ILLMProvider {
public:
    explicit MockProvider(std::string response, std::string name = "mock", std::string model = "mock-1")
        : response_(std::move(response)), name_(std::move(name)), model_(std::move(model)) {}

    std::string complete(const std::string&, const std::string&,
                          double, bool, const json& = json()) const override { return response_; }
    void complete_stream(const std::string&, const std::string&,
                          double, StreamCallback on_chunk) const override {
        on_chunk(response_);
    }
    std::string provider_name() const override { return name_; }
    std::string model_name()    const override { return model_; }

    void set_response(const std::string& r) { response_ = r; }
private:
    std::string response_, name_, model_;
};

// ════════════════════════════════════════════════════════════════
// 3. ModelTier — 高性能 vs 低消耗
// ════════════════════════════════════════════════════════════════

enum class ModelTier { ORCHESTRATOR, SUBAGENT };

// ════════════════════════════════════════════════════════════════
// 4. CircuitBreaker — CLOSED → OPEN → HALF_OPEN
// ════════════════════════════════════════════════════════════════

class CircuitBreaker {
public:
    enum class State { CLOSED, OPEN, HALF_OPEN };

    explicit CircuitBreaker(int failure_threshold = 3, double recovery_sec = 60.0)
        : mu_(std::make_unique<std::mutex>()),
          threshold_(failure_threshold), recovery_sec_(recovery_sec),
          state_(State::CLOSED), failures_(0), half_open_in_flight_(false) {}

    // movable (unique_ptr<mutex> enables this)
    CircuitBreaker(CircuitBreaker&&) = default;
    CircuitBreaker& operator=(CircuitBreaker&&) = default;

    bool   try_allow();
    void   on_success();
    void   on_failure();
    State  state()             const;
    double seconds_until_retry() const;

private:
    mutable std::unique_ptr<std::mutex>   mu_;
    int     threshold_;
    double  recovery_sec_;
    State   state_;
    int     failures_;
    bool    half_open_in_flight_;
    std::chrono::steady_clock::time_point last_open_;
};


// ════════════════════════════════════════════════════════════════
// RateLimiter — Token Bucket（per-provider 限速）
//
// 用法：
//   ProviderConfig::groq(key, model)  // 自动设置 max_rps=0.5 (30 RPM)
//   cfg.max_rps = 1.0;               // 手动设置 60 RPM
// ════════════════════════════════════════════════════════════════

class RateLimiter {
public:
    /** @param max_rps 每秒最大请求数；0 = 不限速 */
    explicit RateLimiter(double max_rps = 0.0)
        : max_rps_(max_rps)
        , tokens_(max_rps > 0 ? 1.0 : 0.0)
        , last_refill_(std::chrono::steady_clock::now()) {}

    /**
     * 获取一个请求 token，如有必要则阻塞等待。
     * @param max_wait_ms 最长等待毫秒数，超时返回 false
     */
    bool acquire(long max_wait_ms = 10000) {
        if (max_rps_ <= 0.0) return true;
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(max_wait_ms);
        while (true) {
            {
                std::lock_guard<std::mutex> lk(*mu_);
                refill();
                if (tokens_ >= 1.0) { tokens_ -= 1.0; return true; }
            }
            if (std::chrono::steady_clock::now() >= deadline) return false;
            // sleep for 1/max_rps seconds, capped at remaining wait
            auto _rem = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now()).count();
            long sleep_ms = static_cast<long>(
                std::max((long long)1,
                    std::min((long long)(1000.0 / max_rps_), (long long)_rem)));
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    bool   enabled() const noexcept { return max_rps_ > 0.0; }
    double max_rps() const noexcept { return max_rps_; }

private:
    void refill() {
        auto now     = std::chrono::steady_clock::now();
        double delta = std::chrono::duration<double>(now - last_refill_).count();
        tokens_      = std::min(1.0, tokens_ + delta * max_rps_);
        last_refill_ = now;
    }
    double   max_rps_;
    double   tokens_;
    std::chrono::steady_clock::time_point last_refill_;
    mutable std::unique_ptr<std::mutex> mu_{std::make_unique<std::mutex>()};
};

// ════════════════════════════════════════════════════════════════
// 5. TierConfig + ProviderStats
// ════════════════════════════════════════════════════════════════

struct TierConfig {
    ProviderConfig              primary;
    std::vector<ProviderConfig> fallbacks;
    int    failure_threshold    = 3;
    double recovery_timeout_sec = 60.0;
};

struct ProviderStats {
    std::string           provider_name;
    std::string           model_name;
    long                  total_calls   = 0;
    long                  successes     = 0;
    long                  failures      = 0;
    CircuitBreaker::State circuit_state = CircuitBreaker::State::CLOSED;
    double                secs_to_retry = 0.0;
};

// ════════════════════════════════════════════════════════════════
// 6. LLMClient — 双 Tier + 熔断降级
// ════════════════════════════════════════════════════════════════

class ResponseCache;  // forward declaration

class LLMClient {
public:
    LLMClient(TierConfig orchestrator, TierConfig subagent);

    /** 测试用构造器 — 直接注入 Provider，跳过 ProviderConfig */
    LLMClient(std::unique_ptr<ILLMProvider> orchestrator_provider,
              std::unique_ptr<ILLMProvider> subagent_provider);

    std::string complete_as(ModelTier tier,
                             const std::string& prompt,
                             const std::string& system     = "",
                             double             temperature = 0.0,
                             bool               force_json  = false,
                             const json&        output_schema = json()) const;

    /** 流式：chunk 通过 on_chunk 回调推送 */
    void complete_as_stream(ModelTier tier,
                             const std::string& prompt,
                             const std::string& system,
                             double             temperature,
                             StreamCallback     on_chunk) const;

    /** 默认走 ORCHESTRATOR（向下兼容） */
    std::string complete(const std::string& prompt,
                          const std::string& system     = "",
                          double             temperature = 0.0) const {
        return complete_as(ModelTier::ORCHESTRATOR, prompt, system, temperature);
    }

    std::vector<ProviderStats> stats(ModelTier tier) const;
    void print_status() const;
    TokenUsage total_usage() const;
    void reset_usage();
    void enable_response_cache(bool on = true);
    long response_cache_hits() const;

private:
    struct Slot {
        std::unique_ptr<ILLMProvider>       provider;
        mutable CircuitBreaker              breaker;
        mutable RateLimiter                 rate_limiter;
        mutable std::unique_ptr<std::mutex> stats_mu{std::make_unique<std::mutex>()};
        mutable long calls = 0, successes = 0, failures = 0;
    };
    std::vector<Slot> orchestrators_, subagents_;
    mutable std::mutex usage_mu_;
    mutable TokenUsage cumulative_usage_;
    mutable std::unique_ptr<ResponseCache> response_cache_;
    bool resp_cache_enabled_ = false;

    static std::vector<Slot> build_slots(const TierConfig& cfg);
    std::string try_slots(const std::vector<Slot>& slots,
                           const std::string& prompt,
                           const std::string& system,
                           double temperature,
                           ModelTier tier,
                           bool force_json = false,
                           const json& output_schema = json()) const;
};

// ════════════════════════════════════════════════════════════════
// 7. 核心数据类型
// ════════════════════════════════════════════════════════════════

enum class StepType { TOOL, LLM, TRANSFORM, CONDITION };
enum class OnError  { FAIL, SKIP, FALLBACK };

struct Step {
    std::string              id;
    StepType                 type;
    std::string              action;
    json                     inputs;
    std::vector<std::string> depends_on;
    int                      retry       = 0;
    double                   timeout_sec = 30.0;
    OnError                  on_error    = OnError::FAIL;
    json                     fallback    = nullptr;
    std::string              fallback_model;   ///< 失败时用此模型重试（空=不重试）
    std::string              description;
    ModelTier                model_tier  = ModelTier::SUBAGENT;
    // ── LLM step extensions ──────────────────────────────
    std::string              system_prompt;   ///< per-step system prompt (LLM steps)
    bool                     json_mode = false;    ///< force JSON output from LLM
    json                     output_schema;      ///< JSON schema to validate LLM output against
    double                   temperature = -1.0; ///< LLM temperature (-1 = use provider default)
};

struct WorkflowPlan {
    std::string       id;
    std::vector<Step> steps;
    json              metadata;

    std::vector<std::vector<Step>> topological_batches() const;
    std::vector<Step>              leaf_steps()          const;
};

/** 结构化的单步追踪记录 (R3) */
struct TraceEntry {
    std::string step_id;
    std::string step_type;    // "tool" | "llm" | "transform" | "condition"
    std::string status;       // "ok" | "failed" | "skipped" | "fallback"
    std::string provider;
    long        duration_ms = 0;
    std::string error;
    long        input_tokens  = 0;  ///< LLM 步骤的 prompt token 数
    long        output_tokens = 0;  ///< LLM 步骤的 completion token 数
};

struct WorkflowState {
    json                               task_input;
    std::map<std::string, json>        step_outputs;
    std::map<std::string, std::string> errors;
    std::vector<TraceEntry>            traces;

    json resolve_ref   (const std::string& ref) const;
    json resolve_inputs(const json& inputs)      const;
    void record_trace  (TraceEntry entry) noexcept;

private:
    json resolve_value(const json& v) const;
};


// ════════════════════════════════════════════════════════════════
// JSON Schema 验证 (output_schema 实际验证，不只是 prompt injection)
// ════════════════════════════════════════════════════════════════

/** 单条 schema 违规 */
struct SchemaViolation {
    std::string path;     ///< JSON Pointer 格式路径（如 ".revenue" 或 ".items[0]"）
    std::string message;  ///< 描述违规原因
};

/**
 * 轻量 JSON Schema 验证器
 * 支持：type, required, properties, items, enum
 * 返回所有违规，空列表 = 通过
 */
std::vector<SchemaViolation> validate_json_schema(const json& value,
                                                    const json& schema,
                                                    const std::string& path = "");

// ════════════════════════════════════════════════════════════════
// 8. ToolRegistry
// ════════════════════════════════════════════════════════════════

using ToolFn = std::function<json(const json& params)>;

struct ToolDef {
    std::string name, description;
    json        input_schema, output_schema;
};

class ToolRegistry {
public:
    void                 register_tool(const ToolDef& def, ToolFn fn);
    json                 call         (const std::string& name, const json& params) const;
    std::vector<ToolDef> list_tools   ()                                            const;
    bool                 has_tool     (const std::string& name)                     const;
    void                 add_guardrail(const std::string& tool_name, GuardrailFn fn);
private:
    std::map<std::string, ToolDef> defs_;
    std::map<std::string, ToolFn>  fns_;
    std::map<std::string, std::vector<GuardrailFn>> guardrails_;
};

// ════════════════════════════════════════════════════════════════
// 9. WorkflowContext — 多轮记忆 (A2)
// ════════════════════════════════════════════════════════════════

/**
 * 在多次 engine.run() 之间传递历史，让 Planner 利用上下文。
 *
 *   WorkflowContext ctx;
 *   auto r1 = engine.run("search Tesla revenue", ctx);
 *   auto r2 = engine.run("compare with BYD", ctx);  // planner 看到 r1 的结果
 */
class WorkflowContext {
public:
    WorkflowContext(int max_history = 5, int max_summary_chars = 400)
        : max_history_(max_history), max_summary_chars_(max_summary_chars) {}

    void record(const std::string& task, const json& output);
    std::string to_prompt_prefix() const;
    bool        empty()           const { return history_.empty(); }
    void        clear()                 { history_.clear(); }

private:
    int max_history_;
    int max_summary_chars_ = 400;
    std::vector<std::pair<std::string, std::string>> history_; // (task, output_str)
};

// ════════════════════════════════════════════════════════════════
// 10. WorkflowPlanner
// ════════════════════════════════════════════════════════════════

class WorkflowPlanner {
public:
    explicit WorkflowPlanner(LLMClient& llm, const std::string& custom_sys = "");

    /** 单次 LLM 调用完成分析+规划（推荐；比 analyze_task+plan_static 快 1-2s）*/
    WorkflowPlan plan        (const std::string& task,
                               const std::vector<ToolDef>& tools,
                               const WorkflowContext& ctx = {},
                               int max_attempts = 3) const;

    static json         extract_json(const std::string& text);
    static WorkflowPlan parse_plan  (const json& raw, const std::string& task);

    void set_system_prompt(const std::string& sys) { custom_sys_ = sys; }

private:
    LLMClient& llm_;
    std::string custom_sys_;
};


// ════════════════════════════════════════════════════════════════
// ThreadPool — 有界线程池（替代 std::async 无限制生成线程）
// ════════════════════════════════════════════════════════════════

/**
 * 有界线程池：固定 N 个工作线程，无限制任务队列。
 * 用于 WorkflowExecutor 内并行步骤执行，避免每步 std::async 生成新线程。
 *
 * 典型大小：hardware_concurrency()，上限 32。
 * WorkflowEngine 持有一个实例，跨多次 run() 调用复用。
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t n = 0) {
        if (n == 0) n = std::max(2u, std::thread::hardware_concurrency());
        n = std::min(n, (size_t)32);
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back([this]{ worker_loop(); });
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template<typename F>
    auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        auto fut  = task->get_future();
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (stop_) throw std::runtime_error("ThreadPool is stopped");
            tasks_.emplace([task]{ (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    size_t size() const { return workers_.size(); }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this]{ return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }
    std::vector<std::thread>           workers_;
    std::queue<std::function<void()>>  tasks_;
    std::mutex                         mu_;
    std::condition_variable            cv_;
    bool                               stop_ = false;
};

// ════════════════════════════════════════════════════════════════
// 11. WorkflowExecutor
// ════════════════════════════════════════════════════════════════


/** DAG 结构无效（重复 ID、未知依赖、环等） */
class DAGValidationError : public AriadneError {
    using AriadneError::AriadneError;
};

/** 单个步骤执行失败 */
class StepExecutionError : public AriadneError {
public:
    StepExecutionError(const std::string& step_id,
                        const std::string& step_desc,
                        const std::string& msg)
        : AriadneError("Step [" + step_id + "] \"" + step_desc + "\": " + msg)
        , step_id(step_id), step_description(step_desc) {}
    std::string step_id, step_description;
};
/// @deprecated Use StepExecutionError
using WorkflowExecutionError = StepExecutionError;



class IMetricsCollector;  // forward declaration

// ════════════════════════════════════════════════════════════════
// LLM 响应缓存 — exact-match，temperature=0 时高命中率
// ════════════════════════════════════════════════════════════════

class ResponseCache {
public:
    explicit ResponseCache(size_t max_size = 200) : max_size_(max_size) {}

    static std::string make_key(const std::string& model, const std::string& system,
                                 const std::string& prompt, double temperature,
                                 bool force_json);
    bool has(const std::string& key) const;
    std::string get(const std::string& key);
    void put(const std::string& key, const std::string& response);
    size_t size() const { return cache_.size(); }
    void clear() { cache_.clear(); order_.clear(); }

    struct Stats { long hits = 0; long misses = 0; };
    Stats stats() const { return stats_; }

private:
    size_t max_size_;
    std::unordered_map<std::string, std::string> cache_;
    std::list<std::string> order_;
    mutable Stats stats_;
    mutable std::mutex mu_;
};

class WorkflowExecutor {
public:
    WorkflowExecutor(LLMClient& llm, ToolRegistry& tools, size_t max_threads = 0,
                      std::shared_ptr<IMetricsCollector> metrics = nullptr);
    WorkflowState execute(const WorkflowPlan& plan,
                           const json& task_input,
                           CancelToken cancel = nullptr) const;
    /** Used by run_stream to execute non-final steps */
    json run_step_pub(const Step& s, const WorkflowState& st, std::vector<TraceEntry>& tr) const {
        return run_step(s, st, tr);
    }
    /** Submit task to executor's thread pool (used by run_stream) */
    template<typename F>
    auto submit_task(F&& f) { return pool_.submit(std::forward<F>(f)); }

private:
    LLMClient&    llm_;
    ToolRegistry& tools_;
    mutable ThreadPool pool_;
    std::shared_ptr<IMetricsCollector> metrics_;

    json run_step      (const Step&, const WorkflowState&, std::vector<TraceEntry>&) const;
    json dispatch      (const Step&, const json& resolved) const;
    json exec_tool     (const Step&, const json&) const;
    json exec_llm      (const Step&, const json&) const;  // fixed: preserves JSON (R2)
    json exec_transform(const Step&, const json&) const;  // fixed: guarded parse_json (B3)
};


// ════════════════════════════════════════════════════════════════
// Metrics / Observability
//
// 默认：NoOpMetrics（零开销）
// 内置：ConsoleMetrics（打印到 stdout）
// 自定义：实现 IMetricsCollector，调用 engine.set_metrics(impl)
// ════════════════════════════════════════════════════════════════

/** 单条指标事件 */
struct MetricEvent {
    enum class Kind {
        WORKFLOW_START, WORKFLOW_END,
        STEP_START,     STEP_END,
        LLM_CALL,       LLM_RESPONSE,
        TOOL_CALL,      TOOL_RESPONSE,
        PROVIDER_ERROR, RATE_LIMIT_WAIT,
    };
    Kind        kind;
    std::string workflow_id;
    std::string step_id;        // 空 = workflow 级别
    std::string provider;       // LLM_CALL/ERROR 时填写
    std::string model;
    long        duration_ms = 0;
    bool        success     = true;
    std::string error;
    json        extra = {};     // 附加上下文（可选）
};

/** 指标收集器接口 */
class IMetricsCollector {
public:
    virtual ~IMetricsCollector() = default;
    virtual void record(const MetricEvent& event) noexcept = 0;
};

/** 默认实现：零开销 no-op */
class NoOpMetrics : public IMetricsCollector {
public:
    void record(const MetricEvent&) noexcept override {}
};

/** 控制台输出：每条事件一行 JSON */
class ConsoleMetrics : public IMetricsCollector {
public:
    void record(const MetricEvent& e) noexcept override {
        static const char* kinds[] = {
            "workflow.start","workflow.end",
            "step.start","step.end",
            "llm.call","llm.response",
            "tool.call","tool.response",
            "provider.error","rate_limit.wait"
        };
        try {
            json j = {
                {"kind",    kinds[(int)e.kind]},
                {"wf_id",   e.workflow_id},
                {"step",    e.step_id},
                {"ok",      e.success},
                {"ms",      e.duration_ms},
            };
            if (!e.provider.empty()) j["provider"] = e.provider;
            if (!e.error.empty())    j["error"]    = e.error;
            std::cout << "[metrics] " << j.dump() << "\n";
        } catch (...) {}
    }
};

// ════════════════════════════════════════════════════════════════
// 12. WorkflowEngine — 主入口
// ════════════════════════════════════════════════════════════════

struct EngineConfig {
    TierConfig  orchestrator;
    TierConfig  subagent;
    int         max_concurrency = 8;

    static EngineConfig from_single(const ProviderConfig& p, int thr = 3) {
        TierConfig t{p, {}, thr, 60.0};
        return {t, t};
    }
    static EngineConfig from_two(const ProviderConfig& orc,
                                  const ProviderConfig& sub, int thr = 3) {
        return {{orc, {}, thr, 60.0}, {sub, {}, thr, 60.0}};
    }
    static EngineConfig with_fallbacks(const TierConfig& orc, const TierConfig& sub) {
        return {orc, sub};
    }
};

/** 完整的 workflow 执行结果 (A1) */
struct WorkflowResult {
    bool        success;
    json        output;
    std::string error_message;

    // 新增字段
    long                       duration_ms   = 0;
    int                        step_count    = 0;
    std::vector<TraceEntry>    traces;
    std::vector<ProviderStats> provider_stats;

    TokenUsage                 token_usage;
    json                       partial_outputs;  ///< 部分成功时的中间结果

    bool        has_output() const { return !output.is_null() && !output.empty(); }
    std::string summary()    const;
};


// ──────────────────────────────────────────────────────────────
// 12b. Provider 自动探测 & 规划
// ──────────────────────────────────────────────────────────────

/** 单个 Provider 探测结果 */
struct ProbeResult {
    std::string name;           // 候选名，如 "groq/llama3.3"
    std::string provider_name;
    std::string model;
    std::string tier;           // "strong" | "fast"
    int         priority = 0;
    bool        alive    = false;
    long        latency_ms = 0;
    std::string error;
};

/**
 * ProviderAutoPlanner — 并发探测所有候选，自动组装最优 EngineConfig
 *
 *   ProviderAutoPlanner planner;
 *   planner.add("claude-opus",   ProviderConfig::anthropic(key),       "strong", 1);
 *   planner.add("groq-llama",    ProviderConfig::openai_compatible(...), "fast",   1);
 *   planner.add("github-models", ProviderConfig::github_models(token),  "fast",   2);
 *
 *   auto r = planner.probe_and_plan();
 *   WorkflowEngine engine(r.config);
 *
 * 执行失败后自修复：
 *   engine.run_with_recovery(task, planner);
 */
class ProviderAutoPlanner {
public:
    void add_candidate(const std::string& name,
                       const ProviderConfig& config,
                       const std::string& tier,   // "strong" | "fast"
                       int priority = 100);

    struct PlanResult {
        EngineConfig             config;
        std::vector<ProbeResult> probe_results;
        std::vector<std::string> alive_strong;
        std::vector<std::string> alive_fast;
        bool                     success = false;
        std::string              error;
    };

    PlanResult probe_and_plan(
        std::chrono::seconds timeout = std::chrono::seconds(8)) const;

    PlanResult repair(
        std::chrono::seconds timeout = std::chrono::seconds(8)) const;

    void print_last_report() const;
    const std::vector<ProbeResult>& last_results() const { return last_results_; }

private:
    struct Candidate {
        std::string name; ProviderConfig config;
        std::string tier; int priority;
    };
    std::vector<Candidate>           candidates_;
    mutable std::vector<ProbeResult> last_results_;

    ProbeResult probe_one(const Candidate& c, std::chrono::seconds timeout) const;
    PlanResult  build_plan(const std::vector<ProbeResult>& results,
                            const std::string& log_prefix) const;
};


// ════════════════════════════════════════════════════════════════
// 16. Agent Loop — ReACT 模式（Reasoning + Acting）
//
// 与纯 DAG workflow 的区别：
//   DAG       → 执行前规划，步骤固定，单向流动
//   AgentLoop → 每步后重新决策；可循环、可重试、可改变方向
// ════════════════════════════════════════════════════════════════

struct AgentAction {
    enum class Type { TOOL_CALL, FINAL_ANSWER, LOOP_BACK, HANDOFF };
    Type        type     = Type::FINAL_ANSWER;
    std::string thought;
    std::string tool_name;
    json        tool_args;
    std::string response;     // FINAL_ANSWER
    std::string reason;       // LOOP_BACK
    std::string target_agent; // HANDOFF
};

/** 多 Agent 编排：Agent 定义 */
struct AgentDef {
    std::string              name;
    std::string              system_prompt;
    std::vector<std::string> allowed_tools;     // 空 = 所有工具
    ModelTier                model_tier = ModelTier::ORCHESTRATOR;
    std::vector<std::string> handoff_targets;   // 可移交的 agent 名
};

struct AgentStep {
    int         iteration;
    std::string thought;
    AgentAction action;
    json        observation;
    long        duration_ms = 0;
};

struct AgentResult {
    bool                   success        = false;
    std::string            final_answer;
    std::vector<AgentStep> steps;
    int                    iterations_used = 0;
    int                    max_iterations  = 0;
    std::string            error;
    long                   duration_ms    = 0;
    TokenUsage             token_usage;

    bool   reached_max() const { return iterations_used >= max_iterations; }
    double avg_step_ms() const {
        if (steps.empty()) return 0.0;
        long t=0; for (const auto& s:steps) t+=s.duration_ms;
        return (double)t/steps.size();
    }
};

// ════════════════════════════════════════════════════════════════
// PlanCache — 计划模板缓存 (基于 NeurIPS 2025 APC 论文)
// 关键词 exact match，LRU 驱逐，跳过 ORCHESTRATOR 规划调用
// ════════════════════════════════════════════════════════════════

class PlanCache {
public:
    explicit PlanCache(size_t max_size = 50) : max_size_(max_size) {}

    static std::string normalize_key(const std::string& task,
                                      const std::vector<ToolDef>& tools);
    bool has(const std::string& key) const;
    WorkflowPlan get(const std::string& key);
    void put(const std::string& key, const WorkflowPlan& plan);
    size_t size() const { return cache_.size(); }
    void clear() { cache_.clear(); order_.clear(); }

    struct Stats { long hits = 0; long misses = 0; };
    Stats stats() const { return stats_; }

private:
    size_t max_size_;
    std::unordered_map<std::string, WorkflowPlan> cache_;
    std::list<std::string> order_;  // front = most recent
    Stats stats_;
    mutable std::mutex mu_;
};

// ════════════════════════════════════════════════════════════════
// MCP Client — Model Context Protocol (stdio transport)
// JSON-RPC 2.0 over subprocess stdin/stdout (NDJSON)
// ════════════════════════════════════════════════════════════════

/** MCP 传输层接口 */
class IMcpTransport {
public:
    virtual ~IMcpTransport() = default;
    virtual void send(const json& message) = 0;
    virtual json receive() = 0;
    virtual void close() = 0;
};

/** Stdio 传输：通过子进程 stdin/stdout 通信 */
class StdioTransport : public IMcpTransport {
public:
    StdioTransport(const std::string& command, const std::vector<std::string>& args = {});
    ~StdioTransport();
    void send(const json& message) override;
    json receive() override;
    void close() override;
private:
    FILE* pipe_ = nullptr;
    bool closed_ = true;
#ifdef _WIN32
    void* process_handle_ = nullptr;
    void* stdin_write_ = nullptr;
    void* stdout_read_ = nullptr;
#else
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    int child_pid_ = -1;
#endif
};

/** MCP 客户端：初始化 → 工具发现 → 工具调用 */
class McpClient {
public:
    explicit McpClient(std::unique_ptr<IMcpTransport> transport);
    ~McpClient();

    json initialize(const std::string& client_name = "ariadne",
                     const std::string& version = "0.3.0");
    std::vector<ToolDef> list_tools();
    json call_tool(const std::string& name, const json& arguments);
    void close();

    void register_all_tools(ToolRegistry& registry);

private:
    std::unique_ptr<IMcpTransport> transport_;
    int next_id_ = 1;
    bool initialized_ = false;
    std::vector<ToolDef> discovered_tools_;

    json make_request(const std::string& method, const json& params = json::object());
};

class WorkflowEngine {
public:
    explicit WorkflowEngine(const EngineConfig& config);

    void           register_tool(const ToolDef& def, ToolFn fn);

    /** 单次运行，无记忆 */
    WorkflowResult run(const std::string& task);

    /** 多轮运行，ctx 在调用之间保留历史 */
    WorkflowResult run(const std::string& task, WorkflowContext& ctx);

    WorkflowPlan   plan_only(const std::string& task);
    void           print_provider_status() const;

    /** 探测当前所有 Provider，返回健康报告 */
    std::vector<ProbeResult> health_check() const;

    /**
     * 带自修复的执行：Provider 故障时调用 planner.repair()，
     * 重建 LLMClient，最多重试 max_repair_attempts 次
     */
    WorkflowResult run_with_recovery(const std::string& task,
                                      ProviderAutoPlanner& planner,
                                      int max_repair_attempts = 2);

    /** 流式执行：最终 LLM 步骤的输出以 chunk 推送给调用方 */
    WorkflowResult run_stream(const std::string& task,
                               StreamCallback on_chunk);

    /**
     * Agent Loop with per-step progress callback.
     * on_step is called after each iteration with a JSON summary:
     *   {"iteration":1,"thought":"...","action":"tool_call","tool":"web_search","status":"ok"}
     */
    AgentResult run_agent(const std::string& task,
                           int max_iterations = 10,
                           std::function<void(const AgentStep&)> on_step = nullptr);

    /**
     * 多 Agent 编排：从 start_agent 开始，agent 可通过 HANDOFF 移交控制权。
     * 全程共享对话历史。
     */
    AgentResult run_multi_agent(const std::string& task,
                                 const std::vector<AgentDef>& agents,
                                 const std::string& start_agent,
                                 int max_iterations = 15,
                                 std::function<void(const AgentStep&)> on_step = nullptr);

    /** Override the planner system prompt (default: PLANNER_SYS constant) */
    void set_planner_prompt(const std::string& sys_prompt);

    /** Override the agent system prompt (default: AGENT_SYS constant) */
    void set_agent_prompt(const std::string& sys_prompt);

    /**
     * Set a custom metrics collector.
     * Thread-safe; can be changed between run() calls.
     * Default: NoOpMetrics (zero overhead).
     */
    void set_metrics(std::shared_ptr<IMetricsCollector> collector);

    /** 取消当前正在执行的 workflow/agent。线程安全。 */
    void cancel();

    /** 重置取消状态（在下次 run 前自动调用） */
    void reset_cancel();

    /** 获取取消令牌（用于外部检查） */
    CancelToken cancel_token() const { return cancel_; }

    /** 设置全局超时（从 run 开始计时，覆盖所有步骤） */
    void set_deadline(std::chrono::seconds timeout);
    void clear_deadline();

    /** Guardrails — 输入/输出/工具验证钩子 */
    void add_input_guardrail(GuardrailFn fn);
    void add_output_guardrail(GuardrailFn fn);
    void add_tool_guardrail(const std::string& tool_name, GuardrailFn fn);

    /** 直接 LLM 调用（跳过 Planner，用于单步执行） */
    std::string llm_complete(const std::string& prompt,
                              const std::string& system = "",
                              ModelTier tier = ModelTier::SUBAGENT,
                              double temperature = 0.0);

    /** 计划缓存控制 */
    void enable_plan_cache(bool on = true) { cache_enabled_ = on; }
    PlanCache::Stats plan_cache_stats() const { return plan_cache_.stats(); }
    void clear_plan_cache() { plan_cache_.clear(); }

    /** LLM 响应缓存控制 */
    void enable_response_cache(bool on = true) { llm_->enable_response_cache(on); }
    long response_cache_hits() const { return llm_->response_cache_hits(); }

    /** MCP 工具注册（通过子进程连接 MCP 服务器，自动发现并注册工具） */
    void connect_mcp(const std::string& command, const std::vector<std::string>& args = {});

private:
    EngineConfig                      config_;
    std::unique_ptr<LLMClient>        llm_;
    std::unique_ptr<ToolRegistry>     tools_;
    std::unique_ptr<WorkflowPlanner>  planner_;
    std::unique_ptr<WorkflowExecutor> executor_;
    std::string                       custom_planner_prompt_;
    std::string                       custom_agent_prompt_;
    std::shared_ptr<IMetricsCollector> metrics_{std::make_shared<NoOpMetrics>()};
    CancelToken                       cancel_{std::make_shared<std::atomic<bool>>(false)};
    std::chrono::steady_clock::time_point deadline_{};
    bool                              has_deadline_ = false;
    PlanCache                         plan_cache_;
    bool                              cache_enabled_ = true;
    std::vector<std::unique_ptr<McpClient>> mcp_clients_;
    std::vector<GuardrailFn>          input_guardrails_;
    std::vector<GuardrailFn>          output_guardrails_;
    std::map<std::string, std::vector<GuardrailFn>> tool_guardrails_;

    WorkflowResult run_internal(const std::string& task, WorkflowContext* ctx);
};

// ════════════════════════════════════════════════════════════════
// 13. DAG 校验
// ════════════════════════════════════════════════════════════════

// DAGValidationError defined earlier in exception hierarchy

void validate_dag(const WorkflowPlan& plan);

} // namespace ariadne
