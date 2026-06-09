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
#include <nlohmann/json.hpp>
#include <curl/curl.h>

using json = nlohmann::json;

namespace ariadne {


// ════════════════════════════════════════════════════════════════
// 异常层次 — 替代散落的 std::runtime_error
// ════════════════════════════════════════════════════════════════

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
                                  bool               force_json  = false) const = 0;
    virtual void complete_stream(const std::string& prompt,
                                  const std::string& system,
                                  double             temperature,
                                  StreamCallback     on_chunk) const = 0;
    virtual std::string provider_name() const = 0;
    virtual std::string model_name()    const = 0;
};

/**
 * HttpProvider — 共享 curl 逻辑的基类 (R1)
 * 子类只需实现 build_request() 和 parse_response()
 */
class HttpProvider : public ILLMProvider {
public:
    explicit HttpProvider(const ProviderConfig& cfg) : cfg_(cfg) {}
protected:
    ProviderConfig cfg_;
    CurlHandle     curl_;

    std::string http_post(const std::string& url,
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
                          bool force_json = false) const override;
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
                          bool force_json = false) const override;
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
                          bool force_json = false) const override;
    void complete_stream(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          StreamCallback on_chunk) const override;
    std::string provider_name() const override { return "openai_responses"; }
    std::string model_name()    const override { return cfg_.model;          }
};

std::unique_ptr<ILLMProvider> make_provider(const ProviderConfig& cfg);

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

class LLMClient {
public:
    LLMClient(TierConfig orchestrator, TierConfig subagent);

    std::string complete_as(ModelTier tier,
                             const std::string& prompt,
                             const std::string& system     = "",
                             double             temperature = 0.0,
                             bool               force_json  = false) const;

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

private:
    struct Slot {
        std::unique_ptr<ILLMProvider>       provider;
        mutable CircuitBreaker              breaker;
        mutable RateLimiter                 rate_limiter;
        mutable std::unique_ptr<std::mutex> stats_mu{std::make_unique<std::mutex>()};
        mutable long calls = 0, successes = 0, failures = 0;
    };
    std::vector<Slot> orchestrators_, subagents_;

    static std::vector<Slot> build_slots(const TierConfig& cfg);
    std::string try_slots(const std::vector<Slot>& slots,
                           const std::string& prompt,
                           const std::string& system,
                           double temperature,
                           ModelTier tier,
                           bool force_json = false) const;
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
    std::string provider;     // 哪个 provider 处理的（仅 llm 步骤）
    long        duration_ms = 0;
    std::string error;        // 失败时填写
};

struct WorkflowState {
    json                               task_input;
    std::map<std::string, json>        step_outputs;
    std::map<std::string, std::string> errors;
    std::vector<TraceEntry>            traces;

    json resolve_ref   (const std::string& ref) const;
    json resolve_inputs(const json& inputs)      const;
    void record_trace  (TraceEntry entry) noexcept;
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
private:
    std::map<std::string, ToolDef> defs_;
    std::map<std::string, ToolFn>  fns_;
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
    explicit WorkflowPlanner(LLMClient& llm);

    /** 单次 LLM 调用完成分析+规划（推荐；比 analyze_task+plan_static 快 1-2s）*/
    WorkflowPlan plan        (const std::string& task,
                               const std::vector<ToolDef>& tools,
                               const WorkflowContext& ctx = {},
                               int max_attempts = 3) const;

    /** @deprecated since v0.2.0 — use plan() directly */
    [[deprecated("use plan() instead")]] json analyze_task(const std::string& task,
                               const std::vector<ToolDef>& tools,
                               const WorkflowContext& ctx = {}) const;
    [[deprecated("use plan() instead")]] WorkflowPlan plan_static(const std::string& task,
                               const json& analysis,
                               const std::vector<ToolDef>& tools,
                               int max_attempts = 3) const;
    [[deprecated("use plan() instead")]] WorkflowPlan replan(const std::string& task,
                               const WorkflowPlan& current,
                               const std::vector<json>& failures) const;

    static json         extract_json(const std::string& text);
    static WorkflowPlan parse_plan  (const json& raw, const std::string& task);

private:
    LLMClient& llm_;
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



class WorkflowExecutor {
public:
    WorkflowExecutor(LLMClient& llm, ToolRegistry& tools, size_t max_threads = 0);
    WorkflowState execute(const WorkflowPlan& plan,
                           const json& task_input) const;
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
    mutable ThreadPool pool_;  ///< bounded thread pool (mutable: execute() is const)

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
    std::string checkpoint_dir  = "";

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
    enum class Type { TOOL_CALL, FINAL_ANSWER, LOOP_BACK };
    Type        type     = Type::FINAL_ANSWER;
    std::string thought;
    std::string tool_name;
    json        tool_args;
    std::string response;   // FINAL_ANSWER
    std::string reason;     // LOOP_BACK
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

    bool   reached_max() const { return iterations_used >= max_iterations; }
    double avg_step_ms() const {
        if (steps.empty()) return 0.0;
        long t=0; for (const auto& s:steps) t+=s.duration_ms;
        return (double)t/steps.size();
    }
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

    // run_agent: see 3-param overload with on_step callback above

private:
    EngineConfig                      config_;
    std::unique_ptr<LLMClient>        llm_;
    std::unique_ptr<ToolRegistry>     tools_;
    std::unique_ptr<WorkflowPlanner>  planner_;
    std::unique_ptr<WorkflowExecutor> executor_;
    std::string                       custom_planner_prompt_;
    std::string                       custom_agent_prompt_;
    std::shared_ptr<IMetricsCollector> metrics_{std::make_shared<NoOpMetrics>()};

    WorkflowResult run_internal(const std::string& task, WorkflowContext* ctx);
};

// ════════════════════════════════════════════════════════════════
// 13. DAG 校验
// ════════════════════════════════════════════════════════════════

// DAGValidationError defined earlier in exception hierarchy

void validate_dag(const WorkflowPlan& plan);

} // namespace ariadne
