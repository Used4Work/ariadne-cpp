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
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <optional>
#include <nlohmann/json.hpp>
#include <curl/curl.h>

using json = nlohmann::json;

namespace ariadne {

/** 粗略 token 估算（英文 ~4 chars/token, 中文 ~2 chars/token） */
inline long estimate_tokens(const std::string& text) {
    long ascii = 0, non_ascii = 0;
    for (unsigned char c : text) { if (c < 128) ++ascii; else ++non_ascii; }
    return ascii / 4 + non_ascii / 2 + 1;
}

/** 库版本号 */
#if __has_include("ariadne_version_gen.hpp")
#include "ariadne_version_gen.hpp"
constexpr const char* ARIADNE_VERSION = ARIADNE_VERSION_STRING;
#else
constexpr const char* ARIADNE_VERSION = "2.3.0";
#endif
inline std::string version() { return ARIADNE_VERSION; }


// ════════════════════════════════════════════════════════════════
// Token 用量追踪
// ════════════════════════════════════════════════════════════════

struct TokenUsage {
    long   input_tokens  = 0;
    long   output_tokens = 0;
    long   total_tokens  = 0;
    double cost_usd      = 0.0;
    TokenUsage& operator+=(const TokenUsage& o) {
        input_tokens += o.input_tokens; output_tokens += o.output_tokens;
        total_tokens += o.total_tokens; cost_usd += o.cost_usd; return *this;
    }
};

/** 模型定价（每 1M tokens，USD） */
struct ModelPricing {
    double input_per_1m  = 0.0;
    double output_per_1m = 0.0;
    double cost(long in_tokens, long out_tokens) const {
        return (in_tokens * input_per_1m + out_tokens * output_per_1m) / 1e6;
    }
    static ModelPricing free() { return {0.0, 0.0}; }
    static ModelPricing gpt4o_mini() { return {0.15, 0.60}; }
    static ModelPricing gpt4o() { return {2.50, 10.00}; }
    static ModelPricing claude_sonnet() { return {3.00, 15.00}; }
    static ModelPricing claude_opus() { return {15.00, 75.00}; }
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

/** Token 预算超出 */
class TokenBudgetError : public AriadneError {
    using AriadneError::AriadneError;
};

/** Human-in-the-loop 中断 — 抛出此异常暂停执行 */
class InterruptError : public AriadneError {
public:
    InterruptError(std::string step_id, std::string reason, json state_snapshot)
        : AriadneError("Interrupted at step '" + step_id + "': " + reason)
        , step_id(std::move(step_id))
        , reason_(std::move(reason))
        , state_snapshot(std::move(state_snapshot)) {}
    std::string step_id;
    std::string reason_;
    json state_snapshot;
};

/** MCP 传输/协议错误 */
class McpError : public AriadneError {
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
// Structured Logging — 替代散落的 cerr/cout
// ════════════════════════════════════════════════════════════════

enum class LogLevel { LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR };

class ILogger {
public:
    virtual ~ILogger() = default;
    virtual void log(LogLevel level, const std::string& component,
                      const std::string& message) noexcept = 0;
};

class NullLogger : public ILogger {
public:
    void log(LogLevel, const std::string&, const std::string&) noexcept override {}
};

class ConsoleLogger : public ILogger {
public:
    explicit ConsoleLogger(LogLevel min_level = LogLevel::LOG_INFO) : min_(min_level) {}
    void log(LogLevel level, const std::string& component,
              const std::string& message) noexcept override {
        if (level < min_) return;
        static const char* names[] = {"DEBUG","INFO","WARN","ERR"};
        try {
            std::cerr << "[" << names[(int)level] << "] "
                      << component << ": " << message << "\n";
        } catch (...) {}
    }
private:
    LogLevel min_;
};

inline std::shared_ptr<ILogger>& global_logger() {
    static auto inst = std::shared_ptr<ILogger>(std::make_shared<NullLogger>());
    return inst;
}
inline void set_logger(std::shared_ptr<ILogger> logger) {
    if (logger) global_logger() = std::move(logger);
}
inline void log_msg(LogLevel level, const std::string& component, const std::string& msg) {
    global_logger()->log(level, component, msg);
}

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

enum class ProviderType { ANTHROPIC, OPENAI_CHAT, OPENAI_RESPONSES, GEMINI };

struct ProviderConfig {
    ProviderType type;
    std::string  api_key;
    std::string  model;
    std::string  base_url;
    int          max_tokens       = 4096;
    double       timeout_sec      = 60.0;
    std::string  completions_path = "";   // 空 = /v1/chat/completions
    double       max_rps          = 0.0;  // 每秒最大请求数；0=不限速
    ModelPricing pricing;                 // 自动成本追踪

    static ProviderConfig anthropic(const std::string& key,
                                     const std::string& model = "claude-opus-4-8") {
        ProviderConfig c{ProviderType::ANTHROPIC, key, model, "", 4096, 60.0, ""};
        c.pricing = (model.find("opus") != std::string::npos) ? ModelPricing::claude_opus() :
                    ModelPricing::claude_sonnet();
        return c;
    }
    static ProviderConfig openai_chat(const std::string& key,
                                       const std::string& model = "gpt-4o") {
        ProviderConfig c{ProviderType::OPENAI_CHAT, key, model, "", 4096, 60.0, ""};
        c.pricing = (model.find("mini") != std::string::npos) ? ModelPricing::gpt4o_mini() :
                    ModelPricing::gpt4o();
        return c;
    }
    static ProviderConfig openai_responses(const std::string& key,
                                            const std::string& model = "gpt-4o") {
        ProviderConfig c{ProviderType::OPENAI_RESPONSES, key, model, "", 4096, 60.0, ""};
        c.pricing = (model.find("mini") != std::string::npos) ? ModelPricing::gpt4o_mini() :
                    ModelPricing::gpt4o();
        return c;
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
                                         double max_rps = 0.1) {
        ProviderConfig c;
        c.type             = ProviderType::OPENAI_CHAT;
        c.api_key          = token;
        c.model            = model;
        c.base_url         = "https://models.github.ai/inference";
        c.completions_path = "/chat/completions";
        c.max_rps          = max_rps;
        c.pricing          = ModelPricing::free();
        return c;
    }
    /** Groq free tier: 30 RPM = 0.5 RPS */
    static ProviderConfig groq(const std::string& key,
                                const std::string& model = "llama-3.3-70b-versatile") {
        return openai_compatible(key, "https://api.groq.com/openai", model, 0.5);
    }
    /** LLM7.io free tier: 30 RPM, no signup needed (key="unused") */
    static ProviderConfig llm7(const std::string& model = "deepseek-v3-0324") {
        return openai_compatible("unused", "https://api.llm7.io", model, 0.5);
    }
    /** Cerebras free tier: 1M tokens/day, 30 RPM */
    static ProviderConfig cerebras(const std::string& key,
                                    const std::string& model = "llama-3.3-70b") {
        return openai_compatible(key, "https://api.cerebras.ai/v1", model, 0.5);
    }
    /** SambaNova free tier: forever-free + $5 credit */
    static ProviderConfig sambanova(const std::string& key,
                                     const std::string& model = "Meta-Llama-3.3-70B-Instruct") {
        return openai_compatible(key, "https://api.sambanova.ai/v1", model, 0.5);
    }
    /** Mistral free tier: ~1B tokens/month */
    static ProviderConfig mistral(const std::string& key,
                                   const std::string& model = "mistral-small-latest") {
        return openai_compatible(key, "https://api.mistral.ai/v1", model, 1.0);
    }
    /** Google Gemini (free tier: 15 RPM for Flash models) */
    static ProviderConfig gemini(const std::string& key,
                                  const std::string& model = "gemini-2.0-flash") {
        ProviderConfig c;
        c.type     = ProviderType::GEMINI;
        c.api_key  = key;
        c.model    = model;
        c.max_rps  = 0.25;
        c.pricing  = (model.find("flash") != std::string::npos)
                     ? ModelPricing{0.075, 0.30} : ModelPricing{1.25, 5.00};
        return c;
    }
};

struct ToolDef;  // forward declaration

// ════════════════════════════════════════════════════════════════
// Vector Memory Store — 轻量级语义检索（无外部依赖）
//
// 用法：
//   InMemoryVectorStore store;
//   store.add("id1", {0.1, 0.2, 0.3}, {{"text","hello"}});
//   auto results = store.query({0.1, 0.2, 0.3}, 5);
//   // results[0].id, results[0].score, results[0].metadata
// ════════════════════════════════════════════════════════════════

struct VectorEntry {
    std::string          id;
    std::vector<float>   embedding;
    json                 metadata;
};

struct VectorResult {
    std::string id;
    float       score = 0.0f;
    json        metadata;
};

class IMemoryStore {
public:
    virtual ~IMemoryStore() = default;
    virtual void add(const std::string& id, const std::vector<float>& embedding,
                      const json& metadata = {}) = 0;
    virtual std::vector<VectorResult> query(const std::vector<float>& embedding,
                                             int top_k = 5) const = 0;
    virtual void remove(const std::string& id) = 0;
    virtual size_t size() const = 0;
    virtual void clear() = 0;
};

class InMemoryVectorStore : public IMemoryStore {
public:
    void add(const std::string& id, const std::vector<float>& embedding,
              const json& metadata = {}) override;
    std::vector<VectorResult> query(const std::vector<float>& embedding,
                                     int top_k = 5) const override;
    void remove(const std::string& id) override;
    size_t size() const override;
    void clear() override;
private:
    mutable std::shared_mutex mu_;
    std::vector<VectorEntry> entries_;
};

// ── Native Tool Calling 数据类型 ─────────────────────────

struct ChatMessage {
    std::string role;           // "system", "user", "assistant", "tool"
    std::string content;
    json        tool_calls;     // assistant 消息的 tool_calls 数组
    std::string tool_call_id;   // tool 结果消息的 call ID
    std::string name;           // tool result 的工具名
    json        content_parts;  // multimodal: [{type,text},{type,image_url}] — 非空时替代 content

    /** 创建纯文本消息 */
    static ChatMessage text(const std::string& role, const std::string& text) {
        return {role, text, {}, "", "", {}};
    }
    /** 创建含图片的多模态消息（base64 编码） */
    static ChatMessage with_image(const std::string& text,
                                   const std::string& base64_data,
                                   const std::string& media_type = "image/jpeg") {
        json parts = json::array();
        parts.push_back({{"type","text"},{"text",text}});
        parts.push_back({{"type","image_url"},
            {"image_url",{{"url","data:" + media_type + ";base64," + base64_data}}}});
        return {"user", "", {}, "", "", parts};
    }
    /** 创建含图片 URL 的多模态消息 */
    static ChatMessage with_image_url(const std::string& text,
                                       const std::string& url) {
        json parts = json::array();
        parts.push_back({{"type","text"},{"text",text}});
        parts.push_back({{"type","image_url"},{"image_url",{{"url",url}}}});
        return {"user", "", {}, "", "", parts};
    }

    bool is_multimodal() const { return !content_parts.is_null() && !content_parts.empty(); }
};

struct LLMToolCall {
    std::string id;             // provider 分配的 call ID
    std::string name;           // 工具名
    json        arguments;      // 工具参数（已解析的 JSON）
};

struct LLMResponse {
    std::string             content;
    std::vector<LLMToolCall> tool_calls;
    bool has_tool_calls() const { return !tool_calls.empty(); }
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

    /** Native tool calling — 多轮对话 + 结构化工具调用
     *  tool_choice: "auto"(default), "none", "required", or tool name
     *  默认实现回退到 complete()（仅用于不支持原生工具的 provider） */
    virtual LLMResponse complete_chat(const std::vector<ChatMessage>& messages,
                                       const std::vector<ToolDef>& tools = {},
                                       double temperature = 0.0,
                                       const std::string& tool_choice = "auto") const;

    /** 是否支持原生工具调用 */
    virtual bool supports_native_tools() const { return false; }
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
        std::string retry_after;
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
    LLMResponse complete_chat(const std::vector<ChatMessage>& messages,
                               const std::vector<ToolDef>& tools = {},
                               double temperature = 0.0,
                               const std::string& tool_choice = "auto") const override;
    bool supports_native_tools() const override { return true; }
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
    LLMResponse complete_chat(const std::vector<ChatMessage>& messages,
                               const std::vector<ToolDef>& tools = {},
                               double temperature = 0.0,
                               const std::string& tool_choice = "auto") const override;
    bool supports_native_tools() const override { return true; }
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

/** POST /v1beta/models/{model}:generateContent → candidates[0].content.parts[0].text */
class GeminiProvider : public HttpProvider {
public:
    explicit GeminiProvider(const ProviderConfig& cfg) : HttpProvider(cfg) {}
    std::string complete(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          bool force_json = false,
                          const json& output_schema = json()) const override;
    void complete_stream(const std::string& prompt,
                          const std::string& system,
                          double temperature,
                          StreamCallback on_chunk) const override;
    LLMResponse complete_chat(const std::vector<ChatMessage>& messages,
                               const std::vector<ToolDef>& tools = {},
                               double temperature = 0.0,
                               const std::string& tool_choice = "auto") const override;
    bool supports_native_tools() const override { return true; }
    std::string provider_name() const override { return "gemini"; }
    std::string model_name()    const override { return cfg_.model; }
};

std::unique_ptr<ILLMProvider> make_provider(const ProviderConfig& cfg);

/** 测试用 Mock Provider — 返回预设响应，支持原生工具调用模拟 */
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
    LLMResponse complete_chat(const std::vector<ChatMessage>& messages,
                               const std::vector<ToolDef>& tools = {},
                               double temperature = 0.0,
                               const std::string& tool_choice = "auto") const override {
        LLMResponse r;
        if (!mock_tool_calls_.empty()) {
            r.tool_calls = mock_tool_calls_;
            mock_tool_calls_.clear();
        } else {
            r.content = response_;
        }
        return r;
    }
    bool supports_native_tools() const override { return true; }
    std::string provider_name() const override { return name_; }
    std::string model_name()    const override { return model_; }

    void set_response(const std::string& r) { response_ = r; }
    void set_tool_calls(const std::vector<LLMToolCall>& calls) { mock_tool_calls_ = calls; }
private:
    std::string response_, name_, model_;
    mutable std::vector<LLMToolCall> mock_tool_calls_;
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
    long                  avg_latency_ms = 0;
    long                  last_latency_ms = 0;
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

    /** Native tool calling — multi-turn chat with structured tool calls
     *  tool_choice: "auto", "none", "required", or specific tool name */
    LLMResponse complete_chat(ModelTier tier,
                               const std::vector<ChatMessage>& messages,
                               const std::vector<ToolDef>& tools = {},
                               double temperature = 0.0,
                               const std::string& tool_choice = "auto") const;
    bool supports_native_tools(ModelTier tier) const;

    std::vector<ProviderStats> stats(ModelTier tier) const;
    void print_status() const;
    TokenUsage total_usage() const;
    void reset_usage();
    void enable_response_cache(bool on = true);
    long response_cache_hits() const;
    void set_token_budget(long max_tokens) { token_budget_ = max_tokens; }
    void clear_token_budget() { token_budget_ = 0; }

private:
    struct Slot {
        std::unique_ptr<ILLMProvider>       provider;
        mutable CircuitBreaker              breaker;
        mutable RateLimiter                 rate_limiter;
        mutable std::unique_ptr<std::mutex> stats_mu{std::make_unique<std::mutex>()};
        mutable long calls = 0, successes = 0, failures = 0;
        mutable long total_latency_ms = 0, last_latency_ms = 0;
    };
    std::vector<Slot> orchestrators_, subagents_;
    mutable std::mutex usage_mu_;
    mutable TokenUsage cumulative_usage_;
    mutable std::unique_ptr<ResponseCache> response_cache_;
    std::atomic<bool> resp_cache_enabled_{false};
    std::atomic<long> token_budget_{0};

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

    /** 序列化为 JSON（可持久化/导出） */
    json to_json() const;
    /** 从 JSON 反序列化 */
    static WorkflowPlan from_json(const json& j);
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

    /** 序列化为 JSON（用于 checkpointing） */
    json to_json() const;
    /** 从 JSON 恢复 */
    static WorkflowState from_json(const json& j);

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
// Checkpointing — WorkflowState 持久化 + 断点恢复
// ════════════════════════════════════════════════════════════════

/** Checkpoint 存储接口 */
class ICheckpointStore {
public:
    virtual ~ICheckpointStore() = default;
    virtual void save(const std::string& workflow_id, const json& state) = 0;
    virtual json load(const std::string& workflow_id) = 0;
    virtual bool exists(const std::string& workflow_id) = 0;
    virtual void remove(const std::string& workflow_id) = 0;
    virtual std::vector<std::string> list() = 0;
};

/** 文件系统 Checkpoint 存储 */
class FileCheckpointStore : public ICheckpointStore {
public:
    explicit FileCheckpointStore(const std::string& directory);
    void save(const std::string& workflow_id, const json& state) override;
    json load(const std::string& workflow_id) override;
    bool exists(const std::string& workflow_id) override;
    void remove(const std::string& workflow_id) override;
    std::vector<std::string> list() override;
private:
    std::string dir_;
};

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
    mutable std::shared_mutex mu_;
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

    /** LLM 调用完成分析+规划 */
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
    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return cache_.size(); }
    void clear() { std::lock_guard<std::mutex> lk(mu_); cache_.clear(); order_.clear(); }

    struct Stats { long hits = 0; long misses = 0; };
    Stats stats() const { std::lock_guard<std::mutex> lk(mu_); return stats_; }

private:
    size_t max_size_;
    std::unordered_map<std::string, std::string> cache_;
    std::list<std::string> order_;
    mutable Stats stats_;
    mutable std::mutex mu_;
};

class WorkflowExecutor {
public:
    using StepInterruptFn = std::function<std::optional<std::string>(const Step&, const WorkflowState&)>;

    WorkflowExecutor(LLMClient& llm, ToolRegistry& tools, size_t max_threads = 0,
                      std::shared_ptr<IMetricsCollector> metrics = nullptr);
    WorkflowState execute(const WorkflowPlan& plan,
                           const json& task_input,
                           CancelToken cancel = nullptr,
                           StepInterruptFn interrupt = nullptr) const;
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

/** 控制台输出：每条事件一行 JSON（通过 ILogger 输出） */
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
            log_msg(LogLevel::LOG_INFO, "metrics", j.dump());
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
    std::vector<TraceEntry> traces;
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
                                      const std::vector<ToolDef>& tools,
                                      const std::string& context = "");
    bool has(const std::string& key) const;
    WorkflowPlan get(const std::string& key);
    void put(const std::string& key, const WorkflowPlan& plan);
    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return cache_.size(); }
    void clear() { std::lock_guard<std::mutex> lk(mu_); cache_.clear(); order_.clear(); }

    struct Stats { long hits = 0; long misses = 0; };
    Stats stats() const { std::lock_guard<std::mutex> lk(mu_); return stats_; }

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

/** HTTP 传输：通过 HTTP POST 与 MCP 服务器通信 (Streamable HTTP) */
class HttpTransport : public IMcpTransport {
public:
    HttpTransport(const std::string& url, const std::string& api_key = "");
    void send(const json& message) override;
    json receive() override;
    void close() override;
private:
    std::string url_;
    std::string api_key_;
    json pending_response_;
    bool has_pending_ = false;
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
                     const std::string& ver = ARIADNE_VERSION);
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

    /** 便利构造器 — 单 provider，一行创建 engine */
    explicit WorkflowEngine(const ProviderConfig& provider)
        : WorkflowEngine(EngineConfig::from_single(provider)) {}

    /** 便利构造器 — 双 provider（orchestrator + subagent） */
    WorkflowEngine(const ProviderConfig& orchestrator, const ProviderConfig& subagent)
        : WorkflowEngine(EngineConfig::from_two(orchestrator, subagent)) {}

    WorkflowEngine(const WorkflowEngine&)            = delete;
    WorkflowEngine& operator=(const WorkflowEngine&) = delete;

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

    /** Native tool calling agent — 使用 provider 原生 tools API
     *  准确率显著高于 prompt-based（97-99% vs ~85%）
     *  自动降级：provider 不支持原生工具时回退到 prompt-based */
    AgentResult run_agent_native(const std::string& task,
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

    /** 直接调用已注册的工具 */
    json call_tool(const std::string& name, const json& params);
    bool has_tool(const std::string& name) const { return tools_->has_tool(name); }
    std::vector<ToolDef> list_tools() const { return tools_->list_tools(); }

    /** Token budget enforcement — 设置最大 token 预算，超出时抛出 TokenBudgetError */
    void set_token_budget(long max_tokens) {
        token_budget_ = max_tokens;
        llm_->set_token_budget(max_tokens);
    }
    void clear_token_budget() {
        token_budget_ = 0;
        llm_->clear_token_budget();
    }
    long token_budget() const { return token_budget_; }
    bool budget_exceeded() const {
        return token_budget_ > 0 && llm_->total_usage().total_tokens >= token_budget_;
    }

    /** 计划缓存控制 */
    void enable_plan_cache(bool on = true) { cache_enabled_ = on; }
    PlanCache::Stats plan_cache_stats() const { return plan_cache_.stats(); }
    void clear_plan_cache() { plan_cache_.clear(); }

    /** LLM 响应缓存控制 */
    void enable_response_cache(bool on = true) { llm_->enable_response_cache(on); }
    long response_cache_hits() const { return llm_->response_cache_hits(); }

    /** MCP 工具注册（通过子进程连接 MCP 服务器，自动发现并注册工具） */
    void connect_mcp(const std::string& command, const std::vector<std::string>& args = {});

    /** MCP over HTTP（Streamable HTTP transport） */
    void connect_mcp_http(const std::string& url, const std::string& api_key = "");

    /** Checkpointing — 设置存储后端 */
    void set_checkpoint_store(std::shared_ptr<ICheckpointStore> store) { checkpoint_store_ = std::move(store); }

    /** Memory store — 语义检索（用于 RAG / 长期记忆） */
    void set_memory_store(std::shared_ptr<IMemoryStore> store) { memory_store_ = std::move(store); }
    std::shared_ptr<IMemoryStore> memory_store() const { return memory_store_; }

    /** Human-in-the-loop 中断回调 — 在每个步骤执行前调用
     *  返回 nullopt=继续, string=暂停原因 (抛出 InterruptError) */
    using InterruptFn = std::function<std::optional<std::string>(const Step& step, const WorkflowState& state)>;
    void set_interrupt_hook(InterruptFn fn) { interrupt_hook_ = std::move(fn); }

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
    long                              token_budget_ = 0;
    std::vector<std::unique_ptr<McpClient>> mcp_clients_;
    std::vector<GuardrailFn>          input_guardrails_;
    std::vector<GuardrailFn>          output_guardrails_;
    std::map<std::string, std::vector<GuardrailFn>> tool_guardrails_;
    std::shared_ptr<ICheckpointStore> checkpoint_store_;
    std::shared_ptr<IMemoryStore> memory_store_;
    InterruptFn interrupt_hook_;

    WorkflowResult run_internal(const std::string& task, WorkflowContext* ctx);
};

// ════════════════════════════════════════════════════════════════
// Dynamic Workflow — Ultracode 级别的动态任务编排
//
// 对标 Claude Code Ultracode 的核心原语：
//   parallel()     → 并行扇出，等待所有结果 (barrier)
//   pipeline()     → 无 barrier 流水线，item 独立流过各阶段
//   map()          → 并行 map，对集合中每个元素应用函数
//   loop_until()   → 动态循环直到条件满足 (loop-until-dry)
//   fan_out_agents → 并行启动多个 ReACT agent
//
// 用法：
//   DynamicWorkflow dw(engine);
//   dw.on_progress([](auto& p, auto& m){ std::cout << "[" << p << "] " << m << "\n"; });
//
//   // 并行扇出 3 个 agent
//   auto results = dw.fan_out_agents({"搜索Tesla","搜索BYD","搜索NIO"}, 8);
//
//   // 流水线：搜索 → 分析 → 验证
//   auto verified = dw.pipeline(items,
//       {[&](auto& x){ return engine.llm_complete("分析: " + x.dump()); },
//        [&](auto& x){ return engine.llm_complete("验证: " + x.dump()); }});
//
//   // 循环直到找到 10 个结果
//   auto all = dw.loop_until(
//       [](int r, auto& acc){ return acc.size() >= 10; },
//       [&](int r){ return engine.run_agent("找更多 bug, round " + std::to_string(r), 5).final_answer; });
// ════════════════════════════════════════════════════════════════

using DynTask  = std::function<json()>;
using StageFn  = std::function<json(const json& input)>;
using StopFn   = std::function<bool(int round, const std::vector<json>& accumulated)>;
using RoundFn  = std::function<std::vector<json>(int round)>;
using ProgressFn = std::function<void(const std::string& phase, const std::string& message)>;

/** 动态编排结果 */
struct DynamicResult {
    bool                success = true;
    std::vector<json>   outputs;
    int                 rounds_used = 0;
    long                duration_ms = 0;
    TokenUsage          token_usage;
    std::string         error;
    std::vector<std::string> log;
};

class DynamicWorkflow {
public:
    explicit DynamicWorkflow(WorkflowEngine& engine, size_t max_concurrency = 0);
    ~DynamicWorkflow();
    DynamicWorkflow(const DynamicWorkflow&)            = delete;
    DynamicWorkflow& operator=(const DynamicWorkflow&) = delete;

    // ── 核心原语 ────────────────────────────────────────

    /** parallel — 并行执行 N 个任务，等待所有完成 (barrier)
     *  失败的任务返回 null，不中断其他任务 */
    std::vector<json> parallel(const std::vector<DynTask>& tasks);

    /** map — 对集合中每个元素并行应用 fn */
    std::vector<json> map(const std::vector<json>& items, StageFn fn);

    /** pipeline — 无 barrier 流水线
     *  每个 item 独立流过所有 stage，item A 可以在 stage 3 而 item B 还在 stage 1
     *  某个 item 在某个 stage 失败则该 item 后续 stage 跳过，结果为 null */
    std::vector<json> pipeline(const std::vector<json>& items,
                                const std::vector<StageFn>& stages);

    /** loop_until — 动态循环直到 stop 返回 true
     *  每轮调用 work(round)，结果追加到累积集合
     *  支持 budget-aware：token 预算耗尽自动停止 */
    DynamicResult loop_until(StopFn stop, RoundFn work, int max_rounds = 50);

    // ── 便利方法 ────────────────────────────────────────

    /** 并行启动多个 ReACT agent，返回各自的 final_answer */
    std::vector<json> fan_out_agents(const std::vector<std::string>& tasks,
                                      int max_iterations = 10);

    /** 对抗验证：对一个 claim 启动 N 个独立验证 agent
     *  每个返回 {verified: bool, reason: string}
     *  多数票决定最终结果 */
    json adversarial_verify(const std::string& claim, int num_voters = 3);

    // ── 进度 & 控制 ────────────────────────────────────

    void phase(const std::string& name);
    void log(const std::string& message);
    void on_progress(ProgressFn fn) { progress_fn_ = std::move(fn); }

    /** 当前阶段名 */
    std::string current_phase() const { return current_phase_; }

private:
    WorkflowEngine& engine_;
    ThreadPool pool_;
    ProgressFn progress_fn_;
    std::string current_phase_ = "init";
    std::vector<std::string> log_;
    std::mutex log_mu_;

    void emit(const std::string& msg);
};

// ════════════════════════════════════════════════════════════════
// Adaptive Orchestrator — 自动策略选择 + 动态编排
//
// 根据任务自动决定最优执行策略：
//   SIMPLE_DAG       → 简单任务，用 engine.run()
//   AGENT_LOOP       → 开放探索，用 engine.run_agent()
//   PARALLEL_RESEARCH → 多子主题，fan_out_agents() + 综合
//   PIPELINE_VERIFY   → 研究+验证，pipeline() 多阶段
//   MULTI_AGENT       → 专业分工，run_multi_agent()
//
// 用法：
//   AdaptiveOrchestrator orch(engine);
//   auto result = orch.run("对比 Tesla 和 BYD 的 Q4 销量");
//   // 自动选择 PARALLEL_RESEARCH，并行搜索 + 对比综合
// ════════════════════════════════════════════════════════════════

enum class OrchestratorStrategy {
    SIMPLE_DAG,         // 直接 DAG workflow
    AGENT_LOOP,         // 单 agent 循环
    PARALLEL_RESEARCH,  // 并行扇出 + 综合
    PIPELINE_VERIFY,    // 流水线：研究 → 分析 → 验证
    MULTI_AGENT         // 多 agent 协作
};

struct OrchestratorPlan {
    OrchestratorStrategy         strategy;
    std::string                  reasoning;
    std::vector<std::string>     subtasks;
    std::string                  synthesis_prompt;
    int                          max_iterations = 8;
};

struct OrchestratorResult {
    bool        success = false;
    json        output;
    std::string strategy_used;
    std::string reasoning;
    long        duration_ms = 0;
    TokenUsage  token_usage;
    std::string error;
    std::vector<std::string> log;
};

class AdaptiveOrchestrator {
public:
    explicit AdaptiveOrchestrator(WorkflowEngine& engine);

    /** 自动分析任务 → 选择策略 → 编排执行 → 返回结果 */
    OrchestratorResult run(const std::string& task);

    /** 仅分析任务并返回推荐策略（不执行） */
    OrchestratorPlan analyze(const std::string& task);

    /** 使用指定策略执行（跳过自动分析） */
    OrchestratorResult run_with_strategy(const std::string& task,
                                          OrchestratorStrategy strategy);

    /** 进度回调 */
    void on_progress(ProgressFn fn) { progress_fn_ = std::move(fn); }

private:
    WorkflowEngine& engine_;
    ProgressFn progress_fn_;

    OrchestratorPlan plan_strategy(const std::string& task);
    OrchestratorResult execute_plan(const std::string& task, const OrchestratorPlan& plan);

    static std::string strategy_name(OrchestratorStrategy s);
};

// ════════════════════════════════════════════════════════════════
// 13. DAG 校验
// ════════════════════════════════════════════════════════════════

// DAGValidationError defined earlier in exception hierarchy

void validate_dag(const WorkflowPlan& plan);

} // namespace ariadne
