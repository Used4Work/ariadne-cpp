#ifdef _WIN32
#  define NOMINMAX
#endif
#include "ariadne.hpp"
#include <thread>
#include <sstream>
#include <regex>
#include <algorithm>
#include <set>
#include <queue>
#include <iomanip>
#include <iostream>
#include <random>
#include <deque>
#include <filesystem>
#include <fstream>

#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#endif

namespace ariadne {

// ════════════════════════════════════════════════════════════════
// HttpProvider — 共享 HTTP 逻辑 (R1: 消除三份重复代码)
// ════════════════════════════════════════════════════════════════

size_t HttpProvider::write_cb(char* p, size_t s, size_t n, void* u) {
    static_cast<std::string*>(u)->append(p, s * n);
    return s * n;
}

static std::string generate_idempotency_key() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t a = dist(rng), b = dist(rng);
    char buf[40];
    snprintf(buf, sizeof(buf), "%016llx-%016llx",
             (unsigned long long)a, (unsigned long long)b);
    return buf;
}

HttpProvider::HttpResponse HttpProvider::http_post(const std::string& url,
                                                    const std::vector<std::string>& hdrs,
                                                    const std::string& body) const {
    CurlHandle curl;
    CURL* c = curl.get();
    HttpResponse resp;
    struct curl_slist* h = nullptr;
    for (const auto& s : hdrs) h = curl_slist_append(h, s.c_str());
    // Idempotency key — prevents double-billing on retried requests
    auto idem_key = "Idempotency-Key: " + generate_idempotency_key();
    h = curl_slist_append(h, idem_key.c_str());
    curl_easy_setopt(c, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER,    h);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS,    body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, (long)body.size());
    // Header callback to capture Retry-After
    std::string resp_headers;
    auto header_cb = [](char* buf, size_t size, size_t nitems, void* userdata) -> size_t {
        static_cast<std::string*>(userdata)->append(buf, size * nitems);
        return size * nitems;
    };
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,     &resp.body);
    curl_easy_setopt(c, CURLOPT_HEADERFUNCTION, +header_cb);
    curl_easy_setopt(c, CURLOPT_HEADERDATA,     &resp_headers);
    // Adaptive timeout: scale with max_tokens to prevent premature timeout on long generations
    double adaptive_timeout = std::max(cfg_.timeout_sec, cfg_.max_tokens / 30.0 + 10.0);
    curl_easy_setopt(c, CURLOPT_TIMEOUT,        (long)adaptive_timeout);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL,       1L);
    CURLcode rc = curl_easy_perform(c);
    curl_slist_free_all(h);
    if (rc != CURLE_OK)
        throw ProviderError(std::string("curl: ") + curl_easy_strerror(rc));
    // Parse Retry-After from response headers
    auto ra_pos = resp_headers.find("retry-after:");
    if (ra_pos == std::string::npos) ra_pos = resp_headers.find("Retry-After:");
    if (ra_pos != std::string::npos) {
        auto val_start = resp_headers.find_first_not_of(" \t", ra_pos + 12);
        if (val_start != std::string::npos) {
            auto val_end = resp_headers.find_first_of("\r\n", val_start);
            size_t len = (val_end == std::string::npos)
                       ? std::string::npos : (val_end - val_start);
            resp.retry_after = resp_headers.substr(val_start, len);
        }
    }
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &resp.status_code);
    return resp;
}

static void extract_token_usage(const json& j) {
    g_last_token_usage = {};
    if (!j.contains("usage")) return;
    const auto& u = j["usage"];
    // OpenAI format
    if (u.contains("prompt_tokens")) {
        g_last_token_usage.input_tokens  = u.value("prompt_tokens", 0L);
        g_last_token_usage.output_tokens = u.value("completion_tokens", 0L);
        g_last_token_usage.total_tokens  = u.value("total_tokens", 0L);
    }
    // Anthropic format
    else if (u.contains("input_tokens")) {
        g_last_token_usage.input_tokens  = u.value("input_tokens", 0L);
        g_last_token_usage.output_tokens = u.value("output_tokens", 0L);
        g_last_token_usage.total_tokens  = g_last_token_usage.input_tokens + g_last_token_usage.output_tokens;
    }
}

// ════════════════════════════════════════════════════════════════
// SSE 解析 + Streaming 实现 (moved before providers for visibility)
// ════════════════════════════════════════════════════════════════

struct SseParser {
    StreamCallback callback;
    std::string    buffer;
    std::string    provider_type;   // "anthropic" | "openai" | "gemini"
    bool           done = false;

    void feed(const char* data, size_t len) {
        if (buffer.size() + len > 16 * 1024 * 1024) {
            log_msg(LogLevel::LOG_ERROR, "SSE",
                "buffer exceeds 16MB, aborting stream (possible malformed SSE)");
            done = true; buffer.clear(); return;
        }
        buffer.append(data, len);
        size_t pos;
        while ((pos = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, pos);
            buffer = buffer.substr(pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.size() < 6 || line.substr(0,6) != "data: ") continue;
            std::string payload = line.substr(6);
            if (payload == "[DONE]") { done = true; continue; }
            try {
                auto j = json::parse(payload);
                std::string chunk;
                if (provider_type == "anthropic") {
                    if (j.value("type","") == "content_block_delta")
                        chunk = j["delta"].value("text","");
                } else if (provider_type == "gemini") {
                    if (j.contains("candidates") && !j["candidates"].empty()) {
                        const auto& parts = j["candidates"][0]["content"]["parts"];
                        if (parts.is_array() && !parts.empty())
                            chunk = parts[0].value("text","");
                    }
                } else {
                    if (j.contains("choices") && !j["choices"].empty())
                        chunk = j["choices"][0]["delta"].value("content","");
                }
                if (!chunk.empty()) callback(chunk);
            } catch (...) {}
        }
    }
};

static size_t sse_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    static_cast<SseParser*>(userdata)->feed(ptr, size * nmemb);
    return size * nmemb;
}

static void do_stream_post(const std::string& url,
                             const std::vector<std::string>& hdrs,
                             const std::string& body,
                             SseParser& parser,
                             double timeout_sec) {
    CurlHandle curl;
    CURL* c = curl.get();
    struct curl_slist* h = nullptr;
    for (const auto& s : hdrs) h = curl_slist_append(h, s.c_str());
    curl_easy_setopt(c, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER,    h);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS,    body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, (long)body.size());
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, sse_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,     &parser);
    curl_easy_setopt(c, CURLOPT_TIMEOUT,       (long)timeout_sec);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL,      1L);
    CURLcode rc = curl_easy_perform(c);
    curl_slist_free_all(h);
    if (rc != CURLE_OK)
        throw ProviderError(std::string("curl stream: ") + curl_easy_strerror(rc));
}

// ── AnthropicProvider ────────────────────────────────────────────

std::string AnthropicProvider::complete(const std::string& prompt,
                                          const std::string& sys,
                                          double temp,
                                          bool force_json,
                                          const json& output_schema) const {
    json body = {{"model", cfg_.model}, {"max_tokens", cfg_.max_tokens},
                  {"temperature", temp},
                  {"messages", {{{"role","user"}, {"content", prompt}}}}};
    if (!sys.empty()) body["system"] = sys;
    // 结构化输出：GA output_config.format（替代旧 response_format）(D85)
    if (!output_schema.is_null() && !output_schema.empty())
        body["output_config"] = anthropic_output_config(output_schema);
    else if (force_json)
        body["response_format"] = {{"type", "json_object"}};
    std::string base = cfg_.base_url.empty() ? "https://api.anthropic.com" : cfg_.base_url;
    auto resp = http_post(base + "/v1/messages",
        {"Content-Type: application/json",
         "x-api-key: " + cfg_.api_key,
         "anthropic-version: 2024-10-22"},
        body.dump());
    if (resp.status_code == 429 || resp.status_code == 503) {
        std::string msg = "Anthropic: " + std::to_string(resp.status_code) + " rate limited";
        if (!resp.retry_after.empty()) msg += " retry_after=" + resp.retry_after;
        throw ProviderError(msg);
    }
    if (resp.status_code == 500 || resp.status_code == 502 || resp.status_code == 504)
        throw ProviderError("Anthropic: " + std::to_string(resp.status_code) + " server error (retryable)");
    json j;
    try { j = json::parse(resp.body); }
    catch (...) { throw ProviderError("Anthropic: invalid JSON (HTTP " + std::to_string(resp.status_code) + ")"); }
    if (j.contains("error"))
        throw ProviderError("Anthropic: " + j["error"].value("message", j["error"].dump()));
    extract_token_usage(j);
    if (!j.contains("content") || !j["content"].is_array() || j["content"].empty())
        throw ProviderError("Anthropic: empty content in response");
    return j["content"][0].value("text", "");
}

// ── AnthropicProvider::complete_chat (native tools) ─────────────

LLMResponse AnthropicProvider::complete_chat(const std::vector<ChatMessage>& messages,
                                               const std::vector<ToolDef>& tools,
                                               double temperature,
                                               const std::string& tool_choice,
                                               const json& output_schema) const {
    // Build messages array (Anthropic format: user/assistant alternating)
    json msgs = json::array();
    std::string system_prompt;
    for (const auto& m : messages) {
        if (m.role == "system") { system_prompt = m.content; continue; }
        if (m.role == "user") {
            // Check if this is a tool_result message
            if (!m.tool_call_id.empty()) {
                msgs.push_back({{"role","user"},{"content", json::array({
                    {{"type","tool_result"},{"tool_use_id",m.tool_call_id},{"content",m.content}}
                })}});
            } else if (m.is_multimodal()) {
                // Convert OpenAI-style image_url to Anthropic image format
                json parts = json::array();
                for (const auto& p : m.content_parts) {
                    if (p.value("type","") == "text") {
                        parts.push_back({{"type","text"},{"text",p.value("text","")}});
                    } else if (p.value("type","") == "image_url") {
                        std::string url = p["image_url"].value("url","");
                        if (url.find("data:") == 0) {
                            auto semi = url.find(';'); auto comma = url.find(',');
                            std::string mt = (semi != std::string::npos) ? url.substr(5, semi-5) : "image/jpeg";
                            std::string data = (comma != std::string::npos) ? url.substr(comma+1) : "";
                            parts.push_back({{"type","image"},{"source",
                                {{"type","base64"},{"media_type",mt},{"data",data}}}});
                        } else {
                            parts.push_back({{"type","image"},{"source",
                                {{"type","url"},{"url",url}}}});
                        }
                    }
                }
                msgs.push_back({{"role","user"},{"content",parts}});
            } else {
                msgs.push_back({{"role","user"},{"content",m.content}});
            }
        } else if (m.role == "tool") {
            // Anthropic: tool results go as user messages with tool_result content
            msgs.push_back({{"role","user"},{"content", json::array({
                {{"type","tool_result"},{"tool_use_id",m.tool_call_id},{"content",m.content}}
            })}});
        } else if (m.role == "assistant") {
            if (!m.tool_calls.is_null() && !m.tool_calls.empty()) {
                // Reconstruct assistant content with tool_use blocks
                json content = json::array();
                if (!m.content.empty())
                    content.push_back({{"type","text"},{"text",m.content}});
                for (const auto& tc : m.tool_calls) {
                    json args = tc.contains("arguments") ? tc["arguments"] : json::object();
                    if (args.is_string()) try { args = json::parse(args.get<std::string>()); } catch (...) {}
                    content.push_back({{"type","tool_use"},
                        {"id", tc.value("id","")},
                        {"name", tc.contains("function") ? tc["function"].value("name","") : tc.value("name","")},
                        {"input", args}});
                }
                msgs.push_back({{"role","assistant"},{"content",content}});
            } else {
                msgs.push_back({{"role","assistant"},{"content",m.content}});
            }
        }
    }

    json body = {{"model", cfg_.model}, {"max_tokens", cfg_.max_tokens},
                  {"temperature", temperature}, {"messages", msgs}};
    if (!system_prompt.empty()) body["system"] = system_prompt;

    // Add tools if provided (Anthropic format)
    if (!tools.empty()) {
        json tools_arr = json::array();
        for (const auto& t : tools) {
            json td = {
                {"name", t.name},
                {"description", t.description},
                {"input_schema", t.input_schema.empty() ? json({{"type","object"}}) : t.input_schema}
            };
            if (cfg_.strict_tools) td["strict"] = true;  // GA 严格工具校验 (D85)
            tools_arr.push_back(std::move(td));
        }
        body["tools"] = tools_arr;
        if (tool_choice == "none") body["tool_choice"] = {{"type","none"}};
        else if (tool_choice == "required") body["tool_choice"] = {{"type","any"}};
        else if (tool_choice != "auto" && !tool_choice.empty())
            body["tool_choice"] = {{"type","tool"},{"name",tool_choice}};
        else body["tool_choice"] = {{"type","auto"}};
    }

    // 结构化输出：GA output_config.format (D85)
    if (!output_schema.is_null() && !output_schema.empty())
        body["output_config"] = anthropic_output_config(output_schema);

    std::string base = cfg_.base_url.empty() ? "https://api.anthropic.com" : cfg_.base_url;
    auto resp = http_post(base + "/v1/messages",
        {"Content-Type: application/json",
         "x-api-key: " + cfg_.api_key,
         "anthropic-version: 2024-10-22"},
        body.dump());
    if (resp.status_code == 429 || resp.status_code == 503) {
        std::string msg = "Anthropic: " + std::to_string(resp.status_code) + " rate limited";
        if (!resp.retry_after.empty()) msg += " retry_after=" + resp.retry_after;
        throw ProviderError(msg);
    }
    if (resp.status_code == 500 || resp.status_code == 502 || resp.status_code == 504)
        throw ProviderError("Anthropic: " + std::to_string(resp.status_code) + " server error (retryable)");
    json j;
    try { j = json::parse(resp.body); }
    catch (...) { throw ProviderError("Anthropic: invalid JSON (HTTP " + std::to_string(resp.status_code) + ")"); }
    if (j.contains("error"))
        throw ProviderError("Anthropic: " + j["error"].value("message", j["error"].dump()));
    extract_token_usage(j);

    LLMResponse result;
    // Parse Anthropic response: content array with text and tool_use blocks
    if (j.contains("content") && j["content"].is_array()) {
        for (const auto& block : j["content"]) {
            if (block.value("type","") == "text")
                result.content += block.value("text","");
            else if (block.value("type","") == "tool_use") {
                LLMToolCall tc;
                tc.id = block.value("id","");
                tc.name = block.value("name","");
                tc.arguments = block.value("input", json::object());
                result.tool_calls.push_back(std::move(tc));
            }
        }
    }
    return result;
}

// ── OpenAIChatProvider ───────────────────────────────────────────

static std::string rtrim_slash(const std::string& s) {
    if (!s.empty() && s.back() == '/') return s.substr(0, s.size()-1);
    return s;
}

/** GPT-5 系列参数适配 + reasoning_effort/verbosity 注入 (D84)。
 *  GPT-5 reasoning 模型要求 max_completion_tokens 且只接受默认 temperature。 */
static void apply_openai_params(json& body, const ProviderConfig& cfg, double temp) {
    if (is_gpt5_family(cfg.model)) {
        body["max_completion_tokens"] = cfg.max_tokens;
    } else {
        body["max_tokens"]  = cfg.max_tokens;
        body["temperature"] = temp;
    }
    if (!cfg.reasoning_effort.empty()) body["reasoning_effort"] = cfg.reasoning_effort;
    if (!cfg.verbosity.empty())        body["verbosity"]        = cfg.verbosity;
}

std::string OpenAIChatProvider::complete(const std::string& prompt,
                                           const std::string& sys,
                                           double temp,
                                           bool force_json,
                                           const json& output_schema) const {
    json msgs = json::array();
    if (!sys.empty()) msgs.push_back({{"role","system"}, {"content", sys}});
    msgs.push_back({{"role","user"}, {"content", prompt}});
    json body = {{"model", cfg_.model}, {"messages", msgs}};
    apply_openai_params(body, cfg_, temp);  // GPT-5 token/temperature + reasoning/verbosity (D84)
    if (!output_schema.empty()) {
        body["response_format"] = {{"type","json_schema"},
            {"json_schema", {{"name","response"},{"strict",true},{"schema",output_schema}}}};
    } else if (force_json) {
        body["response_format"] = {{"type","json_object"}};
    }
    std::string base = cfg_.base_url.empty() ? "https://api.openai.com" : rtrim_slash(cfg_.base_url);
    std::string path = cfg_.completions_path.empty() ? "/v1/chat/completions" : cfg_.completions_path;
    auto resp = http_post(base + path,
        {"Content-Type: application/json",
         "Authorization: Bearer " + cfg_.api_key},
        body.dump());
    if (resp.status_code == 429 || resp.status_code == 503) {
        std::string msg = "OpenAI Chat: " + std::to_string(resp.status_code) + " rate limited";
        if (!resp.retry_after.empty()) msg += " retry_after=" + resp.retry_after;
        throw ProviderError(msg);
    }
    if (resp.status_code == 500 || resp.status_code == 502 || resp.status_code == 504)
        throw ProviderError("OpenAI Chat: " + std::to_string(resp.status_code) + " server error (retryable)");
    json j;
    try { j = json::parse(resp.body); }
    catch (...) { throw ProviderError("OpenAI Chat: invalid JSON (HTTP " + std::to_string(resp.status_code) + ")"); }
    if (j.contains("error"))
        throw ProviderError("OpenAI Chat: " + j["error"].value("message", j["error"].dump()));
    extract_token_usage(j);
    if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty())
        throw ProviderError("OpenAI Chat: empty choices in response");
    return j["choices"][0]["message"].value("content", "");
}

// ── OpenAIResponsesProvider ──────────────────────────────────────

std::string OpenAIResponsesProvider::complete(const std::string& prompt,
                                                const std::string& sys,
                                                double temp,
                                                bool /*force_json*/,
                                                const json& /*output_schema*/) const {
    json body = {{"model", cfg_.model}, {"max_output_tokens", cfg_.max_tokens},
                  {"temperature", temp},
                  {"input", {{{"role","user"}, {"content", prompt}}}}};
    if (!sys.empty()) body["instructions"] = sys;
    std::string base = cfg_.base_url.empty() ? "https://api.openai.com" : cfg_.base_url;
    auto resp = http_post(base + "/v1/responses",
        {"Content-Type: application/json",
         "Authorization: Bearer " + cfg_.api_key},
        body.dump());
    if (resp.status_code == 429 || resp.status_code == 503)
        throw ProviderError("OpenAI Responses: " + std::to_string(resp.status_code) + " rate limited");
    if (resp.status_code == 500 || resp.status_code == 502 || resp.status_code == 504)
        throw ProviderError("OpenAI Responses: " + std::to_string(resp.status_code) + " server error (retryable)");
    json j;
    try { j = json::parse(resp.body); }
    catch (...) { throw ProviderError("OpenAI Responses: invalid JSON (HTTP " + std::to_string(resp.status_code) + ")"); }
    if (j.contains("error"))
        throw ProviderError("OpenAI Responses: " + j["error"].value("message", j["error"].dump()));
    extract_token_usage(j);
    if (!j.contains("output") || !j["output"].is_array() || j["output"].empty())
        throw ProviderError("OpenAI Responses: empty output in response");
    const auto& out = j["output"][0];
    if (out.contains("content") && out["content"].is_array() && !out["content"].empty())
        return out["content"][0].value("text", "");
    return out.value("text", "");
}

// ════════════════════════════════════════════════════════════════
// InMemoryVectorStore — cosine similarity search
// ════════════════════════════════════════════════════════════════

static std::vector<float> normalize_vec(const std::vector<float>& v) {
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm < 1e-9f) return v;
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] / norm;
    return out;
}

void InMemoryVectorStore::add(const std::string& id, const std::vector<float>& embedding,
                                const json& metadata) {
    auto normed = normalize_vec(embedding);
    std::unique_lock<std::shared_mutex> lk(mu_);
    for (auto& e : entries_) {
        if (e.id == id) { e.embedding = normed; e.metadata = metadata; return; }
    }
    entries_.push_back({id, normed, metadata});
}

std::vector<VectorResult> InMemoryVectorStore::query(const std::vector<float>& embedding,
                                                       int top_k) const {
    auto q = normalize_vec(embedding);
    std::shared_lock<std::shared_mutex> lk(mu_);
    std::vector<VectorResult> scored;
    scored.reserve(entries_.size());
    for (const auto& e : entries_) {
        if (q.size() != e.embedding.size()) {
            scored.push_back({e.id, 0.0f, e.metadata});
            continue;
        }
        float dot = 0.0f;
        for (size_t i = 0; i < q.size(); ++i)
            dot += q[i] * e.embedding[i];
        scored.push_back({e.id, dot, e.metadata});
    }
    size_t k = std::min((size_t)top_k, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
        [](const VectorResult& a, const VectorResult& b) { return a.score > b.score; });
    scored.resize(k);
    return scored;
}

void InMemoryVectorStore::remove(const std::string& id) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
        [&](const VectorEntry& e) { return e.id == id; }), entries_.end());
}

size_t InMemoryVectorStore::size() const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    return entries_.size();
}

void InMemoryVectorStore::clear() {
    std::unique_lock<std::shared_mutex> lk(mu_);
    entries_.clear();
}

json InMemoryVectorStore::to_json() const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    json arr = json::array();
    for (const auto& e : entries_) {
        arr.push_back({{"id", e.id}, {"embedding", e.embedding}, {"metadata", e.metadata}});
    }
    return arr;
}

void InMemoryVectorStore::load_json(const json& j) {
    clear();
    if (!j.is_array()) return;
    for (const auto& item : j) {
        std::string id = item.value("id", "");
        auto emb = item.value("embedding", std::vector<float>{});
        json meta = item.value("metadata", json::object());
        add(id, emb, meta);
    }
}

// ── D90: memory scoping + temporal ───────────────────────────────

static long long unix_now_sec() {
    return (long long)std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void InMemoryVectorStore::add_scoped(const std::string& id, const std::vector<float>& embedding,
                                      const std::string& scope, long long ts, const json& metadata) {
    json meta = metadata.is_object() ? metadata : json::object();
    if (!scope.empty()) meta["scope"] = scope;
    meta["ts"] = (ts == 0) ? unix_now_sec() : ts;
    add(id, embedding, meta);
}

std::vector<VectorResult> InMemoryVectorStore::query(const std::vector<float>& embedding,
                                                       const MemoryQuery& opts) const {
    auto q = normalize_vec(embedding);
    long long now = opts.now_ts;
    if (opts.recency_half_life_sec > 0.0 && now == 0) now = unix_now_sec();

    std::shared_lock<std::shared_mutex> lk(mu_);
    std::vector<VectorResult> scored;
    scored.reserve(entries_.size());
    for (const auto& e : entries_) {
        // scope 分段安全过滤（避免 "user:alice" 越界匹配 "user:alice2"）
        if (!opts.scope_prefix.empty()) {
            std::string sc = e.metadata.is_object()
                           ? e.metadata.value("scope", std::string()) : std::string();
            if (!memory_scope_matches(sc, opts.scope_prefix)) continue;
        }
        float dot = 0.0f;
        if (q.size() == e.embedding.size())
            for (size_t i = 0; i < q.size(); ++i) dot += q[i] * e.embedding[i];
        double score = (double)dot;
        // recency 时间衰减：score *= 0.5^(age/half_life)
        if (opts.recency_half_life_sec > 0.0 && e.metadata.is_object() && e.metadata.contains("ts")) {
            long long ts = e.metadata.value("ts", (long long)0);
            double age = (double)(now - ts);
            if (age < 0.0) age = 0.0;
            score *= std::pow(0.5, age / opts.recency_half_life_sec);
        }
        scored.push_back({e.id, (float)score, e.metadata});
    }
    const size_t k = std::min((size_t)std::max(0, opts.top_k), scored.size());
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
        [](const VectorResult& a, const VectorResult& b) { return a.score > b.score; });
    scored.resize(k);
    return scored;
}

// ── ILLMProvider::complete_chat default ──────────────────────────

LLMResponse ILLMProvider::complete_chat(const std::vector<ChatMessage>& messages,
                                          const std::vector<ToolDef>& /*tools*/,
                                          double temperature,
                                          const std::string& /*tool_choice*/,
                                          const json& /*output_schema*/) const {
    // Default: concatenate messages into a single prompt, call complete()
    std::string system, prompt;
    for (const auto& m : messages) {
        if (m.role == "system") system = m.content;
        else if (m.role == "user") prompt += m.content + "\n";
        else if (m.role == "assistant") prompt += "[Assistant]: " + m.content + "\n";
        else if (m.role == "tool") prompt += "[Tool result]: " + m.content + "\n";
    }
    LLMResponse resp;
    resp.content = complete(prompt, system, temperature);
    return resp;
}

// ── OpenAIChatProvider::complete_chat (native tools) ────────────

LLMResponse OpenAIChatProvider::complete_chat(const std::vector<ChatMessage>& messages,
                                                const std::vector<ToolDef>& tools,
                                                double temperature,
                                                const std::string& tool_choice,
                                                const json& output_schema) const {
    // Build messages array
    json msgs = json::array();
    for (const auto& m : messages) {
        json msg = {{"role", m.role}};
        if (m.is_multimodal()) {
            msg["content"] = m.content_parts;
        } else if (!m.content.empty()) {
            msg["content"] = m.content;
        } else {
            msg["content"] = nullptr;
        }
        if (m.role == "assistant" && !m.tool_calls.is_null() && !m.tool_calls.empty())
            msg["tool_calls"] = m.tool_calls;
        if (m.role == "tool") {
            msg["tool_call_id"] = m.tool_call_id;
            if (!m.name.empty()) msg["name"] = m.name;
        }
        msgs.push_back(msg);
    }

    json body = {{"model", cfg_.model}, {"messages", msgs}};
    apply_openai_params(body, cfg_, temperature);  // GPT-5 token/temperature + reasoning/verbosity (D84)

    // Add tools if provided
    if (!tools.empty()) {
        json tools_array = json::array();
        for (const auto& t : tools) {
            json fn = {
                {"name", t.name},
                {"description", t.description},
                {"parameters", t.input_schema.empty() ? json({{"type","object"}}) : t.input_schema}
            };
            if (cfg_.strict_tools) fn["strict"] = true;  // 严格工具校验 (D84)
            tools_array.push_back({{"type","function"},{"function", fn}});
        }
        body["tools"] = tools_array;
        if (tool_choice == "none") body["tool_choice"] = "none";
        else if (tool_choice == "required") body["tool_choice"] = "required";
        else if (tool_choice != "auto" && !tool_choice.empty())
            body["tool_choice"] = {{"type","function"},{"function",{{"name",tool_choice}}}};
        else body["tool_choice"] = "auto";
    }

    // Structured output via response_format
    if (!output_schema.is_null() && !output_schema.empty())
        body["response_format"] = {{"type","json_schema"},
            {"json_schema",{{"name","response"},{"strict",true},{"schema",output_schema}}}};

    std::string base = cfg_.base_url.empty() ? "https://api.openai.com" : rtrim_slash(cfg_.base_url);
    std::string path = cfg_.completions_path.empty() ? "/v1/chat/completions" : cfg_.completions_path;
    auto resp = http_post(base + path,
        {"Content-Type: application/json",
         "Authorization: Bearer " + cfg_.api_key},
        body.dump());
    if (resp.status_code == 429 || resp.status_code == 503) {
        std::string msg = "OpenAI Chat: " + std::to_string(resp.status_code) + " rate limited";
        if (!resp.retry_after.empty()) msg += " retry_after=" + resp.retry_after;
        throw ProviderError(msg);
    }
    if (resp.status_code == 500 || resp.status_code == 502 || resp.status_code == 504)
        throw ProviderError("OpenAI Chat: " + std::to_string(resp.status_code) + " server error (retryable)");
    json j;
    try { j = json::parse(resp.body); }
    catch (...) { throw ProviderError("OpenAI Chat: invalid JSON in chat response"); }
    if (j.contains("error"))
        throw ProviderError("OpenAI Chat: " + j["error"].value("message", j["error"].dump()));
    extract_token_usage(j);

    LLMResponse result;
    if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty())
        throw ProviderError("OpenAI Chat: empty choices in chat response");
    const auto& choice = j["choices"][0]["message"];
    result.content = choice.value("content", "");

    // Parse native tool calls
    if (choice.contains("tool_calls") && !choice["tool_calls"].is_null()) {
        for (const auto& tc : choice["tool_calls"]) {
            LLMToolCall call;
            call.id   = tc.value("id", "");
            call.name = tc["function"].value("name", "");
            std::string args_str = tc["function"].value("arguments", "{}");
            try { call.arguments = json::parse(args_str); }
            catch (...) { call.arguments = json::object(); }
            result.tool_calls.push_back(std::move(call));
        }
    }
    return result;
}

// ── LLMClient::complete_chat ────────────────────────────────────

LLMResponse LLMClient::complete_chat(ModelTier tier,
                                       const std::vector<ChatMessage>& messages,
                                       const std::vector<ToolDef>& tools,
                                       double temperature,
                                       const std::string& tool_choice,
                                       const json& output_schema) const {
    // Token budget check (parity with try_slots)
    if (token_budget_ > 0 && cumulative_usage_.total_tokens >= token_budget_)
        throw TokenBudgetError("Token budget exceeded: " +
            std::to_string(cumulative_usage_.total_tokens) + "/" +
            std::to_string(token_budget_));

    const auto& slots = (tier == ModelTier::ORCHESTRATOR) ? orchestrators_ : subagents_;
    std::string last_error;
    std::vector<std::string> slot_errors;
    for (const auto& slot : slots) {
        std::string slot_name = slot.provider->provider_name() + "/" + slot.provider->model_name();
        if (!slot.breaker.try_allow()) {
            last_error = slot_name + " [OPEN]";
            slot_errors.push_back(last_error);
            continue;
        }
        { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.calls; }
        if (slot.rate_limiter.enabled())
            if (!slot.rate_limiter.acquire(10000)) {
                last_error = slot_name + " [RATE_LIMIT_TIMEOUT]";
                slot_errors.push_back(last_error);
                continue;
            }
        // 429/503 retry with backoff (parity with try_slots)
        constexpr int MAX_RATE_RETRIES = 3;
        for (int rate_retry = 0; rate_retry <= MAX_RATE_RETRIES; ++rate_retry) {
            try {
                auto call_t0 = std::chrono::steady_clock::now();
                auto resp = slot.provider->complete_chat(messages, tools, temperature, tool_choice, output_schema);
                auto call_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - call_t0).count();
                slot.breaker.on_success();
                {
                    std::lock_guard<std::mutex> lk(*slot.stats_mu);
                    ++slot.successes;
                    slot.total_latency_ms += call_ms;
                    slot.last_latency_ms = call_ms;
                }
                { std::lock_guard<std::mutex> lk(usage_mu_); cumulative_usage_ += g_last_token_usage; }
                {   // OTel GenAI span (D87)
                    GenAiSpan span;
                    span.provider_name = slot.provider->provider_name();
                    span.request_model = slot.provider->model_name();
                    span.input_tokens  = g_last_token_usage.input_tokens;
                    span.output_tokens = g_last_token_usage.output_tokens;
                    span.duration_ms   = call_ms;
                    span.finish_reason = resp.has_tool_calls() ? "tool_calls" : "stop";
                    emit_span(span);
                }
                return resp;
            } catch (const std::exception& e) {
                std::string emsg = e.what();
                bool is_rate_limit = emsg.find("429") != std::string::npos
                                  || emsg.find("503") != std::string::npos;
                bool is_fatal = emsg.find("401") != std::string::npos
                             || emsg.find("403") != std::string::npos
                             || emsg.find("invalid_api_key") != std::string::npos;
                if (is_fatal) {
                    { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.failures; }
                    throw AllProvidersExhaustedError(
                        slot_name + ": fatal error — " + emsg);
                } else if (is_rate_limit && rate_retry < MAX_RATE_RETRIES) {
                    int backoff_sec = 3 * (1 << rate_retry);
                    auto ra_pos = emsg.find("retry_after=");
                    if (ra_pos != std::string::npos) {
                        try { backoff_sec = std::stoi(emsg.substr(ra_pos + 12)); }
                        catch (...) {}
                    }
                    if (backoff_sec > 120) backoff_sec = 120;
                    log_msg(LogLevel::LOG_WARN, "LLMClient",
                        slot_name + " chat rate limited, retry "
                        + std::to_string(rate_retry+1) + "/" + std::to_string(MAX_RATE_RETRIES));
                    std::this_thread::sleep_for(std::chrono::seconds(backoff_sec));
                    continue;
                } else {
                    if (is_rate_limit) {
                        last_error = slot_name + " [RATE_LIMITED, exhausted retries]";
                    } else {
                        slot.breaker.on_failure();
                        last_error = slot_name + ": " + emsg;
                    }
                    { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.failures; }
                    slot_errors.push_back(last_error);
                    break;
                }
            }
        }
    }
    std::string tier_name = (tier == ModelTier::ORCHESTRATOR) ? "ORCHESTRATOR" : "SUBAGENT";
    std::string detail = tier_name + " chat providers exhausted (" +
        std::to_string(slot_errors.size()) + " slots tried):";
    for (size_t i = 0; i < slot_errors.size(); ++i)
        detail += " [" + std::to_string(i+1) + "] " + slot_errors[i] + ";";
    throw AllProvidersExhaustedError(detail);
}

bool LLMClient::supports_native_tools(ModelTier tier) const {
    const auto& slots = (tier == ModelTier::ORCHESTRATOR) ? orchestrators_ : subagents_;
    return !slots.empty() && slots[0].provider->supports_native_tools();
}

// ── GeminiProvider ───────────────────────────────────────────────

std::string GeminiProvider::complete(const std::string& prompt,
                                      const std::string& sys,
                                      double temp,
                                      bool force_json,
                                      const json& output_schema) const {
    json contents = json::array();
    contents.push_back({{"role","user"},{"parts",{{{"text", prompt}}}}});
    json body = {{"contents", contents}};
    if (!sys.empty())
        body["systemInstruction"] = {{"parts",{{{"text", sys}}}}};
    json gen_config = {{"temperature", temp}, {"maxOutputTokens", cfg_.max_tokens}};
    { auto tl = gemini_thinking_level(cfg_.reasoning_effort);
      if (!tl.empty()) gen_config["thinking_level"] = tl; }  // Gemini 3 推理深度 (D83)
    if (!output_schema.is_null() && !output_schema.empty()) {
        gen_config["responseMimeType"] = "application/json";
        gen_config["responseSchema"] = output_schema;
    } else if (force_json) gen_config["responseMimeType"] = "application/json";
    body["generationConfig"] = gen_config;

    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/"
                    + cfg_.model + ":generateContent";
    auto resp = http_post(url,
        {"Content-Type: application/json",
         "x-goog-api-key: " + cfg_.api_key},
        body.dump());
    if (resp.status_code == 429 || resp.status_code == 503) {
        std::string msg = "Gemini: " + std::to_string(resp.status_code) + " rate limited";
        if (!resp.retry_after.empty()) msg += " retry_after=" + resp.retry_after;
        throw ProviderError(msg);
    }
    if (resp.status_code == 500 || resp.status_code == 502 || resp.status_code == 504)
        throw ProviderError("Gemini: " + std::to_string(resp.status_code) + " server error (retryable)");
    json j;
    try { j = json::parse(resp.body); }
    catch (...) { throw ProviderError("Gemini: invalid JSON (HTTP " + std::to_string(resp.status_code) + ")"); }
    if (j.contains("error"))
        throw ProviderError("Gemini: " + j["error"].value("message", "unknown error"));
    // Parse Gemini token usage
    if (j.contains("usageMetadata")) {
        const auto& u = j["usageMetadata"];
        g_last_token_usage.input_tokens  = u.value("promptTokenCount", 0L);
        g_last_token_usage.output_tokens = u.value("candidatesTokenCount", 0L);
        g_last_token_usage.total_tokens  = u.value("totalTokenCount", 0L);
    }
    if (!j.contains("candidates") || !j["candidates"].is_array() || j["candidates"].empty())
        throw ProviderError("Gemini: empty candidates in response");
    const auto& parts = j["candidates"][0]["content"]["parts"];
    if (!parts.is_array() || parts.empty())
        throw ProviderError("Gemini: empty parts in response");
    return parts[0].value("text", "");
}

LLMResponse GeminiProvider::complete_chat(const std::vector<ChatMessage>& messages,
                                           const std::vector<ToolDef>& tools,
                                           double temperature,
                                           const std::string& tool_choice,
                                           const json& output_schema) const {
    json contents = json::array();
    std::string system_instruction;
    for (const auto& m : messages) {
        if (m.role == "system") { system_instruction = m.content; continue; }
        std::string role = (m.role == "assistant") ? "model" : "user";
        if (m.role == "tool") role = "user";  // Gemini: tool results as user
        json parts = json::array();
        if (m.is_multimodal()) {
            for (const auto& p : m.content_parts) {
                if (p.value("type","") == "text")
                    parts.push_back({{"text", p.value("text","")}});
                else if (p.value("type","") == "image_url") {
                    std::string url = p["image_url"].value("url","");
                    if (url.find("data:") == 0) {
                        auto comma = url.find(',');
                        auto semi = url.find(';');
                        std::string mt = (semi != std::string::npos) ? url.substr(5,semi-5) : "image/jpeg";
                        std::string data = (comma != std::string::npos) ? url.substr(comma+1) : "";
                        parts.push_back({{"inline_data",{{"mime_type",mt},{"data",data}}}});
                    }
                }
            }
        } else if (m.role == "tool" || !m.tool_call_id.empty()) {
            parts.push_back({{"functionResponse",{
                {"name", m.name.empty() ? "tool" : m.name},
                {"response", {{"result", m.content}}}
            }}});
        } else if (!m.content.empty()) {
            parts.push_back({{"text", m.content}});
        }
        // Gemini: assistant tool_calls → functionCall parts
        if (m.role == "assistant" && !m.tool_calls.is_null()) {
            for (const auto& tc : m.tool_calls) {
                json args = tc.contains("arguments") ? tc["arguments"] : json::object();
                if (args.is_string()) try { args = json::parse(args.get<std::string>()); } catch (...) {}
                std::string fn_name = tc.contains("function")
                    ? tc["function"].value("name","") : tc.value("name","");
                parts.push_back({{"functionCall",{{"name",fn_name},{"args",args}}}});
            }
        }
        if (!parts.empty())
            contents.push_back({{"role",role},{"parts",parts}});
    }

    json body = {{"contents", contents}};
    json gen_config = {{"temperature", temperature}, {"maxOutputTokens", cfg_.max_tokens}};
    { auto tl = gemini_thinking_level(cfg_.reasoning_effort);
      if (!tl.empty()) gen_config["thinking_level"] = tl; }  // Gemini 3 推理深度 (D83)
    // Structured output via responseSchema
    if (!output_schema.is_null() && !output_schema.empty()) {
        gen_config["responseMimeType"] = "application/json";
        gen_config["responseSchema"] = output_schema;
    }
    body["generationConfig"] = gen_config;
    if (!system_instruction.empty())
        body["systemInstruction"] = {{"parts",{{{"text",system_instruction}}}}};

    if (!tools.empty()) {
        json fn_decls = json::array();
        for (const auto& t : tools)
            fn_decls.push_back({{"name",t.name},{"description",t.description},
                {"parameters", t.input_schema.empty() ? json({{"type","object"}}) : t.input_schema}});
        body["tools"] = json::array({{{"functionDeclarations", fn_decls}}});
        if (tool_choice == "none") body["tool_config"] = {{"function_calling_config",{{"mode","NONE"}}}};
        else if (tool_choice == "required") body["tool_config"] = {{"function_calling_config",{{"mode","ANY"}}}};
        else if (tool_choice != "auto" && !tool_choice.empty())
            body["tool_config"] = {{"function_calling_config",{{"mode","ANY"},{"allowed_function_names",json::array({tool_choice})}}}};
        else body["tool_config"] = {{"function_calling_config",{{"mode","AUTO"}}}};
    }

    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/"
                    + cfg_.model + ":generateContent";
    auto resp = http_post(url,
        {"Content-Type: application/json", "x-goog-api-key: " + cfg_.api_key},
        body.dump());
    if (resp.status_code == 429 || resp.status_code == 503) {
        std::string msg = "Gemini: " + std::to_string(resp.status_code) + " rate limited";
        if (!resp.retry_after.empty()) msg += " retry_after=" + resp.retry_after;
        throw ProviderError(msg);
    }
    if (resp.status_code >= 500)
        throw ProviderError("Gemini: " + std::to_string(resp.status_code) + " server error");
    json j;
    try { j = json::parse(resp.body); }
    catch (...) { throw ProviderError("Gemini: invalid JSON"); }
    if (j.contains("error"))
        throw ProviderError("Gemini: " + j["error"].value("message","unknown"));
    if (j.contains("usageMetadata")) {
        const auto& u = j["usageMetadata"];
        g_last_token_usage.input_tokens  = u.value("promptTokenCount", 0L);
        g_last_token_usage.output_tokens = u.value("candidatesTokenCount", 0L);
        g_last_token_usage.total_tokens  = u.value("totalTokenCount", 0L);
    }

    LLMResponse result;
    if (j.contains("candidates") && j["candidates"].is_array() && !j["candidates"].empty()
        && j["candidates"][0].contains("content")
        && j["candidates"][0]["content"].contains("parts")) {
        const auto& parts = j["candidates"][0]["content"]["parts"];
        for (const auto& p : parts) {
            if (p.contains("text")) result.content += p["text"].get<std::string>();
            if (p.contains("functionCall")) {
                LLMToolCall tc;
                tc.name = p["functionCall"].value("name","");
                tc.arguments = p["functionCall"].value("args", json::object());
                tc.id = "gemini_" + tc.name;
                result.tool_calls.push_back(std::move(tc));
            }
        }
    }
    return result;
}

void GeminiProvider::complete_stream(const std::string& prompt,
                                      const std::string& sys,
                                      double temp,
                                      StreamCallback on_chunk) const {
    json contents = json::array();
    contents.push_back({{"role","user"},{"parts",{{{"text", prompt}}}}});
    json body = {{"contents", contents}};
    if (!sys.empty())
        body["systemInstruction"] = {{"parts",{{{"text", sys}}}}};
    json gen_config = {{"temperature", temp}, {"maxOutputTokens", cfg_.max_tokens}};
    { auto tl = gemini_thinking_level(cfg_.reasoning_effort);
      if (!tl.empty()) gen_config["thinking_level"] = tl; }  // Gemini 3 推理深度 (D83)
    body["generationConfig"] = gen_config;

    std::string url = "https://generativelanguage.googleapis.com/v1beta/models/"
                    + cfg_.model + ":streamGenerateContent?alt=sse";
    SseParser parser{on_chunk, "", "gemini"};
    do_stream_post(url,
        {"Content-Type: application/json",
         "x-goog-api-key: " + cfg_.api_key},
        body.dump(), parser, cfg_.timeout_sec);
}

// ── make_provider ────────────────────────────────────────────────

std::unique_ptr<ILLMProvider> make_provider(const ProviderConfig& cfg) {
    switch (cfg.type) {
    case ProviderType::ANTHROPIC:        return std::make_unique<AnthropicProvider>(cfg);
    case ProviderType::OPENAI_CHAT:      return std::make_unique<OpenAIChatProvider>(cfg);
    case ProviderType::OPENAI_RESPONSES: return std::make_unique<OpenAIResponsesProvider>(cfg);
    case ProviderType::GEMINI:           return std::make_unique<GeminiProvider>(cfg);
    default: throw std::invalid_argument("Unknown ProviderType");
    }
}

// ════════════════════════════════════════════════════════════════
// CircuitBreaker
// ════════════════════════════════════════════════════════════════

bool CircuitBreaker::try_allow() {
    std::lock_guard<std::mutex> lk(*mu_);
    switch (state_) {
    case State::CLOSED: return true;
    case State::OPEN: {
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_open_).count();
        if (elapsed >= recovery_sec_) { state_ = State::HALF_OPEN; half_open_in_flight_ = true; return true; }
        return false;
    }
    case State::HALF_OPEN: return !half_open_in_flight_;
    }
    return false;
}
void CircuitBreaker::on_success() {
    std::lock_guard<std::mutex> lk(*mu_);
    state_ = State::CLOSED; failures_ = 0; half_open_in_flight_ = false;
}
void CircuitBreaker::on_failure() {
    std::lock_guard<std::mutex> lk(*mu_);
    ++failures_; half_open_in_flight_ = false;
    if (state_ == State::HALF_OPEN || failures_ >= threshold_)
        { state_ = State::OPEN; last_open_ = std::chrono::steady_clock::now(); }
}
CircuitBreaker::State CircuitBreaker::state() const {
    std::lock_guard<std::mutex> lk(*mu_); return state_;
}
double CircuitBreaker::seconds_until_retry() const {
    std::lock_guard<std::mutex> lk(*mu_);
    if (state_ != State::OPEN) return 0.0;
    double e = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_open_).count();
    return std::max(0.0, recovery_sec_ - e);
}

// ════════════════════════════════════════════════════════════════
// LLMClient
// ════════════════════════════════════════════════════════════════

std::vector<LLMClient::Slot> LLMClient::build_slots(const TierConfig& cfg) {
    std::vector<Slot> slots;
    auto make_slot = [&](const ProviderConfig& pc) {
        Slot s;
        s.provider     = make_provider(pc);
        s.breaker      = CircuitBreaker(cfg.failure_threshold, cfg.recovery_timeout_sec);
        s.rate_limiter = RateLimiter(pc.max_rps);
        return s;
    };
    slots.push_back(make_slot(cfg.primary));
    for (const auto& fb : cfg.fallbacks) slots.push_back(make_slot(fb));
    return slots;
}

LLMClient::LLMClient(TierConfig orc, TierConfig sub)
    : orchestrators_(build_slots(orc)), subagents_(build_slots(sub)) {}

LLMClient::LLMClient(std::unique_ptr<ILLMProvider> orc_prov,
                       std::unique_ptr<ILLMProvider> sub_prov) {
    Slot os; os.provider = std::move(orc_prov);
    orchestrators_.push_back(std::move(os));
    Slot ss; ss.provider = std::move(sub_prov);
    subagents_.push_back(std::move(ss));
}

std::string LLMClient::try_slots(const std::vector<Slot>& slots,
                                   const std::string& prompt,
                                   const std::string& system,
                                   double temperature,
                                   ModelTier tier,
                                   bool force_json,
                                   const json& output_schema) const {
    // 响应缓存查找 (temperature=0 时最有效)
    std::string cache_key;
    if (resp_cache_enabled_ && response_cache_ && !slots.empty()) {
        cache_key = ResponseCache::make_key(
            slots[0].provider->model_name(), system, prompt, temperature, force_json);
        if (response_cache_->has(cache_key)) {
            auto cached = response_cache_->get(cache_key);
            if (!cached.empty()) return cached;
        }
    }
    // Token budget check before making LLM call
    if (token_budget_ > 0 && cumulative_usage_.total_tokens >= token_budget_)
        throw TokenBudgetError("Token budget exceeded: " +
            std::to_string(cumulative_usage_.total_tokens) + "/" +
            std::to_string(token_budget_));

    // Hedged requests: race 2 providers, return first success
    if (hedging_enabled_ && slots.size() >= 2) {
        std::atomic<bool> got_result{false};
        std::string results[2];
        std::string errors[2];
        std::future<void> futs[2];
        for (int h = 0; h < 2; ++h) {
            const auto& slot = slots[h];
            if (!slot.breaker.try_allow()) { errors[h] = "circuit open"; continue; }
            futs[h] = std::async(std::launch::async, [&, h]() {
                try {
                    auto call_t0 = std::chrono::steady_clock::now();
                    auto res = slots[h].provider->complete(prompt, system, temperature, force_json, output_schema);
                    auto call_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - call_t0).count();
                    results[h] = res;
                    got_result = true;
                    slots[h].breaker.on_success();
                    { std::lock_guard<std::mutex> lk(*slots[h].stats_mu);
                      ++slots[h].successes; slots[h].total_latency_ms += call_ms;
                      slots[h].last_latency_ms = call_ms; }
                } catch (const std::exception& e) {
                    errors[h] = e.what();
                    slots[h].breaker.on_failure();
                    { std::lock_guard<std::mutex> lk(*slots[h].stats_mu); ++slots[h].failures; }
                }
            });
        }
        for (int h = 0; h < 2; ++h)
            if (futs[h].valid()) futs[h].wait();
        // Return first successful result (prefer lower-index slot)
        for (int h = 0; h < 2; ++h) {
            if (!results[h].empty()) {
                { std::lock_guard<std::mutex> lk(usage_mu_); cumulative_usage_ += g_last_token_usage; }
                if (resp_cache_enabled_ && response_cache_ && !cache_key.empty())
                    response_cache_->put(cache_key, results[h]);
                return results[h];
            }
        }
        throw AllProvidersExhaustedError("Hedged requests failed: [1] " + errors[0] + "; [2] " + errors[1]);
    }

    std::string last_error;
    std::vector<std::string> slot_errors;
    for (const auto& slot : slots) {
        std::string slot_name = slot.provider->provider_name() + "/" + slot.provider->model_name();
        if (!slot.breaker.try_allow()) {
            last_error = slot_name + " [OPEN, " + std::to_string((int)slot.breaker.seconds_until_retry()) + "s]";
            slot_errors.push_back(last_error);
            continue;
        }
        { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.calls; }
        if (slot.rate_limiter.enabled())
            if (!slot.rate_limiter.acquire(10000)) {
                last_error = slot_name + " [RATE_LIMIT_TIMEOUT]";
                slot_errors.push_back(last_error);
                continue;
            }
        // 429/503: retry same slot with exponential backoff (up to 3 times)
        constexpr int MAX_RATE_RETRIES = 3;
        for (int rate_retry = 0; rate_retry <= MAX_RATE_RETRIES; ++rate_retry) {
            try {
                auto call_t0 = std::chrono::steady_clock::now();
                auto res = slot.provider->complete(prompt, system, temperature, force_json, output_schema);
                auto call_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - call_t0).count();
                slot.breaker.on_success();
                {
                    std::lock_guard<std::mutex> lk(*slot.stats_mu);
                    ++slot.successes;
                    slot.total_latency_ms += call_ms;
                    slot.last_latency_ms = call_ms;
                }
                {
                    std::lock_guard<std::mutex> lk(usage_mu_);
                    cumulative_usage_ += g_last_token_usage;
                }
                {   // OTel GenAI span (D87)
                    GenAiSpan span;
                    span.provider_name = slot.provider->provider_name();
                    span.request_model = slot.provider->model_name();
                    span.input_tokens  = g_last_token_usage.input_tokens;
                    span.output_tokens = g_last_token_usage.output_tokens;
                    span.duration_ms   = call_ms;
                    emit_span(span);
                }
                if (resp_cache_enabled_ && response_cache_ && !cache_key.empty())
                    response_cache_->put(cache_key, res);
                return res;
            } catch (const std::exception& e) {
                std::string emsg = e.what();
                bool is_rate_limit = emsg.find("429") != std::string::npos
                                  || emsg.find("503") != std::string::npos;
                bool is_fatal = emsg.find("401") != std::string::npos
                             || emsg.find("403") != std::string::npos
                             || emsg.find("invalid_api_key") != std::string::npos;

                if (is_fatal) {
                    { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.failures; }
                    throw AllProvidersExhaustedError(
                        slot_name + ": fatal error — " + emsg);
                } else if (is_rate_limit && rate_retry < MAX_RATE_RETRIES) {
                    int backoff_sec = 3 * (1 << rate_retry);  // default: 3s, 6s, 12s
                    auto ra_pos = emsg.find("retry_after=");
                    if (ra_pos != std::string::npos) {
                        try { backoff_sec = std::stoi(emsg.substr(ra_pos + 12)); }
                        catch (...) {}
                    }
                    if (backoff_sec > 120) backoff_sec = 120;
                    log_msg(LogLevel::LOG_WARN, "LLMClient",
                        slot_name + " rate limited, retry "
                        + std::to_string(rate_retry+1) + "/" + std::to_string(MAX_RATE_RETRIES)
                        + ", backoff " + std::to_string(backoff_sec) + "s");
                    std::this_thread::sleep_for(std::chrono::seconds(backoff_sec));
                    continue;
                } else {
                    if (is_rate_limit) {
                        last_error = slot_name + " [RATE_LIMITED, exhausted retries]";
                    } else {
                        slot.breaker.on_failure();
                        last_error = slot_name + ": " + emsg;
                    }
                    { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.failures; }
                    slot_errors.push_back(last_error);
                    break;
                }
            }
        }
    }
    std::string tier_name = (tier == ModelTier::ORCHESTRATOR) ? "ORCHESTRATOR" : "SUBAGENT";
    std::string detail = tier_name + " providers exhausted (" +
        std::to_string(slot_errors.size()) + " slots tried):";
    for (size_t i = 0; i < slot_errors.size(); ++i)
        detail += " [" + std::to_string(i+1) + "] " + slot_errors[i] + ";";
    throw AllProvidersExhaustedError(detail);
}

std::string LLMClient::complete_as(ModelTier tier, const std::string& prompt,
                                     const std::string& system, double temperature,
                                     bool force_json, const json& output_schema) const {
    const auto& slots = (tier == ModelTier::ORCHESTRATOR) ? orchestrators_ : subagents_;
    return try_slots(slots, prompt, system, temperature, tier, force_json, output_schema);
}

std::vector<ProviderStats> LLMClient::stats(ModelTier tier) const {
    const auto& slots = (tier == ModelTier::ORCHESTRATOR) ? orchestrators_ : subagents_;
    std::vector<ProviderStats> out;
    for (const auto& s : slots) {
        ProviderStats ps;
        ps.provider_name = s.provider->provider_name();
        ps.model_name    = s.provider->model_name();
        { std::lock_guard<std::mutex> lk(*s.stats_mu);
          ps.total_calls = s.calls; ps.successes = s.successes; ps.failures = s.failures;
          ps.avg_latency_ms = s.successes > 0 ? s.total_latency_ms / s.successes : 0;
          ps.last_latency_ms = s.last_latency_ms; }
        ps.circuit_state = s.breaker.state();
        ps.secs_to_retry = s.breaker.seconds_until_retry();
        out.push_back(ps);
    }
    return out;
}

TokenUsage LLMClient::total_usage() const {
    std::lock_guard<std::mutex> lk(usage_mu_);
    return cumulative_usage_;
}
void LLMClient::reset_usage() {
    std::lock_guard<std::mutex> lk(usage_mu_);
    cumulative_usage_ = {};
}

void LLMClient::enable_response_cache(bool on) {
    resp_cache_enabled_ = on;
    if (on && !response_cache_) response_cache_ = std::make_unique<ResponseCache>();
}
long LLMClient::response_cache_hits() const {
    return response_cache_ ? response_cache_->stats().hits : 0;
}

void LLMClient::print_status() const {
    auto print_tier = [](const char* name, const std::vector<ProviderStats>& sv) {
        log_msg(LogLevel::LOG_INFO, "LLMClient", std::string("── ") + name + " ──");
        for (const auto& s : sv) {
            const char* cs = s.circuit_state == CircuitBreaker::State::CLOSED ? "CLOSED" :
                             s.circuit_state == CircuitBreaker::State::OPEN   ? "OPEN" : "HALF_OPEN";
            std::string line = "[" + std::string(cs) + "] " + s.provider_name + "/" + s.model_name
                + " calls=" + std::to_string(s.total_calls) + " ok=" + std::to_string(s.successes)
                + " fail=" + std::to_string(s.failures);
            if (s.circuit_state == CircuitBreaker::State::OPEN)
                line += " (retry in " + std::to_string((int)s.secs_to_retry) + "s)";
            log_msg(LogLevel::LOG_INFO, "LLMClient", line);
        }
    };
    print_tier("ORCHESTRATOR", stats(ModelTier::ORCHESTRATOR));
    print_tier("SUBAGENT",     stats(ModelTier::SUBAGENT));
}

// ════════════════════════════════════════════════════════════════
// ToolRegistry
// ════════════════════════════════════════════════════════════════


// ════════════════════════════════════════════════════════════════
// validate_json_schema — 轻量 JSON Schema 验证器
// ════════════════════════════════════════════════════════════════

static std::string json_type_name(const json& v) {
    if (v.is_null())             return "null";
    if (v.is_boolean())          return "boolean";
    if (v.is_number_integer())   return "integer";
    if (v.is_number())           return "number";
    if (v.is_string())           return "string";
    if (v.is_array())            return "array";
    if (v.is_object())           return "object";
    return "unknown";
}

std::vector<SchemaViolation> validate_json_schema(const json& value,
                                                    const json& schema,
                                                    const std::string& path) {
    std::vector<SchemaViolation> out;
    if (!schema.is_object()) return out;

    // type check
    if (schema.contains("type") && schema["type"].is_string()) {
        const std::string exp = schema["type"].get<std::string>();
        const std::string act = json_type_name(value);
        bool ok = (exp == act) || (exp == "number" && act == "integer");
        if (!ok) out.push_back({path.empty()?"(root)":path,
                                 "expected type '" + exp + "', got '" + act + "'"});
    }

    // required
    if (schema.contains("required") && schema["required"].is_array() && value.is_object()) {
        for (const auto& req : schema["required"]) {
            if (!req.is_string()) continue;
            const std::string field = req.get<std::string>();
            if (!value.contains(field))
                out.push_back({(path.empty()?"":path) + "." + field,
                                "required property missing"});
        }
    }

    // enum
    if (schema.contains("enum") && schema["enum"].is_array()) {
        bool found = false;
        for (const auto& e : schema["enum"]) if (e == value) { found = true; break; }
        if (!found) out.push_back({path.empty()?"(root)":path, "value not in enum"});
    }

    // properties (recursive)
    if (schema.contains("properties") && schema["properties"].is_object() && value.is_object()) {
        for (const auto& [prop, pschema] : schema["properties"].items()) {
            if (!value.contains(prop)) continue;
            auto child = validate_json_schema(value[prop], pschema,
                                               (path.empty()?"":path) + "." + prop);
            out.insert(out.end(), child.begin(), child.end());
        }
    }

    // items (recursive, array)
    if (schema.contains("items") && value.is_array()) {
        for (size_t i = 0; i < value.size(); ++i) {
            auto child = validate_json_schema(value[i], schema["items"],
                                               path + "[" + std::to_string(i) + "]");
            out.insert(out.end(), child.begin(), child.end());
        }
    }

    return out;
}

void ToolRegistry::register_tool(const ToolDef& def, ToolFn fn) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    defs_[def.name] = def; fns_[def.name] = std::move(fn);
}
json ToolRegistry::call(const std::string& name, const json& params) const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    auto fit = fns_.find(name);
    if (fit == fns_.end()) throw ToolNotFoundError(name, "not registered");

    auto dit = defs_.find(name);
    if (dit != defs_.end()) {
        const auto& schema = dit->second.input_schema;
        if (schema.is_object() && schema.contains("required")) {
            for (const auto& req : schema["required"]) {
                std::string field = req.get<std::string>();
                if (!params.contains(field))
                    throw ToolInputError(name, "missing required field '" + field + "'");
            }
        }
    }

    auto git = guardrails_.find(name);
    if (git != guardrails_.end()) {
        for (const auto& guard : git->second) {
            auto err = guard(params);
            if (err) throw GuardrailError("Tool '" + name + "' guardrail: " + *err);
        }
    }

    if (!fit->second) return nullptr;
    auto fn_copy = fit->second;
    lk.unlock();
    return fn_copy(params);
}
std::vector<ToolDef> ToolRegistry::list_tools() const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    std::vector<ToolDef> out;
    for (const auto& [_, d] : defs_) out.push_back(d);
    return out;
}
bool ToolRegistry::has_tool(const std::string& n) const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    return fns_.count(n) > 0;
}

void ToolRegistry::add_guardrail(const std::string& tool_name, GuardrailFn fn) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    guardrails_[tool_name].push_back(std::move(fn));
}

// ════════════════════════════════════════════════════════════════
// WorkflowPlan
// ════════════════════════════════════════════════════════════════

std::vector<std::vector<Step>> WorkflowPlan::topological_batches() const {
    std::map<std::string,int> deg;
    std::map<std::string,std::vector<std::string>> adj;
    for (const auto& s : steps) {
        if (!deg.count(s.id)) deg[s.id] = 0;
        for (const auto& d : s.depends_on) { adj[d].push_back(s.id); deg[s.id]++; }
    }
    std::map<std::string, const Step*> sm;
    for (const auto& s : steps) sm[s.id] = &s;
    std::vector<std::vector<Step>> batches;
    while (true) {
        std::vector<Step> batch;
        for (auto& [id, d] : deg) if (d == 0) batch.push_back(*sm[id]);
        if (batch.empty()) break;
        batches.push_back(batch);
        for (const auto& s : batch) { deg.erase(s.id); for (const auto& d : adj[s.id]) deg[d]--; }
    }
    return batches;
}
std::vector<Step> WorkflowPlan::leaf_steps() const {
    std::set<std::string> ref;
    for (const auto& s : steps) for (const auto& d : s.depends_on) ref.insert(d);
    std::vector<Step> out;
    for (const auto& s : steps) if (!ref.count(s.id)) out.push_back(s);
    return out;
}

json WorkflowPlan::to_json() const {
    json j;
    j["id"] = id;
    j["metadata"] = metadata;
    j["steps"] = json::array();
    for (const auto& s : steps) {
        json sj;
        sj["id"] = s.id;
        sj["type"] = (s.type == StepType::TOOL) ? "tool" :
                     (s.type == StepType::LLM) ? "llm" :
                     (s.type == StepType::TRANSFORM) ? "transform" : "condition";
        sj["action"] = s.action;
        sj["inputs"] = s.inputs;
        sj["depends_on"] = s.depends_on;
        sj["retry"] = s.retry;
        sj["timeout_sec"] = s.timeout_sec;
        sj["on_error"] = (s.on_error == OnError::SKIP) ? "skip" :
                         (s.on_error == OnError::FALLBACK) ? "fallback" : "fail";
        sj["description"] = s.description;
        sj["model_tier"] = (s.model_tier == ModelTier::ORCHESTRATOR) ? "orchestrator" : "subagent";
        if (!s.system_prompt.empty()) sj["system_prompt"] = s.system_prompt;
        if (s.json_mode) sj["json_mode"] = true;
        if (!s.output_schema.is_null() && !s.output_schema.empty()) sj["output_schema"] = s.output_schema;
        if (s.temperature >= 0.0) sj["temperature"] = s.temperature;
        if (!s.fallback.is_null()) sj["fallback_value"] = s.fallback;
        if (!s.fallback_model.empty()) sj["fallback_model"] = s.fallback_model;
        j["steps"].push_back(sj);
    }
    return j;
}

WorkflowPlan WorkflowPlan::from_json(const json& j) {
    auto plan = WorkflowPlanner::parse_plan(j, j.value("metadata", json::object()).value("task", ""));
    if (j.contains("id")) plan.id = j["id"].get<std::string>();
    if (j.contains("metadata")) plan.metadata = j["metadata"];
    validate_dag(plan);
    return plan;
}

// ════════════════════════════════════════════════════════════════
// WorkflowState
// ════════════════════════════════════════════════════════════════

json WorkflowState::resolve_ref(const std::string& ref) const {
    if (ref.empty() || ref[0] != '$') return ref;
    std::vector<std::string> parts;
    std::stringstream ss(ref.substr(1)); std::string t;
    while (std::getline(ss, t, '.')) parts.push_back(t);
    if (parts.empty()) return nullptr;
    json obj = (parts[0] == "task_input") ? task_input :
               step_outputs.count(parts[0]) ? step_outputs.at(parts[0]) : json(nullptr);
    for (size_t i = 1; i < parts.size(); ++i) {
        if (obj.is_null()) break;
        if (obj.is_object() && obj.contains(parts[i])) obj = obj[parts[i]];
        else if (obj.is_array()) {
            try {
                int idx = std::stoi(parts[i]);
                if (idx >= 0 && idx < (int)obj.size()) obj = obj[idx];
                else obj = nullptr;
            } catch (...) { obj = nullptr; }
        }
        else obj = nullptr;
    }
    return obj;
}
json WorkflowState::resolve_value(const json& v) const {
    if (v.is_string()) {
        const auto& sv = v.get<std::string>();
        return (!sv.empty() && sv[0] == '$') ? resolve_ref(sv) : v;
    }
    if (v.is_object()) {
        json r = json::object();
        for (auto& [k, val] : v.items()) r[k] = resolve_value(val);
        return r;
    }
    if (v.is_array()) {
        json r = json::array();
        for (const auto& el : v) r.push_back(resolve_value(el));
        return r;
    }
    return v;
}
json WorkflowState::resolve_inputs(const json& inputs) const {
    return resolve_value(inputs);
}
void WorkflowState::record_trace(TraceEntry entry) noexcept {
    traces.push_back(std::move(entry));
}

// ════════════════════════════════════════════════════════════════
// WorkflowContext — 多轮记忆
// ════════════════════════════════════════════════════════════════

void WorkflowContext::record(const std::string& task, const json& output) {
    std::string out_str = output.is_string() ? output.get<std::string>() : output.dump();
    if ((int)out_str.size() > max_summary_chars_) out_str = out_str.substr(0, max_summary_chars_) + "...";
    history_.push_back({task, out_str});
    if ((int)history_.size() > max_history_)
        history_.erase(history_.begin());
}
std::string WorkflowContext::to_prompt_prefix() const {
    if (history_.empty()) return "";
    std::string ctx = "Previous conversation context (use if relevant):\n";
    for (const auto& [t, o] : history_)
        ctx += "  Task: " + t + "\n  Result summary: " + o + "\n\n";
    return ctx;
}

// ════════════════════════════════════════════════════════════════
// WorkflowPlanner
// ════════════════════════════════════════════════════════════════

json WorkflowPlanner::extract_json(const std::string& text) {
    try { return json::parse(text); } catch (const std::exception& e) {
        log_msg(LogLevel::LOG_DEBUG, "Planner", std::string("direct parse failed: ") + e.what());
    }
    std::regex fence(R"(```(?:json)?\s*([\s\S]+?)\s*```)"); std::smatch m;
    if (std::regex_search(text, m, fence)) try { return json::parse(m[1].str()); } catch (const std::exception& e) {
        log_msg(LogLevel::LOG_DEBUG, "Planner", std::string("fence parse failed: ") + e.what());
    }
    std::regex obj(R"(\{[\s\S]+\})");
    if (std::regex_search(text, m, obj)) try { return json::parse(m[0].str()); } catch (const std::exception& e) {
        log_msg(LogLevel::LOG_DEBUG, "Planner", std::string("regex parse failed: ") + e.what());
    }
    throw PlanningError("Cannot extract JSON from LLM response: " + text.substr(0, 300));
}

WorkflowPlan WorkflowPlanner::parse_plan(const json& raw, const std::string& task) {
    WorkflowPlan plan;
    plan.id = "plan_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    plan.metadata = {{"task", task}};
    for (const auto& s : raw["steps"]) {
        Step step;
        step.id     = s.contains("id")     ? s["id"].get<std::string>()     : "";
        step.action = s.contains("action") ? s["action"].get<std::string>() : "";
        if (step.id.empty() || step.action.empty())
            throw PlanningError("step missing required 'id' or 'action' field");
        step.description = s.value("description", "");
        step.inputs      = s.value("inputs", json::object());
        step.depends_on  = s.value("depends_on", std::vector<std::string>{});
        step.retry       = s.value("retry", 0);
        step.timeout_sec = s.value("timeout_sec", 30.0);
        step.fallback    = s.value("fallback_value", json(nullptr));
        const std::string tp = s.value("type", "llm");
        step.type = (tp == "tool") ? StepType::TOOL :
                    (tp == "transform") ? StepType::TRANSFORM :
                    (tp == "condition")  ? StepType::CONDITION : StepType::LLM;
        const std::string oe = s.value("on_error", "fail");
        step.on_error = (oe == "skip") ? OnError::SKIP :
                        (oe == "fallback") ? OnError::FALLBACK : OnError::FAIL;
        step.model_tier = (s.value("model_tier", "subagent") == "orchestrator")
                          ? ModelTier::ORCHESTRATOR : ModelTier::SUBAGENT;
        step.system_prompt = s.value("system_prompt", "");
        step.json_mode     = s.value("json_mode", false);
        step.output_schema = s.value("output_schema", json(nullptr));
        step.temperature   = s.value("temperature", -1.0);
        plan.steps.push_back(std::move(step));
    }
    return plan;
}


static const char* PLANNER_SYS_V2 = R"(
You are a workflow planning engine.
Analyze the task and immediately generate an optimized execution DAG.
Return ONLY {"steps":[...]} JSON — no preamble, no explanation outside the JSON.

STEP SCHEMA:
{"id":"snake_case","type":"tool|llm|transform","action":"tool_name or instruction",
 "description":"one line","inputs":{"key":"value or $step_id.field"},
 "depends_on":["ids"],"retry":0,"timeout_sec":30,"on_error":"fail|skip|fallback",
 "model_tier":"subagent|orchestrator",
 "system_prompt":"","json_mode":false,"output_schema":{}}

Rules:
- Steps without shared dependencies run in parallel automatically
- Use tool steps for data gathering, llm steps for synthesis/generation
- Keep the DAG minimal — prefer 2-4 steps over 8+
- system_prompt / json_mode / output_schema are optional (omit if not needed)
)";

WorkflowPlan WorkflowPlanner::plan(const std::string& task,
                                    const std::vector<ToolDef>& tools,
                                    const WorkflowContext& ctx,
                                    int max_attempts) const {
    // Sort tools alphabetically for stable cache-friendly prompt prefix (D65 parity)
    auto sorted_tools = tools;
    std::sort(sorted_tools.begin(), sorted_tools.end(),
        [](const ToolDef& a, const ToolDef& b) { return a.name < b.name; });
    json tj = json::array();
    for (const auto& t : sorted_tools)
        tj.push_back({{"name", t.name}, {"description", t.description},
                      {"inputs", t.input_schema}});

    std::string prefix = ctx.to_prompt_prefix();
    std::string base = prefix
        + "Available tools:\n" + tj.dump(2)
        + "\n\nTask: " + task
        + "\n\nGenerate the workflow DAG:";

    std::string errors;
    const std::vector<double> temps = {0.0, 0.2, 0.4};
    for (int i = 0; i < max_attempts; ++i) {
        std::string p = base;
        if (!errors.empty())
            p += "\n\nPrevious validation errors (fix them):\n" + errors + "\nReturn ONLY JSON.";
        try {
            const char* sys = custom_sys_.empty() ? PLANNER_SYS_V2 : custom_sys_.c_str();
            auto raw  = extract_json(llm_.complete_as(
                ModelTier::ORCHESTRATOR, p, sys,
                temps[std::min(i, (int)temps.size()-1)], /*force_json=*/true));
            auto plan = parse_plan(raw, task);
            validate_dag(plan);
            return plan;
        } catch (const std::exception& e) {
            errors += "[Attempt " + std::to_string(i+1) + "] " + e.what() + "\n";
        }
    }
    throw PlanningError("plan() failed after " + std::to_string(max_attempts)
                        + " attempts:\n" + errors);
}

WorkflowPlanner::WorkflowPlanner(LLMClient& llm, const std::string& custom_sys)
    : llm_(llm), custom_sys_(custom_sys) {}

// ════════════════════════════════════════════════════════════════
// WorkflowExecutor
// ════════════════════════════════════════════════════════════════

WorkflowExecutor::WorkflowExecutor(LLMClient& llm, ToolRegistry& tools, size_t max_threads,
                                     std::shared_ptr<IMetricsCollector> metrics)
    : llm_(llm), tools_(tools), pool_(max_threads),
      metrics_(metrics ? metrics : std::make_shared<NoOpMetrics>()) {}

WorkflowState WorkflowExecutor::execute(const WorkflowPlan& plan,
                                         const json& task_input,
                                         CancelToken cancel,
                                         StepInterruptFn interrupt) const {
    WorkflowState state;
    state.task_input = task_input;

    // 构建 step 类型索引，用于 CONDITION 分支控制
    std::map<std::string, StepType> step_types;
    for (const auto& s : plan.steps) step_types[s.id] = s.type;

    // 被 CONDITION=false 阻断的步骤集合
    std::set<std::string> skipped_by_condition;

    for (const auto& batch : plan.topological_batches()) {
        if (cancel && cancel->load()) throw WorkflowCancelledError("Workflow cancelled");
        using FutResult = std::tuple<bool, json, std::vector<TraceEntry>>;
        std::vector<std::pair<Step, std::future<FutResult>>> futs;

        for (const auto& step : batch) {
            // Human-in-the-loop: interrupt check before each step
            if (interrupt) {
                auto reason = interrupt(step, state);
                if (reason) {
                    throw InterruptError(step.id, *reason, state.to_json());
                }
            }
            // CONDITION 分支控制：如果任意依赖的 CONDITION 步骤返回 false，跳过此步骤
            bool blocked = false;
            for (const auto& dep : step.depends_on) {
                if (skipped_by_condition.count(dep)) { blocked = true; break; }
                if (step_types.count(dep) && step_types[dep] == StepType::CONDITION) {
                    auto it = state.step_outputs.find(dep);
                    if (it != state.step_outputs.end()) {
                        const auto& v = it->second;
                        if (v.is_null() || v == false || v == 0) { blocked = true; break; }
                    }
                }
            }
            if (blocked) {
                skipped_by_condition.insert(step.id);
                state.step_outputs[step.id] = nullptr;
                state.record_trace({step.id, "condition", "skipped", "", 0, "blocked by false condition"});
                continue;
            }

            futs.emplace_back(step,
                pool_.submit([&, step]() -> FutResult {
                    std::vector<TraceEntry> local_traces;
                    try {
                        auto result = run_step(step, state, local_traces);
                        return {true, result, local_traces};
                    } catch (const std::exception& e) {
                        if (local_traces.empty()) {
                            local_traces.push_back({step.id,
                                step.type == StepType::TOOL ? "tool" : "llm",
                                "failed", "", 0, e.what()});
                        } else {
                            local_traces.back().status = "failed";
                            local_traces.back().error  = e.what();
                        }
                        return {false, {{"error", e.what()}}, local_traces};
                    }
                }));
        }

        for (auto& [step, fut] : futs) {
            auto [ok, result, step_traces] = fut.get();
            for (auto& te : step_traces) state.record_trace(std::move(te));

            if (!ok) {
                switch (step.on_error) {
                case OnError::FAIL:
                    throw StepExecutionError(step.id, step.description,
                                          result.value("error", "unknown"));
                case OnError::SKIP:
                    state.errors[step.id]      = result.value("error", "");
                    state.step_outputs[step.id] = nullptr;
                    break;
                case OnError::FALLBACK:
                    state.errors[step.id]      = result.value("error", "");
                    state.step_outputs[step.id] = step.fallback;
                    break;
                }
            } else {
                state.step_outputs[step.id] = result;
            }
        }
    }
    return state;
}

json WorkflowExecutor::run_step(const Step& step, const WorkflowState& state,
                                  std::vector<TraceEntry>& traces) const {
    auto t0     = std::chrono::steady_clock::now();
    auto inputs = state.resolve_inputs(step.inputs);

    std::string type_str = (step.type == StepType::TOOL) ? "tool" :
                           (step.type == StepType::LLM)  ? "llm"  :
                           (step.type == StepType::TRANSFORM) ? "transform" : "condition";

    metrics_->record({MetricEvent::Kind::STEP_START, "", step.id, "", "", 0, true, ""});

    std::exception_ptr last;
    static thread_local std::mt19937 rng(std::random_device{}());
    for (int i = 0; i <= step.retry; ++i) {
        try {
            auto result = dispatch(step, inputs);
            auto ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            TraceEntry te{step.id, type_str, "ok", "", ms, ""};
            if (step.type == StepType::LLM) {
                te.input_tokens  = g_last_token_usage.input_tokens;
                te.output_tokens = g_last_token_usage.output_tokens;
            }
            traces.push_back(te);
            metrics_->record({MetricEvent::Kind::STEP_END, "", step.id, "", "", ms, true, ""});
            return result;
        } catch (...) {
            last = std::current_exception();
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed >= step.timeout_sec)
                throw StepExecutionError(step.id, step.description,
                    "timed out after " + std::to_string((int)elapsed)
                    + "s (limit: " + std::to_string((int)step.timeout_sec) + "s)");
            if (i < step.retry) {
                std::uniform_int_distribution<int> jitter(0, 200);
                int delay_ms = 500 * (1 << i) + jitter(rng);
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            }
        }
    }
    auto ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    try { std::rethrow_exception(last); } catch (const std::exception& e) {
        traces.push_back({step.id, type_str, "failed", "", ms, e.what()});
        metrics_->record({MetricEvent::Kind::STEP_END, "", step.id, "", "", ms, false, e.what()});
        throw;
    }
}

json WorkflowExecutor::dispatch(const Step& step, const json& inputs) const {
    switch (step.type) {
    case StepType::TOOL:      return exec_tool(step, inputs);
    case StepType::LLM:       return exec_llm (step, inputs);
    case StepType::TRANSFORM: return exec_transform(step, inputs);
    case StepType::CONDITION: {
        auto v = inputs.value("value", json(nullptr));
        return !v.is_null() && v != false && v != 0 && v != "";
    }
    default: throw StepExecutionError(step.id, step.description, "unknown step type");
    }
}

json WorkflowExecutor::exec_tool(const Step& s, const json& in) const {
    metrics_->record({MetricEvent::Kind::TOOL_CALL, "", s.id, "", "", 0, true, "", {{"tool", s.action}}});
    auto t0 = std::chrono::steady_clock::now();
    auto result = tools_.call(s.action, in);
    auto ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    metrics_->record({MetricEvent::Kind::TOOL_RESPONSE, "", s.id, "", "", (long)ms, true, ""});
    return result;
}

json WorkflowExecutor::exec_llm(const Step& s, const json& in) const {
    std::string prompt = s.action;
    if (!in.empty())
        prompt += "\n\nContext:\n" + in.dump(2);

    std::string tier_str = (s.model_tier == ModelTier::ORCHESTRATOR) ? "orchestrator" : "subagent";
    metrics_->record({MetricEvent::Kind::LLM_CALL, "", s.id, tier_str, "", 0, true, ""});
    auto llm_t0 = std::chrono::steady_clock::now();

    // B1: Structured output / JSON mode
    if (s.json_mode || !s.output_schema.empty()) {
        if (!s.output_schema.empty())
            prompt += "\n\nReturn ONLY a JSON object matching this exact schema:\n"
                    + s.output_schema.dump(2)
                    + "\nNo markdown, no explanation — JSON only.";
        else
            prompt += "\n\nReturn ONLY valid JSON. No markdown, no explanation.";

        double temp = (s.temperature >= 0.0) ? s.temperature : 0.0;
        auto raw = llm_.complete_as(s.model_tier, prompt, s.system_prompt, temp, true, s.output_schema);
        try {
            auto result = WorkflowPlanner::extract_json(raw);
            // 实际 schema 验证（不只是 prompt injection）
            if (!s.output_schema.empty()) {
                auto violations = validate_json_schema(result, s.output_schema);
                if (!violations.empty()) {
                    // 记录违规但不抛异常（LLM 输出的容错性）
                    std::string viol_msg = "Step '" + s.id + "' schema violations:";
                    for (const auto& v : violations)
                        viol_msg += " " + v.path + "=" + v.message + ";";
                    log_msg(LogLevel::LOG_WARN, "Executor", viol_msg);
                }
            }
            auto llm_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - llm_t0).count();
            metrics_->record({MetricEvent::Kind::LLM_RESPONSE, "", s.id, "", "", (long)llm_ms, true, ""});
            return result;
        }
        catch (...) {
            auto llm_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - llm_t0).count();
            metrics_->record({MetricEvent::Kind::LLM_RESPONSE, "", s.id, "", "", (long)llm_ms, true, ""});
            return raw;
        }
    }

    double temp = (s.temperature >= 0.0) ? s.temperature : 0.0;
    auto result = llm_.complete_as(s.model_tier, prompt, s.system_prompt, temp);
    auto llm_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - llm_t0).count();
    metrics_->record({MetricEvent::Kind::LLM_RESPONSE, "", s.id, "", "", (long)llm_ms, true, ""});
    return result;
}

// B3: parse_json 加保护
json WorkflowExecutor::exec_transform(const Step& s, const json& in) const {
    if (s.action == "to_list") {
        auto items = in.value("items", json::array());
        if (items.is_object()) {
            json a = json::array();
            for (auto& [_, v] : items.items()) a.push_back(v);
            return a;
        }
        return items;
    }
    if (s.action == "join") {
        std::string sep = in.value("separator", "\n"), r;
        for (const auto& it : in.value("items", json::array())) {
            if (!r.empty()) r += sep;
            r += it.is_string() ? it.get<std::string>() : it.dump();
        }
        return r;
    }
    if (s.action == "parse_json") {
        // B3: 加 try-catch，失败抛出明确错误
        const std::string text = in.value("text", std::string{});
        try { return json::parse(text); }
        catch (const std::exception& e) {
            throw StepExecutionError("", "parse_json", "invalid JSON input — " + std::string(e.what()));
        }
    }
    if (s.action == "head") {
        int n = in.value("n", 5);
        auto items = in.value("items", json::array());
        if (items.is_array())
            return json(items.begin(), items.begin() + std::min(n, (int)items.size()));
        return items;
    }
    return in;
}

// ════════════════════════════════════════════════════════════════
// DynamicWorkflow — Ultracode 级别的动态任务编排
// ════════════════════════════════════════════════════════════════

DynamicWorkflow::DynamicWorkflow(WorkflowEngine& engine, size_t max_concurrency)
    : engine_(engine)
    , pool_(max_concurrency == 0 ? std::min(16u, std::thread::hardware_concurrency()) : max_concurrency)
{}

DynamicWorkflow::~DynamicWorkflow() = default;

void DynamicWorkflow::emit(const std::string& msg) {
    {
        std::lock_guard<std::mutex> lk(log_mu_);
        log_.push_back(msg);
    }
    if (progress_fn_) progress_fn_(current_phase_, msg);
}

void DynamicWorkflow::phase(const std::string& name) {
    current_phase_ = name;
    emit("── Phase: " + name + " ──");
}

void DynamicWorkflow::log(const std::string& message) {
    emit(message);
}

std::vector<json> DynamicWorkflow::parallel(const std::vector<DynTask>& tasks) {
    emit("parallel: " + std::to_string(tasks.size()) + " tasks");
    std::vector<std::future<json>> futs;
    futs.reserve(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        futs.push_back(pool_.submit([&tasks, i, this]() -> json {
            try {
                auto result = tasks[i]();
                emit("  [" + std::to_string(i) + "] done");
                return result;
            } catch (const std::exception& e) {
                emit("  [" + std::to_string(i) + "] failed: " + e.what());
                return nullptr;
            }
        }));
    }
    std::vector<json> results;
    results.reserve(futs.size());
    for (auto& f : futs) results.push_back(f.get());
    return results;
}

std::vector<json> DynamicWorkflow::map(const std::vector<json>& items, StageFn fn) {
    emit("map: " + std::to_string(items.size()) + " items");
    std::vector<std::future<json>> futs;
    futs.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        futs.push_back(pool_.submit([&fn, &items, i, this]() -> json {
            try {
                return fn(items[i]);
            } catch (const std::exception& e) {
                emit("  map[" + std::to_string(i) + "] failed: " + e.what());
                return nullptr;
            }
        }));
    }
    std::vector<json> results;
    results.reserve(futs.size());
    for (auto& f : futs) results.push_back(f.get());
    return results;
}

std::vector<json> DynamicWorkflow::pipeline(const std::vector<json>& items,
                                              const std::vector<StageFn>& stages) {
    emit("pipeline: " + std::to_string(items.size()) + " items × "
         + std::to_string(stages.size()) + " stages");
    // Each item flows through all stages independently — no barrier between stages
    // Pipeline via pool: each item gets a single task that chains all stages
    std::vector<std::future<json>> futs;
    futs.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        futs.push_back(pool_.submit([&stages, &items, i, this]() -> json {
            json current = items[i];
            for (size_t s = 0; s < stages.size(); ++s) {
                if (current.is_null()) return nullptr;  // previous stage failed
                try {
                    current = stages[s](current);
                } catch (const std::exception& e) {
                    emit("  pipe[" + std::to_string(i) + "] failed at stage "
                         + std::to_string(s) + ": " + e.what());
                    return nullptr;
                }
            }
            emit("  pipe[" + std::to_string(i) + "] complete");
            return current;
        }));
    }
    std::vector<json> results;
    results.reserve(futs.size());
    for (auto& f : futs) results.push_back(f.get());
    return results;
}

DynamicResult DynamicWorkflow::loop_until(StopFn stop, RoundFn work, int max_rounds) {
    DynamicResult result;
    auto t0 = std::chrono::steady_clock::now();
    emit("loop_until: max " + std::to_string(max_rounds) + " rounds");

    std::vector<json> accumulated;
    for (int round = 0; round < max_rounds; ++round) {
        // Budget check
        if (engine_.budget_exceeded()) {
            emit("loop_until: budget exceeded at round " + std::to_string(round));
            break;
        }
        // Stop condition
        if (stop(round, accumulated)) {
            emit("loop_until: stop condition met at round " + std::to_string(round));
            break;
        }
        emit("loop_until: round " + std::to_string(round + 1));
        try {
            auto batch = work(round);
            for (auto& item : batch) accumulated.push_back(std::move(item));
            emit("  round " + std::to_string(round + 1) + ": +"
                 + std::to_string(batch.size()) + " items, total "
                 + std::to_string(accumulated.size()));
        } catch (const std::exception& e) {
            emit("  round " + std::to_string(round + 1) + " failed: " + e.what());
            result.error = e.what();
            result.success = false;
            break;
        }
        result.rounds_used = round + 1;
    }
    result.outputs = std::move(accumulated);
    result.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    {
        std::lock_guard<std::mutex> lk(log_mu_);
        result.log = log_;
    }
    return result;
}

std::vector<json> DynamicWorkflow::fan_out_agents(const std::vector<std::string>& tasks,
                                                    int max_iterations) {
    phase("fan_out_agents");
    emit("Spawning " + std::to_string(tasks.size()) + " parallel agents");
    std::vector<DynTask> agent_tasks;
    agent_tasks.reserve(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        agent_tasks.push_back([this, &tasks, i, max_iterations]() -> json {
            auto result = engine_.run_agent_native(tasks[i], max_iterations);
            return {{"task", tasks[i]},
                    {"success", result.success},
                    {"answer", result.final_answer},
                    {"iterations", result.iterations_used},
                    {"duration_ms", result.duration_ms},
                    {"tokens", result.token_usage.total_tokens}};
        });
    }
    return parallel(agent_tasks);
}

json DynamicWorkflow::adversarial_verify(const std::string& claim, int num_voters) {
    phase("adversarial_verify");
    emit("Verifying claim with " + std::to_string(num_voters) + " independent voters");

    std::string verify_prompt =
        "You are a skeptical fact-checker. Analyze this claim and try to REFUTE it.\n"
        "If you cannot refute it, it is likely true.\n\n"
        "Claim: " + claim + "\n\n"
        "Respond with ONLY a JSON object:\n"
        "{\"verified\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"your reasoning\"}";

    std::vector<DynTask> voter_tasks;
    for (int i = 0; i < num_voters; ++i) {
        voter_tasks.push_back([this, &verify_prompt, i]() -> json {
            try {
                auto raw = engine_.llm_complete(verify_prompt, "", ModelTier::SUBAGENT, 0.3);
                return WorkflowPlanner::extract_json(raw);
            } catch (...) {
                return {{"verified", false}, {"confidence", 0.0}, {"reason", "voter failed"}};
            }
        });
    }

    auto votes = parallel(voter_tasks);

    // Majority vote
    int yes = 0, no = 0;
    double total_confidence = 0.0;
    json all_reasons = json::array();
    for (const auto& v : votes) {
        if (v.is_null()) { ++no; continue; }
        bool verified = v.value("verified", false);
        if (verified) ++yes; else ++no;
        total_confidence += v.value("confidence", 0.0);
        all_reasons.push_back(v.value("reason", ""));
    }

    bool final_verdict = yes > no;
    emit("Verdict: " + std::string(final_verdict ? "VERIFIED" : "REFUTED")
         + " (" + std::to_string(yes) + "/" + std::to_string(num_voters) + " votes)");

    return {{"claim", claim},
            {"verified", final_verdict},
            {"votes_yes", yes}, {"votes_no", no},
            {"avg_confidence", votes.empty() ? 0.0 : total_confidence / votes.size()},
            {"reasons", all_reasons}};
}

// ════════════════════════════════════════════════════════════════
// AdaptiveOrchestrator — 自动策略选择 + 动态编排
// ════════════════════════════════════════════════════════════════

static const char* ORCHESTRATOR_STRATEGY_SYS = R"(
You are a workflow strategy planner. Analyze the user's task and decide the BEST execution strategy.

Return ONLY a JSON object (no markdown, no explanation):
{
  "strategy": "simple_dag" | "agent_loop" | "parallel_research" | "pipeline_verify" | "multi_agent",
  "reasoning": "one sentence explaining why this strategy",
  "subtasks": ["subtask1", "subtask2"],
  "synthesis_prompt": "instruction for combining results",
  "max_iterations": 8
}

Strategy guide:
- "simple_dag": Single-step or simple sequential tasks (e.g., "summarize this text", "translate X")
- "agent_loop": Open-ended research or complex reasoning requiring multiple tool calls (e.g., "research topic X")
- "parallel_research": Task involves comparing/analyzing MULTIPLE entities in parallel (e.g., "compare A vs B vs C"). Split into subtasks, one per entity. ALWAYS use this for comparison tasks.
- "pipeline_verify": Task requires research THEN verification/fact-checking (e.g., "find and verify claims about X")
- "multi_agent": Task needs specialized roles (e.g., "research X then write a report" → researcher + writer)

Rules:
- subtasks: list the parallel work items (for parallel_research: one per entity; for pipeline_verify: research topics)
- synthesis_prompt: how to combine the parallel results into a final answer
- max_iterations: 5-15 depending on complexity
- For simple tasks, subtasks can be empty and synthesis_prompt can be empty
)";

AdaptiveOrchestrator::AdaptiveOrchestrator(WorkflowEngine& engine) : engine_(engine) {}

std::string AdaptiveOrchestrator::strategy_name(OrchestratorStrategy s) {
    switch (s) {
    case OrchestratorStrategy::SIMPLE_DAG:        return "simple_dag";
    case OrchestratorStrategy::AGENT_LOOP:        return "agent_loop";
    case OrchestratorStrategy::PARALLEL_RESEARCH: return "parallel_research";
    case OrchestratorStrategy::PIPELINE_VERIFY:   return "pipeline_verify";
    case OrchestratorStrategy::MULTI_AGENT:       return "multi_agent";
    case OrchestratorStrategy::DYNAMIC_COMPOSE:  return "dynamic_compose";
    default: return "unknown";
    }
}

OrchestratorPlan AdaptiveOrchestrator::plan_strategy(const std::string& task) {
    OrchestratorPlan plan;

    // List available tools for context
    auto tools = engine_.list_tools();
    std::string tool_list;
    for (const auto& t : tools) tool_list += t.name + " ";

    std::string prompt = "Task: " + task + "\n";
    if (!tool_list.empty()) prompt += "Available tools: " + tool_list + "\n";
    prompt += "Select the optimal strategy:";

    try {
        auto raw = engine_.llm_complete(prompt, ORCHESTRATOR_STRATEGY_SYS,
                                         ModelTier::ORCHESTRATOR, 0.0);
        auto j = WorkflowPlanner::extract_json(raw);

        const std::string s = j.value("strategy", "agent_loop");
        plan.strategy = (s == "simple_dag")        ? OrchestratorStrategy::SIMPLE_DAG :
                        (s == "parallel_research")  ? OrchestratorStrategy::PARALLEL_RESEARCH :
                        (s == "pipeline_verify")    ? OrchestratorStrategy::PIPELINE_VERIFY :
                        (s == "multi_agent")        ? OrchestratorStrategy::MULTI_AGENT :
                        (s == "dynamic_compose")    ? OrchestratorStrategy::DYNAMIC_COMPOSE :
                                                     OrchestratorStrategy::AGENT_LOOP;
        plan.reasoning = j.value("reasoning", "");
        plan.subtasks  = j.value("subtasks", std::vector<std::string>{});
        plan.synthesis_prompt = j.value("synthesis_prompt", "");
        plan.max_iterations   = j.value("max_iterations", 8);
    } catch (const std::exception& e) {
        // Fallback to agent_loop on parse failure
        plan.strategy = OrchestratorStrategy::AGENT_LOOP;
        plan.reasoning = "LLM strategy planning failed, defaulting to agent loop: " + std::string(e.what());
        plan.max_iterations = 8;
    }
    return plan;
}

OrchestratorPlan AdaptiveOrchestrator::analyze(const std::string& task) {
    return plan_strategy(task);
}

OrchestratorResult AdaptiveOrchestrator::execute_plan(const std::string& task,
                                                       const OrchestratorPlan& plan) {
    OrchestratorResult result;
    result.strategy_used = strategy_name(plan.strategy);
    result.reasoning     = plan.reasoning;
    auto t0 = std::chrono::steady_clock::now();

    auto emit = [&](const std::string& msg) {
        result.log.push_back(msg);
        if (progress_fn_) progress_fn_(result.strategy_used, msg);
    };

    emit("Strategy: " + result.strategy_used + " — " + plan.reasoning);

    try {
        switch (plan.strategy) {

        case OrchestratorStrategy::SIMPLE_DAG: {
            emit("Executing as DAG workflow...");
            auto wr = engine_.run(task);
            result.success = wr.success;
            result.output  = wr.output;
            if (!wr.success) result.error = wr.error_message;
            result.token_usage = wr.token_usage;
            break;
        }

        case OrchestratorStrategy::AGENT_LOOP: {
            emit("Executing as agent loop (" + std::to_string(plan.max_iterations) + " max iterations)...");
            auto ar = engine_.run_agent_native(task, plan.max_iterations,
                [&](const AgentStep& s) {
                    emit("[iter " + std::to_string(s.iteration) + "] " +
                         s.thought.substr(0, 80));
                });
            result.success = ar.success;
            result.output  = ar.final_answer;
            if (!ar.success) result.error = ar.error;
            result.token_usage = ar.token_usage;
            break;
        }

        case OrchestratorStrategy::PARALLEL_RESEARCH: {
            if (plan.subtasks.empty()) {
                // Fallback: run as single agent
                emit("No subtasks found, falling back to agent loop");
                auto ar = engine_.run_agent_native(task, plan.max_iterations);
                result.success = ar.success;
                result.output  = ar.final_answer;
                if (!ar.success) result.error = ar.error;
                result.token_usage = ar.token_usage;
                break;
            }
            DynamicWorkflow dw(engine_);
            dw.on_progress([&](auto&, auto& msg) { emit(msg); });

            // Phase 1: Parallel research
            dw.phase("Research");
            emit("Fan-out " + std::to_string(plan.subtasks.size()) + " parallel agents...");
            auto agent_results = dw.fan_out_agents(plan.subtasks, plan.max_iterations);

            // Phase 2: Synthesis
            dw.phase("Synthesis");
            std::string synthesis_input = "Research results:\n";
            for (size_t i = 0; i < agent_results.size(); ++i) {
                synthesis_input += "\n--- " + plan.subtasks[i] + " ---\n";
                if (!agent_results[i].is_null())
                    synthesis_input += agent_results[i].value("answer", agent_results[i].dump());
                else
                    synthesis_input += "(research failed)";
                synthesis_input += "\n";
            }
            std::string synth_prompt = plan.synthesis_prompt.empty()
                ? "Based on the research results above, provide a comprehensive combined answer to: " + task
                : plan.synthesis_prompt + "\n\n" + synthesis_input;
            if (!plan.synthesis_prompt.empty())
                synth_prompt = synthesis_input + "\n\n" + plan.synthesis_prompt;
            else
                synth_prompt = synthesis_input + "\n\nProvide a comprehensive combined analysis for: " + task;

            emit("Synthesizing results...");
            auto synthesis = engine_.llm_complete(synth_prompt, "", ModelTier::ORCHESTRATOR, 0.0);
            result.success = true;
            result.output  = {{"research", agent_results}, {"synthesis", synthesis}};
            break;
        }

        case OrchestratorStrategy::PIPELINE_VERIFY: {
            DynamicWorkflow dw(engine_);
            dw.on_progress([&](auto&, auto& msg) { emit(msg); });

            // Research phase: fan out agents for each subtask
            dw.phase("Research");
            auto subtasks = plan.subtasks.empty()
                ? std::vector<std::string>{task}
                : plan.subtasks;
            auto research = dw.fan_out_agents(subtasks, plan.max_iterations);

            // Verify phase: pipeline each result through verification
            dw.phase("Verify");
            std::vector<json> to_verify;
            for (size_t i = 0; i < research.size(); ++i) {
                if (!research[i].is_null()) to_verify.push_back(research[i]);
            }
            auto verified = dw.pipeline(to_verify, {
                [&](const json& r) -> json {
                    std::string answer = r.value("answer", r.dump());
                    auto v = dw.adversarial_verify(answer, 3);
                    return {{"research", r}, {"verification", v}};
                }
            });

            result.success = true;
            result.output  = {{"verified_results", verified}};
            break;
        }

        case OrchestratorStrategy::MULTI_AGENT: {
            emit("Executing as multi-agent workflow...");
            // Build agent definitions from subtasks
            std::vector<AgentDef> agents;
            if (plan.subtasks.size() >= 2) {
                // First subtask owner = researcher, last = synthesizer
                agents.push_back({
                    "researcher", "You gather facts using available tools. When done, hand off to writer.",
                    {}, ModelTier::ORCHESTRATOR, {"writer"}
                });
                agents.push_back({
                    "writer", "You write a comprehensive answer from the research. Give a final_answer.",
                    {}, ModelTier::ORCHESTRATOR, {}
                });
            } else {
                agents.push_back({
                    "researcher", "You research the topic and provide a final_answer.",
                    {}, ModelTier::ORCHESTRATOR, {}
                });
            }
            auto ar = engine_.run_multi_agent(task, agents,
                agents[0].name, plan.max_iterations,
                [&](const AgentStep& s) {
                    emit("[iter " + std::to_string(s.iteration) + "] " +
                         s.thought.substr(0, 80));
                });
            result.success = ar.success;
            result.output  = ar.final_answer;
            if (!ar.success) result.error = ar.error;
            result.token_usage = ar.token_usage;
            break;
        }

        case OrchestratorStrategy::DYNAMIC_COMPOSE: {
            emit("Delegating to dynamic workflow composer...");
            auto dr = run_dynamic(task);
            result.success = dr.success;
            result.output  = dr.output;
            result.error   = dr.error;
            result.token_usage = dr.token_usage;
            for (const auto& l : dr.log) result.log.push_back(l);
            break;
        }
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error   = e.what();
        emit("ERROR: " + result.error);
    }

    result.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    emit("Complete: " + std::to_string(result.duration_ms) + "ms, "
         + (result.success ? "success" : "failed"));
    return result;
}

OrchestratorResult AdaptiveOrchestrator::run(const std::string& task) {
    auto plan = plan_strategy(task);
    return execute_plan(task, plan);
}

OrchestratorResult AdaptiveOrchestrator::run_with_strategy(const std::string& task,
                                                            OrchestratorStrategy strategy) {
    OrchestratorPlan plan;
    plan.strategy = strategy;
    plan.reasoning = "User-specified strategy";
    plan.max_iterations = 10;
    return execute_plan(task, plan);
}

// ── run_dynamic: LLM 直接编写自定义工作流 ─────────────────────

static const char* DYNAMIC_COMPOSE_SYS = R"(
You are an expert workflow composer. Given a task and available tools,
design an optimal multi-step execution plan by composing these primitives:

Step types:
- "agent": Run a ReACT agent to research/solve a subtask.
  Required: "task" (string), "max_iterations" (int, 3-15)
- "parallel_agents": Fan out multiple agents in parallel.
  Required: "tasks" (string array), "max_iterations" (int)
- "llm_call": Direct LLM call for synthesis/analysis/writing.
  Required: "prompt" (string). Use $prev to reference previous step output.
  Optional: "system" (string) for role/persona.
- "verify": Adversarial verification of a claim (3-voter majority).
  Required: "claim" (string). Use $prev.field to reference.
- "parallel_calls": Multiple LLM calls in parallel.
  Required: "prompts" (string array)

Rules:
- Steps execute sequentially. Each step's output is available as $step_N (0-indexed).
- $prev refers to the immediately previous step's output.
- Use "parallel_agents" when multiple independent research tasks exist.
- Use "llm_call" for synthesis, analysis, summarization, writing.
- Use "verify" after claims that need fact-checking.
- Keep it minimal: 2-5 steps for most tasks, never more than 8.
- Respond with ONLY a JSON object, no markdown:

{"reasoning": "...", "steps": [...]}
)";

OrchestratorResult AdaptiveOrchestrator::run_dynamic(const std::string& task) {
    OrchestratorResult result;
    result.strategy_used = "dynamic_compose";
    auto t0 = std::chrono::steady_clock::now();

    auto emit = [&](const std::string& msg) {
        result.log.push_back(msg);
        if (progress_fn_) progress_fn_("dynamic", msg);
    };

    // Phase 1: LLM writes the workflow
    auto tools = engine_.list_tools();
    std::string tool_desc;
    for (const auto& t : tools) tool_desc += "- " + t.name + ": " + t.description + "\n";

    std::string compose_prompt = "Task: " + task + "\n";
    if (!tool_desc.empty()) compose_prompt += "\nAvailable tools:\n" + tool_desc;
    compose_prompt += "\nDesign the optimal workflow:";

    emit("Composing dynamic workflow...");
    json workflow;
    try {
        auto raw = engine_.llm_complete(compose_prompt, DYNAMIC_COMPOSE_SYS,
                                         ModelTier::ORCHESTRATOR, 0.0);
        workflow = WorkflowPlanner::extract_json(raw);
    } catch (const std::exception& e) {
        emit("Workflow composition failed, falling back to agent loop: " + std::string(e.what()));
        return run_with_strategy(task, OrchestratorStrategy::AGENT_LOOP);
    }

    result.reasoning = workflow.value("reasoning", "");
    emit("Plan: " + result.reasoning);

    auto steps = workflow.value("steps", json::array());
    if (steps.empty()) {
        emit("Empty workflow, falling back to agent loop");
        return run_with_strategy(task, OrchestratorStrategy::AGENT_LOOP);
    }

    // Phase 2: Execute each step
    DynamicWorkflow dw(engine_);
    dw.on_progress([&](auto&, auto& msg) { emit(msg); });

    std::vector<json> step_outputs;
    json prev_output;

    auto resolve_refs = [&](std::string s) -> std::string {
        // Replace $prev with previous step output
        if (!prev_output.is_null()) {
            auto pos = s.find("$prev");
            while (pos != std::string::npos) {
                std::string replacement = prev_output.is_string()
                    ? prev_output.get<std::string>() : prev_output.dump();
                s.replace(pos, 5, replacement);
                pos = s.find("$prev", pos + replacement.size());
            }
        }
        // Replace $step_N references
        for (size_t i = 0; i < step_outputs.size(); ++i) {
            std::string ref = "$step_" + std::to_string(i);
            auto pos = s.find(ref);
            while (pos != std::string::npos) {
                std::string replacement = step_outputs[i].is_string()
                    ? step_outputs[i].get<std::string>() : step_outputs[i].dump();
                s.replace(pos, ref.size(), replacement);
                pos = s.find(ref, pos + replacement.size());
            }
        }
        return s;
    };

    try {
        for (size_t i = 0; i < steps.size() && i < 8; ++i) {
            const auto& step = steps[i];
            std::string type = step.value("type", "");
            emit("[step " + std::to_string(i) + "] " + type);
            dw.phase("step_" + std::to_string(i));

            json output;

            if (type == "agent") {
                std::string t = resolve_refs(step.value("task", task));
                int max_iter = step.value("max_iterations", 8);
                auto ar = engine_.run_agent_native(t, max_iter);
                output = ar.final_answer;
                emit("[step " + std::to_string(i) + "] agent → " +
                     (ar.success ? "done" : "failed: " + ar.error));

            } else if (type == "parallel_agents") {
                auto tasks = step.value("tasks", std::vector<std::string>{});
                for (auto& t : tasks) t = resolve_refs(t);
                int max_iter = step.value("max_iterations", 8);
                auto results = dw.fan_out_agents(tasks, max_iter);
                output = results;
                emit("[step " + std::to_string(i) + "] parallel_agents → " +
                     std::to_string(results.size()) + " results");

            } else if (type == "llm_call") {
                std::string prompt = resolve_refs(step.value("prompt", ""));
                std::string sys = step.value("system", "");
                auto text = engine_.llm_complete(prompt, sys, ModelTier::ORCHESTRATOR, 0.0);
                output = text;
                emit("[step " + std::to_string(i) + "] llm_call → " +
                     std::to_string(text.size()) + " chars");

            } else if (type == "verify") {
                std::string claim = resolve_refs(step.value("claim", ""));
                int voters = step.value("voters", 3);
                output = dw.adversarial_verify(claim, voters);
                emit("[step " + std::to_string(i) + "] verify → " +
                     std::string(output.value("verified", false) ? "VERIFIED" : "REFUTED"));

            } else if (type == "parallel_calls") {
                auto prompts = step.value("prompts", std::vector<std::string>{});
                std::vector<DynTask> tasks;
                for (auto& p : prompts) {
                    p = resolve_refs(p);
                    tasks.push_back([this, p]() -> json {
                        return engine_.llm_complete(p, "", ModelTier::SUBAGENT, 0.0);
                    });
                }
                output = dw.parallel(tasks);
                emit("[step " + std::to_string(i) + "] parallel_calls → " +
                     std::to_string(prompts.size()) + " results");

            } else {
                emit("[step " + std::to_string(i) + "] unknown type '" + type + "', skipping");
                output = nullptr;
            }

            step_outputs.push_back(output);
            prev_output = output;
        }

        result.success = true;
        result.output = prev_output; // Final step output is the result
    } catch (const std::exception& e) {
        result.error = std::string("Dynamic workflow failed at execution: ") + e.what();
        if (!step_outputs.empty()) result.output = step_outputs.back();
    }

    result.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}

// ════════════════════════════════════════════════════════════════
// validate_dag
// ════════════════════════════════════════════════════════════════

void validate_dag(const WorkflowPlan& plan) {
    std::set<std::string> seen;
    for (const auto& s : plan.steps) {
        if (s.id.empty()) throw DAGValidationError("Step has empty 'id' field");
        if (seen.count(s.id)) throw DAGValidationError("Duplicate step ID: " + s.id);
        seen.insert(s.id);
    }
    for (const auto& s : plan.steps)
        for (const auto& d : s.depends_on)
            if (!seen.count(d))
                throw DAGValidationError("Step '" + s.id + "' depends on unknown step '" + d + "'");
    std::map<std::string, int> deg;
    std::map<std::string, std::vector<std::string>> adj;
    for (const auto& s : plan.steps) {
        if (!deg.count(s.id)) deg[s.id] = 0;
        for (const auto& d : s.depends_on) { adj[d].push_back(s.id); deg[s.id]++; }
    }
    std::queue<std::string> q;
    for (const auto& [id, d] : deg) if (d == 0) q.push(id);
    int vis = 0;
    while (!q.empty()) {
        auto n = q.front(); q.pop(); ++vis;
        for (const auto& d : adj[n]) if (--deg[d] == 0) q.push(d);
    }
    if (vis != (int)plan.steps.size())
        throw DAGValidationError("Cycle detected in workflow DAG");
}

// ════════════════════════════════════════════════════════════════
// WorkflowState serialization
// ════════════════════════════════════════════════════════════════

json WorkflowState::to_json() const {
    json j;
    j["task_input"] = task_input;
    j["step_outputs"] = json::object();
    for (const auto& [k, v] : step_outputs) j["step_outputs"][k] = v;
    j["errors"] = json::object();
    for (const auto& [k, v] : errors) j["errors"][k] = v;
    j["traces"] = json::array();
    for (const auto& t : traces) {
        j["traces"].push_back({
            {"step_id", t.step_id}, {"step_type", t.step_type},
            {"status", t.status}, {"provider", t.provider},
            {"duration_ms", t.duration_ms}, {"error", t.error},
            {"input_tokens", t.input_tokens}, {"output_tokens", t.output_tokens}
        });
    }
    return j;
}

WorkflowState WorkflowState::from_json(const json& j) {
    WorkflowState s;
    s.task_input = j.value("task_input", json::object());
    if (j.contains("step_outputs") && j["step_outputs"].is_object())
        for (const auto& [k, v] : j["step_outputs"].items())
            s.step_outputs[k] = v;
    if (j.contains("errors") && j["errors"].is_object())
        for (const auto& [k, v] : j["errors"].items())
            s.errors[k] = v.get<std::string>();
    if (j.contains("traces") && j["traces"].is_array())
        for (const auto& t : j["traces"])
            s.traces.push_back({
                t.value("step_id",""), t.value("step_type",""),
                t.value("status",""), t.value("provider",""),
                t.value("duration_ms",0L), t.value("error",""),
                t.value("input_tokens",0L), t.value("output_tokens",0L)
            });
    return s;
}

// ════════════════════════════════════════════════════════════════
// FileCheckpointStore
// ════════════════════════════════════════════════════════════════

FileCheckpointStore::FileCheckpointStore(const std::string& directory) : dir_(directory) {
    try { std::filesystem::create_directories(dir_); }
    catch (const std::filesystem::filesystem_error& e) {
        throw AriadneError("Checkpoint directory creation failed: " + std::string(e.what()));
    }
}

void FileCheckpointStore::save(const std::string& workflow_id, const json& state) {
    auto path = dir_ + "/" + workflow_id + ".checkpoint.json";
    std::ofstream f(path);
    if (!f) throw AriadneError("Checkpoint save failed (cannot open): " + path);
    f << state.dump(2);
    if (!f.good()) throw AriadneError("Checkpoint save failed (write error): " + path);
}

json FileCheckpointStore::load(const std::string& workflow_id) {
    auto path = dir_ + "/" + workflow_id + ".checkpoint.json";
    std::ifstream f(path);
    if (!f) throw AriadneError("Checkpoint not found: " + workflow_id);
    try { return json::parse(f); }
    catch (const json::parse_error& e) {
        throw AriadneError("Checkpoint corrupted (" + workflow_id + "): " + std::string(e.what()));
    }
}

bool FileCheckpointStore::exists(const std::string& workflow_id) {
    return std::filesystem::exists(dir_ + "/" + workflow_id + ".checkpoint.json");
}

void FileCheckpointStore::remove(const std::string& workflow_id) {
    std::filesystem::remove(dir_ + "/" + workflow_id + ".checkpoint.json");
}

std::vector<std::string> FileCheckpointStore::list() {
    std::vector<std::string> ids;
    if (!std::filesystem::exists(dir_)) return ids;
    for (const auto& entry : std::filesystem::directory_iterator(dir_)) {
        auto name = entry.path().stem().string();
        if (name.size() > 11 && name.substr(name.size()-11) == ".checkpoint")
            ids.push_back(name.substr(0, name.size()-11));
    }
    return ids;
}

// ════════════════════════════════════════════════════════════════
// WorkflowResult
// ════════════════════════════════════════════════════════════════

std::string WorkflowResult::summary() const {
    std::string s = success ? "OK" : "FAILED";
    s += " steps=" + std::to_string(step_count);
    s += " duration=" + std::to_string(duration_ms) + "ms";
    if (!success) s += " error=" + error_message;
    return s;
}

// ════════════════════════════════════════════════════════════════
// ResponseCache — LLM 响应缓存
// ════════════════════════════════════════════════════════════════

std::string ResponseCache::make_key(const std::string& model, const std::string& system,
                                      const std::string& prompt, double temperature,
                                      bool force_json) {
    return model + "\x00" + system + "\x00" + prompt + "\x00"
         + std::to_string(temperature) + "\x00"
         + (force_json ? "1" : "0");
}

bool ResponseCache::has(const std::string& key) const {
    std::lock_guard<std::mutex> lk(mu_);
    return cache_.count(key) > 0;
}

std::string ResponseCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_.find(key);
    if (it == cache_.end()) { ++stats_.misses; return ""; }
    ++stats_.hits;
    order_.remove(key);
    order_.push_front(key);
    return it->second;
}

void ResponseCache::put(const std::string& key, const std::string& response) {
    std::lock_guard<std::mutex> lk(mu_);
    if (cache_.count(key)) {
        order_.remove(key);
    } else if (cache_.size() >= max_size_) {
        auto lru = order_.back();
        order_.pop_back();
        cache_.erase(lru);
    }
    cache_[key] = response;
    order_.push_front(key);
}

// ════════════════════════════════════════════════════════════════
// PlanCache
// ════════════════════════════════════════════════════════════════

static size_t fnv1a_hash(const std::string& s) {
    size_t h = 14695981039346656037ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

std::string PlanCache::normalize_key(const std::string& task,
                                      const std::vector<ToolDef>& tools,
                                      const std::string& context) {
    std::string key;
    // 工具签名（排序后拼接）
    std::vector<std::string> tnames;
    for (const auto& t : tools) tnames.push_back(t.name);
    std::sort(tnames.begin(), tnames.end());
    for (const auto& n : tnames) key += n + ",";
    key += "|";
    // 任务归一化：小写，去多余空格
    std::string norm;
    bool last_space = true;
    for (char c : task) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!last_space) { norm += ' '; last_space = true; }
        } else { norm += c; last_space = false; }
    }
    while (!norm.empty() && norm.back() == ' ') norm.pop_back();
    key += norm;
    if (!context.empty()) {
        key += "|ctx:" + std::to_string(fnv1a_hash(context));
    }
    return key;
}

bool PlanCache::has(const std::string& key) const {
    std::lock_guard<std::mutex> lk(mu_);
    return cache_.count(key) > 0;
}

WorkflowPlan PlanCache::get(const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_.find(key);
    if (it == cache_.end()) { ++stats_.misses; return {}; }
    ++stats_.hits;
    order_.remove(key);
    order_.push_front(key);
    return it->second;
}

void PlanCache::put(const std::string& key, const WorkflowPlan& plan) {
    std::lock_guard<std::mutex> lk(mu_);
    if (cache_.count(key)) {
        order_.remove(key);
    } else if (cache_.size() >= max_size_) {
        auto lru = order_.back();
        order_.pop_back();
        cache_.erase(lru);
    }
    cache_[key] = plan;
    order_.push_front(key);
}

// ════════════════════════════════════════════════════════════════
// WorkflowEngine
// ════════════════════════════════════════════════════════════════

WorkflowEngine::WorkflowEngine(const EngineConfig& cfg)
    : config_(cfg)
    , llm_(std::make_unique<LLMClient>(cfg.orchestrator, cfg.subagent))
    , tools_(std::make_unique<ToolRegistry>())
    , planner_(std::make_unique<WorkflowPlanner>(*llm_))
    , executor_(std::make_unique<WorkflowExecutor>(*llm_, *tools_, cfg.max_concurrency, metrics_))
{}

void WorkflowEngine::register_tool(const ToolDef& def, ToolFn fn) {
    tools_->register_tool(def, std::move(fn));
}
void WorkflowEngine::print_provider_status() const { llm_->print_status(); }
WorkflowPlan WorkflowEngine::plan_only(const std::string& task) {
    auto tools = tools_->list_tools();
    WorkflowContext empty_ctx;
    auto plan = planner_->plan(task, tools, empty_ctx);  // A1: 单次规划
    for (const auto& step : plan.steps)
        if (step.type == StepType::TOOL && !tools_->has_tool(step.action))
            throw PlanningError("Plan references unregistered tool '" + step.action + "'");
    return plan;
}
WorkflowResult WorkflowEngine::run(const std::string& task) {
    return run_internal(task, nullptr);
}
WorkflowResult WorkflowEngine::run(const std::string& task, WorkflowContext& ctx) {
    return run_internal(task, &ctx);
}

WorkflowResult WorkflowEngine::run_internal(const std::string& task, WorkflowContext* ctx) {
    reset_cancel();
    llm_->reset_usage();
    WorkflowResult res;
    auto t0 = std::chrono::steady_clock::now();
    // Generate a short workflow ID for correlation
    auto wf_id = "wf_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count() % 1000000);
    metrics_->record({MetricEvent::Kind::WORKFLOW_START, wf_id, "", "", "", 0, true, task});
    try {
        // Input guardrails
        json task_json = {{"task", task}};
        for (const auto& guard : input_guardrails_) {
            auto err = guard(task_json);
            if (err) throw GuardrailError("Input guardrail: " + *err);
        }

        auto tools    = tools_->list_tools();
        WorkflowContext empty_ctx;
        WorkflowContext& use_ctx = ctx ? *ctx : empty_ctx;
        // 计划缓存：命中则跳过 ORCHESTRATOR 规划调用
        std::string cache_key;
        WorkflowPlan plan;
        if (cache_enabled_) {
            cache_key = PlanCache::normalize_key(task, tools, use_ctx.to_prompt_prefix());
            if (plan_cache_.has(cache_key)) {
                plan = plan_cache_.get(cache_key);
            } else {
                plan = planner_->plan(task, tools, use_ctx);
                plan_cache_.put(cache_key, plan);
            }
        } else {
            plan = planner_->plan(task, tools, use_ctx);
        }

        // Plan-time tool validation: 提前发现"计划引用了未注册工具"
        for (const auto& step : plan.steps) {
            if (step.type == StepType::TOOL && !tools_->has_tool(step.action)) {
                // 列出已注册工具帮助调试
                auto registered = tools_->list_tools();
                std::string avail;
                for (const auto& t : registered) avail += t.name + " ";
                throw PlanningError(
                    "Plan references unregistered tool '" + step.action +
                    "' in step '" + step.id + "'. Registered tools: [" +
                    (avail.empty() ? "(none)" : avail) + "]");
            }
        }
        if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_)
            throw WorkflowCancelledError("Workflow deadline exceeded");
        auto state    = executor_->execute(plan, {{"task", task}}, cancel_, interrupt_hook_);

        auto leaves = plan.leaf_steps();
        if (leaves.size() == 1)
            res.output = state.step_outputs.count(leaves[0].id)
                         ? state.step_outputs[leaves[0].id] : nullptr;
        else {
            res.output = json::object();
            for (const auto& l : leaves)
                res.output[l.id] = state.step_outputs.count(l.id)
                                   ? state.step_outputs[l.id] : nullptr;
        }
        res.traces     = state.traces;
        res.step_count = (int)plan.steps.size();
        res.success    = true;

        // Output guardrails
        for (const auto& guard : output_guardrails_) {
            auto err = guard(res.output);
            if (err) throw GuardrailError("Output guardrail: " + *err);
        }

        if (ctx && res.has_output()) ctx->record(task, res.output);

        auto orc = llm_->stats(ModelTier::ORCHESTRATOR);
        auto sub = llm_->stats(ModelTier::SUBAGENT);
        res.provider_stats = orc;
        res.provider_stats.insert(res.provider_stats.end(), sub.begin(), sub.end());

    } catch (const StepExecutionError& e) {
        res.success       = false;
        res.error_message = e.what();
        // 部分结果：收集已完成步骤的输出
        res.partial_outputs = json::object();
        // (executor 在异常前已存储完成的 step_outputs)
    } catch (const std::exception& e) {
        res.success       = false;
        res.error_message = e.what();
    }
    res.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    res.token_usage = llm_->total_usage();
    metrics_->record({MetricEvent::Kind::WORKFLOW_END, wf_id, "", "", "",
                      res.duration_ms, res.success, res.error_message});
    return res;
}



void AnthropicProvider::complete_stream(const std::string& prompt,
                                          const std::string& sys,
                                          double temp,
                                          StreamCallback on_chunk) const {
    json body = {{"model",cfg_.model},{"max_tokens",cfg_.max_tokens},
                  {"temperature",temp},{"stream",true},
                  {"messages",{{{"role","user"},{"content",prompt}}}}};
    if (!sys.empty()) body["system"] = sys;
    std::string base = cfg_.base_url.empty() ? "https://api.anthropic.com" : cfg_.base_url;
    SseParser parser{on_chunk, "", "anthropic"};
    do_stream_post(base + "/v1/messages",
        {"Content-Type: application/json",
         "x-api-key: " + cfg_.api_key,
         "anthropic-version: 2024-10-22"},
        body.dump(), parser, cfg_.timeout_sec);
}

void OpenAIChatProvider::complete_stream(const std::string& prompt,
                                           const std::string& sys,
                                           double temp,
                                           StreamCallback on_chunk) const {
    json msgs = json::array();
    if (!sys.empty()) msgs.push_back({{"role","system"},{"content",sys}});
    msgs.push_back({{"role","user"},{"content",prompt}});
    json body = {{"model",cfg_.model},{"max_tokens",cfg_.max_tokens},
                  {"temperature",temp},{"stream",true},{"messages",msgs}};
    std::string base = cfg_.base_url.empty() ? "https://api.openai.com" : rtrim_slash(cfg_.base_url);
    std::string path = cfg_.completions_path.empty() ? "/v1/chat/completions" : cfg_.completions_path;
    SseParser parser{on_chunk, "", "openai"};
    do_stream_post(base + path,
        {"Content-Type: application/json",
         "Authorization: Bearer " + cfg_.api_key},
        body.dump(), parser, cfg_.timeout_sec);
}

// Responses API 尚无原生 SSE，回退到非流式后按词 chunk
void OpenAIResponsesProvider::complete_stream(const std::string& prompt,
                                                const std::string& sys,
                                                double temp,
                                                StreamCallback on_chunk) const {
    auto full = complete(prompt, sys, temp);
    std::istringstream iss(full);
    std::string word;
    while (std::getline(iss, word, ' ')) on_chunk(word + " ");
}

void LLMClient::complete_as_stream(ModelTier tier,
                                     const std::string& prompt,
                                     const std::string& system,
                                     double temperature,
                                     StreamCallback on_chunk) const {
    const auto& slots = (tier == ModelTier::ORCHESTRATOR) ? orchestrators_ : subagents_;
    std::string last_error;
    for (const auto& slot : slots) {
        if (!slot.breaker.try_allow()) continue;
        { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.calls; }
        if (slot.rate_limiter.enabled())
            if (!slot.rate_limiter.acquire(10000)) { continue; }
        try {
            slot.provider->complete_stream(prompt, system, temperature, on_chunk);
            slot.breaker.on_success();
            { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.successes; }
            return;
        } catch (const std::exception& e) {
            slot.breaker.on_failure();
            { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.failures; }
            last_error = e.what();
        }
    }
    throw AllProvidersExhaustedError("Streaming providers exhausted. Last: " + last_error);
}

// ════════════════════════════════════════════════════════════════
// ProviderAutoPlanner
// ════════════════════════════════════════════════════════════════

static const std::string PROBE_PROMPT = "Respond with exactly one word: OK";
static const std::string PROBE_SYS    = "Health check. Reply: OK";

void ProviderAutoPlanner::add_candidate(const std::string& name,
                                         const ProviderConfig& config,
                                         const std::string& tier,
                                         int priority) {
    candidates_.push_back({name, config, tier, priority});
}

ProbeResult ProviderAutoPlanner::probe_one(const Candidate& c,
                                             std::chrono::seconds timeout) const {
    ProbeResult r;
    r.name     = c.name;  r.model = c.config.model;
    r.tier     = c.tier;  r.priority = c.priority;
    r.provider_name = (c.config.type == ProviderType::ANTHROPIC) ? "anthropic" : "openai";

    auto fut = std::async(std::launch::async, [&c]() -> std::pair<bool,std::string> {
        try {
            ProviderConfig pc = c.config;
            pc.timeout_sec = 8.0; pc.max_tokens = 16;
            return {true, make_provider(pc)->complete(PROBE_PROMPT, PROBE_SYS, 0.0)};
        } catch (const std::exception& e) { return {false, e.what()}; }
    });

    auto t0 = std::chrono::steady_clock::now();
    if (fut.wait_for(timeout) == std::future_status::timeout) {
        r.alive = false;
        r.error = "timeout (" + std::to_string(timeout.count()) + "s)";
        r.latency_ms = (long)(timeout.count() * 1000);
        return r;
    }
    auto [ok, msg] = fut.get();
    r.latency_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    r.alive = ok;
    if (!ok) r.error = msg;
    return r;
}

ProviderAutoPlanner::PlanResult
ProviderAutoPlanner::build_plan(const std::vector<ProbeResult>& results,
                                  const std::string& prefix) const {
    log_msg(LogLevel::LOG_INFO, "AutoPlanner",
        prefix + ": " + std::to_string(results.size()) + " candidates probed");
    for (const auto& r : results) {
        std::string line = (r.alive ? "OK  " : "FAIL") + std::string(" [") + r.tier + "] "
            + r.name + " " + std::to_string(r.latency_ms) + "ms";
        if (!r.alive) line += " — " + r.error.substr(0, 60);
        log_msg(r.alive ? LogLevel::LOG_INFO : LogLevel::LOG_WARN, "AutoPlanner", line);
    }

    using PC = std::pair<long, ProviderConfig>;
    std::vector<PC> strong_v, fast_v;
    for (size_t i = 0; i < candidates_.size(); ++i) {
        if (!results[i].alive) continue;
        auto entry = PC{results[i].latency_ms, candidates_[i].config};
        (candidates_[i].tier == "strong" ? strong_v : fast_v).push_back(entry);
    }
    auto by_lat = [](const PC& a, const PC& b){ return a.first < b.first; };
    std::sort(strong_v.begin(), strong_v.end(), by_lat);
    std::sort(fast_v.begin(),   fast_v.end(),   by_lat);

    if (strong_v.empty() && !fast_v.empty())  strong_v = fast_v;
    if (fast_v.empty()   && !strong_v.empty()) fast_v   = strong_v;

    PlanResult plan;
    plan.probe_results = results;
    if (strong_v.empty()) {
        plan.success = false;
        plan.error   = "No providers responded";
        log_msg(LogLevel::LOG_ERROR, "AutoPlanner", prefix + ": " + plan.error);
        return plan;
    }

    auto make_tier = [](const std::vector<PC>& v) {
        TierConfig t;
        t.primary = v[0].second;
        for (size_t i = 1; i < v.size(); ++i) t.fallbacks.push_back(v[i].second);
        t.failure_threshold = 3; t.recovery_timeout_sec = 60.0;
        return t;
    };
    plan.config  = EngineConfig::with_fallbacks(make_tier(strong_v), make_tier(fast_v));
    plan.success = true;
    for (const auto& [_, c] : strong_v) plan.alive_strong.push_back(c.model);
    for (const auto& [_, c] : fast_v)   plan.alive_fast.push_back(c.model);
    log_msg(LogLevel::LOG_INFO, "AutoPlanner",
        prefix + " → ORCHESTRATOR=" + plan.alive_strong[0] + "  SUBAGENT=" + plan.alive_fast[0]);
    return plan;
}

ProviderAutoPlanner::PlanResult
ProviderAutoPlanner::probe_and_plan(std::chrono::seconds timeout) const {
    // L4: Cap concurrent probes at 8 to avoid thread exhaustion
    constexpr int MAX_PARALLEL = 8;
    std::vector<ProbeResult> results;
    results.reserve(candidates_.size());
    for (size_t i = 0; i < candidates_.size(); i += MAX_PARALLEL) {
        std::vector<std::future<ProbeResult>> futs;
        size_t end = std::min(i + (size_t)MAX_PARALLEL, candidates_.size());
        for (size_t j = i; j < end; ++j)
            futs.push_back(std::async(std::launch::async,
                [this, &c=candidates_[j], timeout]() { return probe_one(c, timeout); }));
        for (auto& f : futs) results.push_back(f.get());
    }
    last_results_ = results;
    return build_plan(results, "PROBE");
}

ProviderAutoPlanner::PlanResult
ProviderAutoPlanner::repair(std::chrono::seconds timeout) const {
    log_msg(LogLevel::LOG_WARN, "AutoPlanner", "Re-probing after provider failure...");
    return probe_and_plan(timeout);
}

void ProviderAutoPlanner::print_last_report() const {
    if (last_results_.empty()) { log_msg(LogLevel::LOG_INFO, "AutoPlanner", "(no probe run yet)"); return; }
    for (const auto& r : last_results_) {
        std::string line = (r.alive ? "OK" : "FAIL") + std::string(" [") + r.tier + "] "
            + r.name + "  " + std::to_string(r.latency_ms) + "ms";
        if (!r.alive) line += " — " + r.error;
        log_msg(LogLevel::LOG_INFO, "AutoPlanner", line);
    }
}

// ════════════════════════════════════════════════════════════════
// WorkflowEngine — health_check, run_with_recovery, run_stream
// ════════════════════════════════════════════════════════════════

std::vector<ProbeResult> WorkflowEngine::health_check() const {
    ProviderAutoPlanner planner;
    planner.add_candidate("orchestrator/primary",
                           config_.orchestrator.primary, "strong", 1);
    for (size_t i = 0; i < config_.orchestrator.fallbacks.size(); ++i)
        planner.add_candidate("orchestrator/fb" + std::to_string(i),
                               config_.orchestrator.fallbacks[i], "strong", (int)i+2);
    planner.add_candidate("subagent/primary",
                           config_.subagent.primary, "fast", 1);
    for (size_t i = 0; i < config_.subagent.fallbacks.size(); ++i)
        planner.add_candidate("subagent/fb" + std::to_string(i),
                               config_.subagent.fallbacks[i], "fast", (int)i+2);
    return planner.probe_and_plan().probe_results;
}

WorkflowResult WorkflowEngine::run_with_recovery(const std::string& task,
                                                   ProviderAutoPlanner& planner,
                                                   int max_repair_attempts) {
    WorkflowResult result = run(task);
    if (result.success) return result;

    // A3: 使用类型安全的 lambda，而非字符串匹配
    auto is_provider_err = [](const std::string& msg) {
        return msg.find("providers exhausted") != std::string::npos
            || msg.find("curl:") != std::string::npos;
    };

    for (int attempt = 0; attempt < max_repair_attempts
                        && is_provider_err(result.error_message); ++attempt) {
        log_msg(LogLevel::LOG_WARN, "Recovery",
            "Attempt " + std::to_string(attempt+1) + "/" + std::to_string(max_repair_attempts));
        auto plan = planner.repair();
        if (!plan.success) { result.error_message = "Recovery failed: " + plan.error; break; }
        config_.orchestrator = plan.config.orchestrator;
        config_.subagent     = plan.config.subagent;
        llm_      = std::make_unique<LLMClient>(config_.orchestrator, config_.subagent);
        planner_  = std::make_unique<WorkflowPlanner>(*llm_);
        executor_ = std::make_unique<WorkflowExecutor>(*llm_, *tools_, config_.max_concurrency, metrics_);
        result = run(task);
        if (result.success) {
            log_msg(LogLevel::LOG_INFO, "Recovery", "Success on attempt " + std::to_string(attempt+1));
            return result;
        }
    }
    return result;
}

WorkflowResult WorkflowEngine::run_stream(const std::string& task,
                                           StreamCallback on_chunk) {
    reset_cancel();
    llm_->reset_usage();
    WorkflowResult res;
    auto t0 = std::chrono::steady_clock::now();
    auto wf_id = "wfs_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count() % 1000000);
    metrics_->record({MetricEvent::Kind::WORKFLOW_START, wf_id, "", "", "", 0, true, task});
    try {
        // Input guardrails
        json task_json = {{"task", task}};
        for (const auto& guard : input_guardrails_) {
            auto err = guard(task_json);
            if (err) throw GuardrailError("Input guardrail: " + *err);
        }

        auto tools = tools_->list_tools();
        // Plan caching
        std::string cache_key;
        WorkflowPlan plan;
        if (cache_enabled_) {
            cache_key = PlanCache::normalize_key(task, tools);
            if (plan_cache_.has(cache_key)) {
                plan = plan_cache_.get(cache_key);
            } else {
                WorkflowContext ctx;
                plan = planner_->plan(task, tools, ctx);
                plan_cache_.put(cache_key, plan);
            }
        } else {
            WorkflowContext ctx;
            plan = planner_->plan(task, tools, ctx);
        }

        auto leaves = plan.leaf_steps();
        std::set<std::string> leaf_ids;
        for (const auto& l : leaves) leaf_ids.insert(l.id);

        WorkflowState state;
        state.task_input = {{"task", task}};
        bool streamed_final = false;

        for (const auto& batch : plan.topological_batches()) {
            if (cancel_->load()) throw WorkflowCancelledError("Stream workflow cancelled");
            if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_)
                throw WorkflowCancelledError("Stream workflow deadline exceeded");
            bool is_final = (batch.size() == 1)
                         && leaf_ids.count(batch[0].id)
                         && (batch[0].type == StepType::LLM);

            if (is_final) {
                streamed_final = true;
                const auto& step   = batch[0];
                auto        inputs = state.resolve_inputs(step.inputs);
                std::string prompt = step.action;
                if (!inputs.empty()) prompt += "\n\nContext:\n" + inputs.dump(2);
                std::string acc;
                double temp = (step.temperature >= 0.0) ? step.temperature : 0.0;
                llm_->complete_as_stream(step.model_tier, prompt, step.system_prompt, temp,
                    [&](const std::string& chunk){ acc += chunk; on_chunk(chunk); });
                state.step_outputs[step.id] = acc;
            } else {
                using FT = std::tuple<bool, json, std::vector<TraceEntry>>;
                std::vector<std::pair<Step, std::future<FT>>> futs;
                for (const auto& step : batch)
                    futs.emplace_back(step,
                        executor_->submit_task([this, step, &state]() -> FT {
                            std::vector<TraceEntry> tr;
                            try { return {true, executor_->run_step_pub(step, state, tr), tr}; }
                            catch (const std::exception& e) {
                                tr.push_back({step.id,"","failed","",0,e.what()});
                                return {false,{{"error",e.what()}},tr};
                            }
                        }));
                for (auto& [step, fut] : futs) {
                    auto [ok, rj, tr] = fut.get();
                    for (auto& te : tr) state.record_trace(std::move(te));
                    if (ok) {
                        state.step_outputs[step.id] = rj;
                    } else {
                        switch (step.on_error) {
                        case OnError::FAIL:
                            throw StepExecutionError(step.id, step.description,
                                                  rj.value("error",""));
                        case OnError::SKIP:
                            state.errors[step.id]      = rj.value("error","");
                            state.step_outputs[step.id] = nullptr;
                            break;
                        case OnError::FALLBACK:
                            state.errors[step.id]      = rj.value("error","");
                            state.step_outputs[step.id] = step.fallback;
                            break;
                        }
                    }
                }
            }
        }

        if (leaves.size() == 1)
            res.output = state.step_outputs.count(leaves[0].id)
                         ? state.step_outputs[leaves[0].id] : nullptr;
        else {
            res.output = json::object();
            for (const auto& l : leaves)
                res.output[l.id] = state.step_outputs.count(l.id)
                                   ? state.step_outputs[l.id] : nullptr;
        }
        res.traces = state.traces; res.step_count = (int)plan.steps.size();
        res.success = true;
        // Output guardrails
        for (const auto& guard : output_guardrails_) {
            auto err = guard(res.output);
            if (err) throw GuardrailError("Output guardrail: " + *err);
        }
        if (!streamed_final && res.has_output() && !res.output.is_null()) {
            std::string _out = res.output.is_string()
                ? res.output.get<std::string>() : res.output.dump(2);
            std::istringstream _iss(_out); std::string _w;
            while (std::getline(_iss, _w, ' ')) on_chunk(_w + " ");
        }
    } catch (const std::exception& e) {
        res.success = false; res.error_message = e.what();
    }
    res.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    res.token_usage = llm_->total_usage();
    metrics_->record({MetricEvent::Kind::WORKFLOW_END, wf_id, "", "", "",
                      res.duration_ms, res.success, res.error_message});
    return res;
}


// ════════════════════════════════════════════════════════════════
// Agent Loop — ReACT 实现
// ════════════════════════════════════════════════════════════════

static const char* AGENT_SYS = R"SYS(
You are an AI agent that solves tasks step by step using available tools.
Each response MUST be a single JSON object (no markdown, no explanation outside JSON):
{
  "thought": "your reasoning about what to do next",
  "action": {
    "type": "tool_call" | "final_answer" | "loop_back",
    "tool_name": "name_of_tool",           // required for tool_call
    "tool_args": {"key": "value"},         // required for tool_call
    "response": "your final answer here",  // required for final_answer
    "reason": "why reconsidering"          // required for loop_back
  }
}

Rules:
- Use tool_call to gather information you need
- Use final_answer ONLY when you have enough information to answer completely
- Use loop_back if your current approach is clearly not working
- Never fabricate tool results — always call the tool to get real data
- Be concise in thought; be complete in final_answer
- IMPORTANT: When only 1-2 iterations remain, you MUST provide a final_answer using whatever information you have gathered so far. Do NOT waste the last iterations on more tool calls.
- If repeated tool calls return similar unhelpful results, stop and give a final_answer with what you know.
)SYS";

static AgentAction parse_agent_action(const std::string& raw) {
    AgentAction action;
    try {
        auto j = WorkflowPlanner::extract_json(raw);
        action.thought = j.value("thought", "");
        auto a = j["action"];
        const std::string type = a.value("type", "final_answer");
        if (type == "tool_call") {
            action.type      = AgentAction::Type::TOOL_CALL;
            action.tool_name = a.value("tool_name", "");
            action.tool_args = a.value("tool_args", json::object());
        } else if (type == "loop_back") {
            action.type   = AgentAction::Type::LOOP_BACK;
            action.reason = a.value("reason", "");
        } else if (type == "handoff") {
            action.type         = AgentAction::Type::HANDOFF;
            action.target_agent = a.value("target_agent", a.value("target", ""));
            action.reason       = a.value("reason", "");
        } else {
            action.type     = AgentAction::Type::FINAL_ANSWER;
            action.response = a.value("response", raw);
        }
    } catch (...) {
        // If JSON parsing fails, treat entire response as final answer
        action.type     = AgentAction::Type::FINAL_ANSWER;
        action.response = raw;
        action.thought  = "(parse error — treating as direct answer)";
    }
    return action;
}

struct LoopDetector {
    struct Fingerprint {
        std::string tool_name;
        size_t      args_hash = 0;
        size_t      output_hash = 0;
        bool operator==(const Fingerprint& o) const {
            return tool_name == o.tool_name && args_hash == o.args_hash;
        }
    };
    std::deque<Fingerprint> window;
    static constexpr size_t MAX_WINDOW = 10;

    void record(const std::string& tool, const json& args, const json& obs) {
        Fingerprint fp{tool, std::hash<std::string>{}(args.dump()),
                       std::hash<std::string>{}(obs.dump())};
        window.push_back(fp);
        if (window.size() > MAX_WINDOW) window.pop_front();
    }

    bool exact_repeat(int n = 3) const {
        if ((int)window.size() < n) return false;
        for (int i = 1; i < n; ++i)
            if (!(window[window.size()-1-i] == window.back())) return false;
        return true;
    }

    bool outputs_converged(int n = 3) const {
        if ((int)window.size() < n) return false;
        size_t h = window[window.size()-n].output_hash;
        for (int i = (int)window.size()-n+1; i < (int)window.size(); ++i)
            if (window[i].output_hash != h) return false;
        return true;
    }

    bool is_stuck() const {
        return exact_repeat(3) || (exact_repeat(2) && outputs_converged(2));
    }
};

static std::string mask_old_observations(const std::string& history, int keep_recent = 6) {
    size_t result_count = 0;
    size_t tmp = 0;
    while ((tmp = history.find("Result: ", tmp)) != std::string::npos) { ++result_count; tmp += 8; }
    if (result_count <= (size_t)keep_recent) return history;
    size_t mask_count = result_count - keep_recent;
    size_t seen = 0;
    std::string out;
    std::istringstream iss(history);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("Result: ") == 0 && seen < mask_count) {
            out += "Result: [masked, " + std::to_string(line.size()) + " chars]\n";
            ++seen;
        } else {
            out += line + "\n";
            if (line.find("Result: ") == 0) ++seen;
        }
    }
    return out;
}

static std::string build_agent_prompt(const std::string& task,
                                       std::vector<ToolDef> tools,
                                       const std::string& history,
                                       int iteration,
                                       int max_iterations) {
    // Sort tools alphabetically for stable cache-friendly ordering
    std::sort(tools.begin(), tools.end(),
        [](const ToolDef& a, const ToolDef& b) { return a.name < b.name; });
    json tool_list = json::array();
    for (const auto& t : tools)
        tool_list.push_back({{"name",t.name},{"description",t.description},{"inputs",t.input_schema}});

    std::string prompt = "Available tools:\n" + tool_list.dump(2) + "\n\n";
    prompt += "Task: " + task + "\n";
    int remaining = max_iterations - iteration;
    prompt += "[Iteration " + std::to_string(iteration + 1) + "/" + std::to_string(max_iterations)
            + ", " + std::to_string(remaining) + " remaining]\n";
    if (remaining <= 2)
        prompt += "WARNING: Running low on iterations. Provide a final_answer NOW with whatever information you have.\n";
    // Convergence injection handled by caller via history prefix
    if (!history.empty())
        prompt += "\nWork done so far:\n" + history;
    prompt += "\nRespond with your next action as a JSON object.";
    return prompt;
}

// ════════════════════════════════════════════════════════════════
// Native Tool Calling Agent Loop
// ════════════════════════════════════════════════════════════════

AgentResult WorkflowEngine::run_agent_native(const std::string& task, int max_iterations,
                                               std::function<void(const AgentStep&)> on_step) {
    // Auto-fallback if provider doesn't support native tools
    if (!llm_->supports_native_tools(ModelTier::ORCHESTRATOR)) {
        log_msg(LogLevel::LOG_INFO, "Agent", "Provider lacks native tools, falling back to prompt-based");
        return run_agent(task, max_iterations, on_step);
    }

    llm_->reset_usage();
    reset_cancel();
    AgentResult result;
    result.max_iterations = max_iterations;
    auto t0 = std::chrono::steady_clock::now();
    auto wf_id = "native_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count() % 1000000);
    metrics_->record({MetricEvent::Kind::WORKFLOW_START, wf_id, "", "", "", 0, true, task});

    // Input guardrails
    json task_json = {{"task", task}};
    for (const auto& guard : input_guardrails_) {
        auto err = guard(task_json);
        if (err) {
            result.error = "Input guardrail: " + *err;
            metrics_->record({MetricEvent::Kind::WORKFLOW_END, wf_id, "", "", "", 0, false, result.error});
            return result;
        }
    }

    auto tools = tools_->list_tools();
    // Sort tools alphabetically for stable cache-friendly ordering
    std::sort(tools.begin(), tools.end(),
        [](const ToolDef& a, const ToolDef& b) { return a.name < b.name; });
    LoopDetector loop_detector;

    static const char* NATIVE_AGENT_SYS =
        "You are a helpful AI agent. Use the provided tools to gather information. "
        "When you have enough data, respond with a text message containing your final answer. "
        "You may call multiple tools in parallel when appropriate.";
    std::string sys_prompt = custom_agent_prompt_.empty() ? NATIVE_AGENT_SYS : custom_agent_prompt_;

    std::vector<ChatMessage> messages;
    messages.push_back({"system", sys_prompt, {}, "", ""});
    messages.push_back({"user", task, {}, "", ""});

    for (int iter = 0; iter < max_iterations; ++iter) {
        if (cancel_->load()) { result.error = "Agent cancelled"; break; }
        if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_) {
            result.error = "Agent deadline exceeded"; break;
        }
        auto step_t0 = std::chrono::steady_clock::now();
        AgentStep step;
        step.iteration = iter + 1;

        LLMResponse resp;
        try {
            resp = llm_->complete_chat(ModelTier::ORCHESTRATOR, messages, tools, 0.0);
        } catch (const std::exception& e) {
            result.error = std::string("LLM error: ") + e.what();
            break;
        }

        if (resp.has_tool_calls()) {
            // Build assistant message with ALL tool_calls for conversation history
            json tc_json = json::array();
            for (const auto& c : resp.tool_calls) {
                tc_json.push_back({
                    {"id", c.id}, {"type", "function"},
                    {"function", {{"name", c.name}, {"arguments", c.arguments.dump()}}}
                });
            }
            messages.push_back({"assistant", resp.content, tc_json, "", ""});

            // Execute ALL tool calls (parallel function calling)
            std::string tool_names;
            json all_obs = json::array();
            for (const auto& tc : resp.tool_calls) {
                json obs;
                try { obs = tools_->call(tc.name, tc.arguments); }
                catch (const std::exception& e) {
                    obs = {{"error", std::string("tool failed: ") + e.what()}};
                }
                // Append tool result message for each call
                messages.push_back({"tool", obs.dump(), {}, tc.id, tc.name});
                all_obs.push_back(obs);
                if (!tool_names.empty()) tool_names += "+";
                tool_names += tc.name;

                loop_detector.record(tc.name, tc.arguments, obs);
            }

            // Record step with first tool call's info (for backward compat)
            const auto& first_tc = resp.tool_calls[0];
            step.thought = resp.content.empty() ? "Using " + tool_names : resp.content;
            step.action.type = AgentAction::Type::TOOL_CALL;
            step.action.tool_name = first_tc.name;
            step.action.tool_args = first_tc.arguments;
            step.action.thought = step.thought;
            step.observation = (all_obs.size() == 1) ? all_obs[0] : all_obs;

            // Convergence detection
            if (loop_detector.is_stuck()) {
                messages.push_back({"user",
                    "LOOP DETECTED: You have called tools repeatedly with similar results. "
                    "Please provide your final answer now using the information already gathered.",
                    {}, "", ""});
            }

        } else {
            step.thought = "Providing final answer";
            step.action.type = AgentAction::Type::FINAL_ANSWER;
            step.action.response = resp.content;
            step.action.thought = step.thought;
            step.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - step_t0).count();
            result.steps.push_back(step);
            result.success = true;
            result.final_answer = resp.content;
            result.iterations_used = iter + 1;
            if (on_step) on_step(step);
            break;
        }

        step.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - step_t0).count();
        result.steps.push_back(step);
        result.traces.push_back({
            "native_iter_" + std::to_string(iter+1), "tool", "ok", "",
            step.duration_ms, "",
            g_last_token_usage.input_tokens, g_last_token_usage.output_tokens
        });
        if (on_step) on_step(step);

        // Observation masking: mask old tool result content to save tokens
        long total_msg_tokens = 0;
        for (const auto& m : messages) total_msg_tokens += estimate_tokens(m.content);
        if (total_msg_tokens > 8000) {
            // Mask old tool results AND assistant reasoning (keep last 6 of each)
            int tool_count = 0, asst_count = 0;
            for (const auto& m : messages) {
                if (m.role == "tool") ++tool_count;
                else if (m.role == "assistant") ++asst_count;
            }
            int tool_mask = std::max(0, tool_count - 6);
            int asst_mask = std::max(0, asst_count - 6);
            int t_seen = 0, a_seen = 0, masked = 0;
            for (auto& m : messages) {
                if (m.role == "tool" && t_seen++ < tool_mask) {
                    m.content = "[masked, " + std::to_string(m.content.size()) + " chars]";
                    ++masked;
                } else if (m.role == "assistant" && a_seen++ < asst_mask && !m.content.empty()) {
                    m.content = m.content.substr(0, std::min(m.content.size(), (size_t)100))
                              + "... [masked]";
                    ++masked;
                }
            }
            if (masked > 0)
                log_msg(LogLevel::LOG_DEBUG, "Agent",
                    "masked " + std::to_string(masked) + " old messages");
        }
    }

    if (!result.success && result.error.empty()) {
        result.error = "Max iterations reached without final answer";
        result.iterations_used = max_iterations;
    }
    result.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    result.token_usage = llm_->total_usage();
    // Output guardrails
    if (result.success) {
        for (const auto& guard : output_guardrails_) {
            auto err = guard(json(result.final_answer));
            if (err) { result.success = false; result.error = "Output guardrail: " + *err; }
        }
    }
    metrics_->record({MetricEvent::Kind::WORKFLOW_END, wf_id, "", "", "",
                      result.duration_ms, result.success, result.error});
    return result;
}

AgentResult WorkflowEngine::run_agent(const std::string& task, int max_iterations,
                                           std::function<void(const AgentStep&)> on_step) {
    llm_->reset_usage();
    reset_cancel();
    AgentResult result;
    result.max_iterations = max_iterations;
    auto t0 = std::chrono::steady_clock::now();
    auto wf_id = "agent_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count() % 1000000);
    metrics_->record({MetricEvent::Kind::WORKFLOW_START, wf_id, "", "", "", 0, true, task});

    // Input guardrails
    json task_json = {{"task", task}};
    for (const auto& guard : input_guardrails_) {
        auto err = guard(task_json);
        if (err) {
            result.error = "Input guardrail: " + *err;
            result.duration_ms = 0;
            metrics_->record({MetricEvent::Kind::WORKFLOW_END, wf_id, "", "", "", 0, false, result.error});
            return result;
        }
    }

    auto tools = tools_->list_tools();
    std::string history;
    LoopDetector loop_detector;

    for (int iter = 0; iter < max_iterations; ++iter) {
        if (cancel_->load()) {
            result.error = "Agent cancelled";
            break;
        }
        if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_) {
            result.error = "Agent deadline exceeded";
            break;
        }
        auto step_t0 = std::chrono::steady_clock::now();
        AgentStep step;
        step.iteration = iter + 1;

        // ── Call LLM for next action ──────────────────────────
        std::string raw;
        try {
            const char* sys_to_use = custom_agent_prompt_.empty() ? AGENT_SYS : custom_agent_prompt_.c_str();
            raw = llm_->complete_as(ModelTier::ORCHESTRATOR,
                                    build_agent_prompt(task, tools, history, iter, max_iterations),
                                    sys_to_use, 0.0);
        } catch (const std::exception& e) {
            result.error = std::string("LLM error on iteration ") +
                           std::to_string(iter+1) + ": " + e.what();
            break;
        }

        // ── Parse action ──────────────────────────────────────
        AgentAction action = parse_agent_action(raw);
        step.thought = action.thought;
        step.action  = action;

        // ── Dispatch ──────────────────────────────────────────
        if (action.type == AgentAction::Type::FINAL_ANSWER) {
            step.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - step_t0).count();
            result.steps.push_back(step);
            result.success        = true;
            result.final_answer   = action.response;
            result.iterations_used = iter + 1;
            break;
        }

        if (action.type == AgentAction::Type::TOOL_CALL) {
            // Execute tool
            json obs;
            try {
                obs = tools_->call(action.tool_name, action.tool_args);
            } catch (const std::exception& e) {
                obs = {{"error", std::string("tool '") + action.tool_name + "' failed: " + e.what()}};
            }
            step.observation = obs;

            std::string obs_str = obs.dump();
            if (obs_str.size() > 800) obs_str = obs_str.substr(0, 800) + "...[truncated]";
            history += "[iter " + std::to_string(iter+1) + "]\n"
                    +  "Thought: " + action.thought + "\n"
                    +  "Called: " + action.tool_name + "(" + action.tool_args.dump() + ")\n"
                    +  "Result: " + obs_str + "\n\n";
            if (estimate_tokens(history) > 2000)
                history = mask_old_observations(history);

            // 收敛检测：记录指纹，检测循环
            loop_detector.record(action.tool_name, action.tool_args, obs);
            if (loop_detector.is_stuck()) {
                history += "\n*** LOOP DETECTED: You have called " + action.tool_name
                         + " repeatedly with similar arguments and results. "
                           "You MUST provide a final_answer NOW using the information already gathered. ***\n\n";
            }

        } else if (action.type == AgentAction::Type::LOOP_BACK) {
            history += "[iter " + std::to_string(iter+1) + " — reconsidering]\n"
                    +  "Thought: " + action.thought + "\n"
                    +  "Reason: " + action.reason + "\n\n";
        }

        step.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - step_t0).count();
        result.steps.push_back(step);
        // Agent trace entry
        std::string trace_type = (action.type == AgentAction::Type::TOOL_CALL) ? "tool" :
                                 (action.type == AgentAction::Type::LOOP_BACK) ? "loop_back" : "llm";
        result.traces.push_back({
            "agent_iter_" + std::to_string(iter+1), trace_type,
            "ok", "", step.duration_ms, "",
            g_last_token_usage.input_tokens, g_last_token_usage.output_tokens
        });
        if (on_step) on_step(step);
    }

    // Max iterations reached without final answer
    if (!result.success && result.error.empty()) {
        result.error = "Max iterations (" + std::to_string(max_iterations) +
                       ") reached without final answer";
        result.iterations_used = max_iterations;
    }

    result.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    result.token_usage = llm_->total_usage();

    // Output guardrails (on success)
    if (result.success) {
        for (const auto& guard : output_guardrails_) {
            auto err = guard(json(result.final_answer));
            if (err) { result.success = false; result.error = "Output guardrail: " + *err; }
        }
    }
    metrics_->record({MetricEvent::Kind::WORKFLOW_END, wf_id, "", "", "",
                      result.duration_ms, result.success, result.error});
    return result;
}

static std::string build_multiagent_prompt(const std::string& task,
                                            const AgentDef& agent,
                                            const std::vector<ToolDef>& tools,
                                            const std::string& history,
                                            int iteration, int max_iterations) {
    json tool_list = json::array();
    for (const auto& t : tools)
        tool_list.push_back({{"name",t.name},{"description",t.description},{"inputs",t.input_schema}});

    std::string prompt = "You are agent '" + agent.name + "'.\n";
    if (!agent.handoff_targets.empty()) {
        prompt += "You can hand off to these agents: ";
        for (const auto& t : agent.handoff_targets) prompt += t + " ";
        prompt += "\nTo hand off, respond with action type 'handoff' and a 'target_agent' field.\n";
    }
    prompt += "Available tools:\n" + tool_list.dump(2) + "\n\n";
    prompt += "Task: " + task + "\n";
    int remaining = max_iterations - iteration;
    prompt += "[Iteration " + std::to_string(iteration + 1) + "/" + std::to_string(max_iterations)
            + ", " + std::to_string(remaining) + " remaining]\n";
    if (remaining <= 2)
        prompt += "WARNING: Running low on iterations. Provide a final_answer NOW.\n";
    if (!history.empty()) prompt += "\nWork done so far:\n" + history;
    prompt += "\nRespond with your next action as a JSON object.";
    return prompt;
}

AgentResult WorkflowEngine::run_multi_agent(const std::string& task,
                                             const std::vector<AgentDef>& agents,
                                             const std::string& start_agent,
                                             int max_iterations,
                                             std::function<void(const AgentStep&)> on_step) {
    llm_->reset_usage();
    reset_cancel();
    AgentResult result;
    result.max_iterations = max_iterations;
    auto t0 = std::chrono::steady_clock::now();

    // Input guardrails
    json task_json = {{"task", task}};
    for (const auto& guard : input_guardrails_) {
        auto err = guard(task_json);
        if (err) { result.error = "Input guardrail: " + *err; return result; }
    }

    std::map<std::string, AgentDef> agent_map;
    for (const auto& a : agents) agent_map[a.name] = a;
    if (!agent_map.count(start_agent)) {
        result.error = "Start agent '" + start_agent + "' not found";
        return result;
    }

    std::string current = start_agent;
    std::string history;
    auto all_tools = tools_->list_tools();

    for (int iter = 0; iter < max_iterations; ++iter) {
        if (cancel_->load()) { result.error = "Multi-agent cancelled"; break; }
        const AgentDef& agent = agent_map[current];

        // 过滤当前 agent 允许的工具
        std::vector<ToolDef> agent_tools;
        if (agent.allowed_tools.empty()) {
            agent_tools = all_tools;
        } else {
            for (const auto& t : all_tools)
                if (std::find(agent.allowed_tools.begin(), agent.allowed_tools.end(), t.name)
                    != agent.allowed_tools.end())
                    agent_tools.push_back(t);
        }

        auto step_t0 = std::chrono::steady_clock::now();
        AgentStep step;
        step.iteration = iter + 1;

        std::string raw;
        try {
            std::string sys = agent.system_prompt.empty() ? AGENT_SYS : agent.system_prompt;
            raw = llm_->complete_as(agent.model_tier,
                build_multiagent_prompt(task, agent, agent_tools, history, iter, max_iterations),
                sys, 0.0);
        } catch (const std::exception& e) {
            result.error = std::string("LLM error: ") + e.what();
            break;
        }

        AgentAction action = parse_agent_action(raw);
        step.thought = action.thought;
        step.action  = action;

        if (action.type == AgentAction::Type::FINAL_ANSWER) {
            step.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - step_t0).count();
            result.steps.push_back(step);
            result.success = true;
            result.final_answer = action.response;
            result.iterations_used = iter + 1;
            // Output guardrails
            for (const auto& guard : output_guardrails_) {
                auto err = guard(json(action.response));
                if (err) { result.success = false; result.error = "Output guardrail: " + *err; }
            }
            if (on_step) on_step(step);
            break;
        }

        if (action.type == AgentAction::Type::HANDOFF) {
            std::string target = action.target_agent;
            if (agent_map.count(target)) {
                history += "[iter " + std::to_string(iter+1) + " — handoff]\n"
                        +  current + " → " + target + ": " + action.reason + "\n\n";
                current = target;
            } else {
                history += "[iter " + std::to_string(iter+1) + "]\nInvalid handoff target '"
                        + target + "'. Continue with current agent.\n\n";
            }
        } else if (action.type == AgentAction::Type::TOOL_CALL) {
            json obs;
            try {
                obs = tools_->call(action.tool_name, action.tool_args);
            } catch (const std::exception& e) {
                obs = {{"error", std::string("tool failed: ") + e.what()}};
            }
            step.observation = obs;
            std::string obs_str = obs.dump();
            if (obs_str.size() > 800) obs_str = obs_str.substr(0, 800) + "...[truncated]";
            history += "[iter " + std::to_string(iter+1) + " | " + current + "]\n"
                    +  "Thought: " + action.thought + "\n"
                    +  "Called: " + action.tool_name + "(" + action.tool_args.dump() + ")\n"
                    +  "Result: " + obs_str + "\n\n";
            if (estimate_tokens(history) > 2000)
                history = mask_old_observations(history);
        }

        step.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - step_t0).count();
        result.steps.push_back(step);
        if (on_step) on_step(step);
    }

    if (!result.success && result.error.empty()) {
        result.error = "Max iterations reached without final answer";
        result.iterations_used = max_iterations;
    }
    result.duration_ms = (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    result.token_usage = llm_->total_usage();
    return result;
}

// ════════════════════════════════════════════════════════════════
// MCP HttpTransport — Streamable HTTP (POST JSON-RPC to server)
// ════════════════════════════════════════════════════════════════

HttpTransport::HttpTransport(const std::string& url, const std::string& api_key)
    : url_(url), api_key_(api_key) {}

void HttpTransport::send(const json& message) {
    CurlHandle curl;
    CURL* c = curl.get();
    std::string body = message.dump();
    std::string resp_body;
    struct curl_slist* h = nullptr;
    h = curl_slist_append(h, "Content-Type: application/json");
    h = curl_slist_append(h, "Accept: application/json");
    if (!api_key_.empty())
        h = curl_slist_append(h, ("Authorization: Bearer " + api_key_).c_str());
    auto write_cb = [](char* p, size_t s, size_t n, void* u) -> size_t {
        static_cast<std::string*>(u)->append(p, s * n); return s * n;
    };
    curl_easy_setopt(c, CURLOPT_URL,            url_.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER,     h);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS,     body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE,  (long)body.size());
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION,  +write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,      &resp_body);
    curl_easy_setopt(c, CURLOPT_TIMEOUT,        30L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL,       1L);
    CURLcode rc = curl_easy_perform(c);
    curl_slist_free_all(h);
    if (rc != CURLE_OK)
        throw McpError(std::string("MCP HTTP: ") + curl_easy_strerror(rc));
    if (resp_body.empty()) {
        pending_response_ = json::object();
        has_pending_ = true;
    } else {
        try {
            pending_response_ = json::parse(resp_body);
            has_pending_ = true;
        } catch (...) {
            throw McpError("MCP HTTP: invalid JSON response");
        }
    }
}

json HttpTransport::receive() {
    if (!has_pending_) throw McpError("MCP HTTP: no pending response");
    has_pending_ = false;
    return pending_response_;
}

void HttpTransport::close() {}

// ════════════════════════════════════════════════════════════════
// MCP Client — stdio transport + JSON-RPC 2.0
// ════════════════════════════════════════════════════════════════

#ifdef _WIN32
#include <windows.h>

StdioTransport::StdioTransport(const std::string& command, const std::vector<std::string>& args) {
    std::string cmdline = command;
    for (const auto& a : args) cmdline += " " + a;

    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE child_stdin_rd, child_stdin_wr, child_stdout_rd, child_stdout_wr;
    CreatePipe(&child_stdin_rd, &child_stdin_wr, &sa, 0);
    SetHandleInformation(child_stdin_wr, HANDLE_FLAG_INHERIT, 0);
    CreatePipe(&child_stdout_rd, &child_stdout_wr, &sa, 0);
    SetHandleInformation(child_stdout_rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = child_stdin_rd;
    si.hStdOutput = child_stdout_wr;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, const_cast<char*>(cmdline.c_str()),
                        nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(child_stdin_rd); CloseHandle(child_stdin_wr);
        CloseHandle(child_stdout_rd); CloseHandle(child_stdout_wr);
        throw McpError("MCP: failed to spawn: " + command);
    }
    CloseHandle(child_stdin_rd);
    CloseHandle(child_stdout_wr);
    CloseHandle(pi.hThread);
    process_handle_ = pi.hProcess;
    stdin_write_    = child_stdin_wr;
    stdout_read_    = child_stdout_rd;
    closed_ = false;
}

StdioTransport::~StdioTransport() { close(); }

void StdioTransport::send(const json& message) {
    if (closed_) throw McpError("MCP: transport closed");
    std::string line = message.dump() + "\n";
    DWORD written;
    if (!WriteFile(stdin_write_, line.c_str(), (DWORD)line.size(), &written, nullptr)
        || written != (DWORD)line.size())
        throw McpError("MCP: write to subprocess stdin failed");
}

json StdioTransport::receive() {
    if (closed_) throw McpError("MCP: transport closed");
    std::string line;
    char ch;
    DWORD rd;
    while (ReadFile(stdout_read_, &ch, 1, &rd, nullptr) && rd > 0) {
        if (ch == '\n') break;
        if (ch != '\r') line += ch;
    }
    if (line.empty()) throw McpError("MCP: empty response");
    return json::parse(line);
}

void StdioTransport::close() {
    if (closed_) return;
    closed_ = true;
    if (stdin_write_) { CloseHandle(stdin_write_); stdin_write_ = nullptr; }
    if (stdout_read_) { CloseHandle(stdout_read_); stdout_read_ = nullptr; }
    if (process_handle_) {
        WaitForSingleObject(process_handle_, 5000);
        TerminateProcess(process_handle_, 0);
        CloseHandle(process_handle_);
        process_handle_ = nullptr;
    }
}

#else // POSIX

StdioTransport::StdioTransport(const std::string& command, const std::vector<std::string>& args) {
    int stdin_pipe[2], stdout_pipe[2];
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0)
        throw McpError("MCP: pipe failed");
    pid_t pid = fork();
    if (pid < 0) throw McpError("MCP: fork failed");
    if (pid == 0) {
        ::close(stdin_pipe[1]); dup2(stdin_pipe[0], STDIN_FILENO); ::close(stdin_pipe[0]);
        ::close(stdout_pipe[0]); dup2(stdout_pipe[1], STDOUT_FILENO); ::close(stdout_pipe[1]);
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(command.c_str()));
        for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(command.c_str(), argv.data());
        _exit(127);
    }
    ::close(stdin_pipe[0]); ::close(stdout_pipe[1]);
    stdin_fd_  = stdin_pipe[1];
    stdout_fd_ = stdout_pipe[0];
    child_pid_ = pid;
    closed_ = false;
}

StdioTransport::~StdioTransport() { close(); }

void StdioTransport::send(const json& message) {
    if (closed_) throw McpError("MCP: transport closed");
    std::string line = message.dump() + "\n";
    auto written = ::write(stdin_fd_, line.c_str(), line.size());
    if (written < 0 || (size_t)written != line.size())
        throw McpError("MCP: write to subprocess stdin failed");
}

json StdioTransport::receive() {
    if (closed_) throw McpError("MCP: transport closed");
    std::string line;
    char ch;
    while (::read(stdout_fd_, &ch, 1) == 1) {
        if (ch == '\n') break;
        if (ch != '\r') line += ch;
    }
    if (line.empty()) throw McpError("MCP: empty response");
    return json::parse(line);
}

void StdioTransport::close() {
    if (closed_) return;
    closed_ = true;
    if (stdin_fd_ >= 0) { ::close(stdin_fd_); stdin_fd_ = -1; }
    if (stdout_fd_ >= 0) { ::close(stdout_fd_); stdout_fd_ = -1; }
    if (child_pid_ > 0) {
        kill(child_pid_, SIGTERM);
        int status; waitpid(child_pid_, &status, WNOHANG);
        child_pid_ = -1;
    }
}

#endif // _WIN32 / POSIX

// ── McpClient ────────────────────────────────────────────────

McpClient::McpClient(std::unique_ptr<IMcpTransport> transport)
    : transport_(std::move(transport)) {}

McpClient::~McpClient() { close(); }

json McpClient::make_request(const std::string& method, const json& params) {
    json req = {{"jsonrpc","2.0"},{"id",next_id_++},{"method",method}};
    if (!params.empty()) req["params"] = params;
    transport_->send(req);
    auto resp = transport_->receive();
    if (resp.contains("error"))
        throw McpError("MCP error: " + resp["error"].value("message", "unknown"));
    return resp.value("result", json::object());
}

json McpClient::initialize(const std::string& client_name, const std::string& ver) {
    auto result = make_request("initialize", {
        {"protocolVersion", "2025-06-18"},
        {"capabilities", json::object()},
        {"clientInfo", {{"name", client_name}, {"version", ver}}}
    });
    // Send required initialized notification
    transport_->send({{"jsonrpc","2.0"},{"method","notifications/initialized"}});
    initialized_ = true;
    return result;
}

std::vector<ToolDef> McpClient::list_tools() {
    if (!initialized_) throw McpError("MCP: not initialized");
    auto result = make_request("tools/list");
    discovered_tools_.clear();
    if (result.contains("tools")) {
        for (const auto& t : result["tools"]) {
            ToolDef def;
            def.name        = t.value("name", "");
            def.description = t.value("description", "");
            def.input_schema  = t.value("inputSchema", json::object());
            def.output_schema = t.value("outputSchema", json::object());
            discovered_tools_.push_back(def);
        }
    }
    return discovered_tools_;
}

json McpClient::call_tool(const std::string& name, const json& arguments) {
    if (!initialized_) throw McpError("MCP: not initialized");
    auto result = make_request("tools/call", {{"name", name}, {"arguments", arguments}});
    if (result.value("isError", false)) {
        std::string err;
        if (result.contains("content") && !result["content"].empty())
            err = result["content"][0].value("text", "tool error");
        throw ToolError(name, err);
    }
    if (result.contains("structuredContent")) return result["structuredContent"];
    if (result.contains("content") && !result["content"].empty())
        return result["content"][0].value("text", "");
    return result;
}

void McpClient::close() {
    if (transport_) {
        try { transport_->close(); } catch (...) {}
    }
    initialized_ = false;
}

void McpClient::register_all_tools(ToolRegistry& registry) {
    if (discovered_tools_.empty()) list_tools();
    for (const auto& def : discovered_tools_) {
        auto name = def.name;
        auto* client = this;
        registry.register_tool(def, [client, name](const json& params) -> json {
            return client->call_tool(name, params);
        });
    }
}

void WorkflowEngine::connect_mcp(const std::string& command,
                                   const std::vector<std::string>& args) {
    auto transport = std::make_unique<StdioTransport>(command, args);
    auto client = std::make_unique<McpClient>(std::move(transport));
    client->initialize("ariadne", "0.6.0");
    client->register_all_tools(*tools_);
    log_msg(LogLevel::LOG_INFO, "MCP",
        "Connected to " + command + ", registered " + std::to_string(client->list_tools().size()) + " tools");
    mcp_clients_.push_back(std::move(client));
}

void WorkflowEngine::connect_mcp_http(const std::string& url, const std::string& api_key) {
    auto transport = std::make_unique<HttpTransport>(url, api_key);
    auto client = std::make_unique<McpClient>(std::move(transport));
    client->initialize("ariadne", ARIADNE_VERSION);
    client->register_all_tools(*tools_);
    log_msg(LogLevel::LOG_INFO, "MCP",
        "Connected via HTTP to " + url + ", registered " + std::to_string(client->list_tools().size()) + " tools");
    mcp_clients_.push_back(std::move(client));
}

std::string WorkflowEngine::llm_complete(const std::string& prompt,
                                          const std::string& system,
                                          ModelTier tier, double temperature) {
    return llm_->complete_as(tier, prompt, system, temperature);
}

json WorkflowEngine::call_tool(const std::string& name, const json& params) {
    return tools_->call(name, params);
}

void WorkflowEngine::add_input_guardrail(GuardrailFn fn) { input_guardrails_.push_back(std::move(fn)); }
void WorkflowEngine::add_output_guardrail(GuardrailFn fn) { output_guardrails_.push_back(std::move(fn)); }
void WorkflowEngine::add_tool_guardrail(const std::string& tool_name, GuardrailFn fn) {
    tools_->add_guardrail(tool_name, std::move(fn));
}

void WorkflowEngine::set_planner_prompt(const std::string& sys_prompt) {
    custom_planner_prompt_ = sys_prompt;
    planner_->set_system_prompt(sys_prompt);
}
void WorkflowEngine::set_agent_prompt(const std::string& sys_prompt) {
    custom_agent_prompt_ = sys_prompt;
}

void WorkflowEngine::set_metrics(std::shared_ptr<IMetricsCollector> collector) {
    if (collector) metrics_ = std::move(collector);
    else           metrics_ = std::make_shared<NoOpMetrics>();
    executor_ = std::make_unique<WorkflowExecutor>(*llm_, *tools_, config_.max_concurrency, metrics_);
}

void WorkflowEngine::cancel() { cancel_->store(true); }
void WorkflowEngine::reset_cancel() { cancel_->store(false); }
void WorkflowEngine::set_deadline(std::chrono::seconds timeout) {
    has_deadline_ = true;
    deadline_ = std::chrono::steady_clock::now() + timeout;
}
void WorkflowEngine::clear_deadline() { has_deadline_ = false; }

// ════════════════════════════════════════════════════════════════
// D92 — A2A (Agent2Agent) Client 实现
//   复用每请求一个 CurlHandle 的线程安全模式（同 MCP HttpTransport）。
// ════════════════════════════════════════════════════════════════

static size_t a2a_write_cb(char* p, size_t s, size_t n, void* u) {
    static_cast<std::string*>(u)->append(p, s * n);
    return s * n;
}

static std::string a2a_http_get(const std::string& url, const std::string& api_key) {
    CurlHandle curl;
    CURL* c = curl.get();
    std::string resp;
    struct curl_slist* h = nullptr;
    h = curl_slist_append(h, "Accept: application/json");
    if (!api_key.empty())
        h = curl_slist_append(h, ("Authorization: Bearer " + api_key).c_str());
    curl_easy_setopt(c, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER,     h);
    curl_easy_setopt(c, CURLOPT_HTTPGET,        1L);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION,  +a2a_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,      &resp);
    curl_easy_setopt(c, CURLOPT_TIMEOUT,        30L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL,       1L);
    // 不跟随重定向：携带 Authorization 时跨源重定向会把 Bearer 泄露给第三方主机
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 0L);
    CURLcode rc = curl_easy_perform(c);
    curl_slist_free_all(h);
    if (rc != CURLE_OK)
        throw A2AError(std::string("A2A GET: ") + curl_easy_strerror(rc));
    return resp;
}

static json a2a_http_post(const std::string& url, const std::string& api_key, const json& body) {
    CurlHandle curl;
    CURL* c = curl.get();
    std::string b = body.dump();
    std::string resp;
    struct curl_slist* h = nullptr;
    h = curl_slist_append(h, "Content-Type: application/json");
    h = curl_slist_append(h, "Accept: application/json");
    if (!api_key.empty())
        h = curl_slist_append(h, ("Authorization: Bearer " + api_key).c_str());
    curl_easy_setopt(c, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER,     h);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS,     b.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE,  (long)b.size());
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION,  +a2a_write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,      &resp);
    curl_easy_setopt(c, CURLOPT_TIMEOUT,        60L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL,       1L);
    CURLcode rc = curl_easy_perform(c);
    curl_slist_free_all(h);
    if (rc != CURLE_OK)
        throw A2AError(std::string("A2A POST: ") + curl_easy_strerror(rc));
    if (resp.empty())
        throw A2AError("A2A: empty response");
    try { return json::parse(resp); }
    catch (...) { throw A2AError("A2A: invalid JSON response"); }
}

A2AAgentCard A2AClient::fetch_agent_card() {
    std::string url  = base_url_ + "/.well-known/agent-card.json";
    std::string body = a2a_http_get(url, api_key_);
    json j;
    try { j = json::parse(body); }
    catch (...) { throw A2AError("A2A: agent card is not valid JSON"); }
    A2AAgentCard card = A2AAgentCard::from_json(j);
    // 仅在同源（或显式 opt-in）时把后续 RPC 切到 card 声明的端点，
    // 否则继续用 base_url，避免凭据随 Bearer 泄露到不可信主机 / SSRF。
    if (!card.url.empty()) {
        if (endpoint_pivot_allowed(card.url)) {
            endpoint_ = card.url;
        } else {
            log_msg(LogLevel::LOG_WARN, "a2a",
                "ignoring cross-origin AgentCard url '" + card.url +
                "' (base origin " + origin_of(base_url_) +
                "); call set_allow_cross_origin_endpoint(true) to override");
        }
    }
    return card;
}

json A2AClient::send_message(const A2AMessage& msg) {
    json params = {{"message", msg.to_json()}};
    json req    = make_rpc(next_id_++, "message/send", params);
    json resp   = a2a_http_post(endpoint(), api_key_, req);
    if (resp.contains("error") && !resp["error"].is_null()) {
        json e = resp["error"];
        throw A2AError("A2A RPC error: " + e.value("message", e.dump()));
    }
    return resp.value("result", json::object());
}

std::string A2AClient::ask(const std::string& text) {
    json result = send_message(A2AMessage::user_text(text));
    // result 可能是 Message（含 parts）或 Task（含 status）
    if (result.contains("parts"))
        return A2AMessage::from_json(result).text();
    if (result.contains("status")) {
        A2ATask t = A2ATask::from_json(result);
        for (auto it = t.history.rbegin(); it != t.history.rend(); ++it)
            if (it->role == "agent") return it->text();
    }
    if (result.contains("message"))
        return A2AMessage::from_json(result["message"]).text();
    return result.dump();
}

} // namespace ariadne
