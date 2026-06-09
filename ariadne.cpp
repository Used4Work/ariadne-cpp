#ifdef _WIN32
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
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

namespace ariadne {

// ════════════════════════════════════════════════════════════════
// HttpProvider — 共享 HTTP 逻辑 (R1: 消除三份重复代码)
// ════════════════════════════════════════════════════════════════

size_t HttpProvider::write_cb(char* p, size_t s, size_t n, void* u) {
    static_cast<std::string*>(u)->append(p, s * n);
    return s * n;
}

std::string HttpProvider::http_post(const std::string& url,
                                     const std::vector<std::string>& hdrs,
                                     const std::string& body) const {
    CURL* c = curl_.get();
    std::string resp;
    struct curl_slist* h = nullptr;
    for (const auto& s : hdrs) h = curl_slist_append(h, s.c_str());
    curl_easy_setopt(c, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER,    h);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS,    body.c_str());
    curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, (long)body.size());
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA,     &resp);
    curl_easy_setopt(c, CURLOPT_TIMEOUT,       (long)cfg_.timeout_sec);
    CURLcode rc = curl_easy_perform(c);
    curl_slist_free_all(h);
    if (rc != CURLE_OK)
        throw std::runtime_error(std::string("curl: ") + curl_easy_strerror(rc));
    return resp;
}

// ── AnthropicProvider ────────────────────────────────────────────

std::string AnthropicProvider::complete(const std::string& prompt,
                                          const std::string& sys,
                                          double temp) const {
    json body = {{"model", cfg_.model}, {"max_tokens", cfg_.max_tokens},
                  {"temperature", temp},
                  {"messages", {{{"role","user"}, {"content", prompt}}}}};
    if (!sys.empty()) body["system"] = sys;
    std::string base = cfg_.base_url.empty() ? "https://api.anthropic.com" : cfg_.base_url;
    auto raw = http_post(base + "/v1/messages",
        {"Content-Type: application/json",
         "x-api-key: " + cfg_.api_key,
         "anthropic-version: 2023-06-01"},
        body.dump());
    auto j = json::parse(raw);
    if (j.contains("error"))
        throw std::runtime_error("Anthropic: " + j["error"]["message"].get<std::string>());
    return j["content"][0]["text"].get<std::string>();
}

// ── OpenAIChatProvider ───────────────────────────────────────────

static std::string rtrim_slash(const std::string& s) {
    if (!s.empty() && s.back() == '/') return s.substr(0, s.size()-1);
    return s;
}

std::string OpenAIChatProvider::complete(const std::string& prompt,
                                           const std::string& sys,
                                           double temp) const {
    json msgs = json::array();
    if (!sys.empty()) msgs.push_back({{"role","system"}, {"content", sys}});
    msgs.push_back({{"role","user"}, {"content", prompt}});
    json body = {{"model", cfg_.model}, {"max_tokens", cfg_.max_tokens},
                  {"temperature", temp}, {"messages", msgs}};
    std::string base = cfg_.base_url.empty() ? "https://api.openai.com" : rtrim_slash(cfg_.base_url);
    std::string path = cfg_.completions_path.empty() ? "/v1/chat/completions" : cfg_.completions_path;
    auto raw = http_post(base + path,
        {"Content-Type: application/json",
         "Authorization: Bearer " + cfg_.api_key},
        body.dump());
    auto j = json::parse(raw);
    if (j.contains("error"))
        throw std::runtime_error("OpenAI Chat: " + j["error"]["message"].get<std::string>());
    return j["choices"][0]["message"]["content"].get<std::string>();
}

// ── OpenAIResponsesProvider ──────────────────────────────────────

std::string OpenAIResponsesProvider::complete(const std::string& prompt,
                                                const std::string& sys,
                                                double temp) const {
    json body = {{"model", cfg_.model}, {"max_output_tokens", cfg_.max_tokens},
                  {"temperature", temp},
                  {"input", {{{"role","user"}, {"content", prompt}}}}};
    if (!sys.empty()) body["instructions"] = sys;
    std::string base = cfg_.base_url.empty() ? "https://api.openai.com" : cfg_.base_url;
    auto raw = http_post(base + "/v1/responses",
        {"Content-Type: application/json",
         "Authorization: Bearer " + cfg_.api_key},
        body.dump());
    auto j = json::parse(raw);
    if (j.contains("error"))
        throw std::runtime_error("OpenAI Responses: " + j["error"]["message"].get<std::string>());
    return j["output"][0]["content"][0]["text"].get<std::string>();
}

// ── make_provider ────────────────────────────────────────────────

std::unique_ptr<ILLMProvider> make_provider(const ProviderConfig& cfg) {
    switch (cfg.type) {
    case ProviderType::ANTHROPIC:        return std::make_unique<AnthropicProvider>(cfg);
    case ProviderType::OPENAI_CHAT:      return std::make_unique<OpenAIChatProvider>(cfg);
    case ProviderType::OPENAI_RESPONSES: return std::make_unique<OpenAIResponsesProvider>(cfg);
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

std::string LLMClient::try_slots(const std::vector<Slot>& slots,
                                   const std::string& prompt,
                                   const std::string& system,
                                   double temperature,
                                   ModelTier tier) const {
    std::string last_error;
    for (const auto& slot : slots) {
        if (!slot.breaker.try_allow()) {
            last_error = slot.provider->provider_name() + "/" + slot.provider->model_name()
                       + " [OPEN, " + std::to_string((int)slot.breaker.seconds_until_retry()) + "s]";
            continue;
        }
        { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.calls; }
        // Rate limiting: block until token available (or timeout)
        if (slot.rate_limiter.enabled())
            if (!slot.rate_limiter.acquire(10000)) {
                last_error = slot.provider->provider_name() + "/" +
                             slot.provider->model_name() + " [RATE_LIMIT_TIMEOUT]";
                continue;
            }
        try {
            auto res = slot.provider->complete(prompt, system, temperature);
            slot.breaker.on_success();
            { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.successes; }
            return res;
        } catch (const std::exception& e) {
            slot.breaker.on_failure();
            { std::lock_guard<std::mutex> lk(*slot.stats_mu); ++slot.failures; }
            last_error = slot.provider->provider_name() + "/" + slot.provider->model_name()
                       + ": " + e.what();
        }
    }
    throw AllProvidersExhaustedError(
        std::string(tier == ModelTier::ORCHESTRATOR ? "ORCHESTRATOR" : "SUBAGENT") +
        " providers exhausted. Last: " + last_error);
}

std::string LLMClient::complete_as(ModelTier tier, const std::string& prompt,
                                     const std::string& system, double temperature) const {
    const auto& slots = (tier == ModelTier::ORCHESTRATOR) ? orchestrators_ : subagents_;
    return try_slots(slots, prompt, system, temperature, tier);
}

std::vector<ProviderStats> LLMClient::stats(ModelTier tier) const {
    const auto& slots = (tier == ModelTier::ORCHESTRATOR) ? orchestrators_ : subagents_;
    std::vector<ProviderStats> out;
    for (const auto& s : slots) {
        ProviderStats ps;
        ps.provider_name = s.provider->provider_name();
        ps.model_name    = s.provider->model_name();
        { std::lock_guard<std::mutex> lk(*s.stats_mu);
          ps.total_calls = s.calls; ps.successes = s.successes; ps.failures = s.failures; }
        ps.circuit_state = s.breaker.state();
        ps.secs_to_retry = s.breaker.seconds_until_retry();
        out.push_back(ps);
    }
    return out;
}

void LLMClient::print_status() const {
    auto print_tier = [](const char* name, const std::vector<ProviderStats>& sv) {
        std::cout << "── " << name << " ──
";
        for (const auto& s : sv) {
            const char* cs = s.circuit_state == CircuitBreaker::State::CLOSED ? "CLOSED   " :
                             s.circuit_state == CircuitBreaker::State::OPEN   ? "OPEN     " : "HALF_OPEN";
            std::cout << "  [" << cs << "] " << std::left << std::setw(20)
                      << (s.provider_name + "/" + s.model_name)
                      << " calls=" << s.total_calls << " ok=" << s.successes << " fail=" << s.failures;
            if (s.circuit_state == CircuitBreaker::State::OPEN)
                std::cout << " (retry in " << (int)s.secs_to_retry << "s)";
            std::cout << "
";
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
    defs_[def.name] = def; fns_[def.name] = std::move(fn);
}
json ToolRegistry::call(const std::string& name, const json& params) const {
    auto fit = fns_.find(name);
    if (fit == fns_.end()) throw ToolNotFoundError(name, "not registered");

    // B3: Validate required inputs against schema
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

    if (!fit->second) return nullptr;
    return fit->second(params);
}
std::vector<ToolDef> ToolRegistry::list_tools() const {
    std::vector<ToolDef> out;
    for (const auto& [_, d] : defs_) out.push_back(d);
    return out;
}
bool ToolRegistry::has_tool(const std::string& n) const { return fns_.count(n) > 0; }

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
        else if (obj.is_array()) { try { obj = obj[std::stoi(parts[i])]; } catch (...) { obj = nullptr; } }
        else obj = nullptr;
    }
    return obj;
}
json WorkflowState::resolve_inputs(const json& inputs) const {
    if (!inputs.is_object()) return inputs;
    json r = json::object();
    for (auto& [k, v] : inputs.items()) {
        if (v.is_string()) {
            const auto& sv = v.get<std::string>();
            r[k] = (!sv.empty() && sv[0] == '$') ? resolve_ref(sv) : v;
        } else r[k] = v;
    }
    return r;
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
    std::string ctx = "Previous conversation context (use if relevant):
";
    for (const auto& [t, o] : history_)
        ctx += "  Task: " + t + "
  Result summary: " + o + "

";
    return ctx;
}

// ════════════════════════════════════════════════════════════════
// WorkflowPlanner
// ════════════════════════════════════════════════════════════════

static const char* PLANNER_SYS = R"(
You are a workflow planning engine. Return ONLY {"steps":[...]} JSON, no markdown.
STEP SCHEMA:
{"id":"snake_case","type":"tool|llm|transform","action":"tool_name or instruction",
 "description":"one line","inputs":{"key":"value or $step_id.field"},
 "depends_on":["ids"],"retry":0,"timeout_sec":30.0,"on_error":"fail|skip|fallback",
 "model_tier":"subagent|orchestrator"}
Rules: steps without shared deps run in parallel; no explanation outside the JSON object.
)";

json WorkflowPlanner::extract_json(const std::string& text) {
    try { return json::parse(text); } catch (...) {}
    std::regex fence(R"(```(?:json)?\s*([\s\S]+?)\s*```)"); std::smatch m;
    if (std::regex_search(text, m, fence)) try { return json::parse(m[1].str()); } catch (...) {}
    std::regex obj(R"(\{[\s\S]+\})");
    if (std::regex_search(text, m, obj)) try { return json::parse(m[0].str()); } catch (...) {}
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
    json tj = json::array();
    for (const auto& t : tools)
        tj.push_back({{"name", t.name}, {"description", t.description},
                      {"inputs", t.input_schema}});

    std::string prefix = ctx.to_prompt_prefix();
    std::string base = prefix
        + "Available tools:
" + tj.dump(2)
        + "

Task: " + task
        + "

Generate the workflow DAG:";

    std::string errors;
    const std::vector<double> temps = {0.0, 0.2, 0.4};
    for (int i = 0; i < max_attempts; ++i) {
        std::string p = base;
        if (!errors.empty())
            p += "

Previous validation errors (fix them):
" + errors + "
Return ONLY JSON.";
        try {
            auto raw  = extract_json(llm_.complete_as(
                ModelTier::ORCHESTRATOR, p, PLANNER_SYS_V2,
                temps[std::min(i, (int)temps.size()-1)]));
            auto plan = parse_plan(raw, task);
            validate_dag(plan);
            return plan;
        } catch (const std::exception& e) {
            errors += "[Attempt " + std::to_string(i+1) + "] " + e.what() + "
";
        }
    }
    throw PlanningError("plan() failed after " + std::to_string(max_attempts)
                        + " attempts:
" + errors);
}

WorkflowPlanner::WorkflowPlanner(LLMClient& llm) : llm_(llm) {}

json WorkflowPlanner::analyze_task(const std::string& task,
                                    const std::vector<ToolDef>& tools,
                                    const WorkflowContext& ctx) const {
    json ta = json::array();
    for (const auto& t : tools) ta.push_back({{"name", t.name}, {"description", t.description}});
    std::string prefix = ctx.to_prompt_prefix();
    std::string p = prefix +
        "Analyze the task. Return ONLY JSON:
"
        "{\"intent\":\"...\",\"required_tools\":[...],\"subtasks\":[{\"order\":1,\"description\":\"...\",\"tool\":\"name or null\"}],"
        "\"complexity\":\"simple|moderate|complex\"}
"
        "Available tools: " + ta.dump() + "
Task: " + task;
    return extract_json(llm_.complete_as(ModelTier::ORCHESTRATOR, p));
}

WorkflowPlan WorkflowPlanner::plan_static(const std::string& task, const json& analysis,
                                           const std::vector<ToolDef>& tools,
                                           int max_attempts) const {
    json tj = json::array();
    for (const auto& t : tools)
        tj.push_back({{"name", t.name}, {"description", t.description}, {"inputs", t.input_schema}});
    std::string base = "Tools:
" + tj.dump(2)
                     + "

Task: " + task
                     + "

Analysis:
" + analysis.dump(2)
                     + "

Generate the workflow DAG:";
    std::string errors;
    std::vector<double> temps = {0.0, 0.2, 0.4};
    for (int i = 0; i < max_attempts; ++i) {
        std::string p = base;
        if (!errors.empty())
            p += "

Previous validation errors (fix them):
" + errors + "
Return ONLY JSON.";
        try {
            auto raw  = extract_json(llm_.complete_as(ModelTier::ORCHESTRATOR, p, PLANNER_SYS,
                                     temps[std::min(i, (int)temps.size()-1)]));
            auto plan = parse_plan(raw, task);
            validate_dag(plan);
            return plan;
        } catch (const std::exception& e) {
            errors += "[Attempt " + std::to_string(i+1) + "] " + e.what() + "
";
        }
    }
    throw PlanningError("plan_static failed after " + std::to_string(max_attempts) + " attempts:
" + errors);
}

WorkflowPlan WorkflowPlanner::replan(const std::string& task, const WorkflowPlan& cur,
                                      const std::vector<json>& failures) const {
    std::string fb;
    for (size_t i = 0; i < failures.size(); ++i) {
        const auto& f = failures[i];
        fb += "Case " + std::to_string(i+1) + ": input=" +
              f.value("input", json{}).dump().substr(0, 200) +
              " reason=" + f.value("reason", std::string{}).substr(0, 300) + "
";
    }
    json dag = json::array();
    for (const auto& s : cur.steps) dag.push_back({{"id", s.id}, {"description", s.description}});
    std::string p = "Task: " + task + "
Current DAG:
" + dag.dump(2)
                  + "
Failures:
" + fb + "
Improve the plan. Return ONLY JSON: {\"steps\":[...]}";
    auto raw  = extract_json(llm_.complete_as(ModelTier::ORCHESTRATOR, p, PLANNER_SYS, 0.3));
    auto plan = parse_plan(raw, task);
    validate_dag(plan);
    return plan;
}

// ════════════════════════════════════════════════════════════════
// WorkflowExecutor
// ════════════════════════════════════════════════════════════════

WorkflowExecutor::WorkflowExecutor(LLMClient& llm, ToolRegistry& tools, size_t max_threads)
    : llm_(llm), tools_(tools), pool_(max_threads) {}

WorkflowState WorkflowExecutor::execute(const WorkflowPlan& plan,
                                         const json& task_input) const {
    WorkflowState state;
    state.task_input = task_input;

    for (const auto& batch : plan.topological_batches()) {
        using FutResult = std::tuple<bool, json, std::vector<TraceEntry>>;
        std::vector<std::pair<Step, std::future<FutResult>>> futs;

        for (const auto& step : batch) {
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

    std::exception_ptr last;
    // B4: 移除内层 std::async，curl 的 CURLOPT_TIMEOUT 已处理超时
    // R4: 带 jitter 的退避
    static thread_local std::mt19937 rng(std::random_device{}());
    for (int i = 0; i <= step.retry; ++i) {
        try {
            auto result = dispatch(step, inputs);
            long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            traces.push_back({step.id, type_str, "ok", "", ms, ""});
            return result;
        } catch (...) {
            last = std::current_exception();
            if (i < step.retry) {
                std::uniform_int_distribution<int> jitter(0, 200);
                int delay_ms = 500 * (1 << i) + jitter(rng);
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            }
        }
    }
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    try { std::rethrow_exception(last); } catch (const std::exception& e) {
        traces.push_back({step.id, type_str, "failed", "", ms, e.what()});
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
    default: throw std::runtime_error("Unknown step type");
    }
}

json WorkflowExecutor::exec_tool(const Step& s, const json& in) const {
    return tools_.call(s.action, in);
}

// R2: 保留 JSON 层级结构而不是拍平成字符串
json WorkflowExecutor::exec_llm(const Step& s, const json& in) const {
    std::string prompt = s.action;
    if (!in.empty())
        prompt += "

Context:
" + in.dump(2);

    // B1: Structured output / JSON mode
    if (s.json_mode || !s.output_schema.empty()) {
        if (!s.output_schema.empty())
            prompt += "

Return ONLY a JSON object matching this exact schema:
"
                    + s.output_schema.dump(2)
                    + "
No markdown, no explanation — JSON only.";
        else
            prompt += "

Return ONLY valid JSON. No markdown, no explanation.";

        auto raw = llm_.complete_as(s.model_tier, prompt, s.system_prompt, 0.0);
        try {
            auto result = WorkflowPlanner::extract_json(raw);
            // 实际 schema 验证（不只是 prompt injection）
            if (!s.output_schema.empty()) {
                auto violations = validate_json_schema(result, s.output_schema);
                if (!violations.empty()) {
                    // 记录违规但不抛异常（LLM 输出的容错性）
                    std::cerr << "[schema] Step '" << s.id << "' violations:
";
                    for (const auto& v : violations)
                        std::cerr << "  " << v.path << ": " << v.message << "
";
                }
            }
            return result;
        }
        catch (...) { return raw; }
    }

    // B2: Per-step system prompt
    return llm_.complete_as(s.model_tier, prompt, s.system_prompt);
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
        std::string sep = in.value("separator", "
"), r;
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
            throw std::runtime_error("parse_json: invalid JSON input — " + std::string(e.what()));
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
// WorkflowEngine
// ════════════════════════════════════════════════════════════════

WorkflowEngine::WorkflowEngine(const EngineConfig& cfg) : config_(cfg) {
    pool_     = std::make_unique<ThreadPool>(cfg.max_concurrency);
    llm_      = std::make_unique<LLMClient>(cfg.orchestrator, cfg.subagent);
    tools_    = std::make_unique<ToolRegistry>();
    planner_  = std::make_unique<WorkflowPlanner>(*llm_);
    executor_ = std::make_unique<WorkflowExecutor>(*llm_, *tools_, cfg.max_concurrency);
}

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
    WorkflowResult res;
    auto t0 = std::chrono::steady_clock::now();
    // Generate a short workflow ID for correlation
    auto wf_id = "wf_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count() % 1000000);
    metrics_->record({MetricEvent::Kind::WORKFLOW_START, wf_id, "", "", "", 0, true, task});
    try {
        auto tools    = tools_->list_tools();
        WorkflowContext empty_ctx;
        WorkflowContext& use_ctx = ctx ? *ctx : empty_ctx;
        // A1: 使用单次规划（比 analyze_task+plan_static 少一次 LLM 调用）
        auto plan = planner_->plan(task, tools, use_ctx);

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
        auto state    = executor_->execute(plan, {{"task", task}});

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
        if (ctx && res.has_output()) ctx->record(task, res.output);

        auto orc = llm_->stats(ModelTier::ORCHESTRATOR);
        auto sub = llm_->stats(ModelTier::SUBAGENT);
        res.provider_stats = orc;
        res.provider_stats.insert(res.provider_stats.end(), sub.begin(), sub.end());

    } catch (const std::exception& e) {
        res.success       = false;
        res.error_message = e.what();
    }
    res.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    return res;
}



// ════════════════════════════════════════════════════════════════
// SSE 解析 + Streaming 实现
// ════════════════════════════════════════════════════════════════

struct SseParser {
    StreamCallback callback;
    std::string    buffer;
    std::string    provider_type;   // "anthropic" | "openai"
    bool           done = false;

    void feed(const char* data, size_t len) {
        if (buffer.size() + len > 16 * 1024 * 1024) {
            // L1: Malformed SSE: buffer exceeds 16MB — abort streaming
            done = true; buffer.clear(); return;
        }
        buffer.append(data, len);
        size_t pos;
        while ((pos = buffer.find('
')) != std::string::npos) {
            std::string line = buffer.substr(0, pos);
            buffer = buffer.substr(pos + 1);
            if (!line.empty() && line.back() == '') line.pop_back();
            if (line.size() < 6 || line.substr(0,6) != "data: ") continue;
            std::string payload = line.substr(6);
            if (payload == "[DONE]") { done = true; continue; }
            try {
                auto j = json::parse(payload);
                std::string chunk;
                if (provider_type == "anthropic") {
                    if (j.value("type","") == "content_block_delta")
                        chunk = j["delta"].value("text","");
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

static void do_stream_post(CURL* curl,
                             const std::string& url,
                             const std::vector<std::string>& hdrs,
                             const std::string& body,
                             SseParser& parser,
                             double timeout_sec) {
    struct curl_slist* h = nullptr;
    for (const auto& s : hdrs) h = curl_slist_append(h, s.c_str());
    curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    h);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, sse_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &parser);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,       (long)timeout_sec);
    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(h);
    if (rc != CURLE_OK)
        throw std::runtime_error(std::string("curl stream: ") + curl_easy_strerror(rc));
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
    do_stream_post(curl_.get(), base + "/v1/messages",
        {"Content-Type: application/json",
         "x-api-key: " + cfg_.api_key,
         "anthropic-version: 2023-06-01"},
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
    do_stream_post(curl_.get(), base + path,
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
        r.latency_ms = timeout.count() * 1000;
        return r;
    }
    auto [ok, msg] = fut.get();
    r.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    r.alive = ok;
    if (!ok) r.error = msg;
    return r;
}

ProviderAutoPlanner::PlanResult
ProviderAutoPlanner::build_plan(const std::vector<ProbeResult>& results,
                                  const std::string& prefix) const {
    std::cout << "[" << prefix << "] Probe results (" << results.size() << " candidates):
";
    for (const auto& r : results) {
        std::cout << "  " << (r.alive ? "OK  " : "FAIL") << " [" << r.tier << "] "
                  << std::left << std::setw(22) << r.name << " " << r.latency_ms << "ms";
        if (!r.alive) std::cout << "  — " << r.error.substr(0,60);
        std::cout << "
";
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
        std::cerr << "[" << prefix << "] ERROR: " << plan.error << "
";
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
    std::cout << "[" << prefix << "] → ORCHESTRATOR=" << plan.alive_strong[0]
              << "  SUBAGENT=" << plan.alive_fast[0] << "
";
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
    std::cout << "[REPAIR] Re-probing after provider failure...
";
    return probe_and_plan(timeout);
}

void ProviderAutoPlanner::print_last_report() const {
    if (last_results_.empty()) { std::cout << "(no probe run yet)
"; return; }
    for (const auto& r : last_results_) {
        std::cout << "  " << (r.alive ? "✓" : "✗") << " [" << r.tier << "] "
                  << r.name << "  " << r.latency_ms << "ms";
        if (!r.alive) std::cout << " — " << r.error;
        std::cout << "
";
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
        std::cout << "[RECOVERY] Attempt " << (attempt+1) << "/" << max_repair_attempts << "
";
        auto plan = planner.repair();
        if (!plan.success) { result.error_message = "Recovery failed: " + plan.error; break; }
        config_.orchestrator = plan.config.orchestrator;
        config_.subagent     = plan.config.subagent;
        pool_     = std::make_unique<ThreadPool>(config_.max_concurrency);
        llm_      = std::make_unique<LLMClient>(config_.orchestrator, config_.subagent);
        planner_  = std::make_unique<WorkflowPlanner>(*llm_);
        executor_ = std::make_unique<WorkflowExecutor>(*llm_, *tools_, config_.max_concurrency);
        result = run(task);
        if (result.success) {
            std::cout << "[RECOVERY] Success on attempt " << (attempt+1) << "
";
            return result;
        }
    }
    return result;
}

WorkflowResult WorkflowEngine::run_stream(const std::string& task,
                                           StreamCallback on_chunk) {
    WorkflowResult res;
    auto t0 = std::chrono::steady_clock::now();
    try {
        auto tools = tools_->list_tools();
        WorkflowContext ctx;
        auto plan = planner_->plan(task, tools, ctx);  // A1: 单次规划

        auto leaves = plan.leaf_steps();
        std::set<std::string> leaf_ids;
        for (const auto& l : leaves) leaf_ids.insert(l.id);

        WorkflowState state;
        state.task_input = {{"task", task}};

        for (const auto& batch : plan.topological_batches()) {
            // Check if this is the final single-LLM batch
            bool is_final = (batch.size() == 1)
                         && leaf_ids.count(batch[0].id)
                         && (batch[0].type == StepType::LLM);

            if (is_final) {
                const auto& step   = batch[0];
                auto        inputs = state.resolve_inputs(step.inputs);
                std::string prompt = step.action;
                if (!inputs.empty()) prompt += "

Context:
" + inputs.dump(2);
                std::string acc;
                llm_->complete_as_stream(step.model_tier, prompt, "", 0.0,
                    [&](const std::string& chunk){ acc += chunk; on_chunk(chunk); });
                state.step_outputs[step.id] = acc;
            } else {
                using FT = std::tuple<bool, json, std::vector<TraceEntry>>;
                std::vector<std::pair<Step, std::future<FT>>> futs;
                for (const auto& step : batch)
                    futs.emplace_back(step,
                        pool_->submit([this, step, &state]() -> FT {
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
                    if (ok) state.step_outputs[step.id] = rj;
                    else if (step.on_error == OnError::FAIL)
                        throw StepExecutionError(step.id, step.description,
                                              rj.value("error",""));
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
        // L5: always call on_chunk even if final step is non-LLM
        if (res.has_output() && !res.output.is_null()) {
            std::string _out = res.output.is_string()
                ? res.output.get<std::string>() : res.output.dump(2);
            std::istringstream _iss(_out); std::string _w;
            while (std::getline(_iss, _w, ' ')) on_chunk(_w + " ");
        }
    } catch (const std::exception& e) {
        res.success = false; res.error_message = e.what();
    }
    res.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
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

static std::string build_agent_prompt(const std::string& task,
                                       const std::vector<ToolDef>& tools,
                                       const std::string& history) {
    json tool_list = json::array();
    for (const auto& t : tools)
        tool_list.push_back({{"name",t.name},{"description",t.description},{"inputs",t.input_schema}});

    std::string prompt = "Available tools:
" + tool_list.dump(2) + "

";
    prompt += "Task: " + task + "
";
    if (!history.empty())
        prompt += "
Work done so far:
" + history;
    prompt += "
Respond with your next action as a JSON object.";
    return prompt;
}

AgentResult WorkflowEngine::run_agent(const std::string& task, int max_iterations,
                                           std::function<void(const AgentStep&)> on_step) {
    AgentResult result;
    result.max_iterations = max_iterations;
    auto t0 = std::chrono::steady_clock::now();

    auto tools = tools_->list_tools();
    std::string history;

    for (int iter = 0; iter < max_iterations; ++iter) {
        auto step_t0 = std::chrono::steady_clock::now();
        AgentStep step;
        step.iteration = iter + 1;

        // ── Call LLM for next action ──────────────────────────
        std::string raw;
        try {
            const char* sys_to_use = custom_agent_prompt_.empty() ? AGENT_SYS : custom_agent_prompt_.c_str();
            raw = llm_->complete_as(ModelTier::ORCHESTRATOR,
                                    build_agent_prompt(task, tools, history),
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
            step.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
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

            // Append to history
            // L2: Truncate large observations to avoid context window overflow
            std::string obs_str = obs.dump();
            if (obs_str.size() > 800) obs_str = obs_str.substr(0, 800) + "...[truncated]";
            history += "[iter " + std::to_string(iter+1) + "]
"
                    +  "Thought: " + action.thought + "
"
                    +  "Called: " + action.tool_name + "(" + action.tool_args.dump() + ")
"
                    +  "Result: " + obs_str + "

";
            // L2: Trim history if it exceeds 4000 chars (keep last 2/3)
            if (history.size() > 4000) {
                auto cut = history.find("
[iter ", history.size() / 3);
                if (cut != std::string::npos) history = history.substr(cut);
            }

        } else if (action.type == AgentAction::Type::LOOP_BACK) {
            history += "[iter " + std::to_string(iter+1) + " — reconsidering]
"
                    +  "Thought: " + action.thought + "
"
                    +  "Reason: " + action.reason + "

";
        }

        step.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - step_t0).count();
        result.steps.push_back(step);
        if (on_step) on_step(step);  // B4: 逐步回调
    }

    // Max iterations reached without final answer
    if (!result.success && result.error.empty()) {
        result.error = "Max iterations (" + std::to_string(max_iterations) +
                       ") reached without final answer";
        result.iterations_used = max_iterations;
    }

    result.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}

void WorkflowEngine::set_planner_prompt(const std::string& sys_prompt) {
    custom_planner_prompt_ = sys_prompt;
}
void WorkflowEngine::set_agent_prompt(const std::string& sys_prompt) {
    custom_agent_prompt_ = sys_prompt;
}

void WorkflowEngine::set_metrics(std::shared_ptr<IMetricsCollector> collector) {
    if (collector) metrics_ = std::move(collector);
    else           metrics_ = std::make_shared<NoOpMetrics>();
}
} // namespace ariadne
