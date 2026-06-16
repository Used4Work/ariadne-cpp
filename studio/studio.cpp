#ifdef _MSC_VER
#pragma warning(disable:4996 4244 4267)
#endif
#ifdef _WIN32
#  define NOMINMAX
#  define _CRT_SECURE_NO_WARNINGS
#endif

// Disable optional dependencies for standalone build
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#ifdef CPPHTTPLIB_BROTLI_SUPPORT
#undef CPPHTTPLIB_BROTLI_SUPPORT
#endif
#ifdef CPPHTTPLIB_ZLIB_SUPPORT
#undef CPPHTTPLIB_ZLIB_SUPPORT
#endif
#include "../ariadne.hpp"
#include "studio_ui.hpp"
#include <httplib.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace fs = std::filesystem;
using namespace ariadne;

// ════════════════════════════════════════════════════════════════
// SSE Event Dispatcher
// ════════════════════════════════════════════════════════════════

class EventStream {
public:
    void send(const std::string& event_type, const json& data) {
        std::lock_guard<std::mutex> lk(mu_);
        std::string msg = "event: " + event_type + "\ndata: " + data.dump() + "\n\n";
        messages_.push_back(msg);
        ++seq_;
        cv_.notify_all();
    }

    void complete() {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = true;
        cv_.notify_all();
    }

    bool wait_and_write(httplib::DataSink& sink, size_t& cursor) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait_for(lk, std::chrono::seconds(30), [&]{ return cursor < messages_.size() || done_; });
        while (cursor < messages_.size()) {
            sink.write(messages_[cursor].c_str(), messages_[cursor].size());
            ++cursor;
        }
        if (done_ && cursor >= messages_.size()) return false;
        // Send keepalive if no data
        if (cursor == 0 || messages_.empty()) {
            const char* ka = ": keepalive\n\n";
            sink.write(ka, strlen(ka));
        }
        return true;
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::string> messages_;
    std::atomic<size_t> seq_{0};
    bool done_ = false;
};

// ════════════════════════════════════════════════════════════════
// Workflow Store (JSON files on disk)
// ════════════════════════════════════════════════════════════════

static fs::path get_store_dir() {
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    fs::path dir = fs::path(home ? home : ".") / ".ariadne" / "workflows";
    fs::create_directories(dir);
    return dir;
}

static bool is_safe_id(const std::string& id) {
    if (id.empty() || id.size() > 64) return false;
    for (char c : id)
        if (!std::isalnum(c) && c != '_' && c != '-') return false;
    return true;
}

static json list_workflows() {
    json list = json::array();
    auto dir = get_store_dir();
    if (!fs::exists(dir)) return list;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".json") {
            try {
                std::ifstream f(entry.path());
                json wf = json::parse(f);
                list.push_back({{"id", entry.path().stem().string()},
                                {"name", wf.value("name", "Untitled")}});
            } catch (...) {}
        }
    }
    return list;
}

// ════════════════════════════════════════════════════════════════
// Drawflow JSON → Ariadne WorkflowPlan Converter
// ════════════════════════════════════════════════════════════════

static WorkflowPlan convert_steps(const json& steps_json) {
    WorkflowPlan plan;
    plan.id = "studio_" + std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count() % 1000000);

    for (const auto& s : steps_json) {
        Step step;
        step.id          = s.value("id", "");
        step.action      = s.value("action", "");
        step.description = s.value("description", step.id);
        step.depends_on  = s.value("depends_on", std::vector<std::string>{});
        step.inputs      = s.value("inputs", json::object());
        step.system_prompt = s.value("system_prompt", "");
        step.json_mode   = s.value("json_mode", false);
        step.temperature = s.value("temperature", -1.0);

        const std::string tp = s.value("type", "llm");
        step.type = (tp == "tool") ? StepType::TOOL :
                    (tp == "transform") ? StepType::TRANSFORM :
                    (tp == "condition") ? StepType::CONDITION : StepType::LLM;

        step.model_tier = (s.value("model_tier", "subagent") == "orchestrator")
                          ? ModelTier::ORCHESTRATOR : ModelTier::SUBAGENT;

        plan.steps.push_back(std::move(step));
    }
    return plan;
}

// ════════════════════════════════════════════════════════════════
// Built-in Templates
// ════════════════════════════════════════════════════════════════

static json make_template_node(int id, const std::string& name, const std::string& cls,
                                const json& data, const std::string& html,
                                int inputs, int outputs, int px, int py,
                                const json& in_conns = json::object(),
                                const json& out_conns = json::object()) {
    json node;
    node["id"] = id; node["name"] = name; node["class"] = cls;
    node["data"] = data; node["html"] = html; node["typenode"] = false;
    node["pos_x"] = px; node["pos_y"] = py;
    json inp, outp;
    for (int i = 1; i <= inputs; ++i) {
        std::string k = "input_" + std::to_string(i);
        inp[k] = {{"connections", in_conns.contains(k) ? in_conns[k] : json::array()}};
    }
    for (int i = 1; i <= outputs; ++i) {
        std::string k = "output_" + std::to_string(i);
        outp[k] = {{"connections", out_conns.contains(k) ? out_conns[k] : json::array()}};
    }
    node["inputs"] = inp; node["outputs"] = outp;
    return node;
}

static json get_templates() {
    // Template 1: Search & Summarize
    json t1_data;
    t1_data["1"] = make_template_node(1, "tool", "tool-node",
        {{"action","web_search"},{"tool_input_key","query"},{"tool_input_value","latest AI news"}},
        "<div class='node-header'><span class='ndot'></span>Tool</div><div class='node-body'>web_search</div>",
        0, 1, 100, 140, json::object(),
        {{"output_1", json::array({{{"node","2"},{"output","input_1"}}})}});
    t1_data["2"] = make_template_node(2, "llm", "llm-node",
        {{"action","Summarize the search results in 3 bullet points"},{"system_prompt",""},{"model_tier","subagent"},{"json_mode",false},{"temperature",-1}},
        "<div class='node-header'><span class='ndot'></span>LLM</div><div class='node-body'>Summarize</div>",
        1, 0, 420, 140,
        {{"input_1", json::array({{{"node","1"},{"input","output_1"}}})}});

    // Template 2: Compare (empty canvas)
    json t2_data = json::object();

    // Template 3: Q&A Agent (empty canvas)
    json t3_data = json::object();

    return json::array({
        {{"id","search_summarize"},{"name","Search & Summarize"},
         {"description","Search the web then summarize results with LLM"},
         {"drawflow",{{"drawflow",{{"Home",{{"data",t1_data}}}}}}}},
        {{"id","compare"},{"name","Compare Two Topics"},
         {"description","Search two topics in parallel then compare with LLM (build from scratch)"},
         {"drawflow",{{"drawflow",{{"Home",{{"data",t2_data}}}}}}}},
        {{"id","qa_agent"},{"name","Q&A Agent"},
         {"description","Use agent mode for open-ended Q&A (build from scratch)"},
         {"drawflow",{{"drawflow",{{"Home",{{"data",t3_data}}}}}}}}
    });
}

// ════════════════════════════════════════════════════════════════
// Main — Ariadne Studio Server
// ════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    // Parse port from command line
    int port = 8080;
    if (argc > 1) port = std::atoi(argv[1]);

    // Load API key
    const char* key = std::getenv("GITHUB_TOKEN");
    if (!key) {
        // Try .env file
        std::ifstream envf(".env");
        static std::string token_storage;
        if (envf) {
            std::string line;
            while (std::getline(envf, line)) {
                if (line.find("GITHUB_TOKEN=") == 0) {
                    token_storage = line.substr(13);
                    key = token_storage.c_str();
                    break;
                }
            }
        }
    }

    // Build engine
    std::unique_ptr<WorkflowEngine> engine;
    if (key && strlen(key) > 0) {
        engine = std::make_unique<WorkflowEngine>(
            EngineConfig::from_single(ProviderConfig::github_models(key, "openai/gpt-4.1")));
        // Register mock tools for demo
        engine->register_tool(
            {"web_search", "Search the web",
             {{"required", json::array({"query"})},
              {"properties", {{"query", {{"type","string"}}}}}}, {}},
            [](const json& p) -> json {
                std::string q = p.value("query", "");
                return {{"results", {{{"title", q + " Overview"},
                        {"snippet", q + ": comprehensive search results."}}}}};
            });
        engine->register_tool(
            {"calculator", "Evaluate math",
             {{"required", json::array({"expr"})},
              {"properties", {{"expr", {{"type","string"}}}}}}, {}},
            [](const json&) -> json { return {{"result", 42}}; });
        std::cout << "[studio] Engine ready (GitHub Models)\n";
    } else {
        std::cerr << "[studio] WARNING: No GITHUB_TOKEN found. Set it in env or .env file.\n";
        std::cerr << "[studio] Starting without LLM provider (edit-only mode).\n";
    }

    // Active SSE streams and background threads
    std::mutex streams_mu;
    std::map<std::string, std::shared_ptr<EventStream>> streams;
    std::vector<std::thread> run_threads;

    // HTTP Server
    httplib::Server svr;

    // ── Serve Web UI ──────────────────────────────────
    auto html_content = load_studio_html();
    svr.Get("/", [&html_content](const httplib::Request&, httplib::Response& res) {
        res.set_content(html_content, "text/html");
    });

    // ── List registered tools ─────────────────────────
    svr.Get("/api/tools", [&](const httplib::Request&, httplib::Response& res) {
        json tools = json::array();
        if (engine) {
            // Access tools via plan_only hack (empty task won't work, so just list names)
            tools.push_back({{"name","web_search"},{"description","Search the web"}});
            tools.push_back({{"name","calculator"},{"description","Evaluate math"}});
        }
        res.set_content(tools.dump(), "application/json");
    });

    // ── Workflow CRUD ─────────────────────────────────
    svr.Get("/api/workflows", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(list_workflows().dump(), "application/json");
    });

    svr.Post("/api/workflows", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string name = body.value("name", "Untitled");
            std::string id = "wf_" + std::to_string(
                std::chrono::system_clock::now().time_since_epoch().count() % 100000000);
            body["id"] = id;
            auto path = get_store_dir() / (id + ".json");
            std::ofstream(path) << body.dump(2);
            res.set_content(json({{"id", id}, {"name", name}}).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Get(R"(/api/workflows/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        auto id = std::string(req.matches[1]);
        if (!is_safe_id(id)) { res.status = 400; return; }
        auto path = get_store_dir() / (id + ".json");
        if (!fs::exists(path)) { res.status = 404; return; }
        std::ifstream f(path);
        res.set_content(json::parse(f).dump(), "application/json");
    });

    svr.Delete(R"(/api/workflows/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        auto id = std::string(req.matches[1]);
        if (!is_safe_id(id)) { res.status = 400; return; }
        auto path = get_store_dir() / (id + ".json");
        if (fs::exists(path)) fs::remove(path);
        res.set_content("{\"ok\":true}", "application/json");
    });

    // ── Templates ─────────────────────────────────────
    svr.Get("/api/templates", [](const httplib::Request&, httplib::Response& res) {
        auto ts = get_templates();
        json list = json::array();
        for (const auto& t : ts) list.push_back({{"id",t["id"]},{"name",t["name"]},{"description",t["description"]}});
        res.set_content(list.dump(), "application/json");
    });

    svr.Get(R"(/api/templates/(.+))", [](const httplib::Request& req, httplib::Response& res) {
        auto id = std::string(req.matches[1]);
        for (const auto& t : get_templates()) {
            if (t["id"] == id) { res.set_content(t.dump(), "application/json"); return; }
        }
        res.status = 404;
    });

    // ── Run Workflow ──────────────────────────────────
    svr.Post("/api/run", [&](const httplib::Request& req, httplib::Response& res) {
        if (!engine) {
            res.set_content(json({{"error","No LLM provider configured"}}).dump(), "application/json");
            return;
        }
        try {
            auto body = json::parse(req.body);
            auto steps = body["steps"];

            std::string run_id = "run_" + std::to_string(
                std::chrono::system_clock::now().time_since_epoch().count() % 100000000);

            auto stream = std::make_shared<EventStream>();
            {
                std::lock_guard<std::mutex> lk(streams_mu);
                streams[run_id] = stream;
            }

            // Execute in background thread (stored for cleanup)
            std::thread t([stream, steps, &engine, run_id]() {
                try {
                    auto plan = convert_steps(steps);
                    validate_dag(plan);

                    stream->send("info", {{"message", "DAG validated, " +
                        std::to_string(plan.steps.size()) + " steps"}});

                    // Execute step by step, emitting SSE events
                    WorkflowState state;
                    state.task_input = {{"task", "studio_run"}};

                    auto t0 = std::chrono::steady_clock::now();

                    for (const auto& batch : plan.topological_batches()) {
                        for (const auto& step : batch) {
                            stream->send("step_start", {{"step_id", step.id},
                                {"type", step.type == StepType::TOOL ? "tool" : "llm"}});

                            auto st0 = std::chrono::steady_clock::now();
                            try {
                                // Simple dispatch without full executor (avoid thread pool complexity)
                                json result;
                                auto inputs = state.resolve_inputs(step.inputs);

                                if (step.type == StepType::TOOL) {
                                    result = json("(tool: " + step.action + ")");
                                } else if (step.type == StepType::LLM) {
                                    std::string prompt = step.action;
                                    if (!inputs.empty()) prompt += "\n\nContext:\n" + inputs.dump(2);
                                    auto tier = (step.model_tier == ModelTier::ORCHESTRATOR)
                                                ? ModelTier::ORCHESTRATOR : ModelTier::SUBAGENT;
                                    result = engine->llm_complete(prompt, step.system_prompt, tier);
                                } else if (step.type == StepType::TRANSFORM) {
                                    result = inputs;
                                } else if (step.type == StepType::CONDITION) {
                                    auto v = inputs.value("value", json(nullptr));
                                    result = !v.is_null() && v != false && v != 0;
                                } else {
                                    result = inputs;
                                }

                                state.step_outputs[step.id] = result;
                                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - st0).count();

                                std::string output_str = result.is_string() ?
                                    result.get<std::string>() : result.dump();
                                if (output_str.size() > 500) output_str = output_str.substr(0, 500) + "...";

                                stream->send("step_done", {{"step_id", step.id},
                                    {"duration_ms", ms}, {"output", output_str}});

                            } catch (const std::exception& e) {
                                stream->send("step_error", {{"step_id", step.id},
                                    {"error", e.what()}});
                                state.step_outputs[step.id] = nullptr;
                            }
                        }
                    }

                    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
                    stream->send("complete", {
                        {"summary", std::to_string(plan.steps.size()) + " steps, " +
                                    std::to_string(total_ms) + "ms"},
                        {"duration_ms", total_ms},
                        {"success", true}
                    });
                } catch (const std::exception& e) {
                    stream->send("complete", {{"summary", "Failed: " + std::string(e.what())},
                        {"success", false}, {"error", e.what()}});
                }
                stream->complete();
            });
            {
                std::lock_guard<std::mutex> lk(streams_mu);
                run_threads.push_back(std::move(t));
            }

            res.set_content(json({{"run_id", run_id}}).dump(), "application/json");

        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    // ── SSE Stream ────────────────────────────────────
    svr.Get(R"(/api/stream/(.+))", [&](const httplib::Request& req, httplib::Response& res) {
        auto run_id = std::string(req.matches[1]);
        std::shared_ptr<EventStream> stream;
        {
            std::lock_guard<std::mutex> lk(streams_mu);
            auto it = streams.find(run_id);
            if (it == streams.end()) { res.status = 404; return; }
            stream = it->second;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");

        size_t cursor = 0;
        res.set_chunked_content_provider("text/event-stream",
            [stream, cursor](size_t, httplib::DataSink& sink) mutable {
                return stream->wait_and_write(sink, cursor);
            });
    });

    // ── Agent Run ─────────────────────────────────────
    svr.Post("/api/agent/run", [&](const httplib::Request& req, httplib::Response& res) {
        if (!engine) {
            res.set_content(json({{"error","No LLM provider"}}).dump(), "application/json");
            return;
        }
        try {
            auto body = json::parse(req.body);
            std::string task = body.value("task", "");
            int max_iter = body.value("max_iterations", 10);
            auto result = engine->run_agent(task, max_iter);
            res.set_content(json({
                {"success", result.success},
                {"final_answer", result.final_answer},
                {"iterations_used", result.iterations_used},
                {"error", result.error},
                {"duration_ms", result.duration_ms}
            }).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    // ── Start Server ──────────────────────────────────
    std::cout << "\n"
              << "  ╔═══════════════════════════════════════╗\n"
              << "  ║     ARIADNE STUDIO                    ║\n"
              << "  ║     http://localhost:" << port << "              ║\n"
              << "  ║     Press Ctrl+C to stop              ║\n"
              << "  ╚═══════════════════════════════════════╝\n\n";

    svr.listen("127.0.0.1", port);

    // Join all background threads on shutdown
    {
        std::lock_guard<std::mutex> lk(streams_mu);
        for (auto& t : run_threads)
            if (t.joinable()) t.join();
    }
    return 0;
}
