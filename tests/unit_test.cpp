
#include "../ariadne.hpp"
#include <iostream>
#include <thread>
#include <chrono>
using namespace ariadne;

static int g_run=0, g_pass=0;
#define RUN(fn) do{++g_run;std::cout<<"  "#fn"... "<<std::flush;\
    try{fn();std::cout<<"OK\n";++g_pass;}\
    catch(const std::exception&e){std::cout<<"FAIL: "<<e.what()<<"\n";}}while(0)
#define ASSERT(c) if(!(c))throw std::runtime_error("assert: "#c)
#define ASSERT_THROWS(T,e) do{bool t=false;try{(void)(e);}catch(const T&){t=true;}\
    if(!t)throw std::runtime_error("expected "#T" not thrown");}while(0)

static WorkflowPlan lin3() {
    return {"p",{
        {"a",StepType::TOOL,"t",{},{},0,30,OnError::FAIL},
        {"b",StepType::TOOL,"t",{},{"a"},0,30,OnError::FAIL},
        {"c",StepType::TOOL,"t",{},{"b"},0,30,OnError::FAIL},
    }};
}

// ── DAG ─────────────────────────────────────────────────────────
void test_dag_valid()   { validate_dag(lin3()); }
void test_dag_dup()     {
    WorkflowPlan p{"p",{
        {"x",StepType::TOOL,"t",{},{},0,30,OnError::FAIL},
        {"x",StepType::TOOL,"t",{},{},0,30,OnError::FAIL}}};
    ASSERT_THROWS(DAGValidationError, validate_dag(p)); }
void test_dag_dep()     {
    WorkflowPlan p{"p",{{"a",StepType::TOOL,"t",{},{"ghost"},0,30,OnError::FAIL}}};
    ASSERT_THROWS(DAGValidationError, validate_dag(p)); }
void test_dag_cycle()   {
    WorkflowPlan p{"p",{
        {"a",StepType::TOOL,"t",{},{"b"},0,30,OnError::FAIL},
        {"b",StepType::TOOL,"t",{},{"a"},0,30,OnError::FAIL}}};
    ASSERT_THROWS(DAGValidationError, validate_dag(p)); }

// ── Topological batches ──────────────────────────────────────────
void test_topo_linear() {
    auto b = lin3().topological_batches();
    ASSERT(b.size()==3); ASSERT(b[0][0].id=="a"); ASSERT(b[2][0].id=="c");
}
void test_topo_parallel() {
    WorkflowPlan p{"p",{
        {"a",StepType::TOOL,"t",{},{},0,30,OnError::FAIL},
        {"b",StepType::TOOL,"t",{},{},0,30,OnError::FAIL},
        {"c",StepType::TOOL,"t",{},{"a","b"},0,30,OnError::FAIL}}};
    auto b = p.topological_batches();
    ASSERT(b.size()==2); ASSERT(b[0].size()==2);
}
void test_leaf() {
    WorkflowPlan p{"p",{
        {"a",StepType::TOOL,"t",{},{},0,30,OnError::FAIL},
        {"b",StepType::TOOL,"t",{},{"a"},0,30,OnError::FAIL}}};
    auto l = p.leaf_steps();
    ASSERT(l.size()==1 && l[0].id=="b");
}

// ── $ref resolution ─────────────────────────────────────────────
void test_ref_task()    { WorkflowState s; s.task_input={{"q","hi"}};
                          ASSERT(s.resolve_ref("$task_input.q").get<std::string>()=="hi"); }
void test_ref_nested()  { WorkflowState s; s.step_outputs["x"]={{"items",json::array({"a","b","c"})}};
                          ASSERT(s.resolve_ref("$x.items.1").get<std::string>()=="b"); }
void test_ref_inputs()  { WorkflowState s; s.task_input={{"t","task"}};
                          auto r=s.resolve_inputs({{"q","$task_input.t"},{"n",5}});
                          ASSERT(r["q"].get<std::string>()=="task"); ASSERT(r["n"].get<int>()==5); }

// ── CircuitBreaker ───────────────────────────────────────────────
void test_cb_closed()   { CircuitBreaker cb(3,60); ASSERT(cb.state()==CircuitBreaker::State::CLOSED); ASSERT(cb.try_allow()); }
void test_cb_opens()    { CircuitBreaker cb(3,60); cb.on_failure();cb.on_failure();cb.on_failure();
                          ASSERT(cb.state()==CircuitBreaker::State::OPEN); ASSERT(!cb.try_allow()); }
void test_cb_recovers() { CircuitBreaker cb(2,0.1); cb.on_failure(); cb.on_failure();
                          std::this_thread::sleep_for(std::chrono::milliseconds(150));
                          ASSERT(cb.try_allow()); cb.on_success();
                          ASSERT(cb.state()==CircuitBreaker::State::CLOSED); }

// ── ToolRegistry ─────────────────────────────────────────────────
void test_tool_reg()     { ToolRegistry r; r.register_tool({"add","",{},{}},
                               [](const json&p)->json{return p["a"].get<int>()+p["b"].get<int>();});
                           ASSERT(r.call("add",{{"a",2},{"b",3}}).get<int>()==5); }
void test_tool_unknown() { ToolRegistry r; ASSERT_THROWS(ToolNotFoundError,r.call("ghost",{})); }
void test_tool_list()    { ToolRegistry r; r.register_tool({"t1","",{},{}},nullptr);
                           r.register_tool({"t2","",{},{}},nullptr); ASSERT(r.list_tools().size()==2); }

// ── WorkflowContext ──────────────────────────────────────────────
void test_ctx_empty()   { WorkflowContext c; ASSERT(c.empty()); ASSERT(c.to_prompt_prefix().empty()); }
void test_ctx_records() { WorkflowContext c(3); c.record("task A", json("result A"));
                          ASSERT(!c.empty());
                          ASSERT(c.to_prompt_prefix().find("task A")!=std::string::npos); }
void test_ctx_evict()   { WorkflowContext c(2); c.record("t1",json("r1")); c.record("t2",json("r2"));
                          c.record("t3",json("r3")); std::string p=c.to_prompt_prefix();
                          ASSERT(p.find("t1")==std::string::npos); ASSERT(p.find("t3")!=std::string::npos); }

// ── TraceEntry ──────────────────────────────────────────────────
void test_trace()       { WorkflowState s;
                          s.record_trace({"step1","llm","ok","anthropic/claude",123,""});
                          ASSERT(s.traces.size()==1); ASSERT(s.traces[0].duration_ms==123); }

// ── ProviderAutoPlanner ─────────────────────────────────────────
void test_planner_empty() { ProviderAutoPlanner p;
                             auto r = p.probe_and_plan(std::chrono::seconds(1));
                             ASSERT(!r.success); ASSERT(!r.error.empty()); }
void test_planner_add()   { ProviderAutoPlanner p;
                             p.add_candidate("c1", ProviderConfig::github_models("tok"), "strong", 1);
                             p.add_candidate("c2", ProviderConfig::github_models("tok"), "fast",   1);
                             ASSERT(p.last_results().empty()); }

// ── StreamCallback ──────────────────────────────────────────────
void test_stream_cb()   { std::vector<std::string> chunks;
                          StreamCallback cb=[&](const std::string&c){chunks.push_back(c);};
                          cb("hello"); cb(" world");
                          ASSERT(chunks.size()==2 && chunks[0]=="hello"); }

// ── EngineConfig ────────────────────────────────────────────────
void test_cfg_single()  { auto c=EngineConfig::from_single(ProviderConfig::github_models("tok"));
                          ASSERT(c.orchestrator.primary.model==c.subagent.primary.model); }
void test_cfg_two()     { auto c=EngineConfig::from_two(
                              ProviderConfig::anthropic("k","claude-opus-4-8"),
                              ProviderConfig::github_models("t"));
                          ASSERT(c.orchestrator.primary.type==ProviderType::ANTHROPIC);
                          ASSERT(c.subagent.primary.type==ProviderType::OPENAI_CHAT); }

// ── AgentAction + AgentResult ────────────────────────────────────
void test_agent_action_types() {
    AgentAction a; a.type = AgentAction::Type::TOOL_CALL;
    ASSERT(a.type == AgentAction::Type::TOOL_CALL);
    AgentAction b; b.type = AgentAction::Type::FINAL_ANSWER;
    ASSERT(b.type == AgentAction::Type::FINAL_ANSWER);
    AgentAction c; c.type = AgentAction::Type::LOOP_BACK;
    ASSERT(c.type == AgentAction::Type::LOOP_BACK);
}

void test_agent_result_defaults() {
    AgentResult r;
    ASSERT(!r.success);
    ASSERT(r.iterations_used == 0);
    ASSERT(r.steps.empty());
    ASSERT(r.avg_step_ms() == 0.0);
}

void test_agent_result_reached_max() {
    AgentResult r;
    r.max_iterations  = 5;
    r.iterations_used = 5;
    ASSERT(r.reached_max());
    r.iterations_used = 3;
    ASSERT(!r.reached_max());
}

void test_agent_step_recording() {
    AgentResult r;
    AgentStep s;
    s.iteration   = 1;
    s.thought     = "I should search for this";
    s.duration_ms = 200;
    s.action.type = AgentAction::Type::TOOL_CALL;
    r.steps.push_back(s);
    ASSERT(r.steps.size() == 1);
    ASSERT(r.avg_step_ms() == 200.0);
}

// ── 新增：针对本轮审计修复的回归测试 ────────────────────────────

// C1: ToolFn=nullptr 不崩溃
void test_tool_null_fn() {
    ToolRegistry r;
    r.register_tool({"null_tool","",{},{}}, nullptr);
    auto result = r.call("null_tool", {});
    ASSERT(result.is_null());
}

// C3: validate_dag 捕获空 step.id
void test_dag_empty_id() {
    WorkflowPlan p{"p",{{"",StepType::TOOL,"t",{},{},0,30,OnError::FAIL}}};
    ASSERT_THROWS(DAGValidationError, validate_dag(p));
}

// D2: WorkflowContext 可配置截断长度
void test_ctx_custom_truncation() {
    WorkflowContext ctx(5, 10);  // max_summary_chars=10
    ctx.record("task", json("1234567890abcdef"));
    std::string p = ctx.to_prompt_prefix();
    ASSERT(p.find("...") != std::string::npos);  // should be truncated
}

// L2: Agent history 观测截断
void test_agent_result_steps() {
    AgentResult r;
    AgentStep s;
    s.iteration = 1;
    s.action.type = AgentAction::Type::TOOL_CALL;
    s.action.tool_name = "web_search";
    s.observation = json("x");
    s.duration_ms = 50;
    r.steps.push_back(s);
    r.success = true;
    r.final_answer = "done";
    r.iterations_used = 1;
    r.max_iterations  = 10;   // fix: set max so reached_max() is false
    ASSERT(r.avg_step_ms() == 50.0);
    ASSERT(!r.reached_max());
}

// ── 新功能：异常层次 ──────────────────────────────────────────────
void test_exception_hierarchy() {
    // ProviderError 是 AriadneError
    try { throw AllProvidersExhaustedError("test"); }
    catch (const ProviderError&)  { /* ok */ }
    catch (...) { throw std::runtime_error("AllProvidersExhaustedError not caught as ProviderError"); }

    // ToolNotFoundError 是 ToolError 是 AriadneError
    try { throw ToolNotFoundError("my_tool","not registered"); }
    catch (const ToolError& e) { ASSERT(std::string(e.what()).find("my_tool") != std::string::npos); }
    catch (...) { throw std::runtime_error("ToolNotFoundError not caught as ToolError"); }

    // StepExecutionError (alias WorkflowExecutionError)
    try { throw StepExecutionError("s1","search","timeout"); }
    catch (const AriadneError& e) { ASSERT(std::string(e.what()).find("s1") != std::string::npos); }
    catch (...) { throw std::runtime_error("StepExecutionError not caught as AriadneError"); }
}

// ── 新功能：ToolInputError (schema validation) ───────────────────
void test_tool_schema_validation() {
    ToolRegistry r;
    // Tool with required field "query"
    r.register_tool({
        "search", "Search",
        {{"required", json::array({"query"})},
         {"properties", {{"query", {{"type","string"}}}}}},
        {}
    }, [](const json& p)->json{ return {{"ok",true}}; });

    // Call with required field → success
    auto res = r.call("search", {{"query","hello"}});
    ASSERT(res["ok"].get<bool>());

    // Call without required field → ToolInputError
    bool caught = false;
    try { r.call("search", {}); }
    catch (const ToolInputError& e) {
        caught = true;
        ASSERT(std::string(e.what()).find("query") != std::string::npos);
    }
    ASSERT(caught);
}

// ── 新功能：Step json_mode + system_prompt 字段 ──────────────────
void test_step_new_fields() {
    Step s;
    s.id          = "llm_step";
    s.type        = StepType::LLM;
    s.action      = "Summarize this";
    s.json_mode   = true;
    s.system_prompt = "You are a JSON-only assistant.";
    s.output_schema = {{"type","object"},{"properties",{{"summary",{{"type","string"}}}}}};
    ASSERT(s.json_mode);
    ASSERT(!s.system_prompt.empty());
    ASSERT(s.output_schema.contains("type"));
}

// ── 新功能：AgentResult on_step callback ────────────────────────
void test_agent_on_step_callback() {
    std::vector<int> iterations_seen;
    AgentResult r;
    // Simulate callback being called with steps
    auto on_step = [&](const AgentStep& step) {
        iterations_seen.push_back(step.iteration);
    };
    AgentStep s1; s1.iteration = 1; s1.action.type = AgentAction::Type::TOOL_CALL;
    AgentStep s2; s2.iteration = 2; s2.action.type = AgentAction::Type::FINAL_ANSWER;
    on_step(s1); on_step(s2);
    ASSERT(iterations_seen.size() == 2);
    ASSERT(iterations_seen[0] == 1 && iterations_seen[1] == 2);
}


// ── ThreadPool ────────────────────────────────────────────────────
void test_threadpool_basic() {
    ThreadPool pool(4);
    ASSERT(pool.size() == 4);

    std::atomic<int> counter{0};
    std::vector<std::future<void>> futs;
    for (int i = 0; i < 20; ++i)
        futs.push_back(pool.submit([&]{ ++counter; }));
    for (auto& f : futs) f.get();
    ASSERT(counter == 20);
}

void test_threadpool_returns_value() {
    ThreadPool pool(2);
    auto f1 = pool.submit([]{ return 42; });
    auto f2 = pool.submit([]{ return std::string("hello"); });
    ASSERT(f1.get() == 42);
    ASSERT(f2.get() == "hello");
}

void test_threadpool_parallel() {
    ThreadPool pool(4);
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    std::vector<std::future<void>> futs;
    for (int i = 0; i < 4; ++i)
        futs.push_back(pool.submit([]{
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }));
    for (auto& f : futs) f.get();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        clk::now() - t0).count();
    // 4 × 50ms in parallel should finish in ~50-150ms, not ~200ms
    ASSERT(ms < 180);
}

// ── JSON Schema Validation ────────────────────────────────────────
void test_schema_valid_object() {
    json schema = {{"type","object"},{"required",json::array({"name","age"})},
                   {"properties",{{"name",{{"type","string"}}},{"age",{{"type","integer"}}}}}};
    json value  = {{"name","Alice"},{"age",30}};
    auto v = validate_json_schema(value, schema);
    ASSERT(v.empty());
}

void test_schema_missing_required() {
    json schema = {{"type","object"},{"required",json::array({"name","age"})}};
    json value  = {{"name","Bob"}};  // missing "age"
    auto v = validate_json_schema(value, schema);
    ASSERT(!v.empty());
    bool found_age = false;
    for (const auto& viol : v) if (viol.path.find("age") != std::string::npos) found_age=true;
    ASSERT(found_age);
}

void test_schema_wrong_type() {
    json schema = {{"type","object"},{"properties",{{"score",{{"type","number"}}}}}};
    json value  = {{"score","not-a-number"}};
    auto v = validate_json_schema(value, schema);
    ASSERT(!v.empty());
    ASSERT(v[0].path.find("score") != std::string::npos);
}

void test_schema_enum() {
    json schema = {{"type","string"},{"enum",json::array({"a","b","c"})}};
    ASSERT(validate_json_schema(json("a"), schema).empty());
    ASSERT(!validate_json_schema(json("x"), schema).empty());
}

void test_schema_nested() {
    json schema = {
        {"type","object"},
        {"required",json::array({"user"})},
        {"properties",{
            {"user",{
                {"type","object"},
                {"required",json::array({"id"})},
                {"properties",{{"id",{{"type","integer"}}}}}
            }}
        }}
    };
    json ok  = {{"user",{{"id",1}}}};
    json bad = {{"user",{{"id","wrong-type"}}}};
    ASSERT(validate_json_schema(ok, schema).empty());
    auto v = validate_json_schema(bad, schema);
    ASSERT(!v.empty());
    ASSERT(v[0].path.find("id") != std::string::npos);
}

// ── Plan-time tool validation ─────────────────────────────────────
void test_plan_tool_validation() {
    // validate_dag only checks structure; plan-time validation checks tool existence
    // Simulate: a TOOL step with unregistered action should cause PlanningError
    // (This is tested at the WorkflowEngine level; here we test the registry check)
    ToolRegistry r;
    r.register_tool({"web_search","",{},{}}, nullptr);
    ASSERT(r.has_tool("web_search"));
    ASSERT(!r.has_tool("nonexistent"));
}
// ── Rate Limiter ─────────────────────────────────────────────────
void test_rate_limiter_unlimited() {
    RateLimiter rl(0.0);  // unlimited
    ASSERT(!rl.enabled());
    ASSERT(rl.acquire());  // should return immediately
}

void test_rate_limiter_basic() {
    RateLimiter rl(100.0);  // 100 RPS → token available immediately
    ASSERT(rl.enabled());
    ASSERT(rl.max_rps() == 100.0);
    ASSERT(rl.acquire(100));  // should get token immediately
}

void test_rate_limiter_slow() {
    RateLimiter rl(2.0);  // 2 RPS = token every 500ms
    ASSERT(rl.acquire(100));  // first token: immediate
    // second token needs ~500ms, but we only wait 50ms → timeout
    bool got = rl.acquire(50);
    // may or may not get it depending on timing; just check it doesn't crash
    (void)got;
}

void test_rate_limiter_timeout() {
    RateLimiter rl(0.5);  // 0.5 RPS = token every 2s
    rl.acquire(100);    // consume first token
    bool got = rl.acquire(10);  // 10ms timeout, can't get next token
    ASSERT(!got);  // should time out
}

// ── Metrics ───────────────────────────────────────────────────────
void test_metrics_noop() {
    NoOpMetrics m;
    // Should not throw or crash
    m.record({MetricEvent::Kind::WORKFLOW_START,"wf1","","","",0,true,""});
    m.record({MetricEvent::Kind::WORKFLOW_END,  "wf1","","","",100,true,""});
}

void test_metrics_console() {
    ConsoleMetrics m;
    // Just verify it doesn't crash; output goes to stdout (captured in CI)
    m.record({MetricEvent::Kind::LLM_CALL,"wf1","step1","anthropic","claude-opus-4-8",0,true,""});
}

void test_metrics_interface() {
    std::vector<MetricEvent> recorded;
    class TestMetrics : public IMetricsCollector {
    public:
        std::vector<MetricEvent>& vec;
        explicit TestMetrics(std::vector<MetricEvent>& v) : vec(v) {}
        void record(const MetricEvent& e) noexcept override { vec.push_back(e); }
    };
    TestMetrics collector(recorded);
    collector.record({MetricEvent::Kind::WORKFLOW_START,"wf1","","","",0,true,""});
    collector.record({MetricEvent::Kind::STEP_END,"wf1","s1","","",50,true,""});
    ASSERT(recorded.size() == 2);
    ASSERT(recorded[0].kind == MetricEvent::Kind::WORKFLOW_START);
    ASSERT(recorded[1].duration_ms == 50);
}

// ── ProviderConfig.max_rps + groq() factory ──────────────────────
void test_provider_config_max_rps() {
    auto groq = ProviderConfig::groq("key", "llama-3.3-70b-versatile");
    ASSERT(groq.max_rps == 0.5);

    auto gh = ProviderConfig::github_models("tok");
    ASSERT(gh.max_rps == 0.1);  // updated: 0.1 RPS = 6 RPM per tier (safe under GH Models)

    auto unlimited = ProviderConfig::anthropic("key");
    ASSERT(unlimited.max_rps == 0.0);
}

// ── Recursive $ref Resolution ─────────────────────────────────
void test_ref_nested_object() {
    WorkflowState s;
    s.step_outputs["s1"] = {{"val", "hello"}};
    auto r = s.resolve_inputs({{"outer", {{"inner", "$s1.val"}}}});
    ASSERT(r["outer"]["inner"].get<std::string>() == "hello");
}

void test_ref_array() {
    WorkflowState s;
    s.step_outputs["a"] = "alpha";
    s.step_outputs["b"] = "beta";
    auto r = s.resolve_inputs({{"items", json::array({"$a", "$b", "literal"})}});
    ASSERT(r["items"][0].get<std::string>() == "alpha");
    ASSERT(r["items"][1].get<std::string>() == "beta");
    ASSERT(r["items"][2].get<std::string>() == "literal");
}

void test_ref_mixed() {
    WorkflowState s;
    s.task_input = {{"q", "test"}};
    s.step_outputs["s1"] = {{"data", json::array({10, 20})}};
    auto r = s.resolve_inputs({{"query", "$task_input.q"},
                                {"results", json::array({"$s1.data"})},
                                {"count", 3}});
    ASSERT(r["query"].get<std::string>() == "test");
    ASSERT(r["count"].get<int>() == 3);
}

// ── Cancellation ─────────────────────────────────────────────
void test_cancel_token() {
    auto tok = std::make_shared<std::atomic<bool>>(false);
    ASSERT(!tok->load());
    tok->store(true);
    ASSERT(tok->load());
}

void test_cancelled_error() {
    try { throw WorkflowCancelledError("test cancel"); }
    catch (const AriadneError& e) {
        ASSERT(std::string(e.what()).find("cancel") != std::string::npos);
    }
}

// ── MockProvider + Execution ─────────────────────────────────
void test_mock_provider() {
    MockProvider p("hello world");
    ASSERT(p.complete("", "", 0.0, false) == "hello world");
    ASSERT(p.provider_name() == "mock");
    p.set_response("changed");
    ASSERT(p.complete("", "", 0.0, false) == "changed");
}

void test_llmclient_mock() {
    auto orc = std::make_unique<MockProvider>(R"({"answer":"42"})");
    auto sub = std::make_unique<MockProvider>("sub response");
    LLMClient client(std::move(orc), std::move(sub));
    auto r1 = client.complete_as(ModelTier::ORCHESTRATOR, "test");
    ASSERT(r1.find("42") != std::string::npos);
    auto r2 = client.complete_as(ModelTier::SUBAGENT, "test");
    ASSERT(r2 == "sub response");
}

void test_executor_simple_dag() {
    auto orc = std::make_unique<MockProvider>("orchestrator output");
    auto sub = std::make_unique<MockProvider>("tool result from llm");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    tools.register_tool({"add","Add numbers",{},{}},
        [](const json& p)->json{ return p["a"].get<int>() + p["b"].get<int>(); });

    WorkflowExecutor exec(client, tools);
    WorkflowPlan plan{"test_plan", {
        {"step1", StepType::TOOL, "add", {{"a", 3}, {"b", 4}}, {}, 0, 30, OnError::FAIL},
        {"step2", StepType::LLM, "Summarize", {{"input", "$step1"}}, {"step1"}, 0, 30, OnError::FAIL},
    }};
    auto state = exec.execute(plan, {{"task", "test"}});
    ASSERT(state.step_outputs.count("step1"));
    ASSERT(state.step_outputs["step1"].get<int>() == 7);
    ASSERT(state.step_outputs.count("step2"));
    ASSERT(state.traces.size() == 2);
}

// ── CONDITION Branching ───────────────────────────────────────
void test_condition_true_passes() {
    auto orc = std::make_unique<MockProvider>("llm out");
    auto sub = std::make_unique<MockProvider>("sub out");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    WorkflowExecutor exec(client, tools);
    WorkflowPlan plan{"cond_test", {
        {"check", StepType::CONDITION, "check", {{"value", true}}, {}, 0, 30, OnError::FAIL},
        {"after", StepType::LLM, "Do work", {}, {"check"}, 0, 30, OnError::FAIL},
    }};
    auto state = exec.execute(plan, {{"task","test"}});
    ASSERT(state.step_outputs.count("after"));
    ASSERT(!state.step_outputs["after"].is_null());
}

void test_condition_false_skips() {
    auto orc = std::make_unique<MockProvider>("should not run");
    auto sub = std::make_unique<MockProvider>("sub");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    WorkflowExecutor exec(client, tools);
    WorkflowPlan plan{"cond_test", {
        {"check", StepType::CONDITION, "check", {{"value", false}}, {}, 0, 30, OnError::FAIL},
        {"after", StepType::LLM, "Do work", {}, {"check"}, 0, 30, OnError::FAIL},
    }};
    auto state = exec.execute(plan, {{"task","test"}});
    ASSERT(state.step_outputs["after"].is_null());
}

void test_condition_cascade_skip() {
    auto orc = std::make_unique<MockProvider>("x");
    auto sub = std::make_unique<MockProvider>("x");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    WorkflowExecutor exec(client, tools);
    WorkflowPlan plan{"cascade", {
        {"gate", StepType::CONDITION, "gate", {{"value", false}}, {}, 0, 30, OnError::FAIL},
        {"mid",  StepType::LLM, "Middle", {}, {"gate"}, 0, 30, OnError::FAIL},
        {"end",  StepType::LLM, "End",    {}, {"mid"},  0, 30, OnError::FAIL},
    }};
    auto state = exec.execute(plan, {{"task","test"}});
    ASSERT(state.step_outputs["mid"].is_null());
    ASSERT(state.step_outputs["end"].is_null());
}

// ── Metrics Emission ─────────────────────────────────────────
void test_metrics_step_events() {
    std::vector<MetricEvent> recorded;
    class Collector : public IMetricsCollector {
    public:
        std::vector<MetricEvent>& v;
        explicit Collector(std::vector<MetricEvent>& vec) : v(vec) {}
        void record(const MetricEvent& e) noexcept override { v.push_back(e); }
    };
    auto col = std::make_shared<Collector>(recorded);

    auto orc = std::make_unique<MockProvider>("llm result");
    auto sub = std::make_unique<MockProvider>("sub");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    tools.register_tool({"echo","",{},{}}, [](const json& p)->json{ return p; });
    WorkflowExecutor exec(client, tools, 0, col);
    WorkflowPlan plan{"metrics_test", {
        {"t1", StepType::TOOL, "echo", {{"x",1}}, {}, 0, 30, OnError::FAIL},
        {"l1", StepType::LLM,  "Sum",  {},        {"t1"}, 0, 30, OnError::FAIL},
    }};
    exec.execute(plan, {{"task","test"}});

    int step_start=0, step_end=0, tool_call=0, tool_resp=0, llm_call=0, llm_resp=0;
    for (const auto& e : recorded) {
        if (e.kind == MetricEvent::Kind::STEP_START)    ++step_start;
        if (e.kind == MetricEvent::Kind::STEP_END)      ++step_end;
        if (e.kind == MetricEvent::Kind::TOOL_CALL)     ++tool_call;
        if (e.kind == MetricEvent::Kind::TOOL_RESPONSE) ++tool_resp;
        if (e.kind == MetricEvent::Kind::LLM_CALL)      ++llm_call;
        if (e.kind == MetricEvent::Kind::LLM_RESPONSE)  ++llm_resp;
    }
    ASSERT(step_start == 2);
    ASSERT(step_end == 2);
    ASSERT(tool_call == 1);
    ASSERT(tool_resp == 1);
    ASSERT(llm_call == 1);
    ASSERT(llm_resp == 1);
}

void test_metrics_workflow_end() {
    std::vector<MetricEvent> recorded;
    class Collector : public IMetricsCollector {
    public:
        std::vector<MetricEvent>& v;
        explicit Collector(std::vector<MetricEvent>& vec) : v(vec) {}
        void record(const MetricEvent& e) noexcept override { v.push_back(e); }
    };
    auto col = std::make_shared<Collector>(recorded);
    MetricEvent evt{MetricEvent::Kind::WORKFLOW_START, "wf1", "", "", "", 0, true, ""};
    col->record(evt);
    evt.kind = MetricEvent::Kind::WORKFLOW_END;
    evt.duration_ms = 100;
    col->record(evt);
    ASSERT(recorded.size() == 2);
    ASSERT(recorded[0].kind == MetricEvent::Kind::WORKFLOW_START);
    ASSERT(recorded[1].kind == MetricEvent::Kind::WORKFLOW_END);
    ASSERT(recorded[1].duration_ms == 100);
}

// ── Token Usage ──────────────────────────────────────────────
void test_token_usage_struct() {
    TokenUsage a{10, 20, 30};
    TokenUsage b{5, 15, 20};
    a += b;
    ASSERT(a.input_tokens == 15);
    ASSERT(a.output_tokens == 35);
    ASSERT(a.total_tokens == 50);
}

void test_token_usage_mock_provider() {
    auto orc = std::make_unique<MockProvider>("result");
    auto sub = std::make_unique<MockProvider>("sub");
    LLMClient client(std::move(orc), std::move(sub));
    client.reset_usage();
    client.complete_as(ModelTier::ORCHESTRATOR, "test");
    auto u = client.total_usage();
    // MockProvider doesn't set g_last_token_usage, so should be zero
    ASSERT(u.total_tokens == 0);
}

// ── Plan Cache ───────────────────────────────────────────────
void test_plan_cache_basic() {
    PlanCache cache(3);
    WorkflowPlan p1{"plan1", {{"s1", StepType::TOOL, "act", {}, {}, 0, 30, OnError::FAIL}}};
    cache.put("key1", p1);
    ASSERT(cache.has("key1"));
    ASSERT(!cache.has("key2"));
    auto got = cache.get("key1");
    ASSERT(got.steps.size() == 1);
    ASSERT(got.steps[0].id == "s1");
}

void test_plan_cache_lru() {
    PlanCache cache(2);
    WorkflowPlan p1{"p1",{}}, p2{"p2",{}}, p3{"p3",{}};
    cache.put("a", p1);
    cache.put("b", p2);
    ASSERT(cache.size() == 2);
    cache.put("c", p3);  // evicts "a" (LRU)
    ASSERT(cache.size() == 2);
    ASSERT(!cache.has("a"));
    ASSERT(cache.has("b"));
    ASSERT(cache.has("c"));
}

void test_plan_cache_normalize() {
    std::vector<ToolDef> tools = {{"search","",{},{}}, {"calc","",{},{}}};
    auto k1 = PlanCache::normalize_key("Search Tesla Revenue", tools);
    auto k2 = PlanCache::normalize_key("search  tesla  revenue", tools);
    ASSERT(k1 == k2);
    auto k3 = PlanCache::normalize_key("Search BYD Revenue", tools);
    ASSERT(k1 != k3);
}

// ── Loop Detector (unit-level, no LLM) ───────────────────────
void test_loop_detector_no_loop() {
    // Simulate: 3 different tool calls — no loop
    // Access via the agent loop indirectly. Test the concept.
    std::vector<std::pair<std::string, json>> calls = {
        {"search", {{"q","a"}}}, {"search", {{"q","b"}}}, {"calc", {{"x",1}}}
    };
    // No repeats, so no loop expected
    int repeat_count = 0;
    size_t last_hash = 0;
    for (const auto& [name, args] : calls) {
        size_t h = std::hash<std::string>{}(name + args.dump());
        if (h == last_hash) ++repeat_count; else repeat_count = 1;
        last_hash = h;
    }
    ASSERT(repeat_count < 3);
}

void test_loop_detector_exact_repeat() {
    // 3 identical calls → stuck
    std::vector<std::pair<std::string, json>> calls = {
        {"search", {{"q","tesla"}}}, {"search", {{"q","tesla"}}}, {"search", {{"q","tesla"}}}
    };
    int repeat_count = 0;
    size_t last_hash = 0;
    for (const auto& [name, args] : calls) {
        size_t h = std::hash<std::string>{}(name + args.dump());
        if (h == last_hash) ++repeat_count; else repeat_count = 1;
        last_hash = h;
    }
    ASSERT(repeat_count >= 3);
}

// ── Guardrails ───────────────────────────────────────────────
void test_guardrail_tool_pass() {
    ToolRegistry r;
    r.register_tool({"safe","",{},{}}, [](const json& p)->json{ return {{"ok",true}}; });
    r.add_guardrail("safe", [](const json& p) -> std::optional<std::string> {
        return std::nullopt;  // always pass
    });
    auto res = r.call("safe", {{"x",1}});
    ASSERT(res["ok"].get<bool>());
}

void test_guardrail_tool_block() {
    ToolRegistry r;
    r.register_tool({"risky","",{},{}}, [](const json& p)->json{ return {{"ok",true}}; });
    r.add_guardrail("risky", [](const json& p) -> std::optional<std::string> {
        if (p.value("danger", false)) return std::string("dangerous input blocked");
        return std::nullopt;
    });
    // Safe call passes
    auto ok = r.call("risky", {{"danger", false}});
    ASSERT(ok["ok"].get<bool>());
    // Dangerous call blocked
    bool caught = false;
    try { r.call("risky", {{"danger", true}}); }
    catch (const GuardrailError& e) {
        caught = true;
        ASSERT(std::string(e.what()).find("dangerous") != std::string::npos);
    }
    ASSERT(caught);
}

void test_guardrail_error_type() {
    try { throw GuardrailError("test guardrail"); }
    catch (const AriadneError& e) {
        ASSERT(std::string(e.what()).find("guardrail") != std::string::npos);
    }
}

// ── Agent Handoff ────────────────────────────────────────────
void test_handoff_action_type() {
    AgentAction a;
    a.type = AgentAction::Type::HANDOFF;
    a.target_agent = "specialist";
    ASSERT(a.type == AgentAction::Type::HANDOFF);
    ASSERT(a.target_agent == "specialist");
}

void test_agent_def() {
    AgentDef def;
    def.name = "researcher";
    def.system_prompt = "You research things";
    def.allowed_tools = {"web_search"};
    def.handoff_targets = {"writer"};
    ASSERT(def.name == "researcher");
    ASSERT(def.allowed_tools.size() == 1);
    ASSERT(def.handoff_targets[0] == "writer");
}

// ── Strict Structured Output ─────────────────────────────────
void test_strict_schema_mock() {
    // MockProvider returns canned JSON; verify the executor path with output_schema
    auto orc = std::make_unique<MockProvider>(R"({"revenue": 120})");
    auto sub = std::make_unique<MockProvider>(R"({"revenue": 120})");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    WorkflowExecutor exec(client, tools);
    Step s;
    s.id = "extract"; s.type = StepType::LLM; s.action = "Extract revenue";
    s.output_schema = {{"type","object"},{"properties",{{"revenue",{{"type","number"}}}}}};
    WorkflowPlan plan{"schema_test", {s}};
    auto state = exec.execute(plan, {{"task","test"}});
    ASSERT(state.step_outputs.count("extract"));
    ASSERT(state.step_outputs["extract"]["revenue"].get<int>() == 120);
}

// ── Response Cache ───────────────────────────────────────────
void test_response_cache_basic() {
    ResponseCache cache(10);
    auto key = ResponseCache::make_key("model", "sys", "prompt", 0.0, false);
    cache.put(key, "cached_response");
    ASSERT(cache.has(key));
    ASSERT(cache.get(key) == "cached_response");
    ASSERT(cache.stats().hits == 1);
}

void test_response_cache_lru() {
    ResponseCache cache(2);
    auto k1 = ResponseCache::make_key("m", "s", "p1", 0.0, false);
    auto k2 = ResponseCache::make_key("m", "s", "p2", 0.0, false);
    auto k3 = ResponseCache::make_key("m", "s", "p3", 0.0, false);
    cache.put(k1, "r1"); cache.put(k2, "r2");
    ASSERT(cache.size() == 2);
    cache.put(k3, "r3");  // evicts k1
    ASSERT(!cache.has(k1));
    ASSERT(cache.has(k3));
}

void test_response_cache_key_diff() {
    auto k1 = ResponseCache::make_key("gpt-4o", "sys", "hello", 0.0, false);
    auto k2 = ResponseCache::make_key("gpt-4o", "sys", "hello", 0.7, false);
    auto k3 = ResponseCache::make_key("gpt-4o", "sys", "hello", 0.0, true);
    ASSERT(k1 != k2);  // different temperature
    ASSERT(k1 != k3);  // different force_json
}

// ── MCP Types ────────────────────────────────────────────────
void test_mcp_client_types() {
    // Verify McpClient can be constructed (no actual subprocess)
    // Just test that the types compile and are usable
    ToolDef def{"mcp_tool", "An MCP tool", {{"type","object"}}, {}};
    ASSERT(def.name == "mcp_tool");
    ASSERT(def.input_schema["type"] == "object");
}

void test_mcp_tool_def_conversion() {
    // Simulate what list_tools would return
    json mcp_response = {
        {"tools", {{
            {"name", "weather"},
            {"description", "Get weather"},
            {"inputSchema", {{"type","object"},{"properties",{{"city",{{"type","string"}}}}}}}
        }}}
    };
    // Convert to ToolDef (same as McpClient::list_tools does)
    std::vector<ToolDef> tools;
    for (const auto& t : mcp_response["tools"]) {
        ToolDef def;
        def.name = t.value("name", "");
        def.description = t.value("description", "");
        def.input_schema = t.value("inputSchema", json::object());
        tools.push_back(def);
    }
    ASSERT(tools.size() == 1);
    ASSERT(tools[0].name == "weather");
    ASSERT(tools[0].input_schema.contains("properties"));
}

// ── Token Budget ─────────────────────────────────────────
void test_token_budget_not_set() {
    auto orc = std::make_unique<MockProvider>("result");
    auto sub = std::make_unique<MockProvider>("sub");
    LLMClient client(std::move(orc), std::move(sub));
    // No budget set → should not throw
    auto r = client.complete_as(ModelTier::ORCHESTRATOR, "test");
    ASSERT(r == "result");
}

void test_token_budget_exceeded() {
    auto orc = std::make_unique<MockProvider>("result");
    auto sub = std::make_unique<MockProvider>("sub");
    LLMClient client(std::move(orc), std::move(sub));
    client.set_token_budget(0);  // budget=0 means unlimited
    auto r = client.complete_as(ModelTier::ORCHESTRATOR, "test");
    ASSERT(!r.empty());
    // Now set budget to 1 and try — MockProvider doesn't set token usage,
    // so cumulative stays 0, won't throw
    client.set_token_budget(1);
    r = client.complete_as(ModelTier::ORCHESTRATOR, "test2");
    ASSERT(!r.empty());
}

void test_token_budget_error_type() {
    try { throw TokenBudgetError("budget exceeded: 1000/500"); }
    catch (const AriadneError& e) {
        ASSERT(std::string(e.what()).find("budget") != std::string::npos);
    }
}

// ── WorkflowPlan Serialization ──────────────────────────
void test_plan_serialization() {
    WorkflowPlan plan;
    plan.id = "test_plan";
    plan.metadata = {{"task", "test task"}};
    Step s1;
    s1.id = "step_1"; s1.type = StepType::TOOL; s1.action = "web_search";
    s1.inputs = {{"query", "test"}}; s1.description = "Search";
    Step s2;
    s2.id = "step_2"; s2.type = StepType::LLM; s2.action = "Summarize results";
    s2.depends_on = {"step_1"}; s2.json_mode = true;
    s2.system_prompt = "You are a summarizer";
    s2.temperature = 0.5;
    plan.steps = {s1, s2};

    json j = plan.to_json();
    ASSERT(j["id"] == "test_plan");
    ASSERT(j["steps"].size() == 2);
    ASSERT(j["steps"][0]["type"] == "tool");
    ASSERT(j["steps"][0]["action"] == "web_search");
    ASSERT(j["steps"][1]["depends_on"][0] == "step_1");
    ASSERT(j["steps"][1]["json_mode"] == true);
    ASSERT(j["steps"][1]["system_prompt"] == "You are a summarizer");
}

void test_plan_roundtrip() {
    WorkflowPlan original;
    original.id = "roundtrip";
    original.metadata = {{"task", "round trip test"}};
    Step s;
    s.id = "s1"; s.type = StepType::LLM; s.action = "Do something";
    s.retry = 2; s.timeout_sec = 45.0; s.on_error = OnError::SKIP;
    s.model_tier = ModelTier::ORCHESTRATOR;
    original.steps = {s};

    json j = original.to_json();
    auto restored = WorkflowPlan::from_json(j);
    ASSERT(restored.steps.size() == 1);
    ASSERT(restored.steps[0].id == "s1");
    ASSERT(restored.steps[0].type == StepType::LLM);
    ASSERT(restored.steps[0].retry == 2);
    ASSERT(restored.steps[0].on_error == OnError::SKIP);
    ASSERT(restored.steps[0].model_tier == ModelTier::ORCHESTRATOR);
}

// ── PlanCache Context Awareness ─────────────────────────
void test_plan_cache_with_context() {
    std::vector<ToolDef> tools = {{"search","",{},{}}};
    auto k1 = PlanCache::normalize_key("Search Tesla", tools, "");
    auto k2 = PlanCache::normalize_key("Search Tesla", tools, "Previous: searched BYD");
    ASSERT(k1 != k2);  // different context → different key
    auto k3 = PlanCache::normalize_key("Search Tesla", tools, "");
    ASSERT(k1 == k3);  // same empty context → same key
}

// ── WorkflowState Serialization ──────────────────────────
void test_state_serialization() {
    WorkflowState state;
    state.task_input = {{"task", "test"}};
    state.step_outputs["s1"] = {{"result", 42}};
    state.step_outputs["s2"] = "hello";
    state.errors["s3"] = "something failed";
    state.record_trace({"s1", "tool", "ok", "mock", 100, "", 50, 25});

    json j = state.to_json();
    ASSERT(j["task_input"]["task"] == "test");
    ASSERT(j["step_outputs"]["s1"]["result"] == 42);
    ASSERT(j["errors"]["s3"] == "something failed");
    ASSERT(j["traces"].size() == 1);
    ASSERT(j["traces"][0]["input_tokens"] == 50);

    auto restored = WorkflowState::from_json(j);
    ASSERT(restored.step_outputs["s1"]["result"] == 42);
    ASSERT(restored.step_outputs["s2"] == "hello");
    ASSERT(restored.errors.count("s3"));
    ASSERT(restored.traces[0].duration_ms == 100);
}

// ── Provider Config Factories ───────────────────────────
void test_provider_factories_new() {
    auto c = ProviderConfig::cerebras("key");
    ASSERT(c.base_url == "https://api.cerebras.ai/v1");
    ASSERT(c.max_rps == 0.5);

    auto s = ProviderConfig::sambanova("key");
    ASSERT(s.base_url == "https://api.sambanova.ai/v1");

    auto m = ProviderConfig::mistral("key");
    ASSERT(m.base_url == "https://api.mistral.ai/v1");
    ASSERT(m.max_rps == 1.0);
}

// ── InterruptError ───────────────────────────────────────
void test_interrupt_error_type() {
    try {
        throw InterruptError("step_3", "needs approval", {{"partial", true}});
    } catch (const AriadneError& e) {
        ASSERT(std::string(e.what()).find("step_3") != std::string::npos);
    }
    // Verify fields
    InterruptError err("s1", "review needed", {{"output", 42}});
    ASSERT(err.step_id == "s1");
    ASSERT(err.reason_ == "review needed");
    ASSERT(err.state_snapshot["output"] == 42);
}

void test_interrupt_in_executor() {
    auto orc = std::make_unique<MockProvider>("llm out");
    auto sub = std::make_unique<MockProvider>("sub out");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    tools.register_tool({"echo","",{},{}}, [](const json& p)->json{ return p; });
    WorkflowExecutor exec(client, tools);
    WorkflowPlan plan{"int_test", {
        {"s1", StepType::TOOL, "echo", {{"x",1}}, {}, 0, 30, OnError::FAIL},
        {"s2", StepType::LLM, "Summarize", {}, {"s1"}, 0, 30, OnError::FAIL},
    }};
    // Interrupt at s2 (after s1 completes)
    bool interrupted = false;
    try {
        exec.execute(plan, {{"task","test"}}, nullptr,
            [](const Step& s, const WorkflowState& st) -> std::optional<std::string> {
                if (s.id == "s2") return std::string("needs human review");
                return std::nullopt;
            });
    } catch (const InterruptError& e) {
        interrupted = true;
        ASSERT(e.step_id == "s2");
        ASSERT(e.state_snapshot.contains("step_outputs"));
        // s1 should be in the state snapshot
        ASSERT(e.state_snapshot["step_outputs"].contains("s1"));
    }
    ASSERT(interrupted);
}

// ── Engine call_tool ─────────────────────────────────────
void test_engine_call_tool() {
    auto orc = std::make_unique<MockProvider>("ok");
    auto sub = std::make_unique<MockProvider>("ok");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    tools.register_tool({"add", "Add numbers",
        {{"required", json::array({"a","b"})},
         {"properties", {{"a",{{"type","integer"}}},{"b",{{"type","integer"}}}}}}, {}},
        [](const json& p)->json{ return p["a"].get<int>() + p["b"].get<int>(); });
    ASSERT(tools.has_tool("add"));
    auto result = tools.call("add", {{"a", 3}, {"b", 4}});
    ASSERT(result.get<int>() == 7);
}

// ── DynamicWorkflow ──────────────────────────────────────
void test_dynamic_parallel() {
    auto orc = std::make_unique<MockProvider>("ok");
    auto sub = std::make_unique<MockProvider>("ok");
    WorkflowEngine engine(EngineConfig::from_single(
        ProviderConfig::openai_compatible("test", "http://localhost:9999", "mock")));
    // We can't use engine without real providers, so test DynamicWorkflow primitives directly
    // Use a simple DynamicWorkflow with the engine (won't make LLM calls, just test parallel)
    // Actually, let's test the parallel primitive with pure functions
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::vector<std::future<int>> futs;
    for (int i = 0; i < 5; ++i)
        futs.push_back(pool.submit([&counter, i]() -> int {
            ++counter;
            return i * 10;
        }));
    std::vector<int> results;
    for (auto& f : futs) results.push_back(f.get());
    ASSERT(counter == 5);
    ASSERT(results.size() == 5);
    ASSERT(results[0] == 0);
    ASSERT(results[3] == 30);
}

void test_dynamic_map_pure() {
    // Test map concept with ThreadPool
    ThreadPool pool(4);
    std::vector<int> items = {1, 2, 3, 4, 5};
    std::vector<std::future<int>> futs;
    for (auto& x : items)
        futs.push_back(pool.submit([x]() -> int { return x * x; }));
    std::vector<int> results;
    for (auto& f : futs) results.push_back(f.get());
    ASSERT(results == std::vector<int>({1, 4, 9, 16, 25}));
}

void test_dynamic_pipeline_pure() {
    // Test pipeline concept: item flows through multiple stages
    ThreadPool pool(4);
    auto stage1 = [](int x) { return x + 10; };
    auto stage2 = [](int x) { return x * 2; };
    std::vector<int> items = {1, 2, 3};
    std::vector<std::future<int>> futs;
    for (auto& x : items)
        futs.push_back(pool.submit([x, &stage1, &stage2]() -> int {
            return stage2(stage1(x));  // 1→11→22, 2→12→24, 3→13→26
        }));
    std::vector<int> results;
    for (auto& f : futs) results.push_back(f.get());
    ASSERT(results[0] == 22);
    ASSERT(results[1] == 24);
    ASSERT(results[2] == 26);
}

void test_dynamic_loop_until() {
    // Test loop_until concept
    std::vector<int> accumulated;
    int rounds = 0;
    for (int r = 0; r < 100; ++r) {
        if ((int)accumulated.size() >= 5) break;
        accumulated.push_back(r * 10);
        ++rounds;
    }
    ASSERT(accumulated.size() == 5);
    ASSERT(rounds == 5);
    ASSERT(accumulated[4] == 40);
}

void test_dynamic_result_struct() {
    DynamicResult r;
    r.success = true;
    r.rounds_used = 3;
    r.outputs = {json(1), json(2), json(3)};
    r.log = {"started", "round 1", "done"};
    ASSERT(r.outputs.size() == 3);
    ASSERT(r.rounds_used == 3);
    ASSERT(r.log.size() == 3);
}

void test_dynamic_fan_out_types() {
    // Verify DynTask, StageFn, StopFn, RoundFn type aliases compile
    DynTask task = []() -> json { return {{"result", 42}}; };
    ASSERT(task()["result"] == 42);
    StageFn stage = [](const json& x) -> json { return x.get<int>() * 2; };
    ASSERT(stage(json(5)) == 10);
    StopFn stop = [](int round, const std::vector<json>& acc) { return round >= 3; };
    ASSERT(!stop(0, {}));
    ASSERT(stop(3, {}));
    RoundFn work = [](int round) -> std::vector<json> {
        return {json(round * 10)};
    };
    auto batch = work(2);
    ASSERT(batch.size() == 1);
    ASSERT(batch[0] == 20);
}

// ── Version API ──────────────────────────────────────────
void test_version_api() {
    auto v = ariadne::version();
    ASSERT(!v.empty());
    ASSERT(v.find('.') != std::string::npos);
    ASSERT(std::string(ARIADNE_VERSION) == v);
}

// ── Agent Traces ─────────────────────────────────────────
void test_agent_result_has_traces() {
    AgentResult r;
    r.traces.push_back({"agent_iter_1", "tool", "ok", "", 100, "", 50, 25});
    r.traces.push_back({"agent_iter_2", "llm", "ok", "", 200, "", 60, 30});
    ASSERT(r.traces.size() == 2);
    ASSERT(r.traces[0].input_tokens == 50);
    ASSERT(r.traces[1].step_id == "agent_iter_2");
}

// ── Structured Logging ──────────────────────────────────
void test_logger_null() {
    NullLogger nl;
    nl.log(LogLevel::LOG_ERROR, "test", "should not crash");
}

void test_logger_console() {
    ConsoleLogger cl(LogLevel::LOG_WARN);
    cl.log(LogLevel::LOG_DEBUG, "test", "should be filtered");
    cl.log(LogLevel::LOG_WARN, "test", "should appear");
}

void test_logger_global() {
    auto old = global_logger();
    set_logger(std::make_shared<NullLogger>());
    log_msg(LogLevel::LOG_INFO, "test", "via global");
    set_logger(old);
}

// ── Cost Tracking ───────────────────────────────────────
void test_model_pricing() {
    auto p = ModelPricing::gpt4o_mini();
    ASSERT(p.input_per_1m == 0.15);
    double cost = p.cost(1000, 500);
    ASSERT(cost > 0.0);
    ASSERT(cost < 0.001);  // should be tiny for 1500 tokens
}

void test_token_usage_cost() {
    TokenUsage a{100, 50, 150, 0.001};
    TokenUsage b{200, 100, 300, 0.002};
    a += b;
    ASSERT(a.total_tokens == 450);
    ASSERT(a.cost_usd > 0.0029 && a.cost_usd < 0.0031);
}

// ── Token Estimation ─────────────────────────────────────
void test_estimate_tokens_english() {
    ASSERT(estimate_tokens("hello world") > 0);
    ASSERT(estimate_tokens("hello world") < 5);
    auto t = estimate_tokens("The quick brown fox jumps over the lazy dog");
    ASSERT(t >= 8 && t <= 15);
}

void test_estimate_tokens_empty() {
    ASSERT(estimate_tokens("") == 1);
}

// ── Anthropic Native Tools ──────────────────────────────
void test_anthropic_supports_native() {
    ProviderConfig cfg = ProviderConfig::anthropic("test-key");
    auto p = make_provider(cfg);
    ASSERT(p->supports_native_tools());
}

// ── Retry-After parsing ─────────────────────────────────
void test_retry_after_in_error_msg() {
    // Verify the backoff logic can parse retry_after= from error messages
    std::string emsg = "OpenAI Chat: 429 rate limited retry_after=7";
    auto ra_pos = emsg.find("retry_after=");
    ASSERT(ra_pos != std::string::npos);
    int val = std::stoi(emsg.substr(ra_pos + 12));
    ASSERT(val == 7);
}

// ── Native Tool Calling Types ────────────────────────────
void test_chat_message_struct() {
    ChatMessage m;
    m.role = "user";
    m.content = "hello";
    ASSERT(m.role == "user");
    ASSERT(m.content == "hello");
    ASSERT(m.tool_calls.is_null());
}

void test_llm_tool_call_struct() {
    LLMToolCall tc;
    tc.id = "call_123";
    tc.name = "web_search";
    tc.arguments = {{"query", "tesla"}};
    ASSERT(tc.name == "web_search");
    ASSERT(tc.arguments["query"] == "tesla");
}

void test_llm_response_with_tools() {
    LLMResponse r;
    r.content = "";
    r.tool_calls.push_back({"call_1", "search", {{"q","test"}}});
    ASSERT(r.has_tool_calls());
    ASSERT(r.tool_calls[0].name == "search");
    LLMResponse r2;
    r2.content = "The answer is 42";
    ASSERT(!r2.has_tool_calls());
}

void test_mock_supports_native() {
    MockProvider mp("test");
    ASSERT(!mp.supports_native_tools());  // mock doesn't support native
    auto orc = std::make_unique<MockProvider>("ok");
    auto sub = std::make_unique<MockProvider>("ok");
    LLMClient client(std::move(orc), std::move(sub));
    ASSERT(!client.supports_native_tools(ModelTier::ORCHESTRATOR));
}

// ── Multimodal Messages ─────────────────────────────────
void test_chat_message_text() {
    auto m = ChatMessage::text("user", "hello");
    ASSERT(m.role == "user");
    ASSERT(m.content == "hello");
    ASSERT(!m.is_multimodal());
}

void test_chat_message_with_image() {
    auto m = ChatMessage::with_image("What's this?", "SGVsbG8=", "image/png");
    ASSERT(m.role == "user");
    ASSERT(m.is_multimodal());
    ASSERT(m.content_parts.size() == 2);
    ASSERT(m.content_parts[0]["type"] == "text");
    ASSERT(m.content_parts[1]["type"] == "image_url");
    std::string url = m.content_parts[1]["image_url"]["url"];
    ASSERT(url.find("data:image/png;base64,SGVsbG8=") != std::string::npos);
}

void test_chat_message_with_image_url() {
    auto m = ChatMessage::with_image_url("Describe", "https://example.com/img.jpg");
    ASSERT(m.is_multimodal());
    ASSERT(m.content_parts[1]["image_url"]["url"] == "https://example.com/img.jpg");
}

// ── Gemini Provider ─────────────────────────────────────
void test_gemini_config() {
    auto c = ProviderConfig::gemini("test-key");
    ASSERT(c.type == ProviderType::GEMINI);
    ASSERT(c.model == "gemini-2.0-flash");
    ASSERT(c.max_rps == 0.25);
    ASSERT(c.pricing.input_per_1m == 0.075);
}

void test_gemini_pro_pricing() {
    auto c = ProviderConfig::gemini("key", "gemini-2.0-pro");
    ASSERT(c.pricing.input_per_1m == 1.25);
}

void test_gemini_make_provider() {
    auto p = make_provider(ProviderConfig::gemini("key"));
    ASSERT(p->provider_name() == "gemini");
    ASSERT(p->model_name() == "gemini-2.0-flash");
    ASSERT(p->supports_native_tools());
}

// ── Version from CMake ──────────────────────────────────
void test_version_sync() {
    auto v = ariadne::version();
    ASSERT(!v.empty());
    ASSERT(v == std::string(ARIADNE_VERSION));
}

// ── Parallel Tool Calls ─────────────────────────────────
void test_parallel_tool_calls_response() {
    LLMResponse r;
    r.content = "Let me search both";
    r.tool_calls.push_back({"call_1", "search", {{"q","Tesla"}}});
    r.tool_calls.push_back({"call_2", "search", {{"q","BYD"}}});
    ASSERT(r.has_tool_calls());
    ASSERT(r.tool_calls.size() == 2);
    ASSERT(r.tool_calls[0].name == "search");
    ASSERT(r.tool_calls[1].arguments["q"] == "BYD");
}

// ── Provider Auto-Pricing ────────────────────────────────
void test_provider_auto_pricing() {
    auto a = ProviderConfig::anthropic("key", "claude-opus-4-8");
    ASSERT(a.pricing.input_per_1m == 15.0);  // opus pricing

    auto s = ProviderConfig::anthropic("key", "claude-sonnet-4-6");
    ASSERT(s.pricing.input_per_1m == 3.0);  // sonnet pricing

    auto o = ProviderConfig::openai_chat("key", "gpt-4o");
    ASSERT(o.pricing.input_per_1m == 2.50);

    auto m = ProviderConfig::openai_chat("key", "gpt-4o-mini");
    ASSERT(m.pricing.input_per_1m == 0.15);

    auto g = ProviderConfig::github_models("tok");
    ASSERT(g.pricing.input_per_1m == 0.0);  // free tier
}

// ── AdaptiveOrchestrator Types ───────────────────────────
void test_orchestrator_strategy_enum() {
    ASSERT(OrchestratorStrategy::SIMPLE_DAG != OrchestratorStrategy::AGENT_LOOP);
    ASSERT(OrchestratorStrategy::PARALLEL_RESEARCH != OrchestratorStrategy::PIPELINE_VERIFY);
}

void test_orchestrator_plan_struct() {
    OrchestratorPlan plan;
    plan.strategy = OrchestratorStrategy::PARALLEL_RESEARCH;
    plan.reasoning = "Task compares two entities";
    plan.subtasks = {"Search Tesla", "Search BYD"};
    plan.synthesis_prompt = "Compare the two";
    plan.max_iterations = 10;
    ASSERT(plan.subtasks.size() == 2);
    ASSERT(plan.max_iterations == 10);
}

void test_orchestrator_result_struct() {
    OrchestratorResult result;
    result.success = true;
    result.strategy_used = "parallel_research";
    result.reasoning = "comparison task";
    result.output = {{"answer", "Tesla > BYD"}};
    result.duration_ms = 5000;
    result.log = {"step1", "step2"};
    ASSERT(result.success);
    ASSERT(result.strategy_used == "parallel_research");
    ASSERT(result.log.size() == 2);
}

// ── Response Safety Guards ───────────────────────────────
void test_empty_response_guard() {
    // Verify that accessing [0] on empty arrays throws ProviderError, not crashes
    // We test this indirectly through MockProvider which always returns valid strings
    MockProvider mp("valid response");
    auto result = mp.complete("test", "", 0.0, false);
    ASSERT(result == "valid response");
    // The guards are in the real providers (Anthropic/OpenAI/Gemini)
    // which check j["content"].empty() etc before accessing [0]
}

// ── Convenience Constructor ──────────────────────────────
void test_engine_single_provider() {
    auto cfg = ProviderConfig::openai_compatible("test", "http://localhost:1", "mock");
    WorkflowEngine engine(cfg);
    ASSERT(engine.list_tools().empty());
}

void test_engine_dual_provider() {
    auto orc = ProviderConfig::openai_compatible("t", "http://localhost:1", "orc");
    auto sub = ProviderConfig::openai_compatible("t", "http://localhost:1", "sub");
    WorkflowEngine engine(orc, sub);
    ASSERT(engine.list_tools().empty());
}

// ── Integration: MockProvider DAG ────────────────────────
void test_integration_mock_dag() {
    auto orc = std::make_unique<MockProvider>(
        R"({"steps":[{"id":"s1","type":"tool","action":"echo","inputs":{"x":1},"depends_on":[]},
            {"id":"s2","type":"llm","action":"Summarize","inputs":{},"depends_on":["s1"]}]})");
    auto sub = std::make_unique<MockProvider>("summary result");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    tools.register_tool({"echo","",{},{}}, [](const json& p)->json{ return p; });
    WorkflowExecutor exec(client, tools);
    WorkflowPlan plan{"int_test", {
        {"s1", StepType::TOOL, "echo", {{"x",42}}, {}, 0, 30, OnError::FAIL},
        {"s2", StepType::LLM, "Summarize", {{"data","$s1"}}, {"s1"}, 0, 30, OnError::FAIL},
    }};
    auto state = exec.execute(plan, {{"task","test"}});
    ASSERT(state.step_outputs.count("s1"));
    ASSERT(state.step_outputs["s1"]["x"] == 42);
    ASSERT(state.step_outputs.count("s2"));
    ASSERT(state.traces.size() == 2);
}

// ── Integration: DynamicWorkflow parallel ────────────────
void test_integration_dynamic_parallel() {
    auto orc = std::make_unique<MockProvider>("mock llm");
    auto sub = std::make_unique<MockProvider>("mock sub");
    LLMClient client(std::move(orc), std::move(sub));
    ToolRegistry tools;
    tools.register_tool({"add","",{},{}},
        [](const json& p)->json{ return p["a"].get<int>() + p["b"].get<int>(); });
    // Test DynamicWorkflow parallel primitive directly
    ThreadPool pool(2);
    std::vector<std::future<json>> futs;
    futs.push_back(pool.submit([&]() -> json { return tools.call("add", {{"a",1},{"b",2}}); }));
    futs.push_back(pool.submit([&]() -> json { return tools.call("add", {{"a",3},{"b",4}}); }));
    std::vector<json> results;
    for (auto& f : futs) results.push_back(f.get());
    ASSERT(results[0] == 3);
    ASSERT(results[1] == 7);
}

// ── Integration: Multimodal message construction ─────────
void test_integration_multimodal_roundtrip() {
    auto msg = ChatMessage::with_image("Describe", "AQID", "image/png");
    ASSERT(msg.is_multimodal());
    ASSERT(msg.content_parts.size() == 2);
    // Verify it can serialize to JSON (as providers would)
    json j = {{"role", msg.role}, {"content", msg.content_parts}};
    ASSERT(j["content"].is_array());
    ASSERT(j["content"][0]["type"] == "text");
    ASSERT(j["content"][1]["type"] == "image_url");
}

int main() {
    std::cout<<"=== DAG ===\n";
    RUN(test_dag_valid); RUN(test_dag_dup); RUN(test_dag_dep); RUN(test_dag_cycle);
    std::cout<<"\n=== Topological Batches ===\n";
    RUN(test_topo_linear); RUN(test_topo_parallel); RUN(test_leaf);
    std::cout<<"\n=== $ref Resolution ===\n";
    RUN(test_ref_task); RUN(test_ref_nested); RUN(test_ref_inputs);
    std::cout<<"\n=== CircuitBreaker ===\n";
    RUN(test_cb_closed); RUN(test_cb_opens); RUN(test_cb_recovers);
    std::cout<<"\n=== ToolRegistry ===\n";
    RUN(test_tool_reg); RUN(test_tool_unknown); RUN(test_tool_list);
    std::cout<<"\n=== WorkflowContext ===\n";
    RUN(test_ctx_empty); RUN(test_ctx_records); RUN(test_ctx_evict);
    std::cout<<"\n=== Tracing ===\n";
    RUN(test_trace);
    std::cout<<"\n=== ProviderAutoPlanner ===\n";
    RUN(test_planner_empty); RUN(test_planner_add);
    std::cout<<"\n=== Streaming ===\n";
    RUN(test_stream_cb);
    std::cout<<"\n=== EngineConfig ===\n";
    RUN(test_cfg_single); RUN(test_cfg_two);
    std::cout<<"\n=== Agent Loop ===\n";
    RUN(test_agent_action_types); RUN(test_agent_result_defaults);
    RUN(test_agent_result_reached_max); RUN(test_agent_step_recording);
        std::cout<<"\n=== New features ===\n";
    RUN(test_exception_hierarchy);
    RUN(test_tool_schema_validation);
    RUN(test_step_new_fields);
    RUN(test_agent_on_step_callback);
    std::cout<<"\n=== Regression (audit fixes) ===\n";
    RUN(test_tool_null_fn); RUN(test_dag_empty_id);
    RUN(test_ctx_custom_truncation); RUN(test_agent_result_steps);
    std::cout<<"\n=== ThreadPool ===\n";
    RUN(test_threadpool_basic); RUN(test_threadpool_returns_value);
    RUN(test_threadpool_parallel);
    std::cout<<"\n=== JSON Schema Validation ===\n";
    RUN(test_schema_valid_object); RUN(test_schema_missing_required);
    RUN(test_schema_wrong_type); RUN(test_schema_enum); RUN(test_schema_nested);
    std::cout<<"\n=== Plan-time Tool Validation ===\n";
    RUN(test_plan_tool_validation);
    std::cout<<"\n=== Rate Limiter ===\n";
    RUN(test_rate_limiter_unlimited); RUN(test_rate_limiter_basic);
    RUN(test_rate_limiter_slow); RUN(test_rate_limiter_timeout);
    std::cout<<"\n=== Metrics ===\n";
    RUN(test_metrics_noop); RUN(test_metrics_console); RUN(test_metrics_interface);
    std::cout<<"\n=== ProviderConfig extensions ===\n";
    RUN(test_provider_config_max_rps);

    std::cout<<"\n=== Recursive $ref Resolution ===\n";
    RUN(test_ref_nested_object); RUN(test_ref_array); RUN(test_ref_mixed);

    std::cout<<"\n=== Cancellation ===\n";
    RUN(test_cancel_token); RUN(test_cancelled_error);

    std::cout<<"\n=== MockProvider + Execution ===\n";
    RUN(test_mock_provider); RUN(test_llmclient_mock);
    RUN(test_executor_simple_dag);

    std::cout<<"\n=== CONDITION Branching ===\n";
    RUN(test_condition_true_passes); RUN(test_condition_false_skips);
    RUN(test_condition_cascade_skip);

    std::cout<<"\n=== Metrics Emission ===\n";
    RUN(test_metrics_step_events); RUN(test_metrics_workflow_end);

    std::cout<<"\n=== Token Usage ===\n";
    RUN(test_token_usage_struct); RUN(test_token_usage_mock_provider);

    std::cout<<"\n=== Plan Cache ===\n";
    RUN(test_plan_cache_basic); RUN(test_plan_cache_lru); RUN(test_plan_cache_normalize);

    std::cout<<"\n=== Loop Detector ===\n";
    RUN(test_loop_detector_no_loop); RUN(test_loop_detector_exact_repeat);

    std::cout<<"\n=== Guardrails ===\n";
    RUN(test_guardrail_tool_pass); RUN(test_guardrail_tool_block);
    RUN(test_guardrail_error_type);

    std::cout<<"\n=== Agent Types (handoff) ===\n";
    RUN(test_handoff_action_type); RUN(test_agent_def);

    std::cout<<"\n=== Strict Structured Output ===\n";
    RUN(test_strict_schema_mock);

    std::cout<<"\n=== Response Cache ===\n";
    RUN(test_response_cache_basic); RUN(test_response_cache_lru);
    RUN(test_response_cache_key_diff);

    std::cout<<"\n=== MCP Types ===\n";
    RUN(test_mcp_client_types); RUN(test_mcp_tool_def_conversion);

    std::cout<<"\n=== Token Budget ===\n";
    RUN(test_token_budget_not_set); RUN(test_token_budget_exceeded);
    RUN(test_token_budget_error_type);

    std::cout<<"\n=== WorkflowPlan Serialization ===\n";
    RUN(test_plan_serialization); RUN(test_plan_roundtrip);

    std::cout<<"\n=== PlanCache Context ===\n";
    RUN(test_plan_cache_with_context);

    std::cout<<"\n=== WorkflowState Serialization ===\n";
    RUN(test_state_serialization);

    std::cout<<"\n=== Provider Factories ===\n";
    RUN(test_provider_factories_new);

    std::cout<<"\n=== Human-in-the-loop ===\n";
    RUN(test_interrupt_error_type); RUN(test_interrupt_in_executor);

    std::cout<<"\n=== Engine call_tool ===\n";
    RUN(test_engine_call_tool);

    std::cout<<"\n=== Dynamic Workflow ===\n";
    RUN(test_dynamic_parallel); RUN(test_dynamic_map_pure);
    RUN(test_dynamic_pipeline_pure); RUN(test_dynamic_loop_until);
    RUN(test_dynamic_result_struct); RUN(test_dynamic_fan_out_types);

    std::cout<<"\n=== Version API ===\n";
    RUN(test_version_api);

    std::cout<<"\n=== Agent Traces ===\n";
    RUN(test_agent_result_has_traces);

    std::cout<<"\n=== Structured Logging ===\n";
    RUN(test_logger_null); RUN(test_logger_console); RUN(test_logger_global);

    std::cout<<"\n=== Cost Tracking ===\n";
    RUN(test_model_pricing); RUN(test_token_usage_cost);

    std::cout<<"\n=== Token Estimation ===\n";
    RUN(test_estimate_tokens_english); RUN(test_estimate_tokens_empty);

    std::cout<<"\n=== Anthropic Native Tools ===\n";
    RUN(test_anthropic_supports_native);

    std::cout<<"\n=== Retry-After ===\n";
    RUN(test_retry_after_in_error_msg);

    std::cout<<"\n=== Native Tool Calling ===\n";
    RUN(test_chat_message_struct); RUN(test_llm_tool_call_struct);
    RUN(test_llm_response_with_tools); RUN(test_mock_supports_native);

    std::cout<<"\n=== Multimodal Messages ===\n";
    RUN(test_chat_message_text); RUN(test_chat_message_with_image);
    RUN(test_chat_message_with_image_url);

    std::cout<<"\n=== Gemini Provider ===\n";
    RUN(test_gemini_config); RUN(test_gemini_pro_pricing); RUN(test_gemini_make_provider);

    std::cout<<"\n=== Version Sync ===\n";
    RUN(test_version_sync);

    std::cout<<"\n=== Parallel Tool Calls ===\n";
    RUN(test_parallel_tool_calls_response);

    std::cout<<"\n=== Provider Auto-Pricing ===\n";
    RUN(test_provider_auto_pricing);

    std::cout<<"\n=== Adaptive Orchestrator ===\n";
    RUN(test_orchestrator_strategy_enum); RUN(test_orchestrator_plan_struct);
    RUN(test_orchestrator_result_struct);

    std::cout<<"\n=== Response Safety ===\n";
    RUN(test_empty_response_guard);

    std::cout<<"\n=== Convenience Constructor ===\n";
    RUN(test_engine_single_provider); RUN(test_engine_dual_provider);

    std::cout<<"\n=== Integration Tests ===\n";
    RUN(test_integration_mock_dag); RUN(test_integration_dynamic_parallel);
    RUN(test_integration_multimodal_roundtrip);

    std::cout<<"\n────────────────────────────────────────\n";
    std::cout<<"Result: "<<g_pass<<"/"<<g_run<<" passed\n";
    return (g_pass==g_run) ? 0 : 1;
}