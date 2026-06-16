#include "../ariadne.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
using namespace ariadne;

template<typename Fn>
static void bench(const std::string& name, int iters, Fn fn) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << std::left << std::setw(42) << name
              << std::right << std::setw(7) << iters << " iters  "
              << std::setw(9) << us << " us  "
              << std::setw(6) << (iters > 0 ? us / iters : 0) << " us/op\n";
    std::cout.flush();
}

int main() {
    std::cout << "Ariadne v" << ariadne::version() << " Benchmark\n";
    std::cout << std::string(76, '-') << "\n";

    // DAG validation
    WorkflowPlan plan{"b", {
        {"a", StepType::TOOL, "t", {}, {}, 0, 30, OnError::FAIL},
        {"b", StepType::TOOL, "t", {}, {}, 0, 30, OnError::FAIL},
        {"c", StepType::LLM, "s", {}, {"a","b"}, 0, 30, OnError::FAIL},
    }};
    bench("validate_dag (3 steps)", 50000, [&]{ validate_dag(plan); });
    bench("topological_batches (3 steps)", 50000, [&]{ plan.topological_batches(); });

    // Plan serialization
    bench("WorkflowPlan to_json", 10000, [&]{ plan.to_json(); });
    // Cache operations
    PlanCache pc(100);
    WorkflowPlan cp{"p", {{"s", StepType::LLM, "a", {}, {}, 0, 30, OnError::FAIL}}};
    std::vector<ToolDef> tools = {{"t","",{},{}}};
    auto pk = PlanCache::normalize_key("task", tools);
    pc.put(pk, cp);
    bench("PlanCache get (hit)", 50000, [&]{ pc.get(pk); });

    ResponseCache rc(200);
    auto rk = ResponseCache::make_key("m", "s", "p", 0.0, false);
    rc.put(rk, "cached");
    bench("ResponseCache get (hit)", 50000, [&]{ rc.get(rk); });

    // JSON Schema validation
    json schema = {{"type","object"},{"required",json::array({"name"})},
                   {"properties",{{"name",{{"type","string"}}}}}};
    json value = {{"name","Alice"}};
    bench("validate_json_schema", 50000, [&]{ validate_json_schema(value, schema); });

    // $ref resolution
    WorkflowState state;
    state.task_input = {{"q","test"}};
    state.step_outputs["s1"] = {{"d", json::array({1,2})}};
    bench("resolve_inputs (2 refs)", 50000, [&]{
        state.resolve_inputs({{"a","$task_input.q"},{"b","$s1.d"}});
    });

    // Token estimation
    std::string text(500, 'x');
    bench("estimate_tokens (500 chars)", 100000, [&]{ estimate_tokens(text); });

    // CircuitBreaker
    CircuitBreaker cb(3, 60);
    bench("CircuitBreaker try_allow", 100000, [&]{ cb.try_allow(); });

    // ThreadPool
    {
        ThreadPool pool(4);
        bench("ThreadPool submit+get", 500, [&]{
            auto f = pool.submit([]{ return 1; });
            (void)f.get();
        });
    }

    std::cout << std::string(76, '-') << "\nDone.\n";
    return 0;
}
