#ifdef _MSC_VER
#pragma warning(disable:4996)
#endif
/**
 * eval_runner.cpp — Eval Harness CI
 * Uses GitHub Models (free) via GITHUB_TOKEN — zero secrets needed.
 * In Actions: add  permissions: models: read
 *
 * Scoring per case:
 *   Execution 60%: workflow.success + has_output (tests the framework)
 *   Content   40%: keyword fraction (bonus for relevance quality)
 *   Pass: score >= 0.50
 *   CI gate: 3/5 cases must pass (60%)
 */
#include "../ariadne.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
using namespace ariadne;

static void mock_tools(WorkflowEngine& e) {
    e.register_tool(
        {"web_search","Search the web",
         {{"required",json::array({"query"})},{"properties",{{"query",{{"type","string"}}}}}},{}},
        [](const json& p)->json{
            std::string q=p.value("query","topic");
            return{{"results",{{{"title",q+" Report"},
                {"snippet",q+": revenue $120B 2024, growth +18% YoY."}}}}};});
    e.register_tool(
        {"calculator","Evaluate math",
         {{"required",json::array({"expr"})},{"properties",{{"expr",{{"type","string"}}}}}},{}},
        [](const json&)->json{return{{"result",42}};});
}

struct Case{std::string id,task;std::vector<std::string> kws;};
static const std::vector<Case> CASES={
    {"tc_001","Search Tesla 2024 revenue and summarize in one sentence.",{"Tesla","revenue"}},
    {"tc_002","Search Tesla and BYD Q4 2025 EV sales separately, then compare.",{"Tesla","BYD"}},
    {"tc_003","Search Python 3.13 new features and list the top three.",{"Python","feature"}},
    {"tc_004","Search Apple 2025 market cap and give a one-sentence analysis.",{"Apple","billion"}},
    {"tc_005","Search and summarize the latest OpenAI language model released.",{"OpenAI","model"}},
};

int main(){
    const char* tok=std::getenv("GITHUB_TOKEN");
    if(!tok){
        std::cerr<<"ERROR: GITHUB_TOKEN not set\n"
                   "In Actions: add  permissions: models: read\n";
        return 2;
    }
    const char* menv=std::getenv("EVAL_MODEL");
    std::string model=menv?menv:"openai/gpt-4o-mini";
    double threshold=0.50;

    // Use LLM7.io as fallback if GitHub Models is rate-limited
    auto primary = ProviderConfig::github_models(tok, model);
    auto fallback = ProviderConfig::llm7("deepseek-v3-0324");
    TierConfig tier{primary, {fallback}, 3, 60.0};
    auto config = EngineConfig::with_fallbacks(tier, tier);

    std::cout<<"Eval Harness — GitHub Models + LLM7.io fallback\n"
             <<"Model: "<<model<<" (fallback: deepseek-v3-0324)\n\n";

    WorkflowEngine engine(config);
    mock_tools(engine);

    int passed=0; double total=0;
    json cases=json::array();

    for(const auto& tc:CASES){
        std::cout<<"["<<tc.id<<"] "<<tc.task.substr(0,58)<<"...\n";
        double score=0; bool ok=false; std::string reason;
        try{
            auto r=engine.run(tc.task);
            double exec=0;
            if(r.success)    exec+=0.4;
            if(r.has_output()) exec+=0.2;

            double kw=0;
            if(r.has_output()){
                std::string out=r.output.is_string()?r.output.get<std::string>():r.output.dump();
                std::string lo=out;
                std::transform(lo.begin(),lo.end(),lo.begin(),::tolower);
                int hits=0; std::string miss;
                for(const auto& kw_:tc.kws){
                    std::string kl=kw_;
                    std::transform(kl.begin(),kl.end(),kl.begin(),::tolower);
                    if(lo.find(kl)!=std::string::npos)++hits; else miss+=kw_+" ";
                }
                kw=tc.kws.empty()?0.4:0.4*((double)hits/tc.kws.size());
                reason=r.success?(hits==(int)tc.kws.size()?"all keywords found":
                       hits>0?"partial keywords; missing: "+miss:
                       "no keywords (LLM abstracted output)"):
                       r.error_message;
            } else reason=r.error_message;
            score=exec+kw; ok=(score>=threshold);
        }catch(const std::exception& e){reason=std::string("ex: ")+e.what();}
        if(ok)++passed;
        total+=score;
        std::cout<<"  "<<(ok?"PASS":"FAIL")
                 <<"  score="<<std::fixed<<std::setprecision(2)<<score
                 <<"  "<<reason<<"\n\n";
        cases.push_back({{"id",tc.id},{"passed",ok},{"score",score},{"reason",reason}});
    }

    double pr=(double)passed/CASES.size(), av=total/CASES.size();
    bool ci=(pr>=0.60);
    std::cout<<"══════════════════════════════\n"
             <<"Pass rate: "<<passed<<"/"<<CASES.size()<<" ("<<(int)(pr*100)<<"%)\n"
             <<"Avg score: "<<std::fixed<<std::setprecision(3)<<av<<"\n"
             <<"Result:    "<<(ci?"CI PASS":"CI FAIL")<<"\n";
    json rep={{"provider","github_models"},{"model",model},
               {"pass_rate",pr},{"avg_score",av},{"threshold",threshold},
               {"ci_pass",ci},{"cases",cases}};
    std::ofstream("eval_report.json")<<rep.dump(2);
    std::cout<<"Report → eval_report.json\n";
    return ci?0:1;
}
