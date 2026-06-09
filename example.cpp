#ifdef _MSC_VER
#pragma warning(disable:4996) // getenv
#endif
#include "ariadne.hpp"
#include <iostream>
#include <cstdlib>
using namespace ariadne;

static void add_tools(WorkflowEngine& e) {
    e.register_tool({"web_search","Search the web",
        {{"required",json::array({"query"})},{"properties",{{"query",{{"type","string"}}}}}},{}},
        [](const json& p)->json{
            std::string q=p.value("query","");
            std::cout<<"  [tool] web_search: "<<q<<"\n";
            return{{"results",{{{"title",q+" Overview"},{"snippet",q+": $120B revenue 2024."}}}}};});
}

int main(int argc,char* argv[]) {
    const char* key=std::getenv("GITHUB_TOKEN");
    if(!key){std::cerr<<"Set GITHUB_TOKEN\n";return 1;}
    std::string mode=(argc>1)?argv[1]:"dag";
    WorkflowEngine engine(EngineConfig::from_single(
        ProviderConfig::github_models(key,"openai/gpt-4.1")));
    add_tools(engine);

    if(mode=="dag"){
        auto r=engine.run("Search Tesla Q4 2025 revenue and write a summary");
        if(r.success) std::cout<<"Output: "<<r.output.dump(2)<<"\nDuration: "<<r.duration_ms<<"ms\n";
        else          std::cerr<<"Failed: "<<r.error_message<<"\n";
    } else if(mode=="agent"){
        auto r=engine.run_agent("Research and compare Tesla vs BYD Q4 2025",8,
            [](const AgentStep& s){
                std::cout<<"["<<s.iteration<<"] "<<s.thought.substr(0,60)<<"\n";});
        if(r.success) std::cout<<"Answer: "<<r.final_answer<<"\n";
        else          std::cerr<<"Failed: "<<r.error<<"\n";
    } else if(mode=="probe"){
        ProviderAutoPlanner p;
        p.add_candidate("gpt-4.1",ProviderConfig::github_models(key,"openai/gpt-4.1"),"strong",1);
        p.add_candidate("gpt-4o-mini",ProviderConfig::github_models(key,"openai/gpt-4o-mini"),"fast",1);
        auto r=p.probe_and_plan(std::chrono::seconds(10));
        if(r.success) std::cout<<"ORCHESTRATOR: "<<r.alive_strong[0]<<"\nSUBAGENT: "<<r.alive_fast[0]<<"\n";
    }
}
