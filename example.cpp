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
        ProviderConfig::github_models(key,"openai/gpt-4o-mini")));
    add_tools(engine);

    if(mode=="dag"){
        auto r=engine.run("Search Tesla Q4 2025 revenue and write a summary");
        if(r.success){
            std::cout<<"Output: "<<r.output.dump(2)<<"\nDuration: "<<r.duration_ms<<"ms\n";
            std::cout<<"Tokens: in="<<r.token_usage.input_tokens
                     <<" out="<<r.token_usage.output_tokens
                     <<" total="<<r.token_usage.total_tokens<<"\n";
        } else std::cerr<<"Failed: "<<r.error_message<<"\n";
    } else if(mode=="agent"){
        auto r=engine.run_agent("Research and compare Tesla vs BYD Q4 2025",8,
            [](const AgentStep& s){
                std::cout<<"["<<s.iteration<<"] "<<s.thought.substr(0,60)<<"\n";});
        if(r.success) std::cout<<"Answer: "<<r.final_answer<<"\nTokens: "<<r.token_usage.total_tokens<<"\n";
        else          std::cerr<<"Failed: "<<r.error<<"\n";
    } else if(mode=="stream"){
        std::cout<<"Streaming: ";
        auto r=engine.run_stream("Write a one-sentence summary of electric vehicles",
            [](const std::string& chunk){ std::cout<<chunk<<std::flush; });
        std::cout<<"\n["<<(r.success?"OK":"FAIL")<<"] "<<r.duration_ms<<"ms\n";
    } else if(mode=="multi"){
        std::vector<AgentDef> agents = {
            {"researcher","You gather facts using tools, then hand off to the writer.",
             {"web_search"},ModelTier::ORCHESTRATOR,{"writer"}},
            {"writer","You write a concise summary from the research. Give a final_answer.",
             {},ModelTier::ORCHESTRATOR,{}}
        };
        auto r=engine.run_multi_agent("Research and summarize Tesla Q4 2025",agents,"researcher",10,
            [](const AgentStep& s){
                std::cout<<"["<<s.iteration<<"] "<<s.thought.substr(0,60)<<"\n";});
        if(r.success) std::cout<<"Answer: "<<r.final_answer<<"\n";
        else          std::cerr<<"Failed: "<<r.error<<"\n";
    } else if(mode=="probe"){
        ProviderAutoPlanner p;
        p.add_candidate("gpt-4.1",ProviderConfig::github_models(key,"openai/gpt-4o-mini"),"strong",1);
        p.add_candidate("gpt-4o-mini",ProviderConfig::github_models(key,"openai/gpt-4o-mini"),"fast",1);
        auto r=p.probe_and_plan(std::chrono::seconds(10));
        if(r.success) std::cout<<"ORCHESTRATOR: "<<r.alive_strong[0]<<"\nSUBAGENT: "<<r.alive_fast[0]<<"\n";
    }
}
