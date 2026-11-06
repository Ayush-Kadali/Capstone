#include "placementlens/search.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
int checks=0;
void check(bool yes,const char* why){++checks;if(!yes)throw std::runtime_error(why);}
void rejects(const std::function<void()>& fn,const char* code){
    bool found=false;try{fn();}catch(const std::exception& e){found=std::string(e.what()).find(code)!=std::string::npos;}
    check(found,code);
}
pl::Graph chain(std::size_t count,bool blocked_relu=false){
    pl::Graph graph;
    graph.inputs["X"]={{1,1},{1}};graph.inputs["B"]={{1,1},{1}};
    std::string previous="X";
    for(std::size_t i=0;i<count;++i){
        pl::Node node;node.id="node"+std::to_string(i);node.output="v"+std::to_string(i);
        node.op=blocked_relu && i%3==1?pl::Op::Relu:pl::Op::Add;
        node.inputs=node.op==pl::Op::Relu?std::vector<std::string>{previous}:std::vector<std::string>{previous,"B"};
        node.origins={node.id};graph.nodes.push_back(node);previous=node.output;
    }
    graph.output=previous;return pl::validate(std::move(graph));
}
bool contains(const std::vector<pl::Plan>& plans,const std::string& placement){
    return std::any_of(plans.begin(),plans.end(),[&](const pl::Plan& p){return pl::placement_name(p.placement)==placement;});
}
void invariants(const pl::BoundedSearchResult& result,const pl::BoundedSearchOptions& options){
    check(result.not_exhaustive,"bounded search must not claim exhaustive coverage");
    check(result.evaluated<=options.max_evaluations,"strict total lower-call budget");
    check(result.evaluated==result.evaluations.size(),"one evidence row per evaluation");
    check(result.baseline_evaluated>=1 && result.baseline_evaluated<=2,"baseline dedupe accounting");
    check(result.plans.size()<=options.beam_width+2,"retained full plans bounded");
    std::set<std::string> keys;std::size_t rejected=0;
    for(const auto& row:result.evaluations){
        check(keys.insert(row.placement).second,"each placement evaluated only once");
        if(!row.feasible){++rejected;check(row.rejection.find("E_MEMORY:")==0,"only explicit memory candidate rejection");}
        else check(std::isfinite(row.modeled_ms) && row.modeled_ms>=0,"finite modeled score");
    }
    check(rejected==result.rejected,"rejection count equals attempted rejected candidates");
    if(result.cpu_feasible)check(contains(result.plans,result.cpu_placement),"CPU baseline retained even outside beam");
    if(result.maximal_feasible)check(contains(result.plans,result.maximal_placement),"strict maximal baseline retained");
    else check(!contains(result.plans,result.maximal_placement),"infeasible strict maximal is never substituted");
    for(std::size_t i=1;i<result.plans.size();++i){
        const auto& a=result.plans[i-1];const auto& b=result.plans[i];
        check(a.modeled_ms<b.modeled_ms || (a.modeled_ms==b.modeled_ms && pl::placement_name(a.placement)<pl::placement_name(b.placement)),
              "total ranking is modeled cost then lexicographic placement");
    }
}
void exact_small_comparison(){
    for(std::size_t n=1;n<=6;++n)for(bool blocked:{false,true})for(double boundary:{0.,1.3,10.}){
        const auto graph=chain(n,blocked);pl::Config config;config.transfer_fixed_ms=boundary;config.reuse_buffers=true;
        const pl::BoundedSearchOptions options{32,256};
        const auto result=pl::bounded_search(graph,config,options);
        invariants(result,options);
        const auto exhaustive=pl::enumerate(graph,config);
        check(!result.plans.empty(),"small search has feasible CPU baseline");
        check(result.plans.front().modeled_ms==exhaustive.plans.front().modeled_ms,"small fixture quality matches exhaustive modeled oracle");
        for(const auto& plan:result.plans)check(pl::execute(graph,plan,config).correct,"small retained candidate executes correctly");
    }
}
void deterministic_budget_and_large_graphs(){
    for(const auto n:{32U,64U}){
        const auto graph=chain(n,true);pl::Config config;config.reuse_buffers=true;
        for(const auto budget:{2U,3U,17U,64U}){
            const pl::BoundedSearchOptions options{3,budget};
            const auto a=pl::bounded_search(graph,config,options),b=pl::bounded_search(graph,config,options);
            invariants(a,options);
            check(a.evaluated==budget && a.budget_exhausted,"large search reports budget exhaustion");
            check(a.evaluations.size()==b.evaluations.size(),"deterministic evaluation count");
            for(std::size_t i=0;i<a.evaluations.size();++i){
                check(a.evaluations[i].placement==b.evaluations[i].placement,"deterministic visit order");
                check(a.evaluations[i].modeled_ms==b.evaluations[i].modeled_ms && a.evaluations[i].rejection==b.evaluations[i].rejection,
                      "deterministic scores and rejections");
            }
            float expected=1;
            for(std::size_t i=0;i<n;++i)if(i%3!=1)expected+=1;
            for(const auto& plan:a.plans){
                const auto run=pl::execute(graph,plan,config);
                check(run.correct && run.output.values==std::vector<float>{expected},"large retained candidate matches manual sum");
            }
        }
        rejects([&]{pl::enumerate(graph,config);},"E_LIMIT");
    }
}
void memory_and_disabled_candidates(){
    const auto graph=chain(8);pl::Config config;config.device_bytes=16;
    const pl::BoundedSearchOptions options{4,64};
    const auto result=pl::bounded_search(graph,config,options);invariants(result,options);
    check(result.cpu_feasible && !result.maximal_feasible,"whole-plan memory rejects maximal despite per-node support");
    check(result.maximal_rejection.find("E_MEMORY:")==0,"strict maximal memory reason retained");
    check(result.rejected>0,"memory rejected candidates counted");
    for(const auto& plan:result.plans)check(pl::execute(graph,plan,config).correct,"memory-constrained candidates execute");
    auto disabled=chain(4);
    for(auto& node:disabled.nodes){node.op=pl::Op::Relu;node.inputs.resize(1);}
    disabled=pl::validate(disabled);
    const auto single=pl::bounded_search(disabled,config,{1,2});invariants(single,{1,2});
    check(single.evaluated==1 && single.baseline_evaluated==1 && single.plans.size()==1,"identical CPU/maximal deduped");
    check(!single.budget_exhausted && single.maximal_feasible,"no unseen legal neighborhood remains");
    config.mirrored_inputs={"X","B"};config.device_bytes=4;
    const auto none=pl::bounded_search(graph,config,options);invariants(none,options);
    check(none.plans.empty() && !none.cpu_feasible && !none.maximal_feasible,"unaffordable mandatory preparation rejects even CPU baseline honestly");
}
void invalid_options(){
    const auto graph=chain(1);pl::Config config;
    for(const auto options:{pl::BoundedSearchOptions{0,2},{33,2},{1,1},{1,2049}})
        rejects([&]{pl::bounded_search(graph,config,options);},"E_SEARCH");
    auto malformed=graph;malformed.inputs["X"].values.clear();
    rejects([&]{pl::bounded_search(malformed,config);},"E_VALUE");
}
void bounded_result_is_not_an_optimality_certificate(){
    const auto graph=chain(8);pl::Config config;
    config.device_bytes=16;config.transfer_fixed_ms=0;config.launch_ms=0;config.synchronization_ms=0;
    const auto limited=pl::bounded_search(graph,config,{1,2});
    const auto exhaustive=pl::enumerate(graph,config);
    check(limited.not_exhaustive && limited.budget_exhausted,"small budget has no optimality certificate");
    check(limited.plans.front().modeled_ms>exhaustive.plans.front().modeled_ms,
          "concrete bounded search quality gap is preserved, not mislabeled globally optimal");
}
}
int main(){try{exact_small_comparison();deterministic_budget_and_large_graphs();memory_and_disabled_candidates();invalid_options();bounded_result_is_not_an_optimality_certificate();
    std::cout<<"Bounded search contracts: "<<checks<<" checks passed\n";return 0;
}catch(const std::exception& e){std::cerr<<"Search test failed after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
