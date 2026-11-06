#include "placementlens/search.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace pl {
namespace {
bool better(const Plan& left, const Plan& right) {
    if (left.modeled_ms != right.modeled_ms) return left.modeled_ms < right.modeled_ms;
    return placement_name(left.placement) < placement_name(right.placement);
}
void retain(std::vector<Plan>& plans, const Plan& candidate, std::size_t count) {
    if (std::any_of(plans.begin(),plans.end(),[&](const Plan& p){return p.placement==candidate.placement;})) return;
    plans.push_back(candidate);
    std::sort(plans.begin(),plans.end(),better);
    if(plans.size()>count)plans.resize(count);
}
}
BoundedSearchResult bounded_search(const Graph& input, const Config& config, const BoundedSearchOptions& options) {
    if(options.beam_width<1 || options.beam_width>32 || options.max_evaluations<2 || options.max_evaluations>2048)
        throw std::runtime_error("E_SEARCH: beam width must be 1..32 and total evaluation budget 2..2048");
    validate_config(config);
    const auto graph=validate(input);
    BoundedSearchResult result;
    const std::vector<Backend> cpu(graph.nodes.size(),Backend::Cpu);
    auto maximal=cpu;
    for(std::size_t i=0;i<graph.nodes.size();++i)
        if(capability(graph.nodes[i],graph,config).accepted)maximal[i]=Backend::Device;
    result.cpu_placement=placement_name(cpu);
    result.maximal_placement=placement_name(maximal);
    std::vector<std::pair<std::size_t,std::size_t>> islands;
    for(std::size_t i=0;i<maximal.size();) {
        if(maximal[i]==Backend::Cpu){++i;continue;}
        const auto start=i;
        while(i<maximal.size() && maximal[i]==Backend::Device)++i;
        islands.emplace_back(start,i);
    }
    std::map<std::string,std::size_t> seen;
    std::vector<Plan> best,baselines,frontier;
    auto evaluate=[&](const std::vector<Backend>& placement,std::vector<Plan>& next,bool baseline) {
        const auto key=placement_name(placement);
        if(seen.count(key))return true;
        if(result.evaluated==options.max_evaluations){result.budget_exhausted=true;return false;}
        CandidateEvaluation evidence;evidence.placement=key;
        ++result.evaluated;if(baseline)++result.baseline_evaluated;
        try {
            auto plan=lower(graph,placement,config);
            plan.policy=key==result.cpu_placement?"cpu":key==result.maximal_placement?"maximal":"bounded";
            evidence.feasible=true;evidence.modeled_ms=plan.modeled_ms;
            retain(best,plan,options.beam_width);
            retain(next,plan,options.beam_width);
            if(baseline)baselines.push_back(std::move(plan));
        }catch(const std::runtime_error& error) {
            if(std::string(error.what()).find("E_MEMORY:")!=0)throw;
            ++result.rejected;evidence.rejection=error.what();
        }
        seen.emplace(key,result.evaluations.size());
        result.evaluations.push_back(std::move(evidence));
        return true;
    };
    evaluate(cpu,frontier,true);
    evaluate(maximal,frontier,true);
    const auto& cpu_result=result.evaluations.at(seen.at(result.cpu_placement));
    result.cpu_feasible=cpu_result.feasible;result.cpu_rejection=cpu_result.rejection;
    const auto& maximal_result=result.evaluations.at(seen.at(result.maximal_placement));
    result.maximal_feasible=maximal_result.feasible;result.maximal_rejection=maximal_result.rejection;
    // Each round expands only newly evaluated feasible plans. Dedupe prevents
    // cycles; sorted frontiers and ascending node/island order fix all ties.
    while(!frontier.empty() && !result.budget_exhausted) {
        std::vector<Plan> next;
        for(const auto& current:frontier) {
            for(std::size_t i=0;i<maximal.size() && !result.budget_exhausted;++i) {
                if(maximal[i]!=Backend::Device)continue;
                auto neighbor=current.placement;
                neighbor[i]=neighbor[i]==Backend::Cpu?Backend::Device:Backend::Cpu;
                evaluate(neighbor,next,false);
            }
            for(const auto& island:islands) {
                if(result.budget_exhausted)break;
                for(const auto lane:{Backend::Device,Backend::Cpu}) {
                    auto neighbor=current.placement;
                    std::fill(neighbor.begin()+static_cast<std::ptrdiff_t>(island.first),
                              neighbor.begin()+static_cast<std::ptrdiff_t>(island.second),lane);
                    if(!evaluate(neighbor,next,false))break;
                }
            }
            if(result.budget_exhausted)break;
        }
        frontier=std::move(next);
    }
    result.plans=std::move(best);
    for(const auto& baseline:baselines)retain(result.plans,baseline,options.beam_width+2);
    return result;
}
}
