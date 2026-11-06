#pragma once
#include "placementlens/core.hpp"

namespace pl {
struct BoundedSearchOptions {
    std::size_t beam_width = 8; // 1..32 retained competitive candidates, plus mandatory baselines
    std::size_t max_evaluations = 128; // 2..2048 unique whole-plan lower calls, including baselines
};
struct CandidateEvaluation {
    std::string placement;
    bool feasible = false;
    double modeled_ms = 0; // meaningful only when feasible
    std::string rejection;
};
struct BoundedSearchResult {
    // At most beam_width+2 unique plans: the best candidates plus feasible CPU
    // and strict-maximal baselines. Sorted by modeled_ms then placement string.
    std::vector<Plan> plans;
    std::vector<CandidateEvaluation> evaluations;
    std::string cpu_placement, maximal_placement;
    bool cpu_feasible = false, maximal_feasible = false;
    std::string cpu_rejection, maximal_rejection;
    std::size_t evaluated = 0, rejected = 0, baseline_evaluated = 0;
    bool budget_exhausted = false;
    bool not_exhaustive = true; // Never an exhaustive or measured-oracle claim.
};
// Deterministic complete-placement beam neighborhoods: single-node flips and
// whole capability-island assignments. Every score includes actual lowering,
// memory feasibility, transfers, synchronization and illustrative modeled cost.
// max_evaluations includes CPU and strict-maximal baselines; an identical pair
// is evaluated once. Only E_MEMORY candidate failures are retained as rejections.
BoundedSearchResult bounded_search(const Graph& graph, const Config& config,
                                  const BoundedSearchOptions& options = {});
}
