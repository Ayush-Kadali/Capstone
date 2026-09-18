#include "placementlens/core.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace pl {
namespace {
struct Costs {
    double compute = 0, copies = 0, launch = 0, sync = 0;
};

Costs costs(const Plan& plan, const Config& config) {
    Costs out;
    out.launch = static_cast<double>(plan.islands) * config.launch_ms;
    for (const auto& command : plan.commands) {
        if (command.kind == Kind::Compute) out.compute += command.modeled_ms;
        if (command.kind == Kind::Copy) out.copies += command.modeled_ms;
        if (command.kind == Kind::Synchronize) out.sync += command.modeled_ms;
    }
    // Lowering stores each device-run launch charge in its first compute command.
    out.compute -= out.launch;
    return out;
}

void describe_pass(std::ostream& out, const PassRecord& pass) {
    out << "### " << pass.name << "\n\n";
    for (const auto& change : pass.changes) out << "- " << change << '\n';
    for (const auto& reason : pass.reasons) out << "- Rule: " << reason << '\n';
    for (const auto& consequence : pass.consequences) out << "- Result: " << consequence << '\n';
    out << '\n';
}
}

std::string explain_report(const Graph& source, const Graph& compiled, const Config& config,
                           const std::vector<Plan>& plans, const std::vector<Run>& runs,
                           const SearchResult& search, const std::vector<PassRecord>& frontend_passes) {
    if (plans.empty() || plans.size() != runs.size() || search.plans.empty())
        throw std::runtime_error("E_REPORT: expected executed plans and a nonempty search");
    const auto& selected = plans.back();
    const auto& selected_run = runs.back();
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "# PlacementLens run report\n\n"
        << "Costs are illustrative model values. Arithmetic runs on the host with separate\n"
        << "virtual-device buffers. Host timings below are measured separately.\n\n"
        << "## Input and rules\n\n```text\n" << graph_text(source) << "```\n\n"
        << "- Device ReLU: " << (config.relu_supported ? "supported" : "unsupported") << '\n'
        << "- Retained device memory limit: " << config.device_bytes << " bytes\n"
        << "- Copy model: " << config.transfer_fixed_ms << " ms + bytes / "
        << config.bandwidth_bytes_per_ms << " bytes per ms\n"
        << "- Launch charge per device run: " << config.launch_ms << " ms\n"
        << "- Synchronization charge: " << config.synchronization_ms << " ms\n"
        << "- Compute model: fixed per opcode/backend, independent of matrix size\n\n"
        << "## Compiler decisions\n\n";
    for (const auto& pass : frontend_passes) describe_pass(out, pass);
    if (source.nodes.size() != compiled.nodes.size())
        out << "Compiled graph after fusion:\n\n```text\n" << graph_text(compiled) << "```\n\n";
    for (const auto& pass : selected.passes) describe_pass(out, pass);

    out << "## Placement comparison\n\nC = CPU, A = software device; letters follow compiled node order.\n\n"
        << "| Policy | Map | Compute ms | Copies ms | Launch ms | Sync ms | Total modeled ms | Output |\n"
        << "|---|---|---:|---:|---:|---:|---:|---|\n";
    for (std::size_t i = 0; i < plans.size(); ++i) {
        const auto& plan = plans[i];
        const auto parts = costs(plan, config);
        out << "| " << plan.policy << " | " << placement_name(plan.placement) << " | "
            << parts.compute << " | " << parts.copies << " | " << parts.launch << " | "
            << parts.sync << " | " << plan.modeled_ms << " | " << (runs[i].correct ? "PASS" : "FAIL") << " |\n";
    }
    const auto maximal = std::find_if(plans.begin(), plans.end(), [](const Plan& p) { return p.policy == "maximal"; });
    if (maximal == plans.end()) {
        out << "\nMaximal placement is infeasible under the retained-memory budget and was not executed.\n";
    } else {
        out << "\nChanges from maximal placement to the selected cost plan:\n\n";
        bool changed = false;
        for (std::size_t i = 0; i < compiled.nodes.size(); ++i) {
            if (maximal->placement[i] == selected.placement[i]) continue;
            changed = true;
            out << "- " << compiled.nodes[i].id << " (" << name(compiled.nodes[i].op) << "): "
                << name(maximal->placement[i]) << " -> " << name(selected.placement[i]) << '\n';
        }
        if (!changed) out << "- Both policies selected the same placement.\n";
        out << "- Copy bytes: " << maximal->h2d_bytes + maximal->d2h_bytes << " -> "
            << selected.h2d_bytes + selected.d2h_bytes << '\n'
            << "- Retained device bytes: " << maximal->peak_device_bytes << " -> " << selected.peak_device_bytes << '\n'
            << "- Synchronizations: " << maximal->synchronizations << " -> " << selected.synchronizations << '\n'
            << "- Total modeled cost reduction: " << maximal->modeled_ms - selected.modeled_ms << " ms\n";
    }

    out << "\n## Search evidence\n\nEnumerated " << search.plans.size() + search.infeasible
        << " assignments: " << search.plans.size() << " feasible, " << search.infeasible
        << " rejected by capability or retained-memory checks.\n\n"
        << "Lowest modeled cost: " << placement_name(search.plans.front().placement) << ". "
        << "Search is exhaustive only for the bounded graph and fixed schedule.\n"
        << "Comparison executes the listed policies; use `verify` to execute every feasible assignment.\n\n"
        << "| Feasible map | Modeled ms |\n|---|---:|\n";
    const auto shown = std::min(search.plans.size(), std::size_t{16});
    for (std::size_t i = 0; i < shown; ++i)
        out << "| " << placement_name(search.plans[i].placement) << " | " << search.plans[i].modeled_ms << " |\n";
    if (shown < search.plans.size()) out << "\nFirst 16 shown; `--json` contains the complete ranking.\n";

    out << "\n## Selected executable plan\n\n```text\n" << plan_text(selected) << "```\n\n"
        << "Compute commands retain the following source node IDs:\n\n";
    for (const auto& command : selected.commands) {
        if (command.kind != Kind::Compute) continue;
        out << "- Command " << command.id << " (" << name(command.op) << "):";
        for (const auto& origin : command.origins) out << ' ' << origin;
        out << '\n';
    }
    out << "\n## Execution checks\n\n";
    for (std::size_t i = 0; i < plans.size(); ++i)
        out << "- " << plans[i].policy << ": " << (runs[i].correct ? "PASS" : "FAIL")
            << ", maximum absolute error " << runs[i].max_abs_error << ", " << runs[i].events.size()
            << " commands executed, " << runs[i].host_ms << " ms measured host execution\n";
    out << "\nSelected output, row-major: [";
    const auto values = std::min(selected_run.output.values.size(), std::size_t{16});
    for (std::size_t i = 0; i < values; ++i) out << (i ? ", " : "") << selected_run.output.values[i];
    if (values < selected_run.output.values.size()) out << ", ...; full output in --json";
    out << "]\n\nReference: original graph, double-accumulated MatMul, FP32 results; atol=rtol=1e-5.\n\n"
        << "## Interpretation\n\n"
        << "The emitted commands were executed; bytes, placements and pass changes are inspectable.\n"
        << "The model-optimal policy wins its own objective by construction. That does not validate\n"
        << "hardware speedup. Compute costs do not scale with shape; launch cost is per device run;\n"
        << "execution is synchronous and retains every allocation until run end. There is no\n"
        << "hardware-calibrated cost model, native GPU code generation or asynchronous device queue.\n";
    return out.str();
}
}
