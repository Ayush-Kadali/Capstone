#include "placementlens/core.hpp"
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
const char* help = R"(PlacementLens 0.1 | Explainable hardware-aware graph placement

Usage: placementlens <inspect|compile|compare|run|demo> --graph FILE [options]

  inspect                 Show typed graph and capability decisions
  compile                 Show graph passes and explicit command plan
  compare                 Execute CPU, maximal, and modeled-optimal plans
  run                     Execute one selected policy and check correctness
  demo                    Compare policies and show winning command plan

Options:
  --policy cpu|maximal|cost  Policy for compile/run (default: cost)
  --relu-supported          Enable virtual-device ReLU capability
  --fuse                    Enable guarded MatMul -> ReLU fusion
  --boundary-ms NUMBER      Illustrative fixed cost per copy (default: 1.3)
  --device-bytes INTEGER     Retained device allocation limit (default: 1048576)
  --json FILE               Save complete graph, plans, costs, and results
  --trace FILE              Save measured host trace for the selected/last plan

C = CPU; A = virtual accelerator. Device execution is software on the host.
Modeled costs are illustrative, uncalibrated values, not measured speedups.
Supported input: bounded .plg graphs, f32, rank 2; MatMul, Add, ReLU.
The graph is never executed as source code. See showcase/README.md.
)";
double number(const std::string& s) {
    std::size_t used = 0;
    const auto value = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(value)) throw std::runtime_error("E_OPTION: expected a finite number");
    return value;
}
void save(const std::string& path, const std::string& data) {
    std::ofstream file(path);
    if (!file || !(file << data) || !file.flush()) throw std::runtime_error("E_IO: cannot write " + path);
    std::cout << "Saved: " << path << '\n';
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            std::cout << help; return 0;
        }
        const std::string action = argv[1];
        if (action != "inspect" && action != "compile" && action != "compare" && action != "run" && action != "demo")
            throw std::runtime_error("E_OPTION: unknown command; use --help");
        pl::Config config;
        bool fusion = false;
        std::string graph_path, policy = "cost", json_path, trace_path;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--relu-supported") { config.relu_supported = true; continue; }
            if (arg == "--fuse") { fusion = true; continue; }
            if (arg != "--graph" && arg != "--policy" && arg != "--json" && arg != "--trace" && arg != "--boundary-ms" && arg != "--device-bytes")
                throw std::runtime_error("E_OPTION: unknown option " + arg);
            if (++i == argc) throw std::runtime_error("E_OPTION: missing value for " + arg);
            const std::string value = argv[i];
            if (arg == "--graph") graph_path = value;
            else if (arg == "--policy") policy = value;
            else if (arg == "--json") json_path = value;
            else if (arg == "--trace") trace_path = value;
            else if (arg == "--boundary-ms") config.transfer_fixed_ms = number(value);
            else {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::runtime_error("E_OPTION: device-bytes requires an unsigned integer");
                std::size_t used = 0;
                const auto parsed = std::stoull(value, &used);
                config.device_bytes = static_cast<std::size_t>(parsed);
                if (used != value.size() || config.device_bytes != parsed) throw std::runtime_error("E_OPTION: memory budget out of range");
            }
        }
        if (graph_path.empty()) throw std::runtime_error("E_OPTION: --graph FILE is required");
        if (policy != "cpu" && policy != "maximal" && policy != "cost") throw std::runtime_error("E_POLICY: expected cpu, maximal, or cost");
        auto same_file = [](const std::string& a, const std::string& b) {
            if (a.empty() || b.empty()) return false;
            return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b) ||
                   (std::filesystem::exists(a) && std::filesystem::exists(b) && std::filesystem::equivalent(a,b));
        };
        if (same_file(json_path,graph_path) || same_file(trace_path,graph_path) || same_file(json_path,trace_path))
            throw std::runtime_error("E_OPTION: output files must differ from the input and each other");
        pl::validate_config(config);
        std::ifstream graph_file(graph_path);
        if (!graph_file) throw std::runtime_error("E_IO: cannot open graph " + graph_path);
        std::ostringstream raw;
        raw << graph_file.rdbuf();
        if (raw.str().size() > 16000000) throw std::runtime_error("E_LIMIT: graph file exceeds 16 MB");
        std::istringstream parser_input(raw.str());
        const auto source = pl::parse(parser_input);
        const auto passes = fusion ? pl::fuse_matmul_relu(source, config) : pl::PassResult{source, {"Fusion disabled (enable with --fuse)"}};
        const auto& graph = passes.graph;
        const std::vector<pl::PassRecord> frontend_passes = {
            {"ValidateGraphPass",raw.str(),pl::graph_text(source),
             {"Parse bounded source graph; infer f32 shapes; establish stable topological schedule"},
             {"Require unique definitions, defined operands, compatible shapes, and no cycles"},
             {"Only validated typed IR enters placement"}},
            {"MatMulReluFusionPass",pl::graph_text(source),pl::graph_text(graph),passes.remarks,
             {fusion ? "Require a single-use non-output intermediate and target ReLU support" : "Optional pass disabled"},
             {std::to_string(source.nodes.size()) + " source nodes -> " + std::to_string(graph.nodes.size()) + " compiled nodes"}}
        };
        std::cout << "\nPLACEMENTLENS  /  Boundary Conditions\n"
                     "Explainable compiler decisions, executable command plans\n"
                     "---------------------------------------------------------\n"
                  << pl::graph_text(source);
        for (const auto& remark : passes.remarks) std::cout << "  " << remark << '\n';
        if (graph.nodes.size() != source.nodes.size()) std::cout << pl::graph_text(graph);
        std::cout << "\nVirtual-device capability decisions\n";
        for (const auto& n : graph.nodes) {
            const auto cap = pl::capability(n, graph, config);
            std::cout << "  " << n.id << " : " << (cap.accepted ? "ACCEPT" : "REJECT") << " | " << cap.reason << '\n';
        }
        if (action == "inspect") {
            if (!json_path.empty() || !trace_path.empty()) throw std::runtime_error("E_OPTION: use compare/run/demo for execution reports");
            return 0;
        }
        const auto search = pl::enumerate(graph, config);
        std::vector<pl::Plan> plans;
        if (action == "compare" || action == "demo") {
            for (const std::string p : {"cpu", "maximal", "cost"}) {
                try { plans.push_back(pl::choose(graph, config, p)); }
                catch (const std::runtime_error& e) {
                    if (p != "maximal" || std::string(e.what()).find("E_MEMORY:") != 0) throw;
                    std::cout << "\nmaximal policy INFEASIBLE: " << e.what() << '\n';
                }
            }
        } else plans.push_back(pl::choose(graph, config, policy));
        std::cout << "\nCompiler pass history (selected/last policy)\n";
        auto show_pass = [](const pl::PassRecord& pass) {
            std::cout << "  " << pass.name << '\n';
            for (const auto& consequence : pass.consequences) std::cout << "    -> " << consequence << '\n';
        };
        for (const auto& pass : frontend_passes) show_pass(pass);
        for (const auto& pass : plans.back().passes) show_pass(pass);
        if (action == "compile") {
            if (!json_path.empty() || !trace_path.empty()) throw std::runtime_error("E_OPTION: use run/demo for execution reports");
            std::cout << '\n' << pl::plan_text(plans.back());
            std::cout << "Modeled cost: " << plans.back().modeled_ms << " ms (illustrative). No execution requested.\n";
            return 0;
        }
        std::vector<pl::Run> runs;
        bool correct = true;
        std::cout << "\nPOLICY       MAP       MODEL ms   H2D / D2H bytes   ISLANDS  RESULT\n";
        for (const auto& p : plans) {
            runs.push_back(pl::execute(source, p, config));
            const auto& r = runs.back(); correct = correct && r.correct;
            std::cout << std::left << std::setw(13) << p.policy << std::setw(10) << pl::placement_name(p.placement)
                      << std::right << std::fixed << std::setprecision(4) << std::setw(8) << p.modeled_ms
                      << std::setw(8) << p.h2d_bytes << " / " << std::setw(4) << p.d2h_bytes
                      << std::setw(11) << p.islands << "  " << (r.correct ? "PASS" : "FAIL") << '\n';
        }
        const auto& winner = search.plans.front();
        std::cout << "\nBounded modeled optimum: " << pl::placement_name(winner.placement)
                  << " across " << search.plans.size() << " feasible placements; " << search.infeasible << " rejected.\n"
                  << "The objective sums compute, copies, launch, and synchronization costs.\n"
                  << "No automatic dtype conversion or runtime replanning is performed.\n";
        if (action == "demo") std::cout << '\n' << pl::plan_text(plans.back());
        std::cout << "\nOutput (row-major): [";
        for (std::size_t i = 0; i < runs.back().output.values.size(); ++i)
            std::cout << (i ? ", " : "") << runs.back().output.values[i];
        std::cout << "]\nReference: original graph evaluated independently; atol=rtol=1e-5.\n"
                     "MODEL ms is synthetic. This is NOT a measured accelerator speedup.\n"
                     "Actual host execution time for last plan: " << runs.back().host_ms << " ms.\n";
        if (!json_path.empty()) save(json_path, pl::report_json(source, graph, config, passes.remarks, plans, runs, search, frontend_passes));
        if (!trace_path.empty()) save(trace_path, pl::trace_json(runs.back()));
        return correct ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "PlacementLens error: " << e.what() << '\n';
        return 1;
    }
}
