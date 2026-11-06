#include "placementlens/core.hpp"
#include "placementlens/search.hpp"
#include "placementlens/async.hpp"
#include "placementlens/queue.hpp"
#include "placementlens/device.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
const char* help = R"(PlacementLens | Explainable hardware-aware graph placement

Usage: placementlens <inspect|compile|compare|run|verify|explain|demo> --graph FILE [options]

  inspect                 Show typed graph and capability decisions
  compile                 Show graph passes and explicit command plan
  compare                 Execute CPU, maximal, and modeled-optimal plans
  run                     Execute one selected policy and check correctness
  verify                  Execute every feasible placement against the reference
  explain                 Walk through the decisions, plans, costs, and results
  demo                    Compare policies and show winning command plan

Options:
  --policy cpu|maximal|cost  Policy for compile/run (default: cost)
  --relu-supported          Enable virtual-device ReLU capability
  --fuse                    Enable guarded MatMul -> ReLU fusion
  --async                   Submit execution to the bounded runtime worker
  --runtime serial|command-queue  Executor: serial interpreter (default) or command-level
                            queue simulator with copy/compute engines (software only)
  --backend software|opencl Execution backend (default: software)
  --order-seed INTEGER      Reproducibly shuffle verify execution order
  --search exhaustive|bounded Search strategy for verify (default: exhaustive)
  --search-budget INTEGER    Bounded search whole-plan evaluations, 2..2048
  --beam-width INTEGER       Bounded search competitive candidates, 1..32
  --cnn-supported          Enable scalar software CNN/storage kernels on logical device
  --boundary-ms NUMBER      Illustrative fixed cost per copy (default: 1.3)
  --device-bytes INTEGER     Device allocation/pool limit (default: 1048576)
  --reuse-buffers            Free after last use and reuse exact-size slots
  --tile-size INTEGER        Software-device MatMul tile edge, 0..64 (default: 0)
  --local-bytes INTEGER      Tile scratch-memory limit (default: 32768)
  --sync-ms NUMBER           Illustrative synchronization cost (default: 0.05)
  --launch-ms NUMBER         Illustrative per-device-island launch cost (default: 0.2)
  --bandwidth NUMBER         Illustrative copy bytes/ms (default: 1000000)
  --resident-input NAME      Prepare a mirrored H/D input before timed execution
  --emulate-boundaries       Inject bounded host waits for configured boundary costs
  --device-unsupported OP    Declared target excludes OP (repeatable; e.g. ReLU, Conv2D)
  --device-max-tensor-bytes INTEGER  Declared per-tensor device limit (0 = none)
  --device-no-grouped-conv   Declared target rejects grouped/depthwise Conv2D
  --json FILE               Save complete graph, plans, costs, and results
  --trace FILE              Save measured host trace for the selected/last plan
  --report FILE             Save the explain output as Markdown (explain only)

Try these from the repository root:
  build/placementlens inspect --graph examples/fragmented.plg
  build/placementlens explain --graph examples/fragmented.plg
  build/placementlens explain --graph examples/fragmented.plg --relu-supported --fuse
  build/placementlens compare --graph examples/fragmented.plg --boundary-ms 5
  build/placementlens verify --graph examples/branched.plg

Explore one rule at a time: support, then copy cost, then memory. For example,
--device-bytes 80 rejects the maximal plan for fragmented.plg. The CPU plan
still works. Save evidence with explain --report run.md --json run.json.

C = CPU; A = device lane. Default software backend runs on the host.
OpenCL is an optional real GPU backend, not a BeagleY-AI adapter.
Modeled costs are illustrative, uncalibrated values, not measured speedups.
Supported internal input: up to 64 .plg nodes, contiguous f32 storage; matrix/CNN ops.
Use scripts/explore_model.py or the workbench for bounded standard ONNX input.
Exhaustive verification is limited to 12 nodes; --search bounded samples candidates.
The graph is never executed as source code. See showcase/README.md.
)";
double number(const std::string& s) {
    std::size_t used = 0;
    const auto value = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(value)) throw std::runtime_error("E_OPTION: expected a finite number");
    return value;
}
void validate_utf8(const std::string& text) {
    // Source comments are preserved verbatim in JSON pass snapshots.
    for (std::size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i++]);
        if (first < 0x80) continue;
        unsigned int codepoint = 0, minimum = 0;
        std::size_t trailing = 0;
        if (first >= 0xc2 && first <= 0xdf) { trailing = 1; minimum = 0x80; codepoint = first & 0x1fU; }
        else if (first >= 0xe0 && first <= 0xef) { trailing = 2; minimum = 0x800; codepoint = first & 0x0fU; }
        else if (first >= 0xf0 && first <= 0xf4) { trailing = 3; minimum = 0x10000; codepoint = first & 0x07U; }
        else throw std::runtime_error("E_PARSE: graph source must be valid UTF-8");
        if (trailing > text.size() - i) throw std::runtime_error("E_PARSE: truncated UTF-8 source");
        for (std::size_t j = 0; j < trailing; ++j) {
            const auto next = static_cast<unsigned char>(text[i++]);
            if ((next & 0xc0U) != 0x80U) throw std::runtime_error("E_PARSE: invalid UTF-8 continuation");
            codepoint = (codepoint << 6U) | (next & 0x3fU);
        }
        if (codepoint < minimum || codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff))
            throw std::runtime_error("E_PARSE: invalid UTF-8 code point");
    }
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
        if (action != "inspect" && action != "compile" && action != "compare" && action != "run" && action != "verify" && action != "explain" && action != "demo")
            throw std::runtime_error("E_OPTION: unknown command; use --help");
        pl::Config config;
        bool fusion = false, asynchronous = false, shuffle_order = false;
        std::string runtime = "serial"; // serial | command-queue
        std::uint32_t order_seed = 0;
        pl::BoundedSearchOptions search_options;
        std::string search_mode = "exhaustive";
        bool search_limits_set = false;
        std::string graph_path, policy = "cost", json_path, trace_path, report_path, backend = "software";
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { std::cout << help; return 0; }
            if (arg == "--cnn-supported") { config.cnn_supported = true; continue; }
            if (arg == "--relu-supported") { config.relu_supported = true; continue; }
            if (arg == "--fuse") { fusion = true; continue; }
            if (arg == "--async") { asynchronous = true; continue; }
            if (arg == "--reuse-buffers") { config.reuse_buffers = true; continue; }
            if (arg == "--emulate-boundaries") { config.emulate_boundaries = true; continue; }
            if (arg == "--device-no-grouped-conv") { config.device_grouped_conv = false; continue; }
            if (arg != "--graph" && arg != "--policy" && arg != "--json" && arg != "--trace" && arg != "--report" && arg != "--boundary-ms" && arg != "--device-bytes" && arg != "--backend" && arg != "--order-seed" && arg != "--tile-size" && arg != "--local-bytes" && arg != "--sync-ms" && arg != "--launch-ms" && arg != "--bandwidth" && arg != "--resident-input" && arg != "--search" && arg != "--search-budget" && arg != "--beam-width" && arg != "--device-unsupported" && arg != "--device-max-tensor-bytes" && arg != "--runtime")
                throw std::runtime_error("E_OPTION: unknown option " + arg);
            if (++i == argc) throw std::runtime_error("E_OPTION: missing value for " + arg);
            const std::string value = argv[i];
            if (arg == "--graph") graph_path = value;
            else if (arg == "--policy") policy = value;
            else if (arg == "--search") search_mode = value;
            else if (arg == "--json") json_path = value;
            else if (arg == "--trace") trace_path = value;
            else if (arg == "--report") report_path = value;
            else if (arg == "--backend") backend = value;
            else if (arg == "--runtime") runtime = value;
            else if (arg == "--resident-input") config.mirrored_inputs.push_back(value);
            else if (arg == "--device-unsupported") config.device_unsupported_ops.push_back(value);
            else if (arg == "--order-seed") {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::runtime_error("E_OPTION: order-seed requires an unsigned integer");
                const auto parsed = std::stoull(value);
                if (parsed > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("E_OPTION: order-seed exceeds uint32 range");
                order_seed = static_cast<std::uint32_t>(parsed); shuffle_order = true;
            }
            else if (arg == "--boundary-ms") config.transfer_fixed_ms = number(value);
            else if (arg == "--sync-ms") config.synchronization_ms = number(value);
            else if (arg == "--launch-ms") config.launch_ms = number(value);
            else if (arg == "--bandwidth") config.bandwidth_bytes_per_ms = number(value);
            else {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::runtime_error("E_OPTION: " + arg + " requires an unsigned integer");
                std::size_t used = 0;
                const auto parsed = std::stoull(value, &used);
                const auto bounded = static_cast<std::size_t>(parsed);
                if (used != value.size() || bounded != parsed) throw std::runtime_error("E_OPTION: integer out of range");
                if (arg == "--search-budget") { search_options.max_evaluations = bounded; search_limits_set = true; }
                else if (arg == "--beam-width") { search_options.beam_width = bounded; search_limits_set = true; }
                else if (arg == "--tile-size") config.tile_size = bounded;
                else if (arg == "--local-bytes") config.local_bytes = bounded;
                else if (arg == "--device-max-tensor-bytes") config.device_max_tensor_bytes = bounded;
                else config.device_bytes = bounded;
            }
        }
        if (graph_path.empty()) throw std::runtime_error("E_OPTION: --graph FILE is required");
        if (backend != "software" && backend != "opencl") throw std::runtime_error("E_OPTION: backend must be software or opencl");
        if (backend != "software" && config.emulate_boundaries) throw std::runtime_error("E_OPTION: boundary emulation requires software backend");
        if (runtime != "serial" && runtime != "command-queue") throw std::runtime_error("E_OPTION: runtime must be serial or command-queue");
        if (runtime != "serial" && (backend != "software" || action == "inspect" || action == "compile"))
            throw std::runtime_error("E_OPTION: --runtime command-queue requires an execution action on the software backend");
        if (shuffle_order && action != "verify") throw std::runtime_error("E_OPTION: --order-seed is available only with verify");
        if ((asynchronous || backend != "software") && (action == "inspect" || action == "compile"))
            throw std::runtime_error("E_OPTION: --async and --backend require an execution action");
        if (policy != "cpu" && policy != "maximal" && policy != "cost") throw std::runtime_error("E_POLICY: expected cpu, maximal, or cost");
        if (!report_path.empty() && action != "explain") throw std::runtime_error("E_OPTION: --report is available with explain");
        auto same_file = [](const std::string& a, const std::string& b) {
            if (a.empty() || b.empty()) return false;
            return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b) ||
                   (std::filesystem::exists(a) && std::filesystem::exists(b) && std::filesystem::equivalent(a,b));
        };
        if (same_file(json_path,graph_path) || same_file(trace_path,graph_path) || same_file(json_path,trace_path) ||
            same_file(report_path,graph_path) || same_file(report_path,json_path) || same_file(report_path,trace_path))
            throw std::runtime_error("E_OPTION: output files must differ from the input and each other");
        if (backend == "opencl" && config.cnn_supported) throw std::runtime_error("E_OPTION: --cnn-supported is available only with the software backend");
        if (search_mode != "exhaustive" && search_mode != "bounded") throw std::runtime_error("E_OPTION: search must be exhaustive or bounded");
        if ((search_mode == "bounded" && action != "verify") || (search_limits_set && search_mode != "bounded"))
            throw std::runtime_error("E_OPTION: bounded search controls require verify --search bounded");
        pl::validate_config(config);
        std::sort(config.mirrored_inputs.begin(), config.mirrored_inputs.end());
        std::ifstream graph_file(graph_path);
        if (!graph_file) throw std::runtime_error("E_IO: cannot open graph " + graph_path);
        // Bound storage while reading, including streams without a known file size.
        constexpr std::size_t max_graph_bytes = 16000000;
        std::string raw;
        std::array<char, 8192> chunk{};
        while (graph_file) {
            graph_file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
            const auto count = static_cast<std::size_t>(graph_file.gcount());
            if (count > max_graph_bytes - raw.size())
                throw std::runtime_error("E_LIMIT: graph file exceeds 16 MB");
            raw.append(chunk.data(), count);
        }
        if (graph_file.bad() || !graph_file.eof()) throw std::runtime_error("E_IO: graph read failed");
        validate_utf8(raw);
        std::istringstream parser_input(raw);
        const auto source = pl::parse(parser_input);
        for (const auto& input : config.mirrored_inputs)
            if (!source.inputs.count(input)) throw std::runtime_error("E_RESIDENCY: unknown source input " + input);
        const auto passes = fusion ? pl::fuse_matmul_relu(source, config) : pl::PassResult{source, {"Fusion disabled (enable with --fuse)"}};
        const auto& graph = passes.graph;
        const std::vector<pl::PassRecord> frontend_passes = {
            {"ValidateGraphPass",raw,pl::graph_text(source),
             {"Parse bounded source graph; infer f32 shapes; establish stable topological schedule"},
             {"Require unique definitions, defined operands, compatible shapes, and no cycles"},
             {"Only validated typed IR enters placement"}},
            {"MatMulReluFusionPass",pl::graph_text(source),pl::graph_text(graph),passes.remarks,
             {fusion ? "Require a single-use non-output intermediate and target ReLU support" : "Optional pass disabled"},
             {std::to_string(source.nodes.size()) + " source nodes -> " + std::to_string(graph.nodes.size()) + " compiled nodes"}}
        };
        if (action != "explain") {
            std::cout << "\nPlacementLens | " << action << " | " << graph_path << '\n'
                      << pl::graph_text(source);
            for (const auto& remark : passes.remarks) std::cout << "  " << remark << '\n';
            if (graph.nodes.size() != source.nodes.size()) std::cout << pl::graph_text(graph);
            std::cout << "\nVirtual-device capability decisions\n";
            for (const auto& n : graph.nodes) {
                const auto cap = pl::capability(n, graph, config);
                std::cout << "  " << n.id << " : " << (cap.accepted ? "ACCEPT" : "REJECT") << " | " << cap.reason << '\n';
            }
        }
        if (action == "inspect") {
            if (!json_path.empty() || !trace_path.empty()) throw std::runtime_error("E_OPTION: use compare/run/demo for execution reports");
            return 0;
        }
        pl::SearchResult search;
        if (search_mode == "bounded") {
            const auto candidates = pl::bounded_search(graph, config, search_options);
            search.plans = candidates.plans;
            search.exhaustive = false; search.strategy = "bounded-beam"; search.infeasible = candidates.rejected;
            search.evaluated = candidates.evaluated; search.beam_width = search_options.beam_width;
            search.max_evaluations = search_options.max_evaluations;
            search.budget_exhausted = candidates.budget_exhausted;
            search.maximal_placement = candidates.maximal_placement;
            search.maximal_feasible = candidates.maximal_feasible;
            search.maximal_rejection = candidates.maximal_rejection;
        } else if (graph.nodes.size() > pl::max_exhaustive_nodes &&
                   (action == "run" || action == "compile") && policy != "cost") {
            search.exhaustive = false; search.strategy = "single-policy";
            search.plans.push_back(pl::choose(graph, config, policy));
            search.evaluated = 1;
            search.maximal_rejection = "Not assessed by single-policy search";
        } else search = pl::enumerate(graph, config);
        if (search.plans.empty()) throw std::runtime_error("E_MEMORY: no feasible plan under the initial residency and memory budget");
        auto runner = [backend, runtime](const pl::Graph& original, const pl::Plan& plan, const pl::Config& cfg,
                                const std::atomic<bool>& cancel) {
            const auto started = std::chrono::steady_clock::now();
            const bool uses_device = !plan.initial_buffers.empty() || std::find(plan.placement.begin(), plan.placement.end(), pl::Backend::Device) != plan.placement.end();
            auto device = backend == "opencl" && uses_device ? pl::make_opencl_device() : nullptr;
            auto run = runtime == "command-queue" ? pl::execute_command_queue(original, plan, cfg, 4, &cancel)
                                                  : pl::execute(original, plan, cfg, device.get(), &cancel);
            device.reset();
            run.end_to_end_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count();
            if (!uses_device && runtime == "serial") run.execution_backend = "CPU host (no device work)";
            return run;
        };
        std::unique_ptr<pl::AsyncRuntime> worker;
        if (asynchronous) worker = std::make_unique<pl::AsyncRuntime>(1, runner);
        auto execute_plan = [&](const pl::Plan& plan) {
            if (worker) {
                auto completion = worker->submit(source, plan, config).get();
                completion.run.queue_wait_ms = completion.queue_wait_ms;
                completion.run.end_to_end_ms = completion.total_ms;
                completion.run.execution_backend = "async worker: " + completion.run.execution_backend;
                return completion.run;
            }
            const std::atomic<bool> cancel{false};
            return runner(source, plan, config, cancel);
        };
        if (action == "verify") {
            if (!trace_path.empty()) throw std::runtime_error("E_OPTION: use run for a single-plan trace; verify exports all results with --json");
            auto candidates = search.plans;
            if (shuffle_order) {
                std::mt19937 engine(order_seed);
                std::shuffle(candidates.begin(), candidates.end(), engine);
            }
            std::vector<pl::Run> verified;
            std::size_t failures = 0, commands = 0;
            for (auto& candidate : candidates) {
                candidate.policy = search.exhaustive ? "enumerated" : "bounded-candidate";
                verified.push_back(execute_plan(candidate));
                commands += candidate.commands.size();
                if (!verified.back().correct) {
                    ++failures;
                    std::cout << "FAIL " << pl::placement_name(candidate.placement)
                              << " max_abs_error=" << verified.back().max_abs_error << '\n';
                }
            }
            std::cout << "\nVerified " << candidates.size() << " feasible placements, " << commands
                      << " commands; " << failures << " numerical failures.\n"
                      << search.infeasible << " assignments rejected by capability or retained-memory checks.\n"
                      << (search.exhaustive ? "Every feasible plan was executed against the original graph reference.\n" : "Only retained bounded-search candidates executed; no exhaustive coverage or global optimum claim.\n")
                      << "This checks functional execution, not accelerator performance.\n";
            if (!json_path.empty()) save(json_path, pl::report_json(source, graph, config, passes.remarks,
                                                                   candidates, verified, search, frontend_passes));
            return failures == 0 ? 0 : 2;
        }
        std::vector<pl::Plan> plans;
        if (action == "compare" || action == "demo" || action == "explain") {
            for (const std::string p : {"cpu", "maximal", "cost"}) {
                try { plans.push_back(pl::choose(graph, config, p)); }
                catch (const std::runtime_error& e) {
                    if (p != "maximal" || std::string(e.what()).find("E_MEMORY:") != 0) throw;
                    if (action != "explain") std::cout << "\nmaximal policy INFEASIBLE: " << e.what() << '\n';
                }
            }
        } else plans.push_back(pl::choose(graph, config, policy));
        if (action == "explain") {
            std::vector<pl::Run> runs;
            bool correct = true;
            for (const auto& plan : plans) {
                runs.push_back(execute_plan(plan));
                correct = correct && runs.back().correct;
            }
            const auto report = pl::explain_report(source, graph, config, plans, runs, search, frontend_passes);
            std::cout << report;
            if (!report_path.empty()) save(report_path, report);
            if (!json_path.empty()) save(json_path, pl::report_json(source, graph, config, passes.remarks, plans, runs, search, frontend_passes));
            if (!trace_path.empty()) save(trace_path, pl::trace_json(runs.back()));
            return correct ? 0 : 2;
        }
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
        std::cout << "\nPOLICY       MAP            MODEL ms   H2D / D2H bytes   ISLANDS  RESULT\n";
        for (const auto& p : plans) {
            runs.push_back(execute_plan(p));
            const auto& r = runs.back(); correct = correct && r.correct;
            std::cout << std::left << std::setw(13) << p.policy << std::setw(14) << pl::placement_name(p.placement)
                      << std::right << std::fixed << std::setprecision(4) << std::setw(8) << p.modeled_ms
                      << std::setw(8) << p.h2d_bytes << " / " << std::setw(4) << p.d2h_bytes
                      << std::setw(11) << p.islands << "  " << (r.correct ? "PASS" : "FAIL") << '\n';
        }
        const auto& winner = search.plans.front();
        std::cout << (search.exhaustive ? "\nBounded modeled optimum: " : "\nSelected single-policy placement: ") << pl::placement_name(winner.placement)
                  << " across " << search.plans.size() << " feasible placements; " << search.infeasible << " rejected.\n"
                  << "The objective sums compute, copies, launch, and synchronization costs.\n"
                  << "No automatic dtype conversion or runtime replanning is performed.\n";
        if (action == "demo") std::cout << '\n' << pl::plan_text(plans.back());
        std::cout << "\nOutput (row-major): [";
        const auto output_count = runs.back().output.values.size();
        const auto preview_count = output_count < 16 ? output_count : std::size_t{16};
        for (std::size_t i = 0; i < preview_count; ++i)
            std::cout << (i ? ", " : "") << runs.back().output.values[i];
        if (preview_count < output_count) std::cout << ", ... (" << output_count << " values; full output in --json)";
        std::cout << "]\nReference: original graph evaluated independently; atol=rtol=1e-5.\n"
                     "MODEL ms is synthetic. This is NOT a measured accelerator speedup.\n"
                     "Actual host execution time for last plan: " << runs.back().host_ms << " ms.\n";
        if (runs.back().engine_overlap_us >= 0)
            std::cout << "Command queue (" << runs.back().execution_backend << "): " << runs.back().events.size()
                      << " command events; copy/compute engine overlap " << runs.back().engine_overlap_us << " us"
                      << (config.emulate_boundaries ? " (overlapping intervals are injected synthetic waits, not hardware time)" : "") << ".\n";
        if (!json_path.empty()) save(json_path, pl::report_json(source, graph, config, passes.remarks, plans, runs, search, frontend_passes));
        if (!trace_path.empty()) save(trace_path, pl::trace_json(runs.back()));
        return correct ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "PlacementLens error: " << e.what() << '\n';
        return 1;
    }
}
