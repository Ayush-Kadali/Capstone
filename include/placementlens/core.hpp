#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace pl {
constexpr std::uint32_t command_version = 3;
constexpr std::size_t max_nodes = 64;
constexpr std::size_t max_exhaustive_nodes = 12;
constexpr std::size_t max_elements = 65536;
enum class Op { MatMul, Relu, Add, MatMulRelu, RowSum, Conv2D, MaxPool2D, AveragePool2D, GlobalAveragePool, Reshape };
enum class Backend { Cpu, Device };
enum class Kind { Allocate, Copy, Compute, Synchronize, Free };
constexpr std::size_t no_buffer_slot = static_cast<std::size_t>(-1);
struct Shape {
    std::size_t rows = 0, cols = 0;
    std::size_t elements() const;
    std::size_t bytes() const;
    bool operator==(const Shape& other) const;
};
struct Tensor { Shape shape; std::vector<float> values; };
struct Node {
    std::string id;
    Op op = Op::Add;
    std::vector<std::string> inputs;
    std::string output;
    std::vector<std::string> origins;
    std::vector<std::size_t> parameters = {};
};
struct Graph {
    std::map<std::string, Tensor> inputs;
    std::vector<Node> nodes;
    std::map<std::string, Shape> shapes;
    std::string output;
};
struct Config {
    bool relu_supported = false;
    std::size_t device_bytes = 1024 * 1024;
    double transfer_fixed_ms = 1.3;
    double bandwidth_bytes_per_ms = 1000000.0;
    double launch_ms = 0.2;
    double synchronization_ms = 0.05;
    bool reuse_buffers = false;
    std::size_t tile_size = 0, local_bytes = 32768;
    std::vector<std::string> mirrored_inputs; // host inputs also prepared on device before measured execution
    bool emulate_boundaries = false; // bounded real host waits for configured copy/launch/sync costs
    bool cnn_supported = false; // native scalar software CNN kernels, never an OpenCL capability claim
    // Declared target support rules, normally translated from an architecture profile.
    std::vector<std::string> device_unsupported_ops; // pl::name(Op) spelling, e.g. "ReLU", "Conv2D"
    std::size_t device_max_tensor_bytes = 0;         // 0 = no per-tensor rule
    bool device_grouped_conv = true;                 // false rejects Conv2D with group != 1
};
struct Capability { bool accepted; std::string reason; };
struct PassResult { Graph graph; std::vector<std::string> remarks; };
struct Command {
    std::uint32_t version = command_version;
    std::size_t id = 0;
    Kind kind = Kind::Allocate;
    Backend backend = Backend::Cpu;
    Op op = Op::Add;
    std::string node, output;
    std::vector<std::string> inputs;
    Shape shape;
    double modeled_ms = 0;
    std::vector<std::string> origins;
    std::size_t buffer_slot = no_buffer_slot;
    std::vector<std::size_t> parameters = {};
};
struct PassRecord {
    std::string name, input_ir, output_ir;
    std::vector<std::string> changes, reasons, consequences;
};
struct TransferStep {
    Kind kind = Kind::Compute;
    Backend backend = Backend::Cpu;
    Op op = Op::Add;
    std::string node, output;
    std::vector<std::string> inputs;
    Shape shape;
    double modeled_ms = 0;
    std::vector<std::string> origins;
    std::vector<std::size_t> parameters = {};
};
struct TransferIR {
    std::vector<Backend> placement;
    std::vector<TransferStep> steps;
    std::size_t h2d_bytes = 0, d2h_bytes = 0, islands = 0, synchronizations = 0;
};
struct Plan {
    std::string policy;
    std::vector<Backend> placement;
    std::vector<Command> commands;
    double modeled_ms = 0;
    std::size_t h2d_bytes = 0, d2h_bytes = 0, peak_device_bytes = 0;
    std::size_t islands = 0, synchronizations = 0;
    std::vector<PassRecord> passes;
    std::size_t device_buffer_slots = 0, buffer_reuses = 0, peak_live_device_bytes = 0;
    std::vector<Command> initial_buffers; // ALLOC descriptors in a separate preparation namespace
};
struct SearchResult {
    std::vector<Plan> plans;
    std::size_t infeasible = 0;
    bool exhaustive = true;
    std::string strategy = "exhaustive";
    std::size_t evaluated = 0, beam_width = 0, max_evaluations = 0;
    bool budget_exhausted = false, maximal_feasible = false;
    std::string maximal_placement, maximal_rejection;
};
struct Event {
    std::size_t command_id = 0;
    std::string node, name, lane;
    double start_us = 0, duration_us = 0, modeled_ms = 0;
    double device_us = -1; // negative means no hardware measurement available
    std::string engine; // command-queue lane ("host", "copy-engine", "compute-engine"); empty for serial execute
    std::vector<std::size_t> waits; // command-queue dependency command ids
};
struct Run {
    Tensor output;
    bool correct = false;
    double max_abs_error = 0, host_ms = 0;
    std::vector<Event> events;
    std::string execution_backend = "synchronous host software";
    double queue_wait_ms = -1, end_to_end_ms = -1;
    std::size_t diagnostic_readback_bytes = 0;
    std::size_t software_buffer_reuses = 0, peak_device_reserved_bytes = 0;
    std::size_t tile_count = 0, tile_load_bytes = 0, peak_local_bytes = 0;
    double input_preparation_ms = 0;
    std::size_t preparation_h2d_bytes = 0;
    double injected_boundary_ms = 0; // requested host wait, not a hardware measurement
    double engine_overlap_us = -1; // command queue: copy-engine/compute-engine interval overlap; -1 when not applicable
};
std::string name(Op op);
std::string name(Kind kind);
std::string name(Backend backend);
std::string placement_name(const std::vector<Backend>& placement);
std::string quote(const std::string& text);
Graph parse(std::istream& input);
Graph load(const std::string& path);
Graph validate(Graph graph);
void validate_config(const Config& config);
// Analysis, transformation, lowering, and reference APIs require a graph
// returned by parse/load/validate, without subsequent mutation. Graph is a
// transparent value type, not a serialized or trusted device ABI.
Capability capability(const Node& node, const Graph& graph, const Config& config);
// Declared target support rules (--device-unsupported, --device-max-tensor-bytes,
// --device-no-grouped-conv). Empty string = accepted; shared by capability and both executors.
std::string target_rule_rejection(Op op, const std::vector<Shape>& tensors,
                                  const std::vector<std::size_t>& parameters, const Config& config);
PassResult fuse_matmul_relu(const Graph& graph, const Config& config);
Tensor reference(const Graph& graph, const std::atomic<bool>* cancel = nullptr);
TransferIR insert_transfers(const Graph& graph, const std::vector<Backend>& placement, const Config& config);
Plan lower(const Graph& graph, const std::vector<Backend>& placement, const Config& config);
SearchResult enumerate(const Graph& graph, const Config& config);
Plan choose(const Graph& graph, const Config& config, const std::string& policy);
// Revalidates source storage before kernels; validates commands as consumed.
// Host timing excludes source validation/input copies and reference evaluation.
class DeviceBackend;
Run execute(const Graph& source, const Plan& plan, const Config& config,
            DeviceBackend* device = nullptr, const std::atomic<bool>* cancel = nullptr);
// Shared by execute and execute_command_queue (placementlens/queue.hpp).
Shape compute_shape(Op op, const std::vector<Shape>& inputs, const std::vector<std::size_t>& parameters);
// Sets run.correct/max_abs_error against reference(source); source must be validated.
void check_against_reference(Run& run, const Graph& source, const std::atomic<bool>* cancel = nullptr);
std::string graph_text(const Graph& graph);
std::string plan_text(const Plan& plan);
std::string report_json(const Graph& source, const Graph& compiled, const Config& config,
                        const std::vector<std::string>& remarks,
                        const std::vector<Plan>& plans, const std::vector<Run>& runs,
                        const SearchResult& search, const std::vector<PassRecord>& frontend_passes = {});
std::string trace_json(const Run& run);
std::string explain_report(const Graph& source, const Graph& compiled, const Config& config,
                           const std::vector<Plan>& plans, const std::vector<Run>& runs,
                           const SearchResult& search, const std::vector<PassRecord>& frontend_passes);
}
