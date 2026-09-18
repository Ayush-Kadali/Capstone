#pragma once
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace pl {
constexpr std::uint32_t command_version = 1;
constexpr std::size_t max_nodes = 12;
constexpr std::size_t max_elements = 65536;
enum class Op { MatMul, Relu, Add, MatMulRelu };
enum class Backend { Cpu, Device };
enum class Kind { Allocate, Copy, Compute, Synchronize };
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
};
struct SearchResult { std::vector<Plan> plans; std::size_t infeasible = 0; };
struct Event {
    std::size_t command_id = 0;
    std::string node, name, lane;
    double start_us = 0, duration_us = 0, modeled_ms = 0;
};
struct Run {
    Tensor output;
    bool correct = false;
    double max_abs_error = 0, host_ms = 0;
    std::vector<Event> events;
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
Capability capability(const Node& node, const Graph& graph, const Config& config);
PassResult fuse_matmul_relu(const Graph& graph, const Config& config);
Tensor reference(const Graph& graph);
TransferIR insert_transfers(const Graph& graph, const std::vector<Backend>& placement, const Config& config);
Plan lower(const Graph& graph, const std::vector<Backend>& placement, const Config& config);
SearchResult enumerate(const Graph& graph, const Config& config);
Plan choose(const Graph& graph, const Config& config, const std::string& policy);
Run execute(const Graph& source, const Plan& plan, const Config& config);
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
