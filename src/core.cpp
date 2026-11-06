#include "placementlens/core.hpp"
#include "placementlens/device.hpp"
#include "placementlens/kernels.hpp"
#include "placementlens/cnn.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace pl {
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
void check_cancel(const std::atomic<bool>* cancel) {
    require(!cancel || !cancel->load(std::memory_order_relaxed), "E_CANCELLED: execution cancelled");
}
class DeviceAllocations {
    DeviceBackend* device_;
    std::vector<const std::string*> handles_;
public:
    DeviceAllocations(DeviceBackend* device, std::size_t capacity) : device_(device) { handles_.reserve(capacity); }
    void allocate(const Command& command) {
        device_->allocate(command.output, command.shape);
        handles_.push_back(&command.output); // capacity reserved before allocating on the device
    }
    void release() {
        if (!device_ || handles_.empty()) return;
        device_->synchronize();
        while (!handles_.empty()) {
            device_->release(*handles_.back());
            handles_.pop_back();
        }
    }
    void release(const std::string& handle) {
        const auto found = std::find_if(handles_.begin(), handles_.end(), [&](const std::string* item) { return *item == handle; });
        require(found != handles_.end(), "E_FREE: device allocation is not owned by this run");
        device_->release(handle);
        handles_.erase(found); // explicit FREE must not be released again at teardown
    }
    ~DeviceAllocations() {
        // Preserve an execution exception while still attempting every release.
        if (!device_ || handles_.empty()) return;
        try { device_->synchronize(); } catch (...) {}
        for (const auto* handle : handles_) try { device_->release(*handle); } catch (...) {}
    }
};
bool identifier(const std::string& s) {
    return !s.empty() && s.size() <= 64 && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
    });
}
std::size_t dimension(const std::string& text) {
    std::size_t value = 0;
    require(!text.empty(), "E_SHAPE: missing dimension");
    for (char c : text) {
        require(c >= '0' && c <= '9', "E_SHAPE: dimensions require unsigned decimal integers");
        const auto digit = static_cast<std::size_t>(c - '0');
        require(value <= (max_elements - digit) / 10, "E_SHAPE: dimension exceeds 65536");
        value = value * 10 + digit;
    }
    return value;
}
Shape result_shape(Op op, const std::vector<Shape>& inputs, const std::vector<std::size_t>& parameters = {}) {
    if (cnn_op(op)) return cnn_shape(op, inputs, parameters);
    require(parameters.empty(), "E_ATTRIBUTE: unexpected parameters on matrix operation");
    const bool unary = unary_op(op);
    require(inputs.size() == (unary ? 1U : 2U), "E_ARITY: wrong number of operands");
    if (op == Op::Relu) return inputs[0];
    if (op == Op::RowSum) return {inputs[0].rows, 1};
    if (op == Op::Add) {
        require(inputs[0] == inputs[1], "E_SHAPE: Add requires equal shapes; broadcasting is unsupported");
        return inputs[0];
    }
    require(op == Op::MatMul || op == Op::MatMulRelu, "E_OPCODE: unsupported opcode");
    require(inputs[0].cols == inputs[1].rows, "E_SHAPE: MatMul inner dimensions differ");
    Shape out{inputs[0].rows, inputs[1].cols};
    // elements() throws if the result is too big. We only want that check here,
    // not the number, so the value is thrown away.
    (void)out.elements();
    return out;
}
std::string handle(Backend b, const std::string& tensor) {
    return (b == Backend::Cpu ? "H:" : "D:") + tensor;
}
double compute_cost(Op op, Backend b) {
    // Example costs only. A bigger matrix still gets the same compute cost here.
    // Good enough to test placement decisions, not to predict a real GPU.
    switch (op) {
        case Op::MatMul: return b == Backend::Cpu ? 6.0 : 1.0;
        case Op::Relu: return b == Backend::Cpu ? 1.0 : 0.1;
        case Op::Add: return b == Backend::Cpu ? 2.0 : 0.4;
        case Op::MatMulRelu: return b == Backend::Cpu ? 7.0 : 1.1;
        case Op::RowSum: return b == Backend::Cpu ? 2.0 : 0.5;
        case Op::Conv2D: return b == Backend::Cpu ? 6.0 : 1.0;
        case Op::MaxPool2D: case Op::AveragePool2D: case Op::GlobalAveragePool: return b == Backend::Cpu ? 2.0 : 0.5;
        case Op::Reshape: return b == Backend::Cpu ? 0.1 : 0.1;
    }
    throw std::runtime_error("E_OPCODE: unsupported cost opcode");
}
std::string strings_json(const std::vector<std::string>& xs) {
    std::string out = "[";
    for (std::size_t i = 0; i < xs.size(); ++i) out += (i ? "," : "") + quote(xs[i]);
    return out + "]";
}
std::string parameters_json(const std::vector<std::size_t>& values) {
    std::ostringstream out; out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']'; return out.str();
}
std::string graph_json(const Graph& g) {
    std::ostringstream s;
    s << "{\"output\":" << quote(g.output) << ",\"inputs\":[";
    bool comma = false;
    for (const auto& entry : g.inputs) {
        if (comma) s << ',';
        comma = true;
        s << "{\"name\":" << quote(entry.first) << ",\"dtype\":\"f32\",\"layout\":\"contiguous-row-major\",\"shape\":["
          << entry.second.shape.rows << ',' << entry.second.shape.cols << "],\"values\":[";
        for (std::size_t i = 0; i < entry.second.values.size(); ++i) {
            if (i) s << ',';
            s << std::setprecision(9) << entry.second.values[i];
        }
        s << "]}";
    }
    s << "],\"nodes\":[";
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        if (i) s << ',';
        const auto& n = g.nodes[i];
        const auto shape = g.shapes.at(n.output);
        s << "{\"id\":" << quote(n.id) << ",\"op\":" << quote(name(n.op))
          << ",\"inputs\":" << strings_json(n.inputs) << ",\"output\":" << quote(n.output)
          << ",\"origins\":" << strings_json(n.origins) << ",\"parameters\":" << parameters_json(n.parameters)
          << ",\"dtype\":\"f32\",\"layout\":\"contiguous-row-major\",\"shape\":[" << shape.rows << ',' << shape.cols << "]}";
    }
    return s.str() + "]}";
}
}

Shape compute_shape(Op op, const std::vector<Shape>& inputs, const std::vector<std::size_t>& parameters) {
    return result_shape(op, inputs, parameters);
}
std::size_t Shape::elements() const {
    require(rows > 0 && cols > 0 && rows <= max_elements && cols <= max_elements / rows,
            "E_SHAPE: expected positive dimensions and at most 65536 elements");
    return rows * cols;
}
std::size_t Shape::bytes() const { return elements() * sizeof(float); }
bool Shape::operator==(const Shape& other) const { return rows == other.rows && cols == other.cols; }
std::string name(Op op) {
    switch (op) {
        case Op::MatMul: return "MatMul"; case Op::Relu: return "ReLU";
        case Op::Add: return "Add"; case Op::MatMulRelu: return "MatMulReLU";
        case Op::RowSum: return "RowSum";
        case Op::Conv2D: return "Conv2D"; case Op::MaxPool2D: return "MaxPool2D";
        case Op::AveragePool2D: return "AveragePool2D"; case Op::GlobalAveragePool: return "GlobalAveragePool";
        case Op::Reshape: return "Reshape";
    }
    throw std::runtime_error("E_OPCODE: unknown opcode");
}
std::string name(Kind kind) {
    switch (kind) {
        case Kind::Allocate: return "ALLOC"; case Kind::Copy: return "COPY";
        case Kind::Compute: return "COMPUTE"; case Kind::Synchronize: return "SYNC";
        case Kind::Free: return "FREE";
    }
    throw std::runtime_error("E_COMMAND: unknown command kind");
}
std::string name(Backend backend) {
    require(backend == Backend::Cpu || backend == Backend::Device, "E_BACKEND: unknown backend");
    return backend == Backend::Cpu ? "CPU" : "virtual-device";
}
std::string placement_name(const std::vector<Backend>& placement) {
    std::string s;
    for (auto b : placement) s += b == Backend::Cpu ? 'C' : 'A';
    return s;
}
std::string quote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}
Graph parse(std::istream& input) {
    Graph graph;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        require(line.size() <= 2000000, "E_PARSE: line is too long");
        line = line.substr(0, line.find('#'));
        std::istringstream row(line);
        std::string tag;
        if (!(row >> tag)) continue;
        try {
            if (tag == "tensor") {
                std::string id, dtype, rows, cols;
                Tensor t;
                require(static_cast<bool>(row >> id >> dtype >> rows >> cols), "E_PARSE: tensor needs name, dtype, rows, cols, values");
                require(dtype == "f32", "E_DTYPE: only explicit f32 inputs are supported; no implicit conversion");
                t.shape = {dimension(rows), dimension(cols)};
                require(graph.inputs.size() < 64, "E_LIMIT: at most 64 graph inputs");
                for (std::size_t i = 0; i < t.shape.elements(); ++i) {
                    std::string token;
                    require(static_cast<bool>(row >> token), "E_VALUE: expected finite f32 tensor data");
                    char* end = nullptr;
                    errno = 0;
                    const float v = std::strtof(token.c_str(), &end);
                    // strtof may report ERANGE for a representable subnormal.
                    // Preserve it; reject overflow and underflow rounded to zero.
                    require(end != token.c_str() && end == token.c_str() + token.size() &&
                            std::isfinite(v) && !(errno == ERANGE && v == 0),
                            "E_VALUE: expected finite representable f32 tensor data");
                    t.values.push_back(v);
                }
                require(graph.inputs.emplace(id, std::move(t)).second, "E_SSA: duplicate tensor name");
            } else if (tag == "output") {
                require(graph.output.empty(), "E_PARSE: exactly one output declaration allowed");
                require(static_cast<bool>(row >> graph.output), "E_PARSE: missing output name");
            } else {
                Node n;
                if (tag == "matmul") n.op = Op::MatMul;
                else if (tag == "relu") n.op = Op::Relu;
                else if (tag == "add") n.op = Op::Add;
                else if (tag == "rowsum") n.op = Op::RowSum;
                else if (tag == "conv2d") n.op = Op::Conv2D;
                else if (tag == "maxpool2d") n.op = Op::MaxPool2D;
                else if (tag == "averagepool2d") n.op = Op::AveragePool2D;
                else if (tag == "globalaveragepool") n.op = Op::GlobalAveragePool;
                else if (tag == "reshape") n.op = Op::Reshape;
                else throw std::runtime_error("E_PARSE: unknown statement " + tag);
                std::string a, b;
                require(static_cast<bool>(row >> n.id >> a), "E_PARSE: missing node id or operand");
                n.inputs.push_back(a);
                if (!unary_op(n.op)) {
                    require(static_cast<bool>(row >> b), "E_PARSE: missing second operand");
                    n.inputs.push_back(b);
                }
                require(static_cast<bool>(row >> n.output), "E_PARSE: missing node output");
                if (cnn_op(n.op)) {
                    const auto count = n.op == Op::Conv2D ? 16U : n.op == Op::GlobalAveragePool ? 4U : n.op == Op::Reshape ? 2U : 13U;
                    for (unsigned i = 0; i < count; ++i) {
                        std::string value; require(static_cast<bool>(row >> value), "E_ATTRIBUTE: missing CNN parameter");
                        n.parameters.push_back(dimension(value));
                    }
                }
                n.origins = {n.id};
                graph.nodes.push_back(n);
                require(graph.nodes.size() <= max_nodes, "E_LIMIT: at most 64 nodes");
            }
            std::string extra;
            require(!(row >> extra), "E_PARSE: unexpected trailing token");
        } catch (const std::exception& e) {
            throw std::runtime_error("line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    require(!input.bad(), "E_IO: graph read failed");
    return validate(std::move(graph));
}
Graph load(const std::string& path) {
    std::ifstream in(path);
    require(static_cast<bool>(in), "E_IO: cannot open graph " + path);
    return parse(in);
}
Graph validate(Graph graph) {
    require(!graph.nodes.empty() && graph.nodes.size() <= max_nodes, "E_LIMIT: expected 1..64 nodes");
    require(graph.inputs.size() <= 64, "E_LIMIT: at most 64 graph inputs");
    graph.shapes.clear();
    std::set<std::string> tensor_names, node_names;
    for (const auto& entry : graph.inputs) {
        require(identifier(entry.first), "E_NAME: invalid tensor name");
        require(entry.second.values.size() == entry.second.shape.elements(), "E_VALUE: input storage size mismatch");
        for (float v : entry.second.values) require(std::isfinite(v), "E_VALUE: nonfinite input");
        tensor_names.insert(entry.first);
        graph.shapes[entry.first] = entry.second.shape;
    }
    for (auto& n : graph.nodes) {
        require(identifier(n.id) && identifier(n.output), "E_NAME: invalid node or output name");
        require(node_names.insert(n.id).second, "E_ID: duplicate node id");
        require(tensor_names.insert(n.output).second, "E_SSA: tensor must have a unique definition");
        require(n.inputs.size() == (unary_op(n.op) ? 1U : 2U), "E_ARITY: wrong operand count");
        if (n.origins.empty()) n.origins = {n.id};
    }
    require(tensor_names.count(graph.output) != 0, "E_OUTPUT: graph output is undefined");
    for (const auto& n : graph.nodes) for (const auto& in : n.inputs)
        require(tensor_names.count(in) != 0, "E_REFERENCE: undefined tensor " + in);
    std::vector<Node> ordered;
    std::set<std::string> done;
    while (ordered.size() < graph.nodes.size()) {
        bool progress = false;
        // Stable input-order topological scheduling; no hidden scheduling optimizer.
        for (const auto& n : graph.nodes) {
            if (done.count(n.id)) continue;
            std::vector<Shape> shapes;
            for (const auto& in : n.inputs) if (graph.shapes.count(in)) shapes.push_back(graph.shapes.at(in));
            if (shapes.size() != n.inputs.size()) continue;
            graph.shapes[n.output] = result_shape(n.op, shapes, n.parameters);
            done.insert(n.id);
            ordered.push_back(n);
            progress = true;
        }
        require(progress, "E_CYCLE: graph contains a dependency cycle");
    }
    graph.nodes = std::move(ordered);
    return graph;
}
void validate_config(const Config& c) {
    std::set<std::string> mirrored;
    require(c.mirrored_inputs.size() <= 64, "E_RESIDENCY: at most 64 mirrored inputs");
    for (const auto& input : c.mirrored_inputs)
        require(identifier(input) && mirrored.insert(input).second, "E_RESIDENCY: mirrored input names must be valid and unique");
    require(c.tile_size <= 64, "E_CONFIG: tile_size must be 0 (scalar) or 1..64");
    require(c.tile_size == 0 || 3 * c.tile_size * c.tile_size * sizeof(float) <= c.local_bytes,
            "E_CONFIG: tiled scratch exceeds local memory budget");
    for (double v : {c.transfer_fixed_ms, c.launch_ms, c.synchronization_ms})
        require(std::isfinite(v) && v >= 0 && v <= 1e9, "E_CONFIG: costs must be finite, nonnegative, and <= 1e9 ms");
    require(std::isfinite(c.bandwidth_bytes_per_ms) && c.bandwidth_bytes_per_ms >= 1e-9,
            "E_CONFIG: bandwidth must be finite and >= 1e-9 bytes/ms");
    require(c.device_unsupported_ops.size() <= 16, "E_CONFIG: at most 16 excluded device operators");
    for (const auto& op : c.device_unsupported_ops) {
        bool known = false;
        for (Op o : {Op::MatMul, Op::Relu, Op::Add, Op::MatMulRelu, Op::RowSum, Op::Conv2D, Op::MaxPool2D,
                     Op::AveragePool2D, Op::GlobalAveragePool, Op::Reshape}) known = known || name(o) == op;
        require(known, "E_CONFIG: unknown excluded device operator " + op);
    }
}
// Declared target support rules (from an architecture profile). Empty means accepted.
std::string target_rule_rejection(Op op, const std::vector<Shape>& tensors,
                                         const std::vector<std::size_t>& parameters, const Config& c) {
    if (std::find(c.device_unsupported_ops.begin(), c.device_unsupported_ops.end(), name(op)) != c.device_unsupported_ops.end())
        return "operator excluded by the declared target support rules";
    if (c.device_max_tensor_bytes)
        for (const auto& s : tensors)
            if (s.bytes() > c.device_max_tensor_bytes) return "tensor exceeds the declared per-tensor device limit";
    if (op == Op::Conv2D && !c.device_grouped_conv && parameters.size() > 13 && parameters[13] != 1)
        return "grouped convolution excluded by the declared target support rules";
    return {};
}
Capability capability(const Node& n, const Graph& g, const Config& c) {
    if (cnn_op(n.op) && !c.cnn_supported) return {false, "CNN/storage kernel disabled in this software-device profile"};
    if ((n.op == Op::Relu || n.op == Op::MatMulRelu) && !c.relu_supported)
        return {false, "ReLU is not supported on this device target"};
    if (n.op != Op::Relu && n.op != Op::MatMul && n.op != Op::Add && n.op != Op::MatMulRelu && n.op != Op::RowSum && !cnn_op(n.op))
        return {false, "opcode is unsupported"};
    std::vector<Shape> tensors;
    for (const auto& t : n.inputs) tensors.push_back(g.shapes.at(t));
    tensors.push_back(g.shapes.at(n.output));
    const auto rule = target_rule_rejection(n.op, tensors, n.parameters, c);
    if (!rule.empty()) return {false, rule};
    std::set<std::string> unique(n.inputs.begin(), n.inputs.end());
    unique.insert(n.output);
    std::size_t bytes = 0;
    for (const auto& t : unique) bytes += g.shapes.at(t).bytes();
    if (bytes > c.device_bytes) return {false, "operands plus output exceed device memory budget"};
    return {true, "supported f32 opcode and shape; local buffer footprint fits"};
}
PassResult fuse_matmul_relu(const Graph& graph, const Config& config) {
    PassResult result{graph, {}};
    std::map<std::string, std::size_t> uses;
    for (const auto& n : graph.nodes) for (const auto& in : n.inputs) ++uses[in];
    std::set<std::string> removed;
    std::vector<Node> nodes;
    for (const auto& n : graph.nodes) {
        if (removed.count(n.id)) continue;
        if (n.op != Op::MatMul) { nodes.push_back(n); continue; }
        auto consumer = std::find_if(graph.nodes.begin(), graph.nodes.end(), [&](const Node& candidate) {
            return candidate.op == Op::Relu && candidate.inputs[0] == n.output;
        });
        if (consumer == graph.nodes.end()) { nodes.push_back(n); continue; }
        std::string reason;
        // Another consumer may need the values BEFORE ReLU. Don't remove them.
        if (uses[n.output] != 1) reason = "MatMul output has multiple uses";
        else if (graph.output == n.output) reason = "MatMul output is externally observable";
        else if (!config.relu_supported) reason = "configured target lacks ReLU support";
        if (!reason.empty()) {
            result.remarks.push_back("SKIP " + n.id + ": " + reason);
            nodes.push_back(n);
        } else {
            Node fused = n;
            fused.op = Op::MatMulRelu;
            fused.output = consumer->output;
            fused.origins.insert(fused.origins.end(), consumer->origins.begin(), consumer->origins.end());
            result.remarks.push_back("FUSE " + n.id + " + " + consumer->id + ": single-use intermediate removed; source IDs retained");
            nodes.push_back(fused);
            removed.insert(consumer->id);
        }
    }
    result.graph.nodes = std::move(nodes);
    result.graph = validate(std::move(result.graph));
    if (result.remarks.empty()) result.remarks.push_back("No eligible MatMul -> ReLU pair");
    return result;
}

Tensor reference(const Graph& graph, const std::atomic<bool>* cancel) {
    check_cancel(cancel);
    // Independent high-level evaluator: double accumulation, original unfused graph.
    auto values = graph.inputs;
    for (const auto& n : graph.nodes) {
        if (cnn_op(n.op)) {
            std::vector<const Tensor*> operands;
            for (const auto& id : n.inputs) operands.push_back(&values.at(id));
            values[n.output] = reference_cnn(n.op, operands, n.parameters, cancel);
            continue;
        }
        const auto& a = values.at(n.inputs[0]);
        Tensor out{graph.shapes.at(n.output), {}};
        out.values.resize(out.shape.elements());
        for (std::size_t r = 0; r < out.shape.rows; ++r) {
            check_cancel(cancel);
            for (std::size_t col = 0; col < out.shape.cols; ++col) {
                double v = 0;
                if (n.op == Op::Relu) v = std::max(0.0, static_cast<double>(a.values[r * a.shape.cols + col]));
                else if (n.op == Op::RowSum) {
                    for (std::size_t k = 0; k < a.shape.cols; ++k) {
                        if (k % 256 == 0) check_cancel(cancel);
                        v += static_cast<double>(a.values[r * a.shape.cols + k]);
                    }
                }
                else {
                    const auto& b = values.at(n.inputs[1]);
                    if (n.op == Op::Add) v = static_cast<double>(a.values[r * a.shape.cols + col]) + b.values[r * b.shape.cols + col];
                    else {
                        for (std::size_t k = 0; k < a.shape.cols; ++k) {
                            if (k % 256 == 0) check_cancel(cancel);
                            v += static_cast<double>(a.values[r * a.shape.cols + k]) * b.values[k * b.shape.cols + col];
                        }
                        if (n.op == Op::MatMulRelu) {
                            require(std::isfinite(static_cast<float>(v)), "E_NUMERIC: reference MatMul produced nonfinite f32 output before ReLU");
                            v = std::max(0.0, v);
                        }
                    }
                }
                out.values[r * out.shape.cols + col] = static_cast<float>(v);
                require(std::isfinite(out.values[r * out.shape.cols + col]), "E_NUMERIC: reference produced nonfinite f32 output");
            }
        }
        values[n.output] = std::move(out);
    }
    return values.at(graph.output);
}
TransferIR insert_transfers(const Graph& graph, const std::vector<Backend>& placement, const Config& config) {
    validate_config(config);
    require(placement.size() == graph.nodes.size(), "E_PLACEMENT: placement length mismatch");
    TransferIR plan;
    plan.placement = placement;
    std::set<std::string> valid;
    for (const auto& input : graph.inputs) {
        valid.insert(handle(Backend::Cpu, input.first));
    }
    for (const auto& input : config.mirrored_inputs) {
        require(graph.inputs.count(input) != 0, "E_RESIDENCY: only graph inputs may be initially mirrored");
        valid.insert(handle(Backend::Device, input));
    }
    auto emit = [&](TransferStep step) { plan.steps.push_back(std::move(step)); };
    bool pending = false;
    auto sync = [&]() {
        if (!pending) return;
        TransferStep cmd;
        cmd.kind = Kind::Synchronize; cmd.backend = Backend::Device;
        cmd.modeled_ms = config.synchronization_ms;
        emit(cmd); ++plan.synchronizations; pending = false;
    };
    auto ensure = [&](Backend b, const std::string& t) {
        const auto destination = handle(b, t);
        // Inputs can be shared by several nodes. Copy once, keep that copy.
        if (valid.count(destination)) return;
        const auto other = b == Backend::Cpu ? Backend::Device : Backend::Cpu;
        const auto source = handle(other, t);
        require(valid.count(source) != 0, "E_LOWER: input has no valid resident copy");
        // The host must see completed device work before it reads the result.
        if (b == Backend::Cpu) sync();
        TransferStep cmd;
        cmd.kind = Kind::Copy; cmd.backend = b; cmd.output = destination;
        cmd.inputs = {source}; cmd.shape = graph.shapes.at(t);
        cmd.modeled_ms = config.transfer_fixed_ms + static_cast<double>(cmd.shape.bytes()) / config.bandwidth_bytes_per_ms;
        emit(cmd);
        if (b == Backend::Device) plan.h2d_bytes += cmd.shape.bytes();
        else plan.d2h_bytes += cmd.shape.bytes();
        valid.insert(destination);
    };
    Backend previous = Backend::Cpu;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& n = graph.nodes[i];
        const auto b = placement[i];
        // name() throws on a backend value that is not Cpu or Device. Used as a check.
        (void)name(b);
        if (b == Backend::Device) {
            auto cap = capability(n, graph, config);
            require(cap.accepted, "E_CAPABILITY: " + n.id + ": " + cap.reason);
        } else sync();
        for (const auto& in : n.inputs) ensure(b, in);
        TransferStep cmd;
        cmd.kind = Kind::Compute; cmd.backend = b; cmd.node = n.id; cmd.op = n.op;
        cmd.output = handle(b, n.output); cmd.shape = graph.shapes.at(n.output); cmd.origins = n.origins; cmd.parameters = n.parameters;
        for (const auto& in : n.inputs) cmd.inputs.push_back(handle(b, in));
        cmd.modeled_ms = compute_cost(n.op, b);
        if (b == Backend::Device) {
            // One charge per consecutive device run is a model assumption.
            // We do not generate one real GPU kernel for the whole run.
            if (previous != Backend::Device) { ++plan.islands; cmd.modeled_ms += config.launch_ms; }
            pending = true;
        }
        emit(cmd);
        valid.insert(cmd.output);
        previous = b;
    }
    sync();
    ensure(Backend::Cpu, graph.output);
    return plan;
}

namespace {
std::string placed_text(const Graph& graph, const std::vector<Backend>& placement) {
    std::ostringstream out;
    out << graph_text(graph);
    for (std::size_t i = 0; i < graph.nodes.size(); ++i)
        out << "  place " << graph.nodes[i].id << " -> " << name(placement[i]) << '\n';
    return out.str();
}
std::string transfer_text(const TransferIR& ir) {
    std::ostringstream out;
    out << "Transfer IR (logical storage, prior to allocation lowering)\n";
    for (const auto& step : ir.steps) {
        out << "  " << name(step.kind) << ' ';
        if (step.kind == Kind::Compute) out << name(step.op) << " [" << step.node << "] ";
        for (const auto& in : step.inputs) out << in << ' ';
        if (!step.output.empty()) out << "-> " << step.output;
        out << '\n';
    }
    return out.str();
}
}
Plan lower(const Graph& graph, const std::vector<Backend>& placement, const Config& config) {
    const auto ir = insert_transfers(graph, placement, config);
    Plan plan;
    plan.placement = placement;
    plan.h2d_bytes = ir.h2d_bytes; plan.d2h_bytes = ir.d2h_bytes;
    plan.islands = ir.islands;
    std::set<std::string> allocated;
    for (const auto& input : graph.inputs) allocated.insert(handle(Backend::Cpu, input.first));
    auto emit = [&](Command c) {
        c.id = plan.commands.size();
        plan.modeled_ms += c.modeled_ms;
        if (c.kind == Kind::Synchronize) ++plan.synchronizations;
        plan.commands.push_back(std::move(c));
    };
    std::map<std::string, std::size_t> last_use, handle_slots;
    if (config.reuse_buffers) {
        for (std::size_t i = 0; i < ir.steps.size(); ++i) {
            const auto& step = ir.steps[i];
            if (step.output.rfind("D:", 0) == 0) last_use[step.output] = i;
            for (const auto& input : step.inputs)
                if (input.rfind("D:", 0) == 0) last_use[input] = i;
        }
    }
    struct Slot { std::size_t bytes; bool busy; };
    std::vector<Slot> slots;
    std::size_t live_bytes = 0;
    auto mirrored = config.mirrored_inputs;
    std::sort(mirrored.begin(), mirrored.end());
    for (const auto& input : mirrored) {
        Command allocation;
        allocation.id = plan.initial_buffers.size();
        allocation.kind = Kind::Allocate; allocation.backend = Backend::Device;
        allocation.output = handle(Backend::Device, input); allocation.shape = graph.shapes.at(input);
        const auto bytes = allocation.shape.bytes();
        require(bytes <= config.device_bytes && plan.peak_device_bytes <= config.device_bytes - bytes,
                "E_MEMORY: initial device residency exceeds budget");
        plan.peak_device_bytes += bytes; live_bytes += bytes;
        if (config.reuse_buffers) {
            allocation.buffer_slot = slots.size(); slots.push_back({bytes, true});
            handle_slots[allocation.output] = allocation.buffer_slot;
        }
        allocated.insert(allocation.output); plan.initial_buffers.push_back(allocation);
    }
    plan.peak_live_device_bytes = live_bytes;
    bool pending = false;
    std::vector<std::string> additions;
    for (std::size_t i = 0; i < ir.steps.size(); ++i) {
        const auto& step = ir.steps[i];
        // A lifetime boundary may already have synchronized these commands.
        if (config.reuse_buffers && step.kind == Kind::Synchronize && !pending) continue;
        if (step.kind != Kind::Synchronize && !allocated.count(step.output)) {
            Command allocation;
            allocation.kind = Kind::Allocate; allocation.backend = step.backend;
            allocation.output = step.output; allocation.shape = step.shape;
            if (step.backend == Backend::Device) {
                const auto bytes = step.shape.bytes();
                if (config.reuse_buffers) {
                    auto available = std::find_if(slots.begin(), slots.end(), [&](const Slot& slot) { return !slot.busy && slot.bytes == bytes; });
                    if (available == slots.end()) {
                        require(bytes <= config.device_bytes && plan.peak_device_bytes <= config.device_bytes - bytes,
                                "E_MEMORY: reserved device pool exceeds budget (including idle size classes)");
                        allocation.buffer_slot = slots.size();
                        slots.push_back({bytes, true});
                        plan.peak_device_bytes += bytes;
                    } else {
                        allocation.buffer_slot = static_cast<std::size_t>(available - slots.begin());
                        available->busy = true;
                        ++plan.buffer_reuses;
                    }
                    handle_slots[step.output] = allocation.buffer_slot;
                } else {
                    require(bytes <= config.device_bytes && plan.peak_device_bytes <= config.device_bytes - bytes,
                            "E_MEMORY: plan exceeds retained device allocation budget");
                    plan.peak_device_bytes += bytes;
                }
                live_bytes += bytes;
                plan.peak_live_device_bytes = std::max(plan.peak_live_device_bytes, live_bytes);
            }
            allocated.insert(step.output);
            additions.push_back("ADD ALLOC " + step.output + " " + std::to_string(step.shape.bytes()) + " bytes");
            emit(allocation);
        }
        Command c;
        c.kind=step.kind; c.backend=step.backend; c.op=step.op; c.node=step.node;
        c.output=step.output; c.inputs=step.inputs; c.shape=step.shape;
        c.modeled_ms=step.modeled_ms; c.origins=step.origins; c.parameters=step.parameters;
        emit(c);
        if (step.kind == Kind::Compute && step.backend == Backend::Device) pending = true;
        if (step.kind == Kind::Synchronize) pending = false;
        if (config.reuse_buffers) {
            for (const auto& lifetime : last_use) {
                if (lifetime.second != i) continue;
                if (pending) {
                    Command sync;
                    sync.kind = Kind::Synchronize; sync.backend = Backend::Device;
                    sync.modeled_ms = config.synchronization_ms;
                    emit(sync); pending = false;
                    additions.push_back("ADD SYNC before last-use device release");
                }
                Command free;
                free.kind = Kind::Free; free.backend = Backend::Device;
                free.output = lifetime.first;
                free.shape = graph.shapes.at(lifetime.first.substr(2));
                free.buffer_slot = handle_slots.at(lifetime.first);
                slots[free.buffer_slot].busy = false;
                live_bytes -= free.shape.bytes();
                additions.push_back("ADD FREE " + free.output + " slot " + std::to_string(free.buffer_slot));
                emit(free);
            }
        }
    }
    plan.device_buffer_slots = slots.size();
    std::vector<std::string> transfers;
    for (const auto& step : ir.steps) {
        if (step.kind == Kind::Copy) transfers.push_back("ADD COPY " + step.inputs[0] + " -> " + step.output);
        if (step.kind == Kind::Synchronize) transfers.push_back("ADD SYNC device -> host");
    }
    plan.passes.push_back({"InsertTransfersPass", placed_text(graph,placement), transfer_text(ir), transfers,
        {"Copy only when an operand has no valid resident instance on its selected backend", "Synchronize device work before host reads; retain immutable copies"},
        {std::to_string(plan.h2d_bytes + plan.d2h_bytes) + " bytes transferred", std::to_string(ir.synchronizations) + " synchronization commands before lifetime lowering"}});
    plan.passes.push_back({"LowerToExecutionPlanPass", transfer_text(ir), plan_text(plan), additions,
        {"Materialize distinct H:/D: storage before writes; assign versioned sequential command IDs",
         config.reuse_buffers ? "Free device handles after last use; reuse only idle exact-size slots; count idle pool fragmentation against budget" : "Retain every allocation to run end"},
        {std::to_string(plan.commands.size()) + " executable commands", std::to_string(plan.peak_device_bytes) + " peak reserved device bytes",
         std::to_string(plan.buffer_reuses) + " exact-size slot reuses", std::to_string(plan.synchronizations) + " final synchronization commands"}});
    return plan;
}
SearchResult enumerate(const Graph& graph, const Config& config) {
    validate_config(config);
    require(graph.nodes.size() <= max_exhaustive_nodes, "E_LIMIT: enumeration is bounded to 12 nodes");
    SearchResult result;
    const std::size_t combinations = std::size_t{1} << graph.nodes.size();
    // Small graphs only. Exhaustive search gives us a baseline we can inspect.
    // This will not scale to a full model with hundreds of operators.
    for (std::size_t mask = 0; mask < combinations; ++mask) {
        std::vector<Backend> placement(graph.nodes.size(), Backend::Cpu);
        bool legal = true;
        for (std::size_t i = 0; i < placement.size(); ++i) {
            if ((mask & (std::size_t{1} << i)) == 0) continue;
            placement[i] = Backend::Device;
            if (!capability(graph.nodes[i], graph, config).accepted) legal = false;
        }
        if (!legal) { ++result.infeasible; continue; }
        try { result.plans.push_back(lower(graph, placement, config)); }
        catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("E_MEMORY:") != 0) throw;
            ++result.infeasible;
        }
    }
    std::stable_sort(result.plans.begin(), result.plans.end(), [](const Plan& a, const Plan& b) {
        return a.modeled_ms < b.modeled_ms;
    });
    return result;
}
Plan choose(const Graph& graph, const Config& config, const std::string& policy) {
    Plan plan;
    if (policy == "cost") {
        const auto search = enumerate(graph, config);
        require(!search.plans.empty(), "E_MEMORY: no feasible plan under the initial residency and memory budget");
        plan = search.plans.front();
    }
    else {
        require(policy == "cpu" || policy == "maximal", "E_POLICY: expected cpu, maximal, or cost");
        std::vector<Backend> placement(graph.nodes.size(), Backend::Cpu);
        if (policy == "maximal") for (std::size_t i = 0; i < placement.size(); ++i)
            if (capability(graph.nodes[i], graph, config).accepted) placement[i] = Backend::Device;
        plan = lower(graph, placement, config);
    }
    plan.policy = policy;
    std::vector<std::string> decisions, changes;
    std::ostringstream annotated;
    annotated << graph_text(graph);
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& n = graph.nodes[i];
        const auto cap = capability(n,graph,config);
        decisions.push_back(n.id + (cap.accepted ? " ACCEPT: " : " REJECT: ") + cap.reason);
        annotated << "  " << decisions.back() << '\n';
        changes.push_back("ASSIGN " + n.id + " -> " + name(plan.placement[i]));
    }
    std::vector<PassRecord> before;
    before.push_back({"CapabilityAnalysisPass",graph_text(graph),annotated.str(),
        {"Graph structure unchanged; annotate legal device assignments"}, decisions,
        {"Rejected operators cannot be assigned to the device"}});
    const std::string reason = policy == "cost" ?
        "Enumerate bounded legal placements and minimize sum of lowered compute, copy, launch, and synchronization costs" :
        (policy == "cpu" ? "CPU-only baseline" : "Assign each locally supported node to device; reject if the full plan exceeds memory");
    before.push_back({"PlacementPass",annotated.str(),placed_text(graph,plan.placement),changes,{reason},
        {"Placement " + placement_name(plan.placement),std::to_string(plan.islands) + " device runs in fixed sequential schedule"}});
    before.insert(before.end(),plan.passes.begin(),plan.passes.end());
    plan.passes=std::move(before);
    return plan;
}

Run execute(const Graph& source, const Plan& plan, const Config& config,
            DeviceBackend* device, const std::atomic<bool>* cancel) {
    check_cancel(cancel);
    validate_config(config);
    // Public callers can construct or mutate Graph directly. Validate storage
    // before kernels use its dimensions to index the input vectors.
    const auto checked_source = validate(source);
    require(!device || !config.emulate_boundaries, "E_CONFIG: boundary emulation is available only on the software backend");
    DeviceAllocations device_allocations(device, plan.commands.size() + plan.initial_buffers.size());
    std::map<std::string, Tensor> buffers;
    std::set<std::string> initialized;
    std::set<std::string> allocated_once;
    for (const auto& entry : checked_source.inputs) {
        const auto h = handle(Backend::Cpu, entry.first);
        buffers[h] = entry.second; initialized.insert(h); allocated_once.insert(h);
    }
    std::size_t device_allocated = 0;
    std::size_t live_device_bytes = 0;
    struct RuntimeSlot { std::size_t bytes; std::string owner; std::vector<float> storage; };
    std::vector<RuntimeSlot> slots;
    std::map<std::string, std::size_t> handle_slots;
    KernelStats kernel_stats;
    bool pending = false;
    Run run;
    if (device) run.execution_backend = device->description();
    if (config.emulate_boundaries) run.execution_backend = "host software with injected boundary waits";
    auto emulate = [&](double milliseconds) {
        if (!config.emulate_boundaries || milliseconds == 0) return;
        require(std::isfinite(milliseconds) && milliseconds >= 0 && milliseconds <= 100 &&
                run.injected_boundary_ms + milliseconds <= 10000,
                "E_DELAY: emulation permits <=100 ms per boundary and <=10 seconds per plan");
        run.injected_boundary_ms += milliseconds;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(milliseconds);
        while (std::chrono::steady_clock::now() < deadline) {
            check_cancel(cancel);
            const std::chrono::duration<double> remaining = deadline - std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::min(remaining, std::chrono::duration<double>(0.001)));
        }
        check_cancel(cancel);
    };
    auto mirrored = config.mirrored_inputs;
    std::sort(mirrored.begin(), mirrored.end());
    require(plan.initial_buffers.size() == mirrored.size(), "E_RESIDENCY: initial buffer plan differs from configuration");
    const auto preparation_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < plan.initial_buffers.size(); ++i) {
        check_cancel(cancel);
        const auto& c = plan.initial_buffers[i];
        require(c.parameters.empty(), "E_ATTRIBUTE: preparation cannot carry compute parameters");
        require(checked_source.inputs.count(mirrored[i]) != 0, "E_RESIDENCY: unknown mirrored source input");
        const auto& input = checked_source.inputs.at(mirrored[i]);
        require(c.version == command_version && c.id == i && c.kind == Kind::Allocate && c.backend == Backend::Device &&
                c.output == handle(Backend::Device, mirrored[i]) && c.shape == input.shape && c.inputs.empty(),
                "E_RESIDENCY: invalid preparation descriptor");
        require(c.buffer_slot == (config.reuse_buffers ? i : no_buffer_slot), "E_SLOT: invalid initial residency slot");
        const auto bytes = input.shape.bytes();
        require(bytes <= config.device_bytes && device_allocated <= config.device_bytes - bytes,
                "E_MEMORY: initial device residency exceeds execution budget");
        device_allocated += bytes; live_device_bytes += bytes;
        if (config.reuse_buffers) {
            slots.push_back({bytes, c.output, {}}); handle_slots[c.output] = i;
        }
        if (device) {
            device_allocations.allocate(c);
            device->write(c.output, input);
            buffers.emplace(c.output, Tensor{input.shape, {}});
        } else buffers.emplace(c.output, input);
        emulate(config.transfer_fixed_ms + static_cast<double>(bytes) / config.bandwidth_bytes_per_ms);
        allocated_once.insert(c.output); initialized.insert(c.output);
        run.preparation_h2d_bytes += bytes;
    }
    if (!plan.initial_buffers.empty()) {
        if (device) device->synchronize();
        run.input_preparation_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - preparation_start).count();
    }
    run.peak_device_reserved_bytes = device_allocated;
    const auto diagnostic_start = device ? device->diagnostic_readback_bytes() : 0;
    const auto start = std::chrono::steady_clock::now();
    Backend previous_compute = Backend::Cpu;
    for (std::size_t index = 0; index < plan.commands.size(); ++index) {
        check_cancel(cancel);
        const auto& c = plan.commands[index];
        double device_us = -1;
        const auto event_start = std::chrono::steady_clock::now();
        require(c.version == command_version, "E_VERSION: unsupported command version");
        require(c.id == index, "E_SEQUENCE: command IDs are not contiguous");
        require(c.kind == Kind::Compute || c.parameters.empty(), "E_ATTRIBUTE: noncompute command parameters");
        require(c.kind == Kind::Allocate || c.kind == Kind::Free || c.buffer_slot == no_buffer_slot,
                "E_SLOT: only allocation/free descriptors may name a buffer slot");
        (void)name(c.backend);
        auto in_lane = [&](const std::string& h) {
            return h.size() > 2 && h.substr(0, 2) == (c.backend == Backend::Cpu ? "H:" : "D:");
        };
        if (c.kind == Kind::Synchronize) {
            require(c.backend == Backend::Device, "E_SYNC: invalid synchronization target");
            // Arithmetic already ran on this thread. This flag checks the
            // completion contract; it is not an asynchronous device queue.
            if (device) device->synchronize();
            emulate(config.synchronization_ms);
            pending = false;
        } else if (c.kind == Kind::Allocate) {
            require(in_lane(c.output), "E_HANDLE: allocation backend and handle disagree");
            require(allocated_once.count(c.output) == 0, "E_ALLOC: duplicate logical allocation");
            const auto count = c.shape.elements();
            if (c.backend == Backend::Device) {
                const auto bytes = c.shape.bytes();
                bool reserve = true;
                if (config.reuse_buffers) {
                    require(c.buffer_slot <= slots.size(), "E_SLOT: invalid or noncontiguous buffer slot");
                    if (c.buffer_slot < slots.size()) {
                        auto& slot = slots[c.buffer_slot];
                        require(slot.owner.empty(), "E_SLOT: buffer slot already has a live owner");
                        require(slot.bytes == bytes, "E_SLOT: reused buffer size differs");
                        reserve = false;
                        if (!device) ++run.software_buffer_reuses;
                    }
                } else require(c.buffer_slot == no_buffer_slot, "E_SLOT: buffer reuse is disabled");
                if (reserve) {
                    require(bytes <= config.device_bytes && device_allocated <= config.device_bytes - bytes, "E_MEMORY: device reservation budget exceeded");
                    device_allocated += bytes;
                    if (config.reuse_buffers) slots.push_back({bytes, {}, {}});
                }
                live_device_bytes += bytes;
                run.peak_device_reserved_bytes = std::max(run.peak_device_reserved_bytes, device ? live_device_bytes : device_allocated);
                if (config.reuse_buffers) {
                    slots[c.buffer_slot].owner = c.output;
                    handle_slots.emplace(c.output, c.buffer_slot);
                }
            } else require(c.buffer_slot == no_buffer_slot, "E_SLOT: host allocation cannot claim a device slot");
            allocated_once.insert(c.output);
            if (device && c.backend == Backend::Device) {
                device_allocations.allocate(c);
                buffers.emplace(c.output, Tensor{c.shape, {}});
            } else if (config.reuse_buffers && c.backend == Backend::Device) {
                auto& storage = slots[c.buffer_slot].storage;
                if (storage.empty()) storage.resize(count);
                buffers.emplace(c.output, Tensor{c.shape, std::move(storage)});
            } else buffers.emplace(c.output, Tensor{c.shape, std::vector<float>(count)});
        } else if (c.kind == Kind::Free) {
            require(config.reuse_buffers, "E_FREE: explicit release requires buffer reuse mode");
            require(c.backend == Backend::Device && in_lane(c.output) && c.inputs.empty(), "E_FREE: only a single device handle may be freed");
            require(!pending, "E_SYNC: cannot free pending device operands");
            require(buffers.count(c.output) && initialized.count(c.output), "E_FREE: unknown or uninitialized device handle");
            require(buffers.at(c.output).shape == c.shape, "E_FREE: released shape differs");
            require(handle_slots.count(c.output) && handle_slots.at(c.output) == c.buffer_slot && c.buffer_slot < slots.size(), "E_SLOT: release slot differs from allocation");
            auto& slot = slots[c.buffer_slot];
            require(slot.owner == c.output, "E_SLOT: release does not own buffer slot");
            if (device) device_allocations.release(c.output);
            else slot.storage = std::move(buffers.at(c.output).values);
            live_device_bytes -= c.shape.bytes();
            slot.owner.clear();
            handle_slots.erase(c.output);
            initialized.erase(c.output);
            buffers.erase(c.output);
        } else if (c.kind == Kind::Copy) {
            require(c.inputs.size() == 1 && in_lane(c.output) && !in_lane(c.inputs[0]), "E_COPY: copy must cross backends");
            require(buffers.count(c.output) && initialized.count(c.inputs[0]), "E_BUFFER: copy references unallocated or uninitialized storage");
            require(!initialized.count(c.output), "E_SSA: cannot overwrite initialized storage");
            require(c.output.substr(2) == c.inputs[0].substr(2), "E_COPY: tensor identity changed during copy");
            require(c.backend != Backend::Cpu || !pending, "E_SYNC: device result read before synchronization");
            auto& dst = buffers.at(c.output);
            const auto& src = buffers.at(c.inputs[0]);
            require(dst.shape == src.shape && dst.shape == c.shape, "E_SHAPE: copy shapes differ");
            if (device) {
                if (c.backend == Backend::Device) device->write(c.output, src);
                else {
                    device->read(c.inputs[0], dst);
                    require(dst.shape == c.shape && dst.values.size() == c.shape.elements(), "E_VALUE: backend returned invalid host storage");
                    for (float v : dst.values) require(std::isfinite(v), "E_NUMERIC: backend returned nonfinite output");
                }
                device_us = device->last_elapsed_us();
            } else std::copy(src.values.begin(), src.values.end(), dst.values.begin());
            emulate(config.transfer_fixed_ms + static_cast<double>(c.shape.bytes()) / config.bandwidth_bytes_per_ms);
            initialized.insert(c.output);
        } else if (c.kind == Kind::Compute) {
            require(in_lane(c.output) && buffers.count(c.output), "E_BUFFER: missing output allocation or wrong backend");
            require(!initialized.count(c.output), "E_SSA: compute cannot overwrite initialized storage");
            std::vector<Shape> shapes;
            for (const auto& h : c.inputs) {
                require(in_lane(h) && initialized.count(h), "E_BUFFER: missing initialized operand on compute backend");
                shapes.push_back(buffers.at(h).shape);
            }
            require(result_shape(c.op, shapes, c.parameters) == c.shape && buffers.at(c.output).shape == c.shape, "E_SHAPE: compute output shape mismatch");
            if (c.backend == Backend::Device) {
                require(config.relu_supported || (c.op != Op::Relu && c.op != Op::MatMulRelu), "E_CAPABILITY: device rejects unsupported ReLU command");
                require(!cnn_op(c.op) || config.cnn_supported, "E_CAPABILITY: device CNN kernels disabled");
                {
                    auto tensors = shapes; tensors.push_back(c.shape);
                    const auto rule = target_rule_rejection(c.op, tensors, c.parameters, config);
                    require(rule.empty(), "E_CAPABILITY: device rejects command: " + rule);
                }
                require(!device || !cnn_op(c.op), "E_CAPABILITY: external backend has no CNN kernel contract");
                if (previous_compute != Backend::Device) emulate(config.launch_ms);
                pending = true;
            }
            previous_compute = c.backend;
            auto& out = buffers.at(c.output);
            if (device && c.backend == Backend::Device) {
                device->compute(c);
                device_us = device->last_elapsed_us();
            } else {
                std::vector<const Tensor*> operands;
                for (const auto& input : c.inputs) operands.push_back(&buffers.at(input));
                auto kernel_config = config;
                if (c.backend == Backend::Cpu) kernel_config.tile_size = 0;
                if (cnn_op(c.op)) compute_cnn(c.op, operands, out, c.parameters, cancel);
                else compute_software(c.op, operands, out, kernel_config, kernel_stats, cancel);
            }
            for (float v : out.values) require(std::isfinite(v), "E_NUMERIC: computation produced nonfinite output");
            initialized.insert(c.output);
        } else throw std::runtime_error("E_COMMAND: unknown command kind");
        const auto event_end = std::chrono::steady_clock::now();
        run.events.push_back({c.id, c.node, name(c.kind), name(c.backend),
            std::chrono::duration<double, std::micro>(event_start - start).count(),
            std::chrono::duration<double, std::micro>(event_end - event_start).count(), c.modeled_ms, device_us, {}, {}});
    }
    require(!pending, "E_SYNC: execution ended with unsynchronized device work");
    require(initialized.count(handle(Backend::Cpu, checked_source.output)) != 0, "E_OUTPUT: final output is not available on host");
    run.output = buffers.at(handle(Backend::Cpu, checked_source.output));
    run.tile_count = kernel_stats.tile_count;
    run.tile_load_bytes = kernel_stats.tile_load_bytes;
    run.peak_local_bytes = kernel_stats.peak_local_bytes;
    run.host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (device) {
        require(device->diagnostic_readback_bytes() >= diagnostic_start, "E_BACKEND: diagnostic counter decreased");
        run.diagnostic_readback_bytes = device->diagnostic_readback_bytes() - diagnostic_start;
    }
    check_against_reference(run, checked_source, cancel);
    device_allocations.release(); // explicit success cleanup propagates device release errors
    return run;
}
void check_against_reference(Run& run, const Graph& source, const std::atomic<bool>* cancel) {
    const auto expected = reference(source, cancel);
    check_cancel(cancel);
    require(run.output.shape == expected.shape, "E_OUTPUT: result shape differs from reference");
    run.correct = true;
    for (std::size_t i = 0; i < expected.values.size(); ++i) {
        const double error = std::abs(static_cast<double>(run.output.values[i]) - expected.values[i]);
        run.max_abs_error = std::max(run.max_abs_error, error);
        if (!std::isfinite(expected.values[i]) || !std::isfinite(error) || error > 1e-5 + 1e-5 * std::abs(expected.values[i])) run.correct = false;
    }
}

std::string graph_text(const Graph& graph) {
    std::ostringstream out;
    out << "Typed graph IR | f32, contiguous rank-2 tensors\n";
    for (const auto& n : graph.nodes) {
        const auto s = graph.shapes.at(n.output);
        out << "  " << n.id << " : " << name(n.op) << '(';
        for (std::size_t i = 0; i < n.inputs.size(); ++i) out << (i ? ", " : "") << n.inputs[i];
        out << ") -> " << n.output << " : f32[" << s.rows << ',' << s.cols << "]";
        if (!n.parameters.empty()) out << " parameters=" << parameters_json(n.parameters);
        out << '\n';
    }
    return out.str();
}
std::string plan_text(const Plan& plan) {
    std::ostringstream out;
    out << "Command plan " << placement_name(plan.placement) << " | C=CPU, A=virtual accelerator\n";
    for (const auto& c : plan.initial_buffers)
        out << "  prepare " << c.output << " (" << c.shape.bytes() << " bytes), mirrored before measured commands\n";
    for (const auto& c : plan.commands) {
        out << "  " << std::setw(2) << c.id << "  " << std::left << std::setw(8) << name(c.kind) << std::right;
        if (c.kind == Kind::Compute) out << name(c.op) << " [" << c.node << "] ";
        for (const auto& in : c.inputs) out << in << ' ';
        if (!c.output.empty()) out << "-> " << c.output;
        if (c.kind == Kind::Allocate || c.kind == Kind::Free) out << " (" << c.shape.bytes() << " bytes)";
        if (c.buffer_slot != no_buffer_slot) out << " slot=" << c.buffer_slot;
        if (c.kind == Kind::Synchronize) out << "device completion visible to host";
        out << '\n';
    }
    return out.str();
}
namespace {
std::string passes_json(const std::vector<PassRecord>& passes) {
    std::ostringstream s;
    s << '[';
    for (std::size_t i=0; i<passes.size(); ++i) {
        if (i) s << ',';
        const auto& p=passes[i];
        s << "{\"name\":" << quote(p.name) << ",\"input_ir\":" << quote(p.input_ir)
          << ",\"output_ir\":" << quote(p.output_ir) << ",\"changes\":" << strings_json(p.changes)
          << ",\"reasons\":" << strings_json(p.reasons) << ",\"consequences\":" << strings_json(p.consequences) << '}';
    }
    return s.str()+"]";
}
}
std::string report_json(const Graph& source, const Graph& compiled, const Config& config,
                        const std::vector<std::string>& remarks, const std::vector<Plan>& plans,
                        const std::vector<Run>& runs, const SearchResult& search, const std::vector<PassRecord>& frontend_passes) {
    require(runs.size() == plans.size(), "E_REPORT: each plan needs an execution result");
    std::ostringstream s;
    s << std::setprecision(std::numeric_limits<double>::max_digits10);
    s << "{\"schema_version\":3,\"cost_provenance\":\"illustrative uncalibrated model; not hardware measurements\","
      << "\"execution_backend\":\"see per-result execution_backend; placement letters are logical lanes\","
      << "\"source_graph\":" << graph_json(source) << ",\"compiled_graph\":" << graph_json(compiled)
      << ",\"frontend_passes\":" << passes_json(frontend_passes) << ",\"pass_remarks\":" << strings_json(remarks) << ",\"config\":{\"relu_supported\":" << (config.relu_supported ? "true" : "false")
      << ",\"cnn_supported\":" << (config.cnn_supported ? "true" : "false")
      << ",\"device_unsupported_ops\":" << strings_json(config.device_unsupported_ops)
      << ",\"device_max_tensor_bytes\":" << config.device_max_tensor_bytes
      << ",\"device_grouped_conv\":" << (config.device_grouped_conv ? "true" : "false")
      << ",\"device_bytes\":" << config.device_bytes << ",\"transfer_fixed_ms\":" << config.transfer_fixed_ms
      << ",\"bandwidth_bytes_per_ms\":" << config.bandwidth_bytes_per_ms << ",\"launch_ms\":" << config.launch_ms
      << ",\"synchronization_ms\":" << config.synchronization_ms
      << ",\"reuse_buffers\":" << (config.reuse_buffers ? "true" : "false")
      << ",\"tile_size\":" << config.tile_size << ",\"local_bytes\":" << config.local_bytes
      << ",\"mirrored_inputs\":" << strings_json(config.mirrored_inputs)
      << ",\"emulate_boundaries\":" << (config.emulate_boundaries ? "true" : "false")
      << ",\"compute_ms\":{\"cpu\":[6,1,2,7,2,6,2,2,2,0.1],\"device\":[1,0.1,0.4,1.1,0.5,1,0.5,0.5,0.5,0.1],\"op_order\":[\"MatMul\",\"ReLU\",\"Add\",\"MatMulReLU\",\"RowSum\",\"Conv2D\",\"MaxPool2D\",\"AveragePool2D\",\"GlobalAveragePool\",\"Reshape\"]}}"
      << ",\"capabilities\":[";
    for (std::size_t i = 0; i < compiled.nodes.size(); ++i) {
        if (i) s << ',';
        const auto cap = capability(compiled.nodes[i], compiled, config);
        s << "{\"node_id\":" << quote(compiled.nodes[i].id) << ",\"device_supported\":" << (cap.accepted ? "true" : "false")
          << ",\"reason\":" << quote(cap.reason) << '}';
    }
    s << "],\"search\":{\"exhaustive\":" << (search.exhaustive ? "true" : "false")
      << ",\"strategy\":" << quote(search.strategy)
      << ",\"infeasible\":";
    if (search.exhaustive) s << search.infeasible; else s << "null";
    if (search.strategy == "bounded-beam") {
        s << ",\"evaluated\":" << search.evaluated << ",\"rejected\":" << search.infeasible
          << ",\"beam_width\":" << search.beam_width << ",\"max_evaluations\":" << search.max_evaluations
          << ",\"budget_exhausted\":" << (search.budget_exhausted ? "true" : "false")
          << ",\"maximal_placement\":" << quote(search.maximal_placement)
          << ",\"maximal_feasible\":" << (search.maximal_feasible ? "true" : "false")
          << ",\"maximal_rejection\":" << quote(search.maximal_rejection);
    }
    s << ",\"ranked_feasible\":[";
    for (std::size_t i = 0; i < search.plans.size(); ++i) {
        if (i) s << ',';
        s << "{\"placement\":" << quote(placement_name(search.plans[i].placement)) << ",\"modeled_ms\":" << search.plans[i].modeled_ms << '}';
    }
    s << "]},\"results\":[";
    for (std::size_t i = 0; i < plans.size(); ++i) {
        if (i) s << ',';
        const auto& p = plans[i]; const auto& r = runs[i];
        s << "{\"policy\":" << quote(p.policy) << ",\"placement\":" << quote(placement_name(p.placement))
          << ",\"passes\":" << passes_json(p.passes) << ",\"modeled_ms\":" << p.modeled_ms << ",\"host_execution_ms\":" << r.host_ms
          << ",\"execution_backend\":" << quote(r.execution_backend)
          << ",\"diagnostic_readback_bytes\":" << r.diagnostic_readback_bytes
          << ",\"input_preparation_ms\":" << r.input_preparation_ms << ",\"preparation_h2d_bytes\":" << r.preparation_h2d_bytes
          << ",\"injected_boundary_ms\":" << r.injected_boundary_ms;
        if (r.engine_overlap_us >= 0) s << ",\"engine_overlap_us\":" << r.engine_overlap_us;
        s
          << ",\"software_buffer_reuses\":" << r.software_buffer_reuses
          << ",\"runtime_peak_device_reserved_bytes\":" << r.peak_device_reserved_bytes
          << ",\"tile_count\":" << r.tile_count << ",\"tile_load_bytes\":" << r.tile_load_bytes << ",\"peak_local_bytes\":" << r.peak_local_bytes
          << ",\"queue_wait_ms\":";
        if (r.queue_wait_ms < 0) s << "null"; else s << r.queue_wait_ms;
        s << ",\"end_to_end_ms\":";
        if (r.end_to_end_ms < 0) s << "null"; else s << r.end_to_end_ms;
        s
          << ",\"h2d_bytes\":" << p.h2d_bytes << ",\"d2h_bytes\":" << p.d2h_bytes
          << ",\"peak_allocated_device_bytes\":" << p.peak_device_bytes << ",\"islands\":" << p.islands
          << ",\"device_buffer_slots\":" << p.device_buffer_slots << ",\"buffer_reuses\":" << p.buffer_reuses
          << ",\"peak_live_device_bytes\":" << p.peak_live_device_bytes
          << ",\"synchronizations\":" << p.synchronizations << ",\"correct\":" << (r.correct ? "true" : "false")
          << ",\"max_abs_error\":" << r.max_abs_error << ",\"output\":[";
        for (std::size_t j = 0; j < r.output.values.size(); ++j) { if (j) s << ','; s << r.output.values[j]; }
        s << "],\"initial_buffers\":[";
        for (std::size_t j = 0; j < p.initial_buffers.size(); ++j) {
            if (j) s << ',';
            const auto& c = p.initial_buffers[j];
            s << "{\"id\":" << c.id << ",\"version\":" << c.version << ",\"kind\":\"ALLOC\",\"backend\":\"virtual-device\",\"output\":"
              << quote(c.output) << ",\"shape\":[" << c.shape.rows << ',' << c.shape.cols << "],\"parameters\":[],\"buffer_slot\":";
            if (c.buffer_slot == no_buffer_slot) s << "null"; else s << c.buffer_slot;
            s << '}';
        }
        s << "],\"commands\":[";
        for (std::size_t j = 0; j < p.commands.size(); ++j) {
            if (j) s << ',';
            const auto& c = p.commands[j];
            s << "{\"id\":" << c.id << ",\"version\":" << c.version << ",\"kind\":" << quote(name(c.kind))
              << ",\"backend\":" << quote(name(c.backend)) << ",\"node\":" << quote(c.node)
              << ",\"op\":" << (c.kind == Kind::Compute ? quote(name(c.op)) : "null")
              << ",\"inputs\":" << strings_json(c.inputs) << ",\"output\":" << quote(c.output)
              << ",\"shape\":[" << c.shape.rows << ',' << c.shape.cols << "],\"parameters\":" << parameters_json(c.parameters) << ",\"origins\":" << strings_json(c.origins)
              << ",\"modeled_ms\":" << c.modeled_ms << ",\"buffer_slot\":";
            if (c.buffer_slot == no_buffer_slot) s << "null"; else s << c.buffer_slot;
            s << '}';
        }
        s << "],\"events\":[";
        for (std::size_t j = 0; j < r.events.size(); ++j) {
            if (j) s << ',';
            const auto& e = r.events[j];
            s << "{\"command_id\":" << e.command_id << ",\"node\":" << quote(e.node)
              << ",\"name\":" << quote(e.name) << ",\"lane\":" << quote(e.lane)
              << ",\"start_us\":" << e.start_us << ",\"duration_us\":" << e.duration_us
              << ",\"modeled_ms\":" << e.modeled_ms << ",\"device_us\":";
            if (e.device_us < 0) s << "null"; else s << e.device_us;
            if (!e.engine.empty()) {
                s << ",\"engine\":" << quote(e.engine) << ",\"waits\":[";
                for (std::size_t k = 0; k < e.waits.size(); ++k) s << (k ? "," : "") << e.waits[k];
                s << ']';
            }
            s << '}';
        }
        s << "]}";
    }
    return s.str() + "]}\n";
}
std::string trace_json(const Run& run) {
    std::ostringstream s;
    s << std::setprecision(std::numeric_limits<double>::max_digits10) << "{\"displayTimeUnit\":\"ms\",\"trace_provenance\":\"measured host command durations; hardware event duration is separate when available\",\"execution_backend\":"
      << quote(run.execution_backend) << ",\"traceEvents\":[";
    for (std::size_t i = 0; i < run.events.size(); ++i) {
        if (i) s << ',';
        const auto& e = run.events[i];
        // Command-queue events get one trace row per engine so overlap is visible.
        const int tid = e.engine == "copy-engine" ? 2 : e.engine == "compute-engine" ? 3 : 1;
        s << "{\"name\":" << quote(e.name + " " + e.node) << ",\"cat\":" << quote(e.engine.empty() ? "host execution" : e.engine)
          << ",\"ph\":\"X\",\"pid\":1,\"tid\":" << tid << ",\"ts\":"
          << e.start_us << ",\"dur\":" << e.duration_us << ",\"args\":{\"command_id\":" << e.command_id
          << ",\"logical_backend\":" << quote(e.lane) << ",\"modeled_ms\":" << e.modeled_ms << "}}";
    }
    return s.str() + "]}\n";
}
}
