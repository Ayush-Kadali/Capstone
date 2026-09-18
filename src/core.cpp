#include "placementlens/core.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace pl {
namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
bool identifier(const std::string& s) {
    return !s.empty() && s.size() <= 64 && std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_';
    });
}
Shape result_shape(Op op, const std::vector<Shape>& inputs) {
    const bool unary = op == Op::Relu;
    require(inputs.size() == (unary ? 1U : 2U), "E_ARITY: wrong number of operands");
    if (op == Op::Relu) return inputs[0];
    if (op == Op::Add) {
        require(inputs[0] == inputs[1], "E_SHAPE: Add requires equal shapes; broadcasting is unsupported");
        return inputs[0];
    }
    require(op == Op::MatMul || op == Op::MatMulRelu, "E_OPCODE: unsupported opcode");
    require(inputs[0].cols == inputs[1].rows, "E_SHAPE: MatMul inner dimensions differ");
    Shape out{inputs[0].rows, inputs[1].cols};
    (void)out.elements();
    return out;
}
std::string handle(Backend b, const std::string& tensor) {
    return (b == Backend::Cpu ? "H:" : "D:") + tensor;
}
double compute_cost(Op op, Backend b) {
    // Deliberately illustrative per-op constants, not calibrated performance estimates.
    switch (op) {
        case Op::MatMul: return b == Backend::Cpu ? 6.0 : 1.0;
        case Op::Relu: return b == Backend::Cpu ? 1.0 : 0.1;
        case Op::Add: return b == Backend::Cpu ? 2.0 : 0.4;
        case Op::MatMulRelu: return b == Backend::Cpu ? 7.0 : 1.1;
    }
    throw std::runtime_error("E_OPCODE: unsupported cost opcode");
}
std::string strings_json(const std::vector<std::string>& xs) {
    std::string out = "[";
    for (std::size_t i = 0; i < xs.size(); ++i) out += (i ? "," : "") + quote(xs[i]);
    return out + "]";
}
std::string graph_json(const Graph& g) {
    std::ostringstream s;
    s << "{\"output\":" << quote(g.output) << ",\"inputs\":[";
    bool comma = false;
    for (const auto& entry : g.inputs) {
        if (comma) s << ',';
        comma = true;
        s << "{\"name\":" << quote(entry.first) << ",\"dtype\":\"f32\",\"shape\":["
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
        s << "{\"id\":" << quote(n.id) << ",\"op\":" << quote(name(n.op))
          << ",\"inputs\":" << strings_json(n.inputs) << ",\"output\":" << quote(n.output)
          << ",\"origins\":" << strings_json(n.origins) << '}';
    }
    return s.str() + "]}";
}
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
    }
    throw std::runtime_error("E_OPCODE: unknown opcode");
}
std::string name(Kind kind) {
    switch (kind) {
        case Kind::Allocate: return "ALLOC"; case Kind::Copy: return "COPY";
        case Kind::Compute: return "COMPUTE"; case Kind::Synchronize: return "SYNC";
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
                std::string id, dtype;
                Tensor t;
                require(static_cast<bool>(row >> id >> dtype >> t.shape.rows >> t.shape.cols), "E_PARSE: tensor needs name, dtype, rows, cols, values");
                require(dtype == "f32", "E_DTYPE: only explicit f32 inputs are supported; no implicit conversion");
                require(graph.inputs.size() < 64, "E_LIMIT: at most 64 graph inputs");
                for (std::size_t i = 0; i < t.shape.elements(); ++i) {
                    float v = 0;
                    require(static_cast<bool>(row >> v) && std::isfinite(v), "E_VALUE: expected finite f32 tensor data");
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
                else throw std::runtime_error("E_PARSE: unknown statement " + tag);
                std::string a, b;
                require(static_cast<bool>(row >> n.id >> a), "E_PARSE: missing node id or operand");
                n.inputs.push_back(a);
                if (n.op != Op::Relu) {
                    require(static_cast<bool>(row >> b), "E_PARSE: missing second operand");
                    n.inputs.push_back(b);
                }
                require(static_cast<bool>(row >> n.output), "E_PARSE: missing node output");
                n.origins = {n.id};
                graph.nodes.push_back(n);
                require(graph.nodes.size() <= max_nodes, "E_LIMIT: at most 12 nodes");
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
    require(!graph.nodes.empty() && graph.nodes.size() <= max_nodes, "E_LIMIT: expected 1..12 nodes");
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
        require(n.inputs.size() == (n.op == Op::Relu ? 1U : 2U), "E_ARITY: wrong operand count");
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
            graph.shapes[n.output] = result_shape(n.op, shapes);
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
    for (double v : {c.transfer_fixed_ms, c.launch_ms, c.synchronization_ms})
        require(std::isfinite(v) && v >= 0 && v <= 1e9, "E_CONFIG: costs must be finite, nonnegative, and <= 1e9 ms");
    require(std::isfinite(c.bandwidth_bytes_per_ms) && c.bandwidth_bytes_per_ms >= 1e-9,
            "E_CONFIG: bandwidth must be finite and >= 1e-9 bytes/ms");
}
Capability capability(const Node& n, const Graph& g, const Config& c) {
    if ((n.op == Op::Relu || n.op == Op::MatMulRelu) && !c.relu_supported)
        return {false, "ReLU kernel is absent from the virtual-device capability profile"};
    if (n.op != Op::Relu && n.op != Op::MatMul && n.op != Op::Add && n.op != Op::MatMulRelu)
        return {false, "opcode is unsupported"};
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

Tensor reference(const Graph& graph) {
    // Independent high-level evaluator: double accumulation, original unfused graph.
    auto values = graph.inputs;
    for (const auto& n : graph.nodes) {
        const auto& a = values.at(n.inputs[0]);
        Tensor out{graph.shapes.at(n.output), {}};
        out.values.resize(out.shape.elements());
        for (std::size_t r = 0; r < out.shape.rows; ++r) {
            for (std::size_t col = 0; col < out.shape.cols; ++col) {
                double v = 0;
                if (n.op == Op::Relu) v = std::max(0.0, static_cast<double>(a.values[r * a.shape.cols + col]));
                else {
                    const auto& b = values.at(n.inputs[1]);
                    if (n.op == Op::Add) v = static_cast<double>(a.values[r * a.shape.cols + col]) + b.values[r * b.shape.cols + col];
                    else {
                        for (std::size_t k = 0; k < a.shape.cols; ++k)
                            v += static_cast<double>(a.values[r * a.shape.cols + k]) * b.values[k * b.shape.cols + col];
                        if (n.op == Op::MatMulRelu) v = std::max(0.0, v);
                    }
                }
                out.values[r * out.shape.cols + col] = static_cast<float>(v);
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
        if (valid.count(destination)) return;
        const auto other = b == Backend::Cpu ? Backend::Device : Backend::Cpu;
        const auto source = handle(other, t);
        require(valid.count(source) != 0, "E_LOWER: input has no valid resident copy");
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
        (void)name(b);
        if (b == Backend::Device) {
            auto cap = capability(n, graph, config);
            require(cap.accepted, "E_CAPABILITY: " + n.id + ": " + cap.reason);
        } else sync();
        for (const auto& in : n.inputs) ensure(b, in);
        TransferStep cmd;
        cmd.kind = Kind::Compute; cmd.backend = b; cmd.node = n.id; cmd.op = n.op;
        cmd.output = handle(b, n.output); cmd.shape = graph.shapes.at(n.output); cmd.origins = n.origins;
        for (const auto& in : n.inputs) cmd.inputs.push_back(handle(b, in));
        cmd.modeled_ms = compute_cost(n.op, b);
        if (b == Backend::Device) {
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
    plan.islands = ir.islands; plan.synchronizations = ir.synchronizations;
    std::set<std::string> allocated;
    for (const auto& input : graph.inputs) allocated.insert(handle(Backend::Cpu, input.first));
    auto emit = [&](Command c) {
        c.id = plan.commands.size();
        plan.modeled_ms += c.modeled_ms;
        plan.commands.push_back(std::move(c));
    };
    std::vector<std::string> additions;
    for (const auto& step : ir.steps) {
        if (step.kind != Kind::Synchronize && !allocated.count(step.output)) {
            Command allocation;
            allocation.kind = Kind::Allocate; allocation.backend = step.backend;
            allocation.output = step.output; allocation.shape = step.shape;
            if (step.backend == Backend::Device) {
                plan.peak_device_bytes += step.shape.bytes();
                require(plan.peak_device_bytes <= config.device_bytes, "E_MEMORY: plan exceeds retained device allocation budget");
            }
            allocated.insert(step.output);
            additions.push_back("ADD ALLOC " + step.output + " " + std::to_string(step.shape.bytes()) + " bytes");
            emit(allocation);
        }
        Command c;
        c.kind=step.kind; c.backend=step.backend; c.op=step.op; c.node=step.node;
        c.output=step.output; c.inputs=step.inputs; c.shape=step.shape;
        c.modeled_ms=step.modeled_ms; c.origins=step.origins;
        emit(c);
    }
    std::vector<std::string> transfers;
    for (const auto& step : ir.steps) {
        if (step.kind == Kind::Copy) transfers.push_back("ADD COPY " + step.inputs[0] + " -> " + step.output);
        if (step.kind == Kind::Synchronize) transfers.push_back("ADD SYNC device -> host");
    }
    plan.passes.push_back({"InsertTransfersPass", placed_text(graph,placement), transfer_text(ir), transfers,
        {"Copy only when an operand has no valid resident instance on its selected backend", "Synchronize device work before host reads; retain immutable copies"},
        {std::to_string(plan.h2d_bytes + plan.d2h_bytes) + " bytes transferred", std::to_string(plan.synchronizations) + " synchronization commands"}});
    plan.passes.push_back({"LowerToExecutionPlanPass", transfer_text(ir), plan_text(plan), additions,
        {"Materialize distinct H:/D: storage before writes; assign versioned sequential command IDs"},
        {std::to_string(plan.commands.size()) + " executable commands", std::to_string(plan.peak_device_bytes) + " peak retained device bytes"}});
    return plan;
}
SearchResult enumerate(const Graph& graph, const Config& config) {
    validate_config(config);
    require(graph.nodes.size() <= max_nodes, "E_LIMIT: enumeration is bounded to 12 nodes");
    SearchResult result;
    const std::size_t combinations = std::size_t{1} << graph.nodes.size();
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
    if (policy == "cost") plan = enumerate(graph, config).plans.at(0);
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

Run execute(const Graph& source, const Plan& plan, const Config& config) {
    validate_config(config);
    std::map<std::string, Tensor> buffers;
    std::set<std::string> initialized;
    for (const auto& entry : source.inputs) {
        const auto h = handle(Backend::Cpu, entry.first);
        buffers[h] = entry.second; initialized.insert(h);
    }
    std::size_t device_allocated = 0;
    bool pending = false;
    Run run;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < plan.commands.size(); ++index) {
        const auto& c = plan.commands[index];
        const auto event_start = std::chrono::steady_clock::now();
        require(c.version == command_version, "E_VERSION: unsupported command version");
        require(c.id == index, "E_SEQUENCE: command IDs are not contiguous");
        (void)name(c.backend);
        auto in_lane = [&](const std::string& h) {
            return h.size() > 2 && h.substr(0, 2) == (c.backend == Backend::Cpu ? "H:" : "D:");
        };
        if (c.kind == Kind::Synchronize) {
            require(c.backend == Backend::Device, "E_SYNC: invalid synchronization target");
            pending = false;
        } else if (c.kind == Kind::Allocate) {
            require(in_lane(c.output), "E_HANDLE: allocation backend and handle disagree");
            require(buffers.count(c.output) == 0, "E_ALLOC: duplicate allocation");
            const auto count = c.shape.elements();
            if (c.backend == Backend::Device) {
                require(c.shape.bytes() <= config.device_bytes && device_allocated <= config.device_bytes - c.shape.bytes(), "E_MEMORY: device allocation budget exceeded");
                device_allocated += c.shape.bytes();
            }
            buffers.emplace(c.output, Tensor{c.shape, std::vector<float>(count)});
        } else if (c.kind == Kind::Copy) {
            require(c.inputs.size() == 1 && in_lane(c.output) && !in_lane(c.inputs[0]), "E_COPY: copy must cross backends");
            require(buffers.count(c.output) && initialized.count(c.inputs[0]), "E_BUFFER: copy references unallocated or uninitialized storage");
            require(!initialized.count(c.output), "E_SSA: cannot overwrite initialized storage");
            require(c.output.substr(2) == c.inputs[0].substr(2), "E_COPY: tensor identity changed during copy");
            require(c.backend != Backend::Cpu || !pending, "E_SYNC: device result read before synchronization");
            auto& dst = buffers.at(c.output);
            const auto& src = buffers.at(c.inputs[0]);
            require(dst.shape == src.shape && dst.shape == c.shape, "E_SHAPE: copy shapes differ");
            dst.values = src.values;
            initialized.insert(c.output);
        } else if (c.kind == Kind::Compute) {
            require(in_lane(c.output) && buffers.count(c.output), "E_BUFFER: missing output allocation or wrong backend");
            require(!initialized.count(c.output), "E_SSA: compute cannot overwrite initialized storage");
            std::vector<Shape> shapes;
            for (const auto& h : c.inputs) {
                require(in_lane(h) && initialized.count(h), "E_BUFFER: missing initialized operand on compute backend");
                shapes.push_back(buffers.at(h).shape);
            }
            require(result_shape(c.op, shapes) == c.shape && buffers.at(c.output).shape == c.shape, "E_SHAPE: compute output shape mismatch");
            if (c.backend == Backend::Device) {
                require(config.relu_supported || (c.op != Op::Relu && c.op != Op::MatMulRelu), "E_CAPABILITY: device rejects unsupported ReLU command");
                pending = true;
            }
            auto& out = buffers.at(c.output);
            const auto& a = buffers.at(c.inputs[0]);
            if (c.op == Op::Relu) {
                std::transform(a.values.begin(), a.values.end(), out.values.begin(), [](float x) { return std::max(0.0F, x); });
            } else {
                const auto& b = buffers.at(c.inputs[1]);
                if (c.op == Op::Add) {
                    for (std::size_t i = 0; i < out.values.size(); ++i) out.values[i] = a.values[i] + b.values[i];
                } else {
                    for (std::size_t i = 0; i < a.shape.rows; ++i)
                        for (std::size_t j = 0; j < b.shape.cols; ++j) {
                            float acc = 0;
                            for (std::size_t k = 0; k < a.shape.cols; ++k)
                                acc += a.values[i * a.shape.cols + k] * b.values[k * b.shape.cols + j];
                            out.values[i * out.shape.cols + j] = c.op == Op::MatMulRelu ? std::max(0.0F, acc) : acc;
                        }
                }
            }
            for (float v : out.values) require(std::isfinite(v), "E_NUMERIC: computation produced nonfinite output");
            initialized.insert(c.output);
        } else throw std::runtime_error("E_COMMAND: unknown command kind");
        const auto event_end = std::chrono::steady_clock::now();
        run.events.push_back({c.id, c.node, name(c.kind), name(c.backend),
            std::chrono::duration<double, std::micro>(event_start - start).count(),
            std::chrono::duration<double, std::micro>(event_end - event_start).count(), c.modeled_ms});
    }
    require(!pending, "E_SYNC: execution ended with unsynchronized device work");
    require(initialized.count(handle(Backend::Cpu, source.output)) != 0, "E_OUTPUT: final output is not available on host");
    run.output = buffers.at(handle(Backend::Cpu, source.output));
    run.host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto expected = reference(source);
    require(run.output.shape == expected.shape, "E_OUTPUT: result shape differs from reference");
    run.correct = true;
    for (std::size_t i = 0; i < expected.values.size(); ++i) {
        const double error = std::abs(static_cast<double>(run.output.values[i]) - expected.values[i]);
        run.max_abs_error = std::max(run.max_abs_error, error);
        if (!std::isfinite(expected.values[i]) || !std::isfinite(error) || error > 1e-5 + 1e-5 * std::abs(expected.values[i])) run.correct = false;
    }
    return run;
}

std::string graph_text(const Graph& graph) {
    std::ostringstream out;
    out << "Typed graph IR | f32, contiguous rank-2 tensors\n";
    for (const auto& n : graph.nodes) {
        const auto s = graph.shapes.at(n.output);
        out << "  " << n.id << " : " << name(n.op) << '(';
        for (std::size_t i = 0; i < n.inputs.size(); ++i) out << (i ? ", " : "") << n.inputs[i];
        out << ") -> " << n.output << " : f32[" << s.rows << ',' << s.cols << "]\n";
    }
    return out.str();
}
std::string plan_text(const Plan& plan) {
    std::ostringstream out;
    out << "Command plan " << placement_name(plan.placement) << " | C=CPU, A=virtual accelerator\n";
    for (const auto& c : plan.commands) {
        out << "  " << std::setw(2) << c.id << "  " << std::left << std::setw(8) << name(c.kind) << std::right;
        if (c.kind == Kind::Compute) out << name(c.op) << " [" << c.node << "] ";
        for (const auto& in : c.inputs) out << in << ' ';
        if (!c.output.empty()) out << "-> " << c.output;
        if (c.kind == Kind::Allocate) out << " (" << c.shape.bytes() << " bytes)";
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
    s << std::setprecision(12);
    s << "{\"schema_version\":1,\"cost_provenance\":\"illustrative uncalibrated model; not hardware measurements\","
      << "\"execution_backend\":\"synchronous host execution with separate virtual-device buffers\","
      << "\"source_graph\":" << graph_json(source) << ",\"compiled_graph\":" << graph_json(compiled)
      << ",\"frontend_passes\":" << passes_json(frontend_passes) << ",\"pass_remarks\":" << strings_json(remarks) << ",\"config\":{\"relu_supported\":" << (config.relu_supported ? "true" : "false")
      << ",\"device_bytes\":" << config.device_bytes << ",\"transfer_fixed_ms\":" << config.transfer_fixed_ms
      << ",\"bandwidth_bytes_per_ms\":" << config.bandwidth_bytes_per_ms << ",\"launch_ms\":" << config.launch_ms
      << ",\"synchronization_ms\":" << config.synchronization_ms << ",\"compute_ms\":{\"cpu\":[6,1,2,7],\"device\":[1,0.1,0.4,1.1],\"op_order\":[\"MatMul\",\"ReLU\",\"Add\",\"MatMulReLU\"]}}"
      << ",\"search\":{\"infeasible\":" << search.infeasible << ",\"ranked_feasible\":[";
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
          << ",\"h2d_bytes\":" << p.h2d_bytes << ",\"d2h_bytes\":" << p.d2h_bytes
          << ",\"peak_allocated_device_bytes\":" << p.peak_device_bytes << ",\"islands\":" << p.islands
          << ",\"synchronizations\":" << p.synchronizations << ",\"correct\":" << (r.correct ? "true" : "false")
          << ",\"max_abs_error\":" << r.max_abs_error << ",\"output\":[";
        for (std::size_t j = 0; j < r.output.values.size(); ++j) { if (j) s << ','; s << r.output.values[j]; }
        s << "],\"commands\":[";
        for (std::size_t j = 0; j < p.commands.size(); ++j) {
            if (j) s << ',';
            const auto& c = p.commands[j];
            s << "{\"id\":" << c.id << ",\"version\":" << c.version << ",\"kind\":" << quote(name(c.kind))
              << ",\"backend\":" << quote(name(c.backend)) << ",\"node\":" << quote(c.node)
              << ",\"op\":" << (c.kind == Kind::Compute ? quote(name(c.op)) : "null")
              << ",\"inputs\":" << strings_json(c.inputs) << ",\"output\":" << quote(c.output)
              << ",\"shape\":[" << c.shape.rows << ',' << c.shape.cols << "],\"origins\":" << strings_json(c.origins)
              << ",\"modeled_ms\":" << c.modeled_ms << '}';
        }
        s << "]}";
    }
    return s.str() + "]}\n";
}
std::string trace_json(const Run& run) {
    std::ostringstream s;
    s << std::setprecision(12) << "{\"displayTimeUnit\":\"ms\",\"trace_provenance\":\"measured synchronous host command durations; virtual device is software\",\"traceEvents\":[";
    for (std::size_t i = 0; i < run.events.size(); ++i) {
        if (i) s << ',';
        const auto& e = run.events[i];
        s << "{\"name\":" << quote(e.name + " " + e.node) << ",\"cat\":\"host execution\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":"
          << e.start_us << ",\"dur\":" << e.duration_us << ",\"args\":{\"command_id\":" << e.command_id
          << ",\"logical_backend\":" << quote(e.lane) << ",\"modeled_ms\":" << e.modeled_ms << "}}";
    }
    return s.str() + "]}\n";
}
}
