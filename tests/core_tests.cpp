#include "placementlens/core.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
int checks = 0;
void check(bool value, const std::string& description) {
    ++checks;
    if (!value) throw std::runtime_error("FAILED: " + description);
}
void rejects(const std::function<void()>& f, const std::string& code) {
    bool rejected = false;
    try { f(); } catch (const std::exception& e) { rejected = std::string(e.what()).find(code) != std::string::npos; }
    check(rejected, "expected rejection " + code);
}
pl::Graph graph(const std::string& suffix = "add bias R D Y\noutput Y\n") {
    std::istringstream in("tensor A f32 2 3 1 -2 3 0 4 -1\n"
                         "tensor B f32 3 2 2 1 1 -3 -2 2\n"
                         "tensor D f32 2 2 1 1 1 1\n"
                         "matmul mm A B C\nrelu activation C R\n" + suffix);
    return pl::parse(in);
}
void parse_bad(const std::string& input, const std::string& code) {
    rejects([&] { std::istringstream s(input); (void)pl::parse(s); }, code);
}
pl::Graph parse_graph(const std::string& input) {
    std::istringstream stream(input);
    return pl::parse(stream);
}
void renumber(pl::Plan& plan) {
    for (std::size_t i = 0; i < plan.commands.size(); ++i) plan.commands[i].id = i;
}
}
int main() {
    try {
        const auto g = graph(); pl::Config c;
        check(pl::reference(g).values == std::vector<float>({1,14,7,1}), "hand-computed reference");
        const auto cpu = pl::choose(g,c,"cpu"), maximal = pl::choose(g,c,"maximal"), cost = pl::choose(g,c,"cost");
        check(pl::placement_name(cost.placement) == "ACC", "boundary-aware assignment");
        check(pl::placement_name(maximal.placement) == "ACA", "maximal fragments");
        check(maximal.modeled_ms > cpu.modeled_ms && cost.modeled_ms < cpu.modeled_ms, "modeled decision reversal");
        check(maximal.h2d_bytes == 80 && maximal.d2h_bytes == 32 && maximal.islands == 2, "fragmented transfer accounting");
        check(cost.h2d_bytes == 48 && cost.d2h_bytes == 16 && cost.islands == 1, "boundary-aware transfer accounting");
        for (const auto& p : pl::enumerate(g,c).plans) check(pl::execute(g,p,c).correct, "all default legal plans execute correctly");
        check(pl::fuse_matmul_relu(g,c).graph.nodes.size() == 3, "fusion cannot bypass missing ReLU capability");
        c.relu_supported = true;
        const auto fused = pl::fuse_matmul_relu(g,c).graph;
        check(fused.nodes.size() == 2 && fused.nodes[0].origins.size() == 2, "fusion retains source lineage");
        check(pl::execute(g,pl::choose(fused,c,"cost"),c).correct, "fusion matches original unfused reference");
        check(pl::fuse_matmul_relu(graph("add branch C R Y\noutput Y\n"),c).graph.nodes.size() == 3, "shared intermediate blocks fusion");
        check(pl::fuse_matmul_relu(graph("add bias R D Y\noutput C\n"),c).graph.nodes.size() == 3, "observable intermediate blocks fusion");
        for (const auto& p : pl::enumerate(g,c).plans) check(pl::execute(g,p,c).correct, "all full-capability placements");
        c.transfer_fixed_ms = 0;
        check(pl::placement_name(pl::choose(g,c,"cost").placement) == "AAA", "zero transfer cost selects accelerator");
        c.transfer_fixed_ms = 100;
        check(pl::placement_name(pl::choose(g,c,"cost").placement) == "CCC", "large transfer cost selects CPU");
        c = pl::Config{}; c.device_bytes = 1;
        check(pl::placement_name(pl::choose(g,c,"cost").placement) == "CCC", "tiny memory budget selects CPU");
        c = pl::Config{}; c.device_bytes = 80;
        rejects([&] { (void)pl::choose(g,c,"maximal"); }, "E_MEMORY");
        check(pl::execute(g,pl::choose(g,c,"cost"),c).correct, "infeasible maximal still permits safe cost plan");
        c = pl::Config{};
        auto bad = cost; bad.commands.front().version = 999;
        rejects([&] { (void)pl::execute(g,bad,c); }, "E_VERSION");
        bad = cost; bad.commands.front().id = 999;
        rejects([&] { (void)pl::execute(g,bad,c); }, "E_SEQUENCE");
        bad = cost;
        bad.commands.erase(std::remove_if(bad.commands.begin(),bad.commands.end(),[](const pl::Command& x){return x.kind==pl::Kind::Synchronize;}),bad.commands.end());
        renumber(bad);
        rejects([&] { (void)pl::execute(g,bad,c); }, "E_SYNC");
        bad = cost;
        auto compute = std::find_if(bad.commands.begin(),bad.commands.end(),[](const pl::Command& x){return x.kind==pl::Kind::Compute;});
        compute->inputs[0] = "D:missing";
        rejects([&] { (void)pl::execute(g,bad,c); }, "E_BUFFER");
        c.relu_supported=true;
        bad = pl::choose(fused,c,"maximal"); c.relu_supported=false;
        rejects([&] { (void)pl::execute(g,bad,c); }, "E_CAPABILITY");
        bad = cost;
        compute = std::find_if(bad.commands.begin(),bad.commands.end(),[](const pl::Command& x){return x.kind==pl::Kind::Compute;});
        compute->shape.cols = 9;
        rejects([&] { (void)pl::execute(g,bad,c); }, "E_SHAPE");
        bad = cost;
        bad.commands.insert(bad.commands.begin()+1,bad.commands.front()); renumber(bad);
        rejects([&] { (void)pl::execute(g,bad,c); }, "E_ALLOC");
        bad = cpu;
        auto relu = std::find_if(bad.commands.begin(),bad.commands.end(),[](const pl::Command& x){return x.kind==pl::Kind::Compute && x.op==pl::Op::Relu;});
        relu->op = pl::Op::Add; relu->inputs.push_back(relu->inputs[0]);
        check(!pl::execute(g,bad,c).correct, "reference detects a numerically wrong but well-formed command plan");
        c.transfer_fixed_ms = std::numeric_limits<double>::quiet_NaN();
        rejects([&] { (void)pl::choose(g,c,"cost"); }, "E_CONFIG");
        c = pl::Config{}; c.bandwidth_bytes_per_ms = 0;
        rejects([&] { pl::validate_config(c); }, "E_CONFIG");
        parse_bad("tensor A f16 1 1 1\n", "E_DTYPE");
        parse_bad("tensor A f32 0 1\n", "E_SHAPE");
        parse_bad("tensor A f32 999999999 999999999\n", "E_SHAPE");
        parse_bad("tensor A f32 -18446744073709551615 1 1\nrelu r A Y\noutput Y\n", "E_SHAPE");
        parse_bad("tensor A f32 1 -18446744073709551615 1\nrelu r A Y\noutput Y\n", "E_SHAPE");
        parse_bad("tensor A f32 -1 1\n", "E_SHAPE");
        parse_bad("tensor A f32 18446744073709551616 1\n", "E_SHAPE");
        parse_bad("tensor A f32 1.0 1 1\n", "E_SHAPE");
        parse_bad("tensor A f32 1 1 1 2\n", "E_PARSE");
        parse_bad("tensor A f32 1 2 1\n", "E_VALUE");
        parse_bad("tensor A f32 1 1 1\nrelu n X Y\noutput Y\n", "E_REFERENCE");
        parse_bad("relu a Y X\nrelu b X Y\noutput Y\n", "E_CYCLE");
        parse_bad("tensor A f32 1 1 1\nrelu a A A\noutput A\n", "E_SSA");
        parse_bad("tensor A f32 1 2 1 2\ntensor B f32 3 1 1 2 3\nmatmul m A B Y\noutput Y\n", "E_SHAPE");
        // Scheduling scans source order, then retries earlier blocked nodes.
        const auto reordered = parse_graph("tensor X f32 1 1 -2\nrelu consumer A Y\nadd producer X X A\nrelu independent X Z\noutput Y\n");
        check(reordered.nodes[0].id == "producer" && reordered.nodes[1].id == "independent" &&
              reordered.nodes[2].id == "consumer", "deterministic source-order topological scans");
        c = pl::Config{};
        for (const auto& p : pl::enumerate(reordered,c).plans)
            check(pl::execute(reordered,p,c).output.values == std::vector<float>({0}), "forward references execute in dependency order");
        // A public Graph can be mutated after validation. Reject malformed
        // storage before ReLU's transform or MatMul's indexing reaches it.
        const auto unary = parse_graph("tensor X f32 1 1 2\nrelu r X Y\noutput Y\n");
        const auto unary_plan = pl::choose(unary,c,"cpu");
        auto malformed = unary;
        malformed.inputs.at("X").values.push_back(3);
        rejects([&] { (void)pl::execute(malformed,unary_plan,c); }, "E_VALUE");
        malformed.inputs.at("X").values.clear();
        rejects([&] { (void)pl::execute(malformed,unary_plan,c); }, "E_VALUE");
        malformed = unary; malformed.inputs.at("X").shape.rows = 0;
        rejects([&] { (void)pl::execute(malformed,unary_plan,c); }, "E_SHAPE");
        malformed = unary; malformed.inputs.at("X").values[0] = std::numeric_limits<float>::infinity();
        rejects([&] { (void)pl::execute(malformed,unary_plan,c); }, "E_VALUE");
        // ReLU must not hide a nonfinite MatMul result. These finite inputs
        // produce negative infinity and NaN respectively in f32 accumulation.
        c.relu_supported = true;
        for (const std::string operands : {
                "tensor A f32 1 1 3e38\ntensor B f32 1 1 -2\n",
                "tensor A f32 1 2 3e38 3e38\ntensor B f32 2 1 -2 2\n"}) {
            const auto overflow = parse_graph(operands + "matmul m A B M\nrelu r M Y\noutput Y\n");
            const auto overflow_fused = pl::fuse_matmul_relu(overflow,c).graph;
            for (const auto& p : pl::enumerate(overflow,c).plans)
                rejects([&] { (void)pl::execute(overflow,p,c); }, "E_NUMERIC");
            for (const auto& p : pl::enumerate(overflow_fused,c).plans)
                rejects([&] { (void)pl::execute(overflow,p,c); }, "E_NUMERIC");
        }
        const auto negative_overflow = parse_graph("tensor A f32 1 1 3e38\ntensor B f32 1 1 -2\nmatmul m A B M\nrelu r M Y\noutput Y\n");
        rejects([&] { (void)pl::reference(negative_overflow); }, "E_NUMERIC");
        const auto negative_fused = pl::fuse_matmul_relu(negative_overflow,c).graph;
        rejects([&] { (void)pl::reference(negative_fused); }, "E_NUMERIC");
        // The f32 runtime sum is finite here, while double accumulation rounds
        // to infinity when stored as f32. Reject the unusable reference before
        // it could become an infinite error and invalid JSON numeric literal.
        std::string overflow_text = "tensor A f32 1 12";
        for (int i = 0; i < 12; ++i) overflow_text += " 2.83568648e37";
        overflow_text += "\ntensor B f32 12 1 1 1 1 1 1 1 1 1 1 1 1 1\nmatmul m A B Y\noutput Y\n";
        const auto oracle_overflow = parse_graph(overflow_text);
        rejects([&] { (void)pl::reference(oracle_overflow); }, "E_NUMERIC");
        rejects([&] { (void)pl::execute(oracle_overflow,pl::choose(oracle_overflow,c,"cpu"),c); }, "E_NUMERIC");
        // Different graph topology: shared input reused on the same backend is copied only once.
        std::istringstream shared("tensor X f32 1 2 -1 2\nadd a X X A\nadd b A X Y\noutput Y\n");
        const auto sg = pl::parse(shared); c = pl::Config{};
        const auto sp = pl::choose(sg,c,"maximal");
        check(sp.h2d_bytes == 8 && sp.d2h_bytes == 8, "resident input reused without duplicate copies");
        check(pl::execute(sg,sp,c).output.values == std::vector<float>({-3,6}), "shared-input graph numerical output");
        // A deterministic randomized numerical suite exercises every placement before/after fusion.
        std::mt19937 rng(20260917); std::uniform_real_distribution<float> dist(-2,2);
        for (int trial=0; trial<24; ++trial) {
            auto random_graph = g;
            const auto m = static_cast<std::size_t>(1 + trial % 4);
            const auto k = static_cast<std::size_t>(1 + (trial / 4) % 3);
            const auto n = static_cast<std::size_t>(1 + (trial / 3) % 5);
            random_graph.inputs["A"].shape = {m,k};
            random_graph.inputs["B"].shape = {k,n};
            random_graph.inputs["D"].shape = {m,n};
            for (auto& entry : random_graph.inputs) entry.second.values.resize(entry.second.shape.elements());
            for (auto& entry : random_graph.inputs) for (auto& x : entry.second.values) x=dist(rng);
            random_graph = pl::validate(random_graph);
            c.relu_supported=true;
            for (const auto& p : pl::enumerate(random_graph,c).plans)
                check(pl::execute(random_graph,p,c).correct,"random original graph/all placements");
            auto fg=pl::fuse_matmul_relu(random_graph,c).graph;
            for (const auto& p : pl::enumerate(fg,c).plans)
                check(pl::execute(random_graph,p,c).correct,"random fused graph/all placements");
        }
        {   // Declared target support rules: capability reasons, execute-time recheck, config validation.
            const auto rg = graph();
            pl::Config open; open.relu_supported = true;
            const auto all_device = pl::lower(rg, std::vector<pl::Backend>(rg.nodes.size(), pl::Backend::Device), open);
            check(pl::execute(rg, all_device, open).correct, "rule-free all-device plan executes");
            auto excluded = open; excluded.device_unsupported_ops = {"MatMul"};
            check(pl::capability(rg.nodes[0], rg, excluded).reason == "operator excluded by the declared target support rules",
                  "operator exclusion reason");
            check(pl::capability(rg.nodes[1], rg, excluded).accepted, "exclusion is per operator");
            auto small = open; small.device_max_tensor_bytes = 8;  // A is 2x3 = 24 B
            check(pl::capability(rg.nodes[0], rg, small).reason == "tensor exceeds the declared per-tensor device limit",
                  "per-tensor limit reason");
            for (const auto& cfg : {excluded, small})
                rejects([&] { (void)pl::execute(rg, all_device, cfg); }, "E_CAPABILITY: device rejects command");
            const auto conv = parse_graph("tensor X f32 1 4 1 2 3 4\ntensor W f32 2 1 1 1\n"
                                          "conv2d dw X W Y 1 2 1 2 2 1 1 1 1 0 0 0 0 2 1 1\noutput Y\n");
            pl::Config cnn; cnn.cnn_supported = true;
            check(pl::capability(conv.nodes[0], conv, cnn).accepted, "grouped conv accepted by default");
            const auto conv_plan = pl::lower(conv, {pl::Backend::Device}, cnn);
            auto ungrouped = cnn; ungrouped.device_grouped_conv = false;
            check(pl::capability(conv.nodes[0], conv, ungrouped).reason == "grouped convolution excluded by the declared target support rules",
                  "grouped-conv rule reason");
            rejects([&] { (void)pl::execute(conv, conv_plan, ungrouped); }, "E_CAPABILITY: device rejects command");
            auto unknown = open; unknown.device_unsupported_ops = {"Sigmoid"};
            rejects([&] { pl::validate_config(unknown); }, "E_CONFIG: unknown excluded device operator");
            auto many = open; many.device_unsupported_ops.assign(17, "Add");
            rejects([&] { pl::validate_config(many); }, "E_CONFIG: at most 16");
        }
        std::cout << "PASS: " << checks << " core assertions\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
