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
        parse_bad("tensor A f32 1 1 1 2\n", "E_PARSE");
        parse_bad("tensor A f32 1 2 1\n", "E_VALUE");
        parse_bad("tensor A f32 1 1 1\nrelu n X Y\noutput Y\n", "E_REFERENCE");
        parse_bad("relu a Y X\nrelu b X Y\noutput Y\n", "E_CYCLE");
        parse_bad("tensor A f32 1 1 1\nrelu a A A\noutput A\n", "E_SSA");
        parse_bad("tensor A f32 1 2 1 2\ntensor B f32 3 1 1 2 3\nmatmul m A B Y\noutput Y\n", "E_SHAPE");
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
        std::cout << "PASS: " << checks << " core assertions\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
