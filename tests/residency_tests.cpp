#include "placementlens/core.hpp"
#include "placementlens/async.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main() {
    std::size_t checks = 0;
    auto check = [&](bool condition) { ++checks; if (!condition) throw std::runtime_error("residency assertion failed"); };
    auto fails = [&](auto action, const std::string& code) {
        bool rejected = false;
        try { action(); } catch (const std::runtime_error& e) { rejected = std::string(e.what()).find(code) == 0; }
        check(rejected);
    };
    std::istringstream input("tensor X f32 2 2 1 -2 3 4\ntensor W f32 2 2 1 0 0 1\nmatmul m X W M\nrowsum r M Y\noutput Y\n");
    const auto graph = pl::parse(input);
    for (bool reuse : {false, true}) {
        pl::Config host; host.reuse_buffers = reuse; host.tile_size = 2;
        for (const auto& baseline : pl::enumerate(graph, host).plans) {
            for (const auto& names : {std::vector<std::string>{"X"}, std::vector<std::string>{"W", "X"}}) {
                auto config = host; config.mirrored_inputs = names;
                const auto plan = pl::lower(graph, baseline.placement, config);
                const auto result = pl::execute(graph, plan, config);
                check(result.correct && result.output.values == std::vector<float>({-1, 7}));
                check(result.preparation_h2d_bytes == names.size() * 16);
                check(result.input_preparation_ms >= 0);
                check(plan.h2d_bytes <= baseline.h2d_bytes);
                check(result.events.size() == plan.commands.size());
                check(result.peak_device_reserved_bytes == plan.peak_device_bytes);
                check(plan.initial_buffers.size() == names.size());
                if (baseline.placement[0] == pl::Backend::Device)
                    check(baseline.h2d_bytes - plan.h2d_bytes == result.preparation_h2d_bytes);
                auto bad = plan; bad.initial_buffers[0].shape = {1, 1};
                fails([&] { pl::execute(graph, bad, config); }, "E_RESIDENCY:");
                bad = plan; bad.initial_buffers[0].id = 99;
                fails([&] { pl::execute(graph, bad, config); }, "E_RESIDENCY:");
                bad = plan; bad.initial_buffers.pop_back();
                fails([&] { pl::execute(graph, bad, config); }, "E_RESIDENCY:");
            }
        }
    }
    pl::Config config; config.mirrored_inputs = {"X", "X"};
    fails([&] { pl::validate_config(config); }, "E_RESIDENCY:");
    config.mirrored_inputs = {"M"};
    fails([&] { pl::lower(graph, {pl::Backend::Cpu, pl::Backend::Cpu}, config); }, "E_RESIDENCY:");
    config.mirrored_inputs = {"X"}; config.device_bytes = 15;
    check(pl::enumerate(graph, config).plans.empty());
    config.device_bytes = 16;
    const auto cpu = pl::choose(graph, config, "cpu");
    check(pl::execute(graph, cpu, config).correct);
    pl::AsyncRuntime runtime(2);
    auto fence = runtime.submit(graph, cpu, config);
    check(fence.get().run.correct);
    config.mirrored_inputs.clear(); config.device_bytes = 1024;
    config.emulate_boundaries = true;
    config.transfer_fixed_ms = 0.1; config.synchronization_ms = 0.1; config.launch_ms = 0.1;
    const auto delayed_plan = pl::choose(graph, config, "maximal");
    const auto delayed = pl::execute(graph, delayed_plan, config);
    const auto expected_delay = 3 * config.transfer_fixed_ms +
        static_cast<double>(delayed_plan.h2d_bytes + delayed_plan.d2h_bytes) / config.bandwidth_bytes_per_ms +
        static_cast<double>(delayed_plan.synchronizations) * config.synchronization_ms + config.launch_ms;
    check(std::abs(delayed.injected_boundary_ms - expected_delay) < 1e-9);
    check(delayed.host_ms >= delayed.injected_boundary_ms && delayed.correct);
    check(delayed.execution_backend.find("injected") != std::string::npos);
    config.transfer_fixed_ms = 101;
    fails([&] { pl::execute(graph, pl::choose(graph, config, "maximal"), config); }, "E_DELAY:");
    std::cout << "PASS: " << checks << " residency assertions\n";
}
