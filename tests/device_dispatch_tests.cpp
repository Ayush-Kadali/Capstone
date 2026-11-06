#include "placementlens/core.hpp"
#include "placementlens/device.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
int checks = 0;
void check(bool condition, const std::string& label) {
    ++checks;
    if (!condition) throw std::runtime_error(label);
}
void rejects(const std::function<void()>& action, const std::string& code) {
    try { action(); } catch (const std::exception& error) {
        check(std::string(error.what()).find(code) != std::string::npos, error.what()); return;
    }
    throw std::runtime_error("Expected " + code);
}
class SpyDevice final : public pl::DeviceBackend {
public:
    std::map<std::string, pl::Tensor> buffers;
    std::vector<std::string> calls;
    bool fail_compute = false, invalid_size = false, nonfinite = false;
    std::atomic<bool>* cancel_after_write = nullptr;
    std::string description() const override { return "test-only buffer backend"; }
    double last_elapsed_us() const override { return -1; }
    void allocate(const std::string& h, pl::Shape s) override {
        calls.push_back("ALLOC"); buffers.emplace(h, pl::Tensor{s, std::vector<float>(s.elements())});
    }
    void release(const std::string& h) override { calls.push_back("FREE"); buffers.erase(h); }
    void write(const std::string& h, const pl::Tensor& t) override {
        calls.push_back("COPY"); buffers.at(h) = t;
        if (cancel_after_write) cancel_after_write->store(true);
    }
    void read(const std::string& h, pl::Tensor& t) override {
        calls.push_back("COPY"); t = buffers.at(h);
        if (invalid_size) t.values.clear();
        if (nonfinite) t.values[0] = std::numeric_limits<float>::infinity();
    }
    void synchronize() override { calls.push_back("SYNC"); }
    void compute(const pl::Command& command) override {
        calls.push_back("COMPUTE");
        if (fail_compute) throw std::runtime_error("E_TEST_DEVICE: injected worker failure");
        check(command.op == pl::Op::Add, "fixture supports Add only");
        auto& output = buffers.at(command.output);
        const auto& a = buffers.at(command.inputs[0]);
        const auto& b = buffers.at(command.inputs[1]);
        for (std::size_t i=0; i<output.values.size(); ++i) output.values[i] = a.values[i]+b.values[i];
    }
};
}
int main() {
    try {
        std::istringstream text("tensor X f32 1 2 -1 2\nadd a X X A\nadd b A X Y\noutput Y\n");
        const auto graph = pl::parse(text); const pl::Config config;
        for (const auto& plan : pl::enumerate(graph,config).plans) {
            SpyDevice backend;
            const auto run = pl::execute(graph,plan,config,&backend);
            check(run.correct && run.output.values == std::vector<float>({-3,6}), "external backend output");
            std::vector<std::string> expected;
            for (const auto& command : plan.commands)
                if (command.backend == pl::Backend::Device || command.kind == pl::Kind::Copy)
                    expected.push_back(pl::name(command.kind));
            const auto allocations = std::count_if(plan.commands.begin(),plan.commands.end(),[](const auto& c){return c.kind==pl::Kind::Allocate && c.backend==pl::Backend::Device;});
            if (allocations) {
                expected.push_back("SYNC");
                for (std::size_t i=0; i<static_cast<std::size_t>(allocations); ++i) expected.push_back("FREE");
            }
            check(backend.calls == expected, "backend sees exact device commands and scoped teardown");
            check(backend.buffers.empty(), "successful run releases all owned device buffers");
            check(pl::execute(graph,plan,config,&backend).correct,"backend reusable across executions");
            check(run.events.size() == plan.commands.size(), "one event per command");
            check(std::all_of(run.events.begin(),run.events.end(),[](const auto& e){return e.device_us < 0;}), "unavailable hardware duration remains unavailable");
        }
        const auto plan = pl::choose(graph,config,"maximal");
        SpyDevice failure; failure.fail_compute=true;
        rejects([&]{(void)pl::execute(graph,plan,config,&failure);},"E_TEST_DEVICE");
        check(failure.buffers.empty(),"failed run releases device buffers");
        SpyDevice invalid; invalid.invalid_size=true;
        rejects([&]{(void)pl::execute(graph,plan,config,&invalid);},"E_VALUE");
        SpyDevice nonfinite; nonfinite.nonfinite=true;
        rejects([&]{(void)pl::execute(graph,plan,config,&nonfinite);},"E_NUMERIC");
        std::atomic<bool> cancel{true}; SpyDevice pre_cancel;
        rejects([&]{(void)pl::execute(graph,plan,config,&pre_cancel,&cancel);},"E_CANCELLED");
        check(pre_cancel.calls.empty(),"pre-cancel starts no device operations");
        cancel.store(false); SpyDevice mid_cancel; mid_cancel.cancel_after_write=&cancel;
        rejects([&]{(void)pl::execute(graph,plan,config,&mid_cancel,&cancel);},"E_CANCELLED");
        check(mid_cancel.calls == std::vector<std::string>({"ALLOC","COPY","SYNC","FREE"}),"cancellation stops commands and tears down owned resources");
        check(mid_cancel.buffers.empty(),"cancelled run releases device buffers");
        std::cout << "PASS: " << checks << " backend-dispatch assertions\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
