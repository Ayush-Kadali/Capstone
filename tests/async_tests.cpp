#include "placementlens/async.hpp"
#include <condition_variable>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
using namespace std::chrono_literals;
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) throw std::runtime_error(std::string("FAILED: ") + description);
}
template <typename F> void rejects(F&& action, const char* code) {
    bool rejected = false;
    try { action(); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find(code) != std::string::npos; }
    check(rejected, code);
}
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false;
    void enter() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        changed.notify_all();
        if (!changed.wait_for(lock, 10s, [&] { return released; }))
            throw std::runtime_error("test gate watchdog expired");
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!changed.wait_for(lock, 10s, [&] { return entered; }))
            throw std::runtime_error("test worker watchdog expired");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
};
struct ReleaseOnExit { Gate& gate; ~ReleaseOnExit() { gate.release(); } };
pl::Run success() { pl::Run run; run.correct = true; run.host_ms = 7; return run; }
pl::Graph graph() {
    std::istringstream input("tensor A f32 1 2 -3 4\nrelu r A Y\noutput Y\n");
    return pl::parse(input);
}
void capacity_timeout_cancellation() {
    Gate gate;
    int invocations = 0;
    pl::AsyncRuntime runtime(2, [&](const pl::Graph&, const pl::Plan&, const pl::Config&, const std::atomic<bool>&) {
        if (++invocations == 1) gate.enter();
        return success();
    });
    ReleaseOnExit release{gate};
    auto first = runtime.submit({}, {}, {});
    gate.wait_entered();
    auto second = runtime.submit({}, {}, {});
    check(first.state() == pl::JobState::Running, "first job running behind gate");
    check(second.state() == pl::JobState::Queued, "second job queued behind gate");
    check(!runtime.try_submit({}, {}, {}), "capacity includes running and queued jobs");
    rejects([&] { (void)runtime.submit({}, {}, {}); }, "E_QUEUE_FULL");
    check(first.wait_for(0ms) == std::future_status::timeout, "timeout observes unfinished job");
    check(first.state() == pl::JobState::Running, "timeout does not cancel job");
    check(second.cancel(), "queued cancellation accepted");
    check(!second.cancel(), "queued cancellation idempotent");
    check(second.state() == pl::JobState::Cancelled, "queued cancellation is immediately terminal");
    check(second.wait_for(0ms) == std::future_status::ready, "queued cancellation resolves fence");
    rejects([&] { (void)second.get(); }, "E_CANCELLED");
    auto replacement = runtime.submit({}, {}, {});
    gate.release();
    const auto result = first.get();
    check(result.run.correct && result.run.host_ms == 7, "runner result and timing retained");
    check(result.queue_wait_ms >= 0 && result.total_ms >= result.queue_wait_ms, "separate monotonic queue timings");
    check(!first.cancel(), "completed result cannot be cancelled");
    check(first.get().run.correct, "shared fence get can repeat");
    check(replacement.get().run.correct, "queued cancellation frees capacity");
    runtime.shutdown();
    check(invocations == 2, "cancelled queued runner never invoked");
    runtime.shutdown();
    rejects([&] { (void)runtime.submit({}, {}, {}); }, "E_CLOSED");
    rejects([&] { (void)runtime.try_submit({}, {}, {}); }, "E_CLOSED");
}
void running_cancellation_and_errors() {
    Gate gate;
    pl::AsyncRuntime runtime(2, [&](const pl::Graph&, const pl::Plan& plan, const pl::Config&, const std::atomic<bool>& token) {
        if (plan.policy == "blocked") {
            gate.enter();
            if (!token.load()) throw std::runtime_error("test cancellation token missing");
            return success(); // Accepted cancellation must beat even a successful return.
        }
        if (plan.policy == "fail") throw std::runtime_error("injected runner failure");
        if (plan.policy == "nonstandard") throw 42;
        return success();
    });
    ReleaseOnExit release{gate};
    pl::Plan blocked; blocked.policy = "blocked";
    auto cancelled = runtime.submit({}, blocked, {});
    gate.wait_entered();
    check(cancelled.cancel(), "running cancellation accepted");
    check(!cancelled.cancel(), "running cancellation requested once");
    check(cancelled.state() == pl::JobState::Running, "running cancellation remains pending until worker returns");
    check(cancelled.wait_for(0ms) == std::future_status::timeout, "cancel does not abandon running storage");
    gate.release();
    rejects([&] { (void)cancelled.get(); }, "E_CANCELLED");
    check(cancelled.state() == pl::JobState::Cancelled, "running cancellation terminal state");
    pl::Plan bad; bad.policy = "fail";
    auto failed = runtime.submit({}, bad, {});
    rejects([&] { (void)failed.get(); }, "injected runner failure");
    check(failed.state() == pl::JobState::Failed, "runner error resolves failed fence");
    check(!failed.cancel(), "failure cannot be cancelled retroactively");
    bad.policy = "nonstandard";
    auto unusual = runtime.submit({}, bad, {});
    bool exact_exception = false;
    try { (void)unusual.get(); } catch (int value) { exact_exception = value == 42; }
    check(exact_exception, "nonstandard runner exception preserved");
    check(runtime.submit({}, {}, {}).get().run.correct, "worker survives runner failures");
}
void ownership_and_drain() {
    Gate gate;
    std::vector<int> order;
    pl::AsyncRuntime runtime(4, [&](const pl::Graph& input, const pl::Plan& plan, const pl::Config& config, const std::atomic<bool>&) {
        gate.enter();
        order.push_back(static_cast<int>(input.inputs.at("A").values[0]));
        if (plan.policy != "owned" || config.device_bytes != 123)
            throw std::runtime_error("job input ownership lost");
        return success();
    });
    ReleaseOnExit release{gate};
    auto input = graph();
    pl::Plan plan; plan.policy = "owned";
    pl::Config config; config.device_bytes = 123;
    auto first = runtime.submit(input, plan, config);
    gate.wait_entered();
    input.inputs.at("A").values[0] = 2;
    auto second = runtime.submit(input, plan, config);
    input.inputs.at("A").values[0] = 3;
    auto third = runtime.submit(input, plan, config);
    input.inputs.clear(); plan.policy.clear(); config.device_bytes = 0;
    gate.release();
    runtime.shutdown(pl::ShutdownMode::Drain);
    check(order == std::vector<int>({-3, 2, 3}), "FIFO and value ownership through drain");
    for (auto fence : {first, second, third}) {
        check(fence.get().run.correct, "drain publishes all accepted jobs");
        check(fence.state() == pl::JobState::Succeeded, "drained job succeeded");
    }
}
void shutdown_and_destructor() {
    std::optional<pl::Fence> survivor, queued;
    Gate started;
    {
        pl::AsyncRuntime runtime(2, [&](const pl::Graph&, const pl::Plan&, const pl::Config&, const std::atomic<bool>& token) {
            { std::lock_guard<std::mutex> lock(started.mutex); started.entered = true; started.changed.notify_all(); }
            const auto watchdog = std::chrono::steady_clock::now() + 10s;
            while (!token.load()) {
                if (std::chrono::steady_clock::now() > watchdog) throw std::runtime_error("cancel watchdog expired");
                std::this_thread::yield();
            }
            return success();
        });
        survivor = runtime.submit({}, {}, {});
        started.wait_entered();
        queued = runtime.submit({}, {}, {});
        // Two shutdown callers may join safely; Cancel can escalate Drain.
        std::thread drain([&] { runtime.shutdown(pl::ShutdownMode::Drain); });
        runtime.shutdown(pl::ShutdownMode::Cancel);
        drain.join();
    }
    rejects([&] { (void)survivor->get(); }, "E_CANCELLED");
    rejects([&] { (void)queued->get(); }, "E_CANCELLED");
    check(survivor->state() == pl::JobState::Cancelled, "fence survives runtime destruction");
    Gate destructor_started;
    {
        pl::AsyncRuntime runtime(1, [&](const pl::Graph&, const pl::Plan&, const pl::Config&, const std::atomic<bool>& token) {
            { std::lock_guard<std::mutex> lock(destructor_started.mutex); destructor_started.entered = true; destructor_started.changed.notify_all(); }
            const auto watchdog = std::chrono::steady_clock::now() + 10s;
            while (!token.load()) {
                if (std::chrono::steady_clock::now() > watchdog) throw std::runtime_error("destructor watchdog expired");
                std::this_thread::yield();
            }
            return success();
        });
        survivor = runtime.submit({}, {}, {});
        destructor_started.wait_entered();
    }
    rejects([&] { (void)survivor->get(); }, "E_CANCELLED");
    pl::AsyncRuntime* address = nullptr;
    pl::AsyncRuntime self_join(1, [&](const pl::Graph&, const pl::Plan& plan, const pl::Config&, const std::atomic<bool>&) {
        if (plan.policy == "self") address->shutdown();
        return success();
    });
    address = &self_join;
    pl::Plan self; self.policy = "self";
    auto failed = self_join.submit({}, self, {});
    rejects([&] { (void)failed.get(); }, "E_SELF_JOIN");
    check(self_join.submit({}, {}, {}).get().run.correct, "self-join rejection leaves runtime usable");
}
void concurrent_production_jobs() {
    const auto input = graph();
    pl::Config config;
    const auto plan = pl::choose(input, config, "cpu");
    pl::AsyncRuntime runtime(64);
    std::vector<pl::Fence> fences;
    std::mutex mutex;
    std::exception_ptr failure;
    std::vector<std::thread> producers;
    for (int producer = 0; producer < 8; ++producer) {
        producers.emplace_back([&] {
            try {
                for (int job = 0; job < 8; ++job) {
                    auto fence = runtime.submit(input, plan, config);
                    std::lock_guard<std::mutex> lock(mutex);
                    fences.push_back(std::move(fence));
                }
            } catch (...) { std::lock_guard<std::mutex> lock(mutex); failure = std::current_exception(); }
        });
    }
    for (auto& producer : producers) producer.join();
    if (failure) std::rethrow_exception(failure);
    runtime.shutdown();
    check(fences.size() == 64, "all concurrent submissions accepted");
    for (const auto& fence : fences) {
        const auto result = fence.get();
        check(result.run.correct && result.run.output.values == std::vector<float>({0, 4}), "production runner concurrent submission correctness");
    }
    auto invalid = input; invalid.inputs.at("A").values.clear();
    pl::AsyncRuntime errors(1);
    auto bad = errors.submit(invalid, plan, config);
    rejects([&] { (void)bad.get(); }, "E_VALUE");
    check(errors.submit(input, plan, config).get().run.correct, "production error does not stop worker");
}
}
int main() {
    try {
        rejects([] { pl::AsyncRuntime invalid(0); }, "E_CAPACITY");
        capacity_timeout_cancellation();
        running_cancellation_and_errors();
        ownership_and_drain();
        shutdown_and_destructor();
        concurrent_production_jobs();
        std::cout << "PASS " << checks << " async assertions\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
