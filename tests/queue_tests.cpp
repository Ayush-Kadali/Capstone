#include "placementlens/queue.hpp"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <set>
#include <sstream>

namespace {
using namespace std::chrono_literals;
using pl::Backend; using pl::Kind; using pl::JobState;
std::atomic<int> checks{0}; // incremented from producer threads too
std::string source_dir = ".";
void check(bool value, const std::string& description) {
    ++checks;
    if (!value) throw std::runtime_error("FAILED: " + description);
}
template <typename F> void rejects(F&& action, const char* code) {
    bool rejected = false;
    try { action(); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find(code) != std::string::npos; }
    check(rejected, code);
}
// Blocks the hooked command until released or cancelled. Polls the cancel token
// because the queue cannot notify this condition variable.
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false;
    void enter(const std::atomic<bool>& cancel) {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        changed.notify_all();
        const auto watchdog = std::chrono::steady_clock::now() + 10s;
        while (!released && !cancel.load())
            if (changed.wait_until(lock, std::min(watchdog, std::chrono::steady_clock::now() + 1ms)) == std::cv_status::timeout &&
                std::chrono::steady_clock::now() >= watchdog) throw std::runtime_error("test gate watchdog expired");
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!changed.wait_for(lock, 10s, [&] { return entered; })) throw std::runtime_error("test watchdog expired");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
};
struct ReleaseOnExit { Gate& gate; ~ReleaseOnExit() { gate.release(); } };

pl::Command command(std::size_t id, Kind kind, Backend backend, std::vector<std::string> inputs, std::string output,
                    pl::Shape shape = {1, 2}, pl::Op op = pl::Op::Add, std::size_t slot = pl::no_buffer_slot) {
    pl::Command c;
    c.id = id; c.kind = kind; c.backend = backend; c.inputs = std::move(inputs); c.output = std::move(output);
    c.shape = shape; c.op = op; c.buffer_slot = slot; c.node = "n" + std::to_string(id);
    return c;
}
pl::Command alloc(std::size_t id, const std::string& handle, std::size_t slot = pl::no_buffer_slot) {
    return command(id, Kind::Allocate, handle[0] == 'D' ? Backend::Device : Backend::Cpu, {}, handle, {1, 2}, pl::Op::Add, slot);
}
pl::Command copy(std::size_t id, const std::string& from, const std::string& to) {
    return command(id, Kind::Copy, to[0] == 'D' ? Backend::Device : Backend::Cpu, {from}, to);
}
pl::Command add(std::size_t id, const std::string& a, const std::string& b, const std::string& out) {
    return command(id, Kind::Compute, out[0] == 'D' ? Backend::Device : Backend::Cpu, {a, b}, out);
}
pl::Tensor pair(float a, float b) { return {{1, 2}, {a, b}}; }
pl::CommandEvent submit(pl::CommandQueue& queue, const pl::Command& c) {
    auto event = queue.submit(c, 5s);
    check(event.has_value(), "submission admitted within 5 s");
    return *event;
}
// (a) every command that ran started after all of its dependencies finished.
void check_happens_before(const std::vector<pl::QueueRecord>& records) {
    for (const auto& r : records) {
        if (r.start_us < 0) continue;
        for (auto w : r.waits)
            check(records[w].state == JobState::Succeeded && records[w].end_us <= r.start_us,
                  "command " + std::to_string(r.command_id) + " starts after dependency " + std::to_string(records[w].command_id));
    }
}
bool overlaps(const pl::QueueRecord& a, const pl::QueueRecord& b) { return a.start_us < b.end_us && b.start_us < a.end_us; }

void equivalence_with_serial_execute() {
    std::vector<std::pair<std::string, pl::Graph>> graphs;
    for (const char* file : {"add", "branched", "fragmented"})
        graphs.emplace_back(file, pl::load(source_dir + "/examples/" + file + ".plg"));
    std::istringstream cnn("tensor X f32 1 4 -1 2 3 4\ntensor W f32 1 1 2\n"
        "conv2d conv X W C 1 1 2 2 1 1 1 1 1 0 0 0 0 1 1 1\n"
        "maxpool2d max C M 1 1 2 2 2 2 2 2 0 0 0 0 0\n"
        "averagepool2d avg M A 1 1 1 1 1 1 1 1 0 0 0 0 0\n"
        "globalaveragepool global A G 1 1 1 1\nreshape reshape G Y 1 1\noutput Y\n");
    graphs.emplace_back("cnn", pl::parse(cnn));
    std::size_t plans = 0;
    for (const auto& entry : graphs) {
        const auto& graph = entry.second;
        std::vector<pl::Config> configs(5);
        configs[1].relu_supported = true;
        configs[2].relu_supported = configs[2].reuse_buffers = true;
        configs[3] = configs[2];
        configs[3].mirrored_inputs = {graph.inputs.begin()->first};
        configs[4].relu_supported = true; configs[4].tile_size = 1; configs[4].local_bytes = 64;
        for (auto& config : configs) {
            config.cnn_supported = entry.first == "cnn";
            for (const auto& plan : pl::enumerate(graph, config).plans) {
                const auto serial = pl::execute(graph, plan, config);
                const auto queued = pl::execute_command_queue(graph, plan, config, 2);
                const auto label = entry.first + " " + pl::placement_name(plan.placement);
                check(queued.correct && queued.output.values == serial.output.values, label + " matches serial execute bitwise");
                check(queued.execution_backend.find("command-queue simulator") == 0, "queue backend label");
                check(queued.events.size() == plan.commands.size() && queued.engine_overlap_us >= 0, label + " one event per command");
                check(queued.peak_device_reserved_bytes == serial.peak_device_reserved_bytes &&
                      queued.software_buffer_reuses == serial.software_buffer_reuses &&
                      queued.tile_count == serial.tile_count && queued.preparation_h2d_bytes == serial.preparation_h2d_bytes,
                      label + " memory/tile accounting matches serial execute");
                for (const auto& e : queued.events) {
                    const auto& c = plan.commands[e.command_id];
                    const std::string lane = c.kind == Kind::Copy ? "copy-engine" :
                        (c.kind == Kind::Compute && c.backend == Backend::Device ? "compute-engine" : "host");
                    check(e.engine == lane, label + " lane assignment");
                    for (auto w : e.waits)
                        check(queued.events[w].start_us + queued.events[w].duration_us <= e.start_us, label + " happens-before in Run.events");
                    if (c.kind == Kind::Compute || c.kind == Kind::Copy)
                        for (const auto& input : c.inputs)
                            for (const auto& producer : plan.commands)
                                if ((producer.kind == Kind::Compute || producer.kind == Kind::Copy) && producer.output == input)
                                    check(std::count(e.waits.begin(), e.waits.end(), producer.id) == 1, label + " RAW dependency recorded");
                }
                ++plans;
            }
        }
    }
    check(plans >= 60, "equivalence covered many placements and configurations");
}

// (b) independent branches with synthetic copy/launch waits overlap in time.
void copy_compute_overlap() {
    pl::Config config;
    config.emulate_boundaries = true; config.transfer_fixed_ms = 30; config.launch_ms = 30;
    pl::CommandQueue queue(config, 4);
    queue.import("H:a", pair(1, 2)); queue.import("H:b", pair(3, 4));
    for (auto c : {alloc(0, "D:a"), alloc(1, "D:b"), alloc(2, "D:r")}) submit(queue, c);
    submit(queue, copy(3, "H:a", "D:a"));
    submit(queue, add(4, "D:a", "D:a", "D:r")); // waits only for copy 3
    submit(queue, copy(5, "H:b", "D:b"));       // independent of compute 4
    auto sync = submit(queue, command(6, Kind::Synchronize, Backend::Device, {}, ""));
    check(sync.state() == JobState::Succeeded, "host SYNC returns after device work");
    const auto r = queue.records();
    check_happens_before(r);
    check(r[4].lane == "compute-engine" && r[5].lane == "copy-engine", "independent commands on separate engines");
    check(r[4].start_us >= r[3].end_us, "compute starts after its copy");
    check(overlaps(r[4], r[5]), "copy and compute intervals overlap");
    check(r[3].injected_ms >= 30 && r[4].injected_ms == 30, "synthetic waits are labelled");
    check(std::count(r[6].waits.begin(), r[6].waits.end(), 4U) == 1 && std::count(r[6].waits.begin(), r[6].waits.end(), 5U) == 1 &&
          r[6].start_us >= std::max(r[4].end_us, r[5].end_us), "SYNC is a barrier over both engines");

    // The same effect on a lowered plan: COPY D overlaps the MatMul launch wait.
    const auto graph = pl::load(source_dir + "/examples/fragmented.plg");
    pl::Config plan_config;
    plan_config.relu_supported = plan_config.emulate_boundaries = true;
    plan_config.transfer_fixed_ms = 20; plan_config.launch_ms = 20;
    const auto plan = pl::choose(graph, plan_config, "maximal");
    const auto run = pl::execute_command_queue(graph, plan, plan_config);
    check(run.correct && run.engine_overlap_us > 0 && run.injected_boundary_ms > 0, "lowered plan shows engine overlap");
    check(run.execution_backend.find("injected boundary waits") != std::string::npos, "synthetic waits named in backend label");
}

void saturation_timeout_and_cancellation() {
    Gate gate;
    std::set<std::size_t> hooked;
    std::mutex hooked_mutex;
    pl::CommandQueue queue({}, 1, [&](const pl::Command& c, const std::atomic<bool>& cancel) {
        { std::lock_guard<std::mutex> lock(hooked_mutex); hooked.insert(c.id); }
        if (c.id == 10) gate.enter(cancel);
    });
    ReleaseOnExit release{gate};
    queue.import("H:x", pair(1, 2)); queue.import("H:q", pair(5, 6));
    for (auto c : {alloc(0, "D:x"), alloc(1, "D:y"), alloc(2, "D:z"), alloc(3, "D:q"), alloc(4, "D:w")}) submit(queue, c);
    submit(queue, copy(5, "H:x", "D:x")).wait();
    auto running = submit(queue, add(10, "D:x", "D:x", "D:y"));
    gate.wait_entered();
    check(running.state() == JobState::Running, "gated compute is running");
    check(!running.wait_for(0ms), "wait_for times out on unfinished command");
    check(running.state() == JobState::Running, "timeout does not cancel");
    check(!queue.try_submit(add(11, "D:x", "D:x", "D:z")), "try_submit reports a saturated compute engine");
    check(!queue.submit(add(11, "D:x", "D:x", "D:z"), 20ms), "blocking submit times out on saturation");
    check(queue.try_submit(copy(12, "H:q", "D:q")).has_value(), "copy engine has independent capacity");
    rejects([&] { (void)queue.try_submit(add(13, "D:missing", "D:x", "D:w")); }, "E_HANDLE"); // validated before capacity
    gate.release();
    running.wait();
    auto queued = submit(queue, add(11, "D:x", "D:x", "D:z"));
    queued.wait();
    check(queued.state() == JobState::Succeeded, "capacity recovers after completion");

    // Cancel a queued command and a running command; dependents fail, never run.
    Gate second;
    pl::CommandQueue q2({}, 4, [&](const pl::Command& c, const std::atomic<bool>& cancel) {
        { std::lock_guard<std::mutex> lock(hooked_mutex); hooked.insert(100 + c.id); }
        if (c.id == 10) second.enter(cancel);
    });
    ReleaseOnExit release_second{second};
    q2.import("H:x", pair(1, 2));
    for (auto c : {alloc(0, "D:x"), alloc(1, "D:y"), alloc(2, "D:z"), alloc(3, "H:z")}) submit(q2, c);
    submit(q2, copy(4, "H:x", "D:x"));
    auto gated = submit(q2, add(10, "D:x", "D:x", "D:y"));
    second.wait_entered();
    auto behind = submit(q2, add(11, "D:x", "D:x", "D:z"));
    auto dependent = submit(q2, copy(12, "D:z", "H:z"));
    check(behind.state() == JobState::Queued, "second compute queued behind gate");
    check(behind.cancel() && !behind.cancel(), "queued cancellation accepted once");
    check(behind.state() == JobState::Cancelled && behind.error().find("E_CANCELLED") == 0, "queued cancel is immediate");
    check(gated.cancel(), "running cancellation accepted");
    gated.wait();
    check(gated.state() == JobState::Cancelled, "running command observes cooperative cancel");
    dependent.wait();
    check(dependent.state() == JobState::Failed && dependent.error().find("E_DEPENDENCY: command 11 cancelled") == 0,
          "dependent of cancelled command fails with propagated cause");
    std::lock_guard<std::mutex> lock(hooked_mutex);
    check(!hooked.count(111) && !hooked.count(112), "cancelled and dependent commands never ran");
}

void invalid_commands_and_memory() {
    pl::Config config;
    config.reuse_buffers = true; config.device_bytes = 16;
    pl::CommandQueue queue(config, 2);
    queue.import("H:x", pair(1, 2));
    auto old = alloc(0, "D:x", 0); old.version = 2;
    rejects([&] { (void)queue.try_submit(old); }, "E_VERSION");
    rejects([&] { (void)queue.try_submit(copy(0, "H:nope", "D:nope")); }, "E_HANDLE");
    rejects([&] { (void)queue.try_submit(alloc(0, "H:x")); }, "E_ALLOC");
    submit(queue, alloc(0, "D:x", 0));
    submit(queue, alloc(1, "D:y", 1));
    rejects([&] { (void)queue.try_submit(command(2, Kind::Allocate, Backend::Device, {}, "D:big", {1, 1}, pl::Op::Add, 2)); }, "E_MEMORY");
    rejects([&] { (void)queue.try_submit(alloc(2, "D:z", 0)); }, "E_SLOT");
    auto wrong_shape = copy(2, "H:x", "D:x"); wrong_shape.shape = {2, 1};
    rejects([&] { (void)queue.try_submit(wrong_shape); }, "E_SHAPE");
    submit(queue, copy(2, "H:x", "D:x"));
    rejects([&] { (void)queue.try_submit(copy(3, "H:x", "D:x")); }, "E_SSA");
    auto bad_compute = add(3, "D:x", "D:x", "D:y"); bad_compute.shape = {2, 1};
    rejects([&] { (void)queue.try_submit(bad_compute); }, "E_SHAPE");
    rejects([&] { (void)queue.try_submit(add(3, "D:x", "H:x", "D:y")); }, "E_BUFFER");
    auto reader = submit(queue, add(3, "D:x", "D:x", "D:y"));
    auto release = submit(queue, command(4, Kind::Free, Backend::Device, {}, "D:x", {1, 2}, pl::Op::Add, 0));
    check(release.state() == JobState::Succeeded, "FREE completes on host");
    rejects([&] { (void)queue.try_submit(add(5, "D:x", "D:x", "D:y")); }, "E_FREED");
    auto reuse = submit(queue, alloc(5, "D:w", 0));
    const auto r = queue.records();
    check_happens_before(r);
    check(r[4].waits == std::vector<std::size_t>{0, 2, 3} && r[4].start_us >= r[3].end_us, "FREE waits for allocation, writer and reader");
    check(r[5].waits == std::vector<std::size_t>{4}, "slot reuse waits for the previous owner's FREE");
    check(queue.stats().software_buffer_reuses == 1 && queue.stats().peak_device_reserved_bytes == 16, "slot reuse keeps budget");
    (void)reader; (void)reuse;

    // A plan that fit one budget is rejected under a smaller one.
    const auto graph = pl::load(source_dir + "/examples/fragmented.plg");
    pl::Config roomy;
    const auto plan = pl::choose(graph, roomy, "maximal");
    pl::Config tight = roomy; tight.device_bytes = 32;
    rejects([&] { pl::execute_command_queue(graph, plan, tight); }, "E_MEMORY");
    auto renumbered = plan; renumbered.commands[1].id = 7;
    rejects([&] { pl::execute_command_queue(graph, renumbered, roomy); }, "E_SEQUENCE");
    auto versioned = plan; versioned.commands.back().version = 1;
    rejects([&] { pl::execute_command_queue(graph, versioned, roomy); }, "E_VERSION");
    const std::atomic<bool> stop{true};
    rejects([&] { pl::execute_command_queue(graph, plan, roomy, 4, &stop); }, "E_CANCELLED");
    // Declared target support rules apply to queued device compute, as in the serial executor.
    pl::Config relu; relu.relu_supported = true;
    const auto all_device = pl::lower(graph, std::vector<pl::Backend>(graph.nodes.size(), pl::Backend::Device), relu);
    auto excluded = relu; excluded.device_unsupported_ops = {"MatMul"};
    auto small = relu; small.device_max_tensor_bytes = 4;
    for (const auto& cfg : {excluded, small})
        rejects([&] { pl::execute_command_queue(graph, all_device, cfg); }, "E_CAPABILITY: device rejects command");
}

void failure_propagation() {
    std::set<std::size_t> ran;
    std::mutex ran_mutex;
    pl::CommandQueue queue({}, 4, [&](const pl::Command& c, const std::atomic<bool>&) {
        { std::lock_guard<std::mutex> lock(ran_mutex); ran.insert(c.id); }
        if (c.id == 5) throw std::runtime_error("E_INJECTED: simulated engine fault");
    });
    queue.import("H:x", pair(1, 2));
    for (auto c : {alloc(0, "D:x"), alloc(1, "D:y"), alloc(2, "D:z"), alloc(3, "H:z"), alloc(4, "D:u")}) submit(queue, c);
    auto source = submit(queue, copy(9, "H:x", "D:x"));
    auto failed = submit(queue, add(5, "D:x", "D:x", "D:y"));
    auto child = submit(queue, add(6, "D:y", "D:x", "D:z"));
    auto grandchild = submit(queue, copy(7, "D:z", "H:z"));
    auto independent = submit(queue, add(8, "D:x", "D:x", "D:u"));
    submit(queue, alloc(10, "H:w"));
    auto host = submit(queue, add(11, "H:z", "H:x", "H:w")); // CPU fallback waits on the failed device chain
    for (auto* e : {&failed, &child, &grandchild, &independent}) e->wait();
    check(failed.state() == JobState::Failed && failed.error() == "E_INJECTED: simulated engine fault", "root failure recorded");
    check(child.state() == JobState::Failed && child.error() == "E_DEPENDENCY: command 5 failed: E_INJECTED: simulated engine fault", "child gets propagated cause");
    check(grandchild.state() == JobState::Failed && grandchild.error() == child.error(), "propagation keeps the root cause");
    check(independent.state() == JobState::Succeeded && source.state() == JobState::Succeeded, "independent work unaffected");
    check(host.state() == JobState::Failed && host.error() == child.error(), "host CPU fallback never runs on bad data");
    rejects([&] { (void)queue.read("H:z"); }, "E_BUFFER");
    std::lock_guard<std::mutex> lock(ran_mutex);
    check(!ran.count(6) && !ran.count(7) && !ran.count(11), "dependents of a failure never ran");
}

void shutdown_and_lifetime() {
    // Drain completes accepted work, then admission is closed.
    std::optional<pl::CommandEvent> kept;
    {
        pl::CommandQueue queue({}, 4);
        queue.import("H:x", pair(1, 2));
        for (auto c : {alloc(0, "D:x"), alloc(1, "D:y"), alloc(2, "H:y")}) submit(queue, c);
        auto in = submit(queue, copy(3, "H:x", "D:x"));
        submit(queue, add(4, "D:x", "D:x", "D:y"));
        kept = submit(queue, copy(5, "D:y", "H:y"));
        queue.shutdown(pl::ShutdownMode::Drain);
        check(kept->state() == JobState::Succeeded && in.state() == JobState::Succeeded, "drain completes queued work");
        check(queue.read("H:y").values == std::vector<float>{2, 4}, "drained result readable");
        rejects([&] { (void)queue.try_submit(alloc(6, "D:q")); }, "E_CLOSED");
        queue.shutdown(pl::ShutdownMode::Cancel); // idempotent
    }
    check(kept->state() == JobState::Succeeded && kept->wait_for(0ms), "event outlives its queue");

    // Cancel: running command is cooperatively cancelled, queued ones immediately.
    Gate gate;
    std::optional<pl::CommandEvent> running, queued, dependent;
    {
        pl::CommandQueue queue({}, 4, [&](const pl::Command& c, const std::atomic<bool>& cancel) {
            if (c.id == 4) gate.enter(cancel);
        });
        ReleaseOnExit release{gate};
        queue.import("H:x", pair(1, 2));
        for (auto c : {alloc(0, "D:x"), alloc(1, "D:y"), alloc(2, "D:z"), alloc(5, "H:z")}) submit(queue, c);
        submit(queue, copy(3, "H:x", "D:x"));
        running = submit(queue, add(4, "D:x", "D:x", "D:y"));
        gate.wait_entered();
        queued = submit(queue, add(6, "D:x", "D:x", "D:z"));
        dependent = submit(queue, copy(7, "D:z", "H:z"));
        queue.shutdown(pl::ShutdownMode::Cancel);
        check(running->state() == JobState::Cancelled && queued->state() == JobState::Cancelled, "cancel shutdown");
        check(dependent->state() == JobState::Cancelled || dependent->state() == JobState::Failed, "dependent never succeeds");
    } // destructor after explicit shutdown joins nothing new
    check(queued->error().find("E_CANCELLED") == 0, "cancelled event readable after queue destroyed");

    // Destructor without explicit shutdown cancels and joins.
    Gate last;
    std::optional<pl::CommandEvent> orphan;
    {
        pl::CommandQueue queue({}, 2, [&](const pl::Command& c, const std::atomic<bool>& cancel) {
            if (c.kind == Kind::Copy) last.enter(cancel); // host-lane hooks run on this thread: never gate them
        });
        ReleaseOnExit release{last};
        queue.import("H:x", pair(1, 2));
        submit(queue, alloc(0, "D:x"));
        orphan = submit(queue, copy(1, "H:x", "D:x"));
        last.wait_entered();
    }
    check(orphan->state() == JobState::Cancelled, "destructor cancels running command and joins");
    rejects([] { pl::CommandQueue bad({}, 0); }, "E_CAPACITY");
}

void concurrent_producers() {
    // Several host threads submit independent chains; TSan evidence for the shared state.
    pl::CommandQueue queue({}, 2);
    constexpr int producers = 4, chains = 8;
    for (int p = 0; p < producers; ++p)
        for (int i = 0; i < chains; ++i) queue.import("H:x" + std::to_string(p) + "_" + std::to_string(i), pair(float(p), float(i)));
    std::vector<std::thread> threads;
    for (int p = 0; p < producers; ++p)
        threads.emplace_back([&queue, p] {
            for (int i = 0; i < chains; ++i) {
                const auto t = std::to_string(p) + "_" + std::to_string(i);
                const auto base = static_cast<std::size_t>(1000 * p + 10 * i);
                for (auto c : {alloc(base, "D:x" + t), alloc(base + 1, "D:y" + t), alloc(base + 2, "H:y" + t)}) submit(queue, c);
                submit(queue, copy(base + 3, "H:x" + t, "D:x" + t));
                submit(queue, add(base + 4, "D:x" + t, "D:x" + t, "D:y" + t));
                submit(queue, copy(base + 5, "D:y" + t, "H:y" + t)).wait();
            }
        });
    for (auto& thread : threads) thread.join();
    for (int p = 0; p < producers; ++p)
        for (int i = 0; i < chains; ++i)
            check(queue.read("H:y" + std::to_string(p) + "_" + std::to_string(i)).values == std::vector<float>{2.0f * float(p), 2.0f * float(i)},
                  "concurrent producer result");
    check_happens_before(queue.records());
}
}

int main(int argc, char** argv) {
    if (argc > 1) source_dir = argv[1];
    try {
        equivalence_with_serial_execute();
        copy_compute_overlap();
        saturation_timeout_and_cancellation();
        invalid_commands_and_memory();
        failure_propagation();
        shutdown_and_lifetime();
        concurrent_producers();
    } catch (const std::exception& error) {
        std::cerr << error.what() << " after " << checks << " checks\n";
        return 1;
    }
    std::cout << "queue_contracts: " << checks << " checks passed\n";
    return 0;
}
