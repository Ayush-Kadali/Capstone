#include "placementlens/queue.hpp"
#include "placementlens/cnn.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <map>

namespace pl {
namespace {
using Clock = std::chrono::steady_clock;
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
bool terminal(JobState state) {
    return state == JobState::Succeeded || state == JobState::Failed || state == JobState::Cancelled;
}
std::size_t index(Lane lane) { return static_cast<std::size_t>(lane); }
}
std::string name(Lane lane) {
    switch (lane) {
        case Lane::Host: return "host";
        case Lane::CopyEngine: return "copy-engine";
        case Lane::ComputeEngine: return "compute-engine";
    }
    throw std::runtime_error("E_LANE: unknown lane");
}

namespace queue_detail {
// Mutable fields are guarded by Shared::mutex, except that the executing lane
// owns the tensors behind inputs/output while state == Running.
struct Item {
    Command command;
    Lane lane = Lane::Host;
    std::size_t sequence = 0; // submission index; ids may repeat across preparation and plan
    std::vector<std::shared_ptr<Item>> deps; // must reach Succeeded before this runs
    std::vector<const Tensor*> inputs;
    Tensor* output = nullptr;
    double injected_ms = 0;
    JobState state = JobState::Queued;
    std::string error;
    std::atomic<bool> cancel{false};
    double start_us = -1, end_us = -1;
    KernelStats stats;
};
// Per logical handle. Handles are SSA: one writer, then any number of readers.
struct Buffer {
    Tensor tensor; // std::map node: address stable until the queue is destroyed
    std::shared_ptr<Item> allocation, writer;
    std::vector<std::shared_ptr<Item>> readers;
    bool written = false, freed = false; // written: a writer was accepted (or imported)
    std::size_t slot = no_buffer_slot;
};
struct Slot { std::size_t bytes = 0; std::string owner; std::shared_ptr<Item> released; };
struct Shared {
    std::mutex mutex;
    std::condition_variable changed; // admission, start, completion, capacity, shutdown
    const Config config;
    const std::size_t capacity;
    const CommandQueue::Hook hook;
    const Clock::time_point epoch = Clock::now();
    std::map<std::string, Buffer> buffers;
    std::vector<Slot> slots;
    std::array<std::deque<std::shared_ptr<Item>>, 3> fifo; // indexed by Lane; Host unused
    std::array<std::shared_ptr<Item>, 3> running;
    std::vector<std::shared_ptr<Item>> history, device_work;
    std::size_t reserved = 0;
    QueueStats stats;
    bool closed = false;
    Shared(Config c, std::size_t n, CommandQueue::Hook h) : config(std::move(c)), capacity(n), hook(std::move(h)) {}
    double now_us() const { return std::chrono::duration<double, std::micro>(Clock::now() - epoch).count(); }
};

namespace {
bool ready(const Item& item) {
    return std::all_of(item.deps.begin(), item.deps.end(), [](const auto& d) { return terminal(d->state); });
}
bool has_room(const Shared& s, Lane lane) {
    return s.fifo[index(lane)].size() + (s.running[index(lane)] ? 1U : 0U) < s.capacity;
}
void finish(Shared& s, Item& item, JobState state, std::string error) {
    item.state = state;
    item.error = std::move(error);
    if (item.start_us >= 0) item.end_us = s.now_us();
    s.stats.kernels.tile_count += item.stats.tile_count;
    s.stats.kernels.tile_load_bytes += item.stats.tile_load_bytes;
    s.stats.kernels.peak_local_bytes = std::max(s.stats.kernels.peak_local_bytes, item.stats.peak_local_bytes);
    s.changed.notify_all();
}
// Synthetic boundary cost (Config::emulate_boundaries): a real host wait on the
// executing lane, polled for cancellation. Never hardware evidence.
void synthetic_wait(double milliseconds, const std::atomic<bool>& cancel) {
    const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(milliseconds));
    while (Clock::now() < deadline) {
        require(!cancel.load(), "E_CANCELLED: execution cancelled");
        std::this_thread::sleep_for(std::min<Clock::duration>(deadline - Clock::now(), std::chrono::milliseconds(1)));
    }
}
void perform(const Config& config, Item& item) {
    const auto& c = item.command;
    synthetic_wait(item.injected_ms, item.cancel);
    if (c.kind == Kind::Copy) {
        std::copy(item.inputs[0]->values.begin(), item.inputs[0]->values.end(), item.output->values.begin());
    } else if (c.kind == Kind::Compute) {
        auto kernel_config = config;
        if (c.backend == Backend::Cpu) kernel_config.tile_size = 0;
        if (cnn_op(c.op)) compute_cnn(c.op, item.inputs, *item.output, c.parameters, &item.cancel);
        else compute_software(c.op, item.inputs, *item.output, kernel_config, item.stats, &item.cancel);
        for (float v : item.output->values) require(std::isfinite(v), "E_NUMERIC: computation produced nonfinite output");
    } else if (c.kind == Kind::Free) {
        std::vector<float>().swap(item.output->values); // every prior user has finished
    } // ALLOC bookkeeping happened at admission; SYNC is only its dependency wait.
}
// Called with the lock held and every dependency terminal. A dependency that did
// not succeed fails this command without running it (no work on bad data).
void run(Shared& s, Item& item, std::unique_lock<std::mutex>& lock) {
    for (const auto& d : item.deps) {
        if (d->state == JobState::Succeeded) continue;
        finish(s, item, JobState::Failed, d->error.rfind("E_DEPENDENCY", 0) == 0 ? d->error :
               "E_DEPENDENCY: command " + std::to_string(d->command.id) +
               (d->state == JobState::Cancelled ? " cancelled: " : " failed: ") + d->error);
        return;
    }
    item.state = JobState::Running;
    item.start_us = s.now_us();
    s.changed.notify_all();
    lock.unlock();
    std::string error;
    try {
        if (s.hook) s.hook(item.command, item.cancel);
        perform(s.config, item);
    } catch (const std::exception& e) { error = e.what(); }
    catch (...) { error = "E_UNKNOWN: non-standard exception"; }
    lock.lock();
    // Accepted cancellation wins over a racing successful return, as in Fence.
    if (item.cancel.load()) finish(s, item, JobState::Cancelled, "E_CANCELLED: command cancelled while running");
    else finish(s, item, error.empty() ? JobState::Succeeded : JobState::Failed, error);
}
void engine(std::shared_ptr<Shared> s, Lane lane) {
    std::unique_lock<std::mutex> lock(s->mutex);
    auto& fifo = s->fifo[index(lane)];
    auto& running = s->running[index(lane)];
    for (;;) {
        // In-order engine: the head waits for its dependencies. Submission order
        // is a topological order, so the oldest unfinished command is always at
        // some engine head with complete dependencies: no deadlock.
        s->changed.wait(lock, [&] { return fifo.empty() ? s->closed : ready(*fifo.front()); });
        if (fifo.empty()) return;
        running = fifo.front();
        fifo.pop_front();
        run(*s, *running, lock);
        running.reset();
        s->changed.notify_all(); // capacity released
    }
}
// Validates a command against the admitted state and derives its wait-list.
// Changes nothing; commit() applies the bookkeeping once capacity is available.
std::shared_ptr<Item> prepare(Shared& s, const Command& c) {
    require(c.version == command_version, "E_VERSION: unsupported command version");
    require(c.kind == Kind::Compute || c.parameters.empty(), "E_ATTRIBUTE: noncompute command parameters");
    require(c.kind == Kind::Allocate || c.kind == Kind::Free || c.buffer_slot == no_buffer_slot,
            "E_SLOT: only allocation/free descriptors may name a buffer slot");
    auto item = std::make_shared<Item>();
    item->command = c;
    item->lane = c.kind == Kind::Copy ? Lane::CopyEngine :
                 (c.kind == Kind::Compute && c.backend == Backend::Device ? Lane::ComputeEngine : Lane::Host);
    const std::string prefix = c.backend == Backend::Cpu ? "H:" : "D:";
    auto in_lane = [&](const std::string& h) { return h.size() > 2 && h.compare(0, 2, prefix) == 0; };
    auto wait = [&](const std::shared_ptr<Item>& d) {
        if (d && std::find(item->deps.begin(), item->deps.end(), d) == item->deps.end()) item->deps.push_back(d);
    };
    auto live = [&](const std::string& h) -> Buffer& {
        const auto found = s.buffers.find(h);
        require(found != s.buffers.end(), "E_HANDLE: unknown buffer handle " + h);
        require(!found->second.freed, "E_FREED: use of freed buffer " + h);
        return found->second;
    };
    auto read = [&](const std::string& h) { // read-after-write
        auto& b = live(h);
        require(b.written, "E_BUFFER: operand has no writer: " + h);
        wait(b.writer); // null for imported host inputs
        item->inputs.push_back(&b.tensor);
    };
    // SSA: a second write is rejected, so logical WAW/WAR cannot occur. Physical
    // WAW/WAR through a reused slot is ordered by FREE -> ALLOC -> writer.
    auto write = [&](const std::string& h) -> const Shape& {
        auto& b = live(h);
        require(!b.written, "E_SSA: cannot overwrite initialized storage " + h);
        wait(b.allocation);
        item->output = &b.tensor;
        return b.tensor.shape;
    };
    double boundary_ms = 0;
    switch (c.kind) {
    case Kind::Allocate: {
        require(in_lane(c.output), "E_HANDLE: allocation backend and handle disagree");
        require(!s.buffers.count(c.output), "E_ALLOC: duplicate logical allocation");
        const auto bytes = c.shape.bytes();
        bool reserve = c.backend == Backend::Device;
        if (c.backend == Backend::Device && s.config.reuse_buffers) {
            require(c.buffer_slot <= s.slots.size(), "E_SLOT: invalid or noncontiguous buffer slot");
            if (c.buffer_slot < s.slots.size()) {
                const auto& slot = s.slots[c.buffer_slot];
                require(slot.owner.empty(), "E_SLOT: buffer slot already has a live owner");
                require(slot.bytes == bytes, "E_SLOT: reused buffer size differs");
                wait(slot.released); // the previous owner's FREE, which waited for all its users
                reserve = false;
            }
        } else require(c.buffer_slot == no_buffer_slot, c.backend == Backend::Device ?
                       "E_SLOT: buffer reuse is disabled" : "E_SLOT: host allocation cannot claim a device slot");
        if (reserve) require(bytes <= s.config.device_bytes && s.reserved <= s.config.device_bytes - bytes,
                             "E_MEMORY: device reservation budget exceeded");
        break;
    }
    case Kind::Free: {
        require(s.config.reuse_buffers, "E_FREE: explicit release requires buffer reuse mode");
        require(c.backend == Backend::Device && in_lane(c.output) && c.inputs.empty(), "E_FREE: only a single device handle may be freed");
        auto& b = live(c.output);
        require(b.written, "E_FREE: uninitialized device handle");
        require(b.tensor.shape == c.shape, "E_FREE: released shape differs");
        require(b.slot == c.buffer_slot && c.buffer_slot < s.slots.size(), "E_SLOT: release slot differs from allocation");
        wait(b.allocation); wait(b.writer); // FREE waits for every prior user
        for (const auto& reader : b.readers) wait(reader);
        item->output = &b.tensor;
        break;
    }
    case Kind::Copy:
        require(c.inputs.size() == 1 && in_lane(c.output) && c.inputs[0].size() > 2 && !in_lane(c.inputs[0]),
                "E_COPY: copy must cross backends");
        require(c.output.substr(2) == c.inputs[0].substr(2), "E_COPY: tensor identity changed during copy");
        read(c.inputs[0]);
        require(write(c.output) == c.shape && item->inputs[0]->shape == c.shape, "E_SHAPE: copy shapes differ");
        boundary_ms = s.config.transfer_fixed_ms + static_cast<double>(c.shape.bytes()) / s.config.bandwidth_bytes_per_ms;
        break;
    case Kind::Compute: {
        require(in_lane(c.output), "E_BUFFER: compute output on the wrong backend");
        std::vector<Shape> shapes;
        for (const auto& h : c.inputs) {
            require(in_lane(h), "E_BUFFER: compute operand on the wrong backend");
            read(h);
            shapes.push_back(item->inputs.back()->shape);
        }
        require(compute_shape(c.op, shapes, c.parameters) == c.shape && write(c.output) == c.shape,
                "E_SHAPE: compute output shape mismatch");
        if (c.backend == Backend::Device) {
            require(s.config.relu_supported || (c.op != Op::Relu && c.op != Op::MatMulRelu), "E_CAPABILITY: device rejects unsupported ReLU command");
            require(!cnn_op(c.op) || s.config.cnn_supported, "E_CAPABILITY: device CNN kernels disabled");
            auto tensors = shapes; tensors.push_back(c.shape);
            const auto rule = target_rule_rejection(c.op, tensors, c.parameters, s.config);
            require(rule.empty(), "E_CAPABILITY: device rejects command: " + rule);
            boundary_ms = s.config.launch_ms;
        }
        break;
    }
    case Kind::Synchronize:
        require(c.backend == Backend::Device, "E_SYNC: invalid synchronization target");
        for (const auto& d : s.device_work) if (!terminal(d->state)) wait(d); // barrier over both engines
        boundary_ms = s.config.synchronization_ms;
        break;
    default:
        throw std::runtime_error("E_COMMAND: unknown command kind");
    }
    if (s.config.emulate_boundaries && boundary_ms > 0) {
        require(boundary_ms <= 100 && s.stats.injected_ms + boundary_ms <= 10000,
                "E_DELAY: emulation permits <=100 ms per boundary and <=10 seconds per queue");
        item->injected_ms = boundary_ms;
    }
    return item;
}
void commit(Shared& s, const std::shared_ptr<Item>& item) {
    const auto& c = item->command;
    if (c.kind == Kind::Allocate) {
        auto& b = s.buffers[c.output];
        b.tensor = Tensor{c.shape, std::vector<float>(c.shape.elements())};
        b.allocation = item;
        item->output = &b.tensor;
        if (c.backend == Backend::Device) {
            const auto bytes = c.shape.bytes();
            if (!s.config.reuse_buffers) s.reserved += bytes;
            else {
                if (c.buffer_slot == s.slots.size()) {
                    s.slots.push_back({bytes, {}, {}});
                    s.reserved += bytes;
                } else ++s.stats.software_buffer_reuses;
                s.slots[c.buffer_slot].owner = c.output;
                b.slot = c.buffer_slot;
            }
            s.stats.peak_device_reserved_bytes = std::max(s.stats.peak_device_reserved_bytes, s.reserved);
        }
    } else if (c.kind == Kind::Free) {
        s.buffers.at(c.output).freed = true;
        auto& slot = s.slots[c.buffer_slot];
        slot.owner.clear();
        slot.released = item;
    } else if (c.kind == Kind::Copy || c.kind == Kind::Compute) {
        auto& out = s.buffers.at(c.output);
        out.written = true;
        out.writer = item;
        for (const auto& h : c.inputs) s.buffers.at(h).readers.push_back(item);
    } else if (c.kind == Kind::Synchronize) {
        auto& work = s.device_work;
        work.erase(std::remove_if(work.begin(), work.end(), [](const auto& d) { return terminal(d->state); }), work.end());
    }
    if (item->lane != Lane::Host) s.device_work.push_back(item);
    s.stats.injected_ms += item->injected_ms;
    item->sequence = s.history.size();
    s.history.push_back(item);
}
}
}

using queue_detail::Item;
using queue_detail::Shared;

CommandEvent::CommandEvent(std::shared_ptr<Shared> shared, std::shared_ptr<Item> item)
    : shared_(std::move(shared)), item_(std::move(item)) {}
JobState CommandEvent::state() const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    return item_->state;
}
void CommandEvent::wait() const {
    std::unique_lock<std::mutex> lock(shared_->mutex);
    shared_->changed.wait(lock, [&] { return terminal(item_->state); });
}
bool CommandEvent::wait_for(std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(shared_->mutex);
    return shared_->changed.wait_for(lock, timeout, [&] { return terminal(item_->state); });
}
bool CommandEvent::cancel() const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    if (terminal(item_->state) || item_->cancel.exchange(true)) return false;
    if (item_->state == JobState::Queued) {
        auto& fifo = shared_->fifo[index(item_->lane)];
        fifo.erase(std::remove(fifo.begin(), fifo.end(), item_), fifo.end());
        queue_detail::finish(*shared_, *item_, JobState::Cancelled, "E_CANCELLED: command cancelled before start");
    }
    return true;
}
std::string CommandEvent::error() const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    return item_->error;
}
std::size_t CommandEvent::command_id() const { return item_->command.id; }

CommandQueue::CommandQueue(Config config, std::size_t capacity, Hook hook) {
    require(capacity > 0, "E_CAPACITY: command queue capacity must be positive");
    validate_config(config);
    shared_ = std::make_shared<Shared>(std::move(config), capacity, std::move(hook));
    engines_.emplace_back(queue_detail::engine, shared_, Lane::CopyEngine);
    engines_.emplace_back(queue_detail::engine, shared_, Lane::ComputeEngine);
}
CommandQueue::~CommandQueue() { shutdown(ShutdownMode::Cancel); }

void CommandQueue::import(const std::string& host_handle, Tensor value) {
    require(host_handle.size() > 2 && host_handle.compare(0, 2, "H:") == 0, "E_HANDLE: only host tensors can be imported");
    require(value.values.size() == value.shape.elements(), "E_SHAPE: imported storage differs from its shape");
    std::lock_guard<std::mutex> lock(shared_->mutex);
    require(!shared_->closed, "E_CLOSED: command queue is closed");
    require(!shared_->buffers.count(host_handle), "E_ALLOC: duplicate logical allocation");
    auto& b = shared_->buffers[host_handle];
    b.tensor = std::move(value);
    b.written = true;
}
std::optional<CommandEvent> CommandQueue::try_submit(const Command& command) {
    return submit(command, std::chrono::milliseconds(0));
}
std::optional<CommandEvent> CommandQueue::submit(const Command& command, std::chrono::milliseconds timeout) {
    auto& s = *shared_;
    const auto deadline = Clock::now() + timeout;
    std::unique_lock<std::mutex> lock(s.mutex);
    std::shared_ptr<Item> item;
    for (;;) {
        require(!s.closed, "E_CLOSED: command queue is closed");
        item = queue_detail::prepare(s, command); // revalidate: state may change while waiting
        if (item->lane == Lane::Host || queue_detail::has_room(s, item->lane)) break;
        if (Clock::now() >= deadline) return std::nullopt; // engine saturated
        s.changed.wait_until(lock, deadline);
    }
    queue_detail::commit(s, item);
    CommandEvent event(shared_, item);
    if (item->lane != Lane::Host) {
        s.fifo[index(item->lane)].push_back(item);
        s.changed.notify_all();
        return event;
    }
    // Host lane: the submitting thread is the executor. Waiting here on device
    // events is the queue's Synchronize boundary.
    s.changed.wait(lock, [&] { return terminal(item->state) || queue_detail::ready(*item); });
    if (!terminal(item->state)) queue_detail::run(s, *item, lock);
    return event;
}
Tensor CommandQueue::read(const std::string& host_handle) const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    const auto found = shared_->buffers.find(host_handle);
    require(host_handle.compare(0, 2, "H:") == 0 && found != shared_->buffers.end() && !found->second.freed &&
            found->second.written && (!found->second.writer || found->second.writer->state == JobState::Succeeded),
            "E_BUFFER: host tensor is not available: " + host_handle);
    return found->second.tensor;
}
std::vector<QueueRecord> CommandQueue::records() const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    std::vector<QueueRecord> out;
    for (const auto& item : shared_->history) {
        QueueRecord r;
        const auto& c = item->command;
        r.command_id = c.id; r.node = c.node; r.kind = name(c.kind); r.backend = name(c.backend);
        r.lane = name(item->lane);
        for (const auto& d : item->deps) r.waits.push_back(d->sequence);
        r.state = item->state; r.error = item->error;
        r.start_us = item->start_us; r.end_us = item->end_us; r.injected_ms = item->injected_ms;
        out.push_back(std::move(r));
    }
    return out;
}
QueueStats CommandQueue::stats() const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    return shared_->stats;
}
void CommandQueue::shutdown(ShutdownMode mode) {
    {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        shared_->closed = true;
        if (mode == ShutdownMode::Cancel) {
            for (auto& fifo : shared_->fifo) {
                for (auto& item : fifo) {
                    item->cancel.store(true);
                    queue_detail::finish(*shared_, *item, JobState::Cancelled, "E_CANCELLED: queue shut down before start");
                }
                fifo.clear();
            }
            for (auto& item : shared_->running) if (item) item->cancel.store(true);
        }
        shared_->changed.notify_all();
    }
    // Never hold the queue mutex across join; a Cancel may escalate a Drain.
    std::lock_guard<std::mutex> joining(join_mutex_);
    for (auto& engine : engines_) if (engine.joinable()) engine.join();
}

Run execute_command_queue(const Graph& source, const Plan& plan, const Config& config,
                          std::size_t capacity, const std::atomic<bool>* cancel) {
    auto check_cancel = [&] { require(!cancel || !cancel->load(), "E_CANCELLED: execution cancelled"); };
    check_cancel();
    validate_config(config);
    const auto checked = validate(source);
    auto mirrored = config.mirrored_inputs;
    std::sort(mirrored.begin(), mirrored.end());
    require(plan.initial_buffers.size() == mirrored.size(), "E_RESIDENCY: initial buffer plan differs from configuration");
    CommandQueue queue(config, capacity); // destructor cancels and joins if anything below throws
    for (const auto& input : checked.inputs) queue.import("H:" + input.first, input.second);
    std::vector<CommandEvent> events;
    auto submit = [&](const Command& c) {
        for (;;) { // blocking submission that still observes the caller's cancel token
            check_cancel();
            if (auto event = queue.submit(c, std::chrono::milliseconds(10))) { events.push_back(*event); return; }
        }
    };
    auto wait_all = [&] {
        for (const auto& event : events) while (!event.wait_for(std::chrono::milliseconds(10))) check_cancel();
        const CommandEvent* failed = nullptr; // report the root cause, not a propagated E_DEPENDENCY
        for (const auto& event : events) {
            if (event.state() == JobState::Succeeded) continue;
            if (!failed || (failed->error().rfind("E_DEPENDENCY", 0) == 0 && event.error().rfind("E_DEPENDENCY", 0) != 0)) failed = &event;
        }
        if (failed) throw std::runtime_error(failed->error());
    };
    Run run;
    run.execution_backend = std::string("command-queue simulator: copy+compute engines") +
                            (config.emulate_boundaries ? " with injected boundary waits" : "");
    // Initial residency = ALLOC + H2D COPY through the same queue, measured separately.
    const auto preparation_start = Clock::now();
    for (std::size_t i = 0; i < plan.initial_buffers.size(); ++i) {
        const auto& c = plan.initial_buffers[i];
        require(checked.inputs.count(mirrored[i]) != 0, "E_RESIDENCY: unknown mirrored source input");
        require(c.version == command_version && c.id == i && c.kind == Kind::Allocate && c.backend == Backend::Device &&
                c.output == "D:" + mirrored[i] && c.shape == checked.inputs.at(mirrored[i]).shape &&
                c.inputs.empty() && c.parameters.empty(), "E_RESIDENCY: invalid preparation descriptor");
        require(c.buffer_slot == (config.reuse_buffers ? i : no_buffer_slot), "E_SLOT: invalid initial residency slot");
        submit(c);
        Command copy;
        copy.id = i; copy.kind = Kind::Copy; copy.backend = Backend::Device; copy.node = "preparation";
        copy.inputs = {"H:" + mirrored[i]}; copy.output = c.output; copy.shape = c.shape;
        submit(copy);
        run.preparation_h2d_bytes += c.shape.bytes();
    }
    wait_all();
    if (!plan.initial_buffers.empty())
        run.input_preparation_ms = std::chrono::duration<double, std::milli>(Clock::now() - preparation_start).count();
    const auto first = events.size();
    const auto start = Clock::now();
    for (std::size_t i = 0; i < plan.commands.size(); ++i) {
        require(plan.commands[i].id == i, "E_SEQUENCE: command IDs are not contiguous");
        submit(plan.commands[i]);
    }
    wait_all();
    run.host_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    run.output = queue.read("H:" + checked.output);
    queue.shutdown(ShutdownMode::Drain);
    const auto records = queue.records();
    double origin = -1;
    for (std::size_t i = first; i < records.size(); ++i)
        if (records[i].start_us >= 0 && (origin < 0 || records[i].start_us < origin)) origin = records[i].start_us;
    std::vector<std::pair<double, double>> copies, computes;
    for (std::size_t i = first; i < records.size(); ++i) {
        const auto& r = records[i];
        const auto& c = plan.commands[r.command_id];
        std::vector<std::size_t> waits; // plan command ids; preparation finished before execution began
        for (auto w : r.waits) if (w >= first) waits.push_back(records[w].command_id);
        run.events.push_back({c.id, c.node, r.kind, r.backend, r.start_us - origin, r.end_us - r.start_us,
                              c.modeled_ms, -1, r.lane, waits});
        if (r.lane == "copy-engine") copies.emplace_back(r.start_us, r.end_us);
        if (r.lane == "compute-engine") computes.emplace_back(r.start_us, r.end_us);
    }
    // Each engine is serial, so summing pairwise intersections measures the time
    // both engines were busy at once. O(copies x computes); plans are <= hundreds.
    run.engine_overlap_us = 0;
    for (const auto& a : copies)
        for (const auto& b : computes)
            run.engine_overlap_us += std::max(0.0, std::min(a.second, b.second) - std::max(a.first, b.first));
    const auto stats = queue.stats();
    run.peak_device_reserved_bytes = stats.peak_device_reserved_bytes;
    run.software_buffer_reuses = stats.software_buffer_reuses;
    run.injected_boundary_ms = stats.injected_ms;
    run.tile_count = stats.kernels.tile_count;
    run.tile_load_bytes = stats.kernels.tile_load_bytes;
    run.peak_local_bytes = stats.kernels.peak_local_bytes;
    check_against_reference(run, checked, cancel);
    return run;
}
}
