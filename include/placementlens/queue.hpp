#pragma once
#include "placementlens/async.hpp"
#include "placementlens/kernels.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Command-level simulated device queue. See docs/runtime/COMMAND_QUEUE.md.
// The host submits individual lowered commands; each returns an event. Copies
// run on a copy-engine thread, device computes on a compute-engine thread,
// and ALLOC/FREE/SYNC plus CPU computes run on the submitting host thread
// after their dependency events complete.
namespace pl {
namespace queue_detail { struct Shared; struct Item; }

enum class Lane { Host, CopyEngine, ComputeEngine };
std::string name(Lane lane); // "host", "copy-engine", "compute-engine"

// One per submitted command, in submission order. Times are microseconds since
// queue construction on a monotonic host clock; -1 means the command never ran.
struct QueueRecord {
    std::size_t command_id = 0;
    std::string node, kind, backend, lane;
    std::vector<std::size_t> waits; // indices into records() (submission order) of its dependencies
    JobState state = JobState::Queued;
    std::string error;
    double start_us = -1, end_us = -1;
    double injected_ms = 0; // synthetic host wait (emulate_boundaries), not hardware time
};

// Copyable completion handle; remains valid after its queue is destroyed.
class CommandEvent {
public:
    JobState state() const;
    void wait() const;
    // True when the command reached a terminal state within the timeout.
    // A timeout is observational: it neither cancels nor fails the command.
    bool wait_for(std::chrono::milliseconds timeout) const;
    // True only if this call newly requests cancellation before completion.
    // Queued commands are cancelled immediately; running ones cooperatively.
    bool cancel() const;
    std::string error() const; // empty unless Failed or Cancelled
    std::size_t command_id() const;
private:
    friend class CommandQueue;
    CommandEvent(std::shared_ptr<queue_detail::Shared>, std::shared_ptr<queue_detail::Item>);
    std::shared_ptr<queue_detail::Shared> shared_;
    std::shared_ptr<queue_detail::Item> item_;
};

struct QueueStats {
    std::size_t peak_device_reserved_bytes = 0, software_buffer_reuses = 0;
    double injected_ms = 0;
    KernelStats kernels;
};

class CommandQueue {
public:
    // Test hook run on the executing lane just before a command's work. It may
    // block (gates) and must poll the cancel token to stop promptly, or throw
    // to inject a failure.
    using Hook = std::function<void(const Command&, const std::atomic<bool>& cancel)>;
    // capacity bounds unfinished commands per engine (queued + running).
    CommandQueue(Config config, std::size_t capacity, Hook hook = {});
    ~CommandQueue(); // shutdown(Cancel)
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;

    // Registers an initialized host tensor (a graph input); not a command.
    void import(const std::string& host_handle, Tensor value);
    // Invalid commands throw an E_* runtime_error and are never enqueued.
    // Engine commands: nullopt when that engine is full (never blocks).
    // Host-lane commands never consume capacity; they execute on this thread
    // after their dependencies, so try_submit can block on those events.
    std::optional<CommandEvent> try_submit(const Command& command);
    // As try_submit, but waits up to timeout for engine capacity.
    std::optional<CommandEvent> submit(const Command& command, std::chrono::milliseconds timeout);
    // Copy of a host tensor whose writer succeeded (E_BUFFER otherwise).
    Tensor read(const std::string& host_handle) const;
    std::vector<QueueRecord> records() const;
    QueueStats stats() const;
    // Stops admission (E_CLOSED) and joins both engines. Drain runs accepted
    // commands; Cancel cancels queued ones and requests running cancellation.
    void shutdown(ShutdownMode mode = ShutdownMode::Drain);
private:
    std::shared_ptr<queue_detail::Shared> shared_;
    std::vector<std::thread> engines_;
    std::mutex join_mutex_;
};

// Runs an existing plan through a CommandQueue (software device only) and
// checks it against reference() exactly like execute(). Run.events carry the
// engine lane, dependency ids and timestamps of every command.
Run execute_command_queue(const Graph& source, const Plan& plan, const Config& config,
                          std::size_t capacity = 4, const std::atomic<bool>* cancel = nullptr);
}
