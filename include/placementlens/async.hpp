#pragma once
#include "placementlens/core.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace pl {
enum class JobState { Queued, Running, Succeeded, Failed, Cancelled };
enum class ShutdownMode { Drain, Cancel };
class AsyncCancelled : public std::runtime_error {
public:
    AsyncCancelled() : std::runtime_error("E_CANCELLED: asynchronous job cancelled") {}
};
struct Completion {
    Run run;
    double queue_wait_ms = 0;
    double total_ms = 0;
};
namespace async_detail { struct Shared; struct Job; }

// A copyable completion fence; it remains valid after its runtime is destroyed.
class Fence {
public:
    Completion get() const;
    void wait() const;
    std::future_status wait_for(std::chrono::milliseconds timeout) const;
    JobState state() const;
    // Returns true only if this call newly requests cancellation before completion.
    bool cancel() const;
private:
    friend class AsyncRuntime;
    Fence(std::shared_ptr<async_detail::Shared>, std::shared_ptr<async_detail::Job>);
    std::shared_ptr<async_detail::Shared> shared_;
    std::shared_ptr<async_detail::Job> job_;
};

// Capacity bounds unfinished jobs, including the running job. Jobs own their
// graph, plan and configuration. A runner must return/throw eventually and
// poll cancellation if prompt cancellation is required.
class AsyncRuntime {
public:
    using Runner = std::function<Run(const Graph&, const Plan&, const Config&, const std::atomic<bool>&)>;
    explicit AsyncRuntime(std::size_t capacity, Runner runner = {});
    ~AsyncRuntime();
    AsyncRuntime(const AsyncRuntime&) = delete;
    AsyncRuntime& operator=(const AsyncRuntime&) = delete;
    Fence submit(Graph graph, Plan plan, Config config);
    // Queue saturation returns nullopt; closed runtimes throw E_CLOSED.
    std::optional<Fence> try_submit(Graph graph, Plan plan, Config config);
    // Stops admission and joins. Drain completes accepted jobs; Cancel requests
    // cooperative running cancellation and immediately cancels queued jobs.
    void shutdown(ShutdownMode mode = ShutdownMode::Drain);
private:
    std::shared_ptr<async_detail::Shared> shared_;
    std::thread worker_;
    std::mutex join_mutex_;
};
}
