#include "placementlens/async.hpp"
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <utility>

namespace pl {
namespace async_detail {
using Clock = std::chrono::steady_clock;
struct Job {
    Graph graph;
    Plan plan;
    Config config;
    JobState state = JobState::Queued;
    std::atomic<bool> cancel{false};
    std::promise<Completion> promise;
    std::shared_future<Completion> future = promise.get_future().share();
    Clock::time_point submitted = Clock::now();
    Job(Graph g, Plan p, Config c) : graph(std::move(g)), plan(std::move(p)), config(c) {}
};
struct Shared {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::shared_ptr<Job>> queue;
    std::shared_ptr<Job> running;
    const std::size_t capacity;
    bool closed = false;
    AsyncRuntime::Runner runner;
    std::thread::id worker_id;
    Shared(std::size_t n, AsyncRuntime::Runner r) : capacity(n), runner(std::move(r)) {}
};
bool terminal(JobState state) {
    return state == JobState::Succeeded || state == JobState::Failed || state == JobState::Cancelled;
}
void cancelled(Job& job) {
    job.state = JobState::Cancelled;
    job.promise.set_exception(std::make_exception_ptr(AsyncCancelled{}));
}
double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}
void work(const std::shared_ptr<Shared>& shared) {
    std::unique_lock<std::mutex> lock(shared->mutex);
    shared->worker_id = std::this_thread::get_id();
    for (;;) {
        shared->ready.wait(lock, [&] { return shared->closed || !shared->queue.empty(); });
        if (shared->queue.empty()) {
            shared->worker_id = {};
            return;
        }
        auto job = shared->queue.front();
        shared->queue.pop_front();
        shared->running = job;
        job->state = JobState::Running;
        const auto started = Clock::now();
        lock.unlock();
        Completion completion;
        std::exception_ptr failure;
        try {
            completion.run = shared->runner(job->graph, job->plan, job->config, job->cancel);
        } catch (...) { failure = std::current_exception(); }
        const auto finished = Clock::now();
        completion.queue_wait_ms = milliseconds(started - job->submitted);
        completion.total_ms = milliseconds(finished - job->submitted);
        lock.lock();
        // Publishing completion and accepting cancellation use the same mutex:
        // once cancel() succeeds, a racing successful return cannot win.
        if (job->cancel.load(std::memory_order_relaxed)) cancelled(*job);
        else if (failure) {
            job->state = JobState::Failed;
            job->promise.set_exception(failure);
        } else {
            job->state = JobState::Succeeded;
            job->promise.set_value(std::move(completion));
        }
        shared->running.reset();
    }
}
}

Fence::Fence(std::shared_ptr<async_detail::Shared> shared, std::shared_ptr<async_detail::Job> job)
    : shared_(std::move(shared)), job_(std::move(job)) {}
Completion Fence::get() const { return job_->future.get(); }
void Fence::wait() const { job_->future.wait(); }
std::future_status Fence::wait_for(std::chrono::milliseconds timeout) const {
    return job_->future.wait_for(timeout);
}
JobState Fence::state() const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    return job_->state;
}
bool Fence::cancel() const {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    if (async_detail::terminal(job_->state) || job_->cancel.exchange(true)) return false;
    if (job_->state == JobState::Queued) {
        const auto found = std::find(shared_->queue.begin(), shared_->queue.end(), job_);
        shared_->queue.erase(found);
        async_detail::cancelled(*job_);
    }
    return true;
}

AsyncRuntime::AsyncRuntime(std::size_t capacity, Runner runner) {
    if (capacity == 0) throw std::invalid_argument("E_CAPACITY: asynchronous capacity must be positive");
    if (!runner) runner = [](const Graph& graph, const Plan& plan, const Config& config,
                             const std::atomic<bool>& cancel) {
        return execute(graph, plan, config, nullptr, &cancel);
    };
    shared_ = std::make_shared<async_detail::Shared>(capacity, std::move(runner));
    worker_ = std::thread(async_detail::work, shared_);
}
AsyncRuntime::~AsyncRuntime() { shutdown(ShutdownMode::Cancel); }

std::optional<Fence> AsyncRuntime::try_submit(Graph graph, Plan plan, Config config) {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    if (shared_->closed) throw std::runtime_error("E_CLOSED: asynchronous runtime is closed");
    if (shared_->queue.size() + (shared_->running ? 1U : 0U) >= shared_->capacity) return std::nullopt;
    auto job = std::make_shared<async_detail::Job>(std::move(graph), std::move(plan), config);
    shared_->queue.push_back(job);
    shared_->ready.notify_one();
    return Fence(shared_, std::move(job));
}
Fence AsyncRuntime::submit(Graph graph, Plan plan, Config config) {
    auto fence = try_submit(std::move(graph), std::move(plan), config);
    if (!fence) throw std::runtime_error("E_QUEUE_FULL: asynchronous runtime capacity exhausted");
    return std::move(*fence);
}
void AsyncRuntime::shutdown(ShutdownMode mode) {
    {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        if (shared_->worker_id == std::this_thread::get_id())
            throw std::logic_error("E_SELF_JOIN: shutdown must be called outside the worker");
        shared_->closed = true;
        if (mode == ShutdownMode::Cancel) {
            for (auto& job : shared_->queue) {
                job->cancel.store(true);
                async_detail::cancelled(*job);
            }
            shared_->queue.clear();
            if (shared_->running) shared_->running->cancel.store(true);
        }
        shared_->ready.notify_one();
    }
    // A second shutdown caller may escalate Drain to Cancel while the first
    // joins. Serialize only thread::join, never hold the queue mutex across it.
    std::lock_guard<std::mutex> joining(join_mutex_);
    if (worker_.joinable()) worker_.join();
}
}
