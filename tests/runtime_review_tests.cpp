#include "placementlens/async.hpp"
#include "placementlens/device.hpp"
#include "placementlens/kernels.hpp"
#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace {
using namespace std::chrono_literals;
int checks = 0;
void check(bool ok, const char* reason) {
    ++checks;
    if (!ok) throw std::runtime_error(reason);
}
template <typename F> void rejects(F&& action, const char* code) {
    bool rejected = false;
    try { action(); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find(code) != std::string::npos; }
    check(rejected, code);
}
pl::Graph parse(const std::string& text) { std::istringstream input(text); return pl::parse(input); }
void renumber(pl::Plan& plan) {
    for (std::size_t i=0;i<plan.commands.size();++i) plan.commands[i].id=i;
}
std::size_t command(const pl::Plan& plan, pl::Kind kind, const std::string& output = {}) {
    for (std::size_t i=0;i<plan.commands.size();++i)
        if (plan.commands[i].kind==kind && (output.empty() || plan.commands[i].output==output)) return i;
    throw std::runtime_error("review fixture command missing");
}
pl::Graph chain() {
    return parse("tensor A f32 1 2 2 -3\nadd first A A X\nadd second X A Y\nadd third Y A Z\noutput Z\n");
}
// This spy checks ownership and fault behavior; its arithmetic deliberately uses
// the same helper. Independent numerical evidence comes from hand outputs below.
class FaultDevice final : public pl::DeviceBackend {
public:
    std::map<std::string,pl::Tensor> buffers;
    std::map<std::string,int> release_attempts;
    std::string fail;
    bool fail_once = true, persist_release_failure = false;
    std::size_t allocations = 0, fail_allocation_at = 0;
    std::atomic<bool>* cancel_after_compute = nullptr;
    void inject(const char* operation) {
        if (fail==operation) {
            if (fail_once) fail.clear();
            throw std::runtime_error(std::string("injected ")+operation);
        }
    }
    std::string description() const override { return "review-only fault backend"; }
    void allocate(const std::string& handle, pl::Shape shape) override {
        if (++allocations == fail_allocation_at) throw std::runtime_error("injected allocate");
        if (buffers.count(handle)) throw std::runtime_error("duplicate allocation");
        buffers.emplace(handle,pl::Tensor{shape,std::vector<float>(shape.elements())});
    }
    void release(const std::string& handle) override {
        ++release_attempts[handle];
        if (persist_release_failure) throw std::runtime_error("persistent release failure");
        inject("release");
        if (buffers.erase(handle)!=1) throw std::runtime_error("duplicate release");
    }
    void write(const std::string& handle,const pl::Tensor& tensor) override {
        inject("write"); buffers.at(handle)=tensor;
    }
    void read(const std::string& handle,pl::Tensor& tensor) override {
        inject("read"); tensor=buffers.at(handle);
        if (fail=="shape") tensor.shape={tensor.shape.cols,tensor.shape.rows};
    }
    void compute(const pl::Command& descriptor) override {
        inject("compute");
        std::vector<const pl::Tensor*> inputs;
        for (const auto& input:descriptor.inputs) inputs.push_back(&buffers.at(input));
        pl::KernelStats stats;
        pl::compute_software(descriptor.op,inputs,buffers.at(descriptor.output),{},stats);
        if (cancel_after_compute) cancel_after_compute->store(true);
    }
    void synchronize() override { inject("sync"); }
    double last_elapsed_us() const override { return -1; }
};
void protocol_negatives() {
    const auto graph=chain(); pl::Config config; config.reuse_buffers=true;
    const auto good=pl::choose(graph,config,"maximal");
    const auto free_index=command(good,pl::Kind::Free);
    auto bad=good;
    bad.commands[0].buffer_slot=pl::no_buffer_slot;
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_SLOT");
    bad=good; bad.commands[free_index].inputs={"D:A"};
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_FREE");
    bad=good;
    auto early=bad.commands[free_index];
    const auto allocation=command(bad,pl::Kind::Allocate,early.output);
    bad.commands.insert(bad.commands.begin()+static_cast<std::ptrdiff_t>(allocation+1),early); renumber(bad);
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_FREE");
    bad=good;
    auto resurrect=bad.commands[command(bad,pl::Kind::Allocate,bad.commands[free_index].output)];
    bad.commands.insert(bad.commands.begin()+static_cast<std::ptrdiff_t>(free_index+1),resurrect); renumber(bad);
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_ALLOC");
    bad=good;
    const auto freed=bad.commands[free_index].output;
    auto next_compute=std::find_if(bad.commands.begin()+static_cast<std::ptrdiff_t>(free_index+1),bad.commands.end(),[](const auto& c){return c.kind==pl::Kind::Compute;});
    check(next_compute!=bad.commands.end(),"fixture has a use after first lifetime boundary");
    next_compute->inputs[0]=freed;
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_BUFFER");
    bad=good;
    const auto host=std::find_if(bad.commands.begin(),bad.commands.end(),[](const auto& c){return c.kind==pl::Kind::Allocate && c.backend==pl::Backend::Cpu;});
    check(host!=bad.commands.end(),"fixture has host output allocation");
    host->buffer_slot=0;
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_SLOT");
    bad=good; bad.commands[free_index].version=pl::command_version+1;
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_VERSION");
    bad=good; bad.commands[free_index].kind=static_cast<pl::Kind>(99); bad.commands[free_index].buffer_slot=pl::no_buffer_slot;
    rejects([&]{(void)pl::execute(graph,bad,config);},"E_COMMAND");
    auto constrained=config; constrained.device_bytes=good.peak_device_bytes-1;
    rejects([&]{(void)pl::execute(graph,good,constrained);},"E_MEMORY");
    constrained=config; constrained.tile_size=2; constrained.local_bytes=47;
    FaultDevice device;
    rejects([&]{(void)pl::execute(graph,good,constrained,&device);},"E_CONFIG");
    check(device.allocations==0,"invalid local configuration fails before backend allocation");
}
void external_failure_cleanup() {
    const auto graph=chain(); pl::Config config;
    const auto retained=pl::choose(graph,config,"maximal");
    for (const auto* failure: {"write","compute","read","sync","release","shape"}) {
        FaultDevice device; device.fail=failure;
        rejects([&]{(void)pl::execute(graph,retained,config,&device);},std::string(failure)=="shape"?"E_VALUE":(std::string("injected ")+failure).c_str());
        check(device.buffers.empty(),"all resources released after backend fault");
        device.fail.clear();
        check(pl::execute(graph,retained,config,&device).correct,"same backend recovers after injected fault");
        check(device.buffers.empty(),"recovery run also cleans resources");
    }
    FaultDevice allocation_failure; allocation_failure.fail_allocation_at=2;
    rejects([&]{(void)pl::execute(graph,retained,config,&allocation_failure);},"injected allocate");
    check(allocation_failure.buffers.empty() && allocation_failure.release_attempts.size()==1,"partial allocation rollback releases only successful allocations");
    FaultDevice failed_cleanup; failed_cleanup.fail="compute"; failed_cleanup.persist_release_failure=true;
    rejects([&]{(void)pl::execute(graph,retained,config,&failed_cleanup);},"injected compute");
    check(failed_cleanup.release_attempts.size()==failed_cleanup.buffers.size(),"cleanup attempts every owned handle even when releases fail");
    for (const auto& entry:failed_cleanup.release_attempts) check(entry.second==1,"failed cleanup visits each allocation once");
    failed_cleanup.persist_release_failure=false;
    while (!failed_cleanup.buffers.empty()) failed_cleanup.release(failed_cleanup.buffers.begin()->first);
    std::atomic<bool> cancel{false}; FaultDevice cancellation;
    cancellation.cancel_after_compute=&cancel;
    rejects([&]{(void)pl::execute(graph,retained,config,&cancellation,&cancel);},"E_CANCELLED");
    check(cancellation.buffers.empty(),"postcompute cancellation synchronizes and frees all resources");
    config.reuse_buffers=true;
    const auto pooled=pl::choose(graph,config,"maximal");
    FaultDevice pooled_fault; pooled_fault.fail="release";
    rejects([&]{(void)pl::execute(graph,pooled,config,&pooled_fault);},"injected release");
    check(pooled_fault.buffers.empty(),"failed explicit FREE is retried without losing owned handles");
}
void initial_residency_contracts() {
    const auto graph=chain();
    for (const bool pooled:{false,true}) {
        pl::Config config; config.reuse_buffers=pooled; config.mirrored_inputs={"A"};
        const auto search=pl::enumerate(graph,config);
        check(search.plans.size()==8,"initial residency preserves all feasible placement masks");
        for (const auto& plan:search.plans) {
            const auto run=pl::execute(graph,plan,config);
            check(run.correct && run.output.values==std::vector<float>({8,-12}),"prepared input preserves hand result for mixed placements");
            check(run.preparation_h2d_bytes==8,"preparation traffic is counted separately");
            check(std::none_of(plan.commands.begin(),plan.commands.end(),[](const auto& descriptor){return descriptor.kind==pl::Kind::Copy && descriptor.output=="D:A";}),"mirrored shared input suppresses its in-plan upload");
            check(run.peak_device_reserved_bytes==plan.peak_device_bytes,"prepared pool counted in reserved memory");
            FaultDevice device;
            check(pl::execute(graph,plan,config,&device).correct && device.buffers.empty(),"prepared backend allocation included in scoped cleanup");
        }
        const auto good=pl::choose(graph,config,"maximal");
        auto bad=good; bad.initial_buffers[0].shape={2,1};
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_RESIDENCY");
        bad=good; bad.initial_buffers[0].output="D:X";
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_RESIDENCY");
        bad=good; bad.initial_buffers[0].id=1;
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_RESIDENCY");
        bad=good; bad.initial_buffers[0].version=99;
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_RESIDENCY");
        bad=good; bad.initial_buffers[0].inputs={"H:A"};
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_RESIDENCY");
        bad=good; bad.initial_buffers[0].buffer_slot=pooled?pl::no_buffer_slot:0;
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_SLOT");
        bad=good; bad.initial_buffers.clear();
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_RESIDENCY");
        FaultDevice failure; failure.fail="write";
        rejects([&]{(void)pl::execute(graph,good,config,&failure);},"injected write");
        check(failure.buffers.empty(),"preparation failure releases its allocation");
        auto small=config;small.device_bytes=7;
        rejects([&]{(void)pl::execute(graph,good,small);},"E_MEMORY");
        rejects([&]{(void)pl::choose(graph,small,"cost");},"E_MEMORY");
        auto mismatch=config;mismatch.mirrored_inputs={};
        rejects([&]{(void)pl::execute(graph,good,mismatch);},"E_RESIDENCY");
        mismatch=config;mismatch.mirrored_inputs={"A","A"};
        rejects([&]{(void)pl::execute(graph,good,mismatch);},"E_RESIDENCY");
    }
    // Sorting the user list must produce stable preparation descriptors.
    const auto pair=parse("tensor A f32 1 1 2\ntensor B f32 1 1 3\nadd sum A B Y\noutput Y\n");
    pl::Config config;config.mirrored_inputs={"B","A"};config.reuse_buffers=true;
    const auto plan=pl::choose(pair,config,"maximal");
    check(plan.initial_buffers[0].output=="D:A" && plan.initial_buffers[1].output=="D:B","preparation IDs are independent of configuration input ordering");
    check(pl::execute(pair,plan,config).output.values==std::vector<float>{5},"several prepared sources execute correctly");
    FaultDevice second_failure;second_failure.fail_allocation_at=2;
    rejects([&]{(void)pl::execute(pair,plan,config,&second_failure);},"injected allocate");
    check(second_failure.buffers.empty() && second_failure.release_attempts.size()==1,"second preparation allocation failure cleans first prepared source");
}
void integrated_numeric_and_async() {
    const auto graph=parse("tensor A f32 2 3 1 -2 3 4 5 -6\ntensor B f32 3 2 1 2 3 4 5 6\n"
        "matmul mm A B X\nrelu activation X R\nrowsum sum R S\nadd twice S S Y\nrowsum side X D\nadd final Y D O\noutput O\n");
    pl::Config config; config.relu_supported=true; config.reuse_buffers=true;
    check(pl::reference(graph).values==std::vector<float>({66,-19}),"independent hand-computed branch/reduction reference");
    check(pl::fuse_matmul_relu(graph,config).graph.nodes.size()==6,"shared MatMul intermediate blocks fusion");
    for (const auto tile: {std::size_t{0},std::size_t{2},std::size_t{5}}) {
        config.tile_size=tile;
        const auto search=pl::enumerate(graph,config);
        check(search.plans.size()==64 && search.infeasible==0,"all six-node placements covered");
        pl::AsyncRuntime runtime(search.plans.size());
        std::vector<pl::Fence> fences;
        for (const auto& plan:search.plans) fences.push_back(runtime.submit(graph,plan,config));
        runtime.shutdown(pl::ShutdownMode::Drain);
        for (std::size_t i=0;i<fences.size();++i) {
            const auto run=fences[i].get().run;
            check(run.correct && run.output.values==std::vector<float>({66,-19}),"async pooled mixed placement agrees with hand output");
            const bool tiled=tile!=0 && search.plans[i].placement[0]==pl::Backend::Device;
            check((run.tile_count>0)==tiled,"only device MatMul contributes tile counters");
            check(run.peak_device_reserved_bytes==search.plans[i].peak_device_bytes,"queue preserves independent pool accounting");
        }
    }
    // Cancellation-sensitive arithmetic must not turn a failed oracle comparison
    // into a successful result just because execution was queued or pooled.
    const auto cancellation_graph=parse("tensor X f32 1 3 100000000 1 -100000000\nrowsum reduce X Y\noutput Y\n");
    config.tile_size=2;
    pl::AsyncRuntime runtime(2);
    for (const auto& plan:pl::enumerate(cancellation_graph,config).plans) {
        const auto run=runtime.submit(cancellation_graph,plan,config).get().run;
        check(!run.correct && run.max_abs_error==1 && run.output.values==std::vector<float>{0},"f32 reduction mismatch remains visible through async completion");
    }
    const auto overflow=parse("tensor A f32 1 1 3e38\ntensor B f32 1 1 -2\nmatmul mm A B X\nrelu activation X Y\noutput Y\n");
    const auto fused=pl::fuse_matmul_relu(overflow,config).graph;
    check(fused.nodes.size()==1,"overflow fixture exercises fused path");
    for (const auto& plan:pl::enumerate(fused,config).plans)
        rejects([&]{(void)runtime.submit(overflow,plan,config).get();},"E_NUMERIC");
    const auto healthy=chain();
    check(runtime.submit(healthy,pl::choose(healthy,config,"maximal"),config).get().run.correct,"async worker survives fused numerical failures");
}
void queued_failure_and_cancellation_precedence() {
    std::mutex mutex; std::condition_variable changed;
    bool entered=false,released=false;
    pl::AsyncRuntime runtime(3,[&](const pl::Graph& source,const pl::Plan& plan,const pl::Config& config,const std::atomic<bool>& cancel) {
        if (plan.policy=="blocked") {
            std::unique_lock<std::mutex> lock(mutex); entered=true; changed.notify_all();
            if (!changed.wait_for(lock,10s,[&]{return released;})) throw std::runtime_error("test watchdog");
            throw std::runtime_error("runner error after cancellation");
        }
        return pl::execute(source,plan,config,nullptr,&cancel);
    });
    struct Release { std::mutex& mutex; std::condition_variable& changed; bool& released; ~Release(){std::lock_guard<std::mutex> lock(mutex);released=true;changed.notify_all();} } release{mutex,changed,released};
    const auto graph=chain(); pl::Config config; config.reuse_buffers=true;
    auto blocked_plan=pl::choose(graph,config,"maximal"); blocked_plan.policy="blocked";
    auto first=runtime.submit(graph,blocked_plan,config);
    { std::unique_lock<std::mutex> lock(mutex); check(changed.wait_for(lock,10s,[&]{return entered;}),"worker reached deterministic gate"); }
    auto invalid=graph; invalid.inputs.at("A").values.clear();
    auto queued=runtime.submit(invalid,pl::choose(graph,config,"maximal"),config);
    auto healthy=runtime.submit(graph,pl::choose(graph,config,"maximal"),config);
    check(queued.state()==pl::JobState::Queued,"malformed job is queued before its validation failure");
    check(first.cancel(),"running cancellation accepted before runner error");
    { std::lock_guard<std::mutex> lock(mutex);released=true;changed.notify_all(); }
    rejects([&]{(void)first.get();},"E_CANCELLED");
    rejects([&]{(void)queued.get();},"E_VALUE");
    check(queued.state()==pl::JobState::Failed,"queued malformed job fails after worker starts it");
    check(healthy.get().run.correct,"later accepted work survives queued validation failure");
}
}
int main() {
    try {
        protocol_negatives(); external_failure_cleanup(); initial_residency_contracts(); integrated_numeric_and_async(); queued_failure_and_cancellation_precedence();
        std::cout<<"PASS "<<checks<<" runtime integration review assertions\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
