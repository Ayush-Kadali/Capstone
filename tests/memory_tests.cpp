#include "placementlens/core.hpp"
#include "placementlens/device.hpp"
#include "placementlens/kernels.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
int checks = 0;
void check(bool ok, const std::string& why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
void rejects(const std::function<void()>& action, const std::string& code) {
    try { action(); }
    catch (const std::exception& e) {
        check(std::string(e.what()).find(code) != std::string::npos, "wrong error: " + std::string(e.what()));
        return;
    }
    throw std::runtime_error("missing error: " + code);
}
pl::Graph parse(const std::string& text) { std::istringstream in(text); return pl::parse(in); }
void renumber(pl::Plan& plan) { for (std::size_t i=0; i<plan.commands.size(); ++i) plan.commands[i].id=i; }
std::size_t first(const pl::Plan& plan, pl::Kind kind) {
    for (std::size_t i=0; i<plan.commands.size(); ++i) if (plan.commands[i].kind==kind) return i;
    throw std::runtime_error("missing command in test fixture");
}
// Strict external backend spy: real ownership rules, software arithmetic.
// It lets failure/cancellation tests observe whether interpreter cleanup leaks
// or double-releases allocations without needing a physical accelerator.
class Spy final : public pl::DeviceBackend {
public:
    std::map<std::string, pl::Tensor> buffers;
    std::map<std::string, int> releases;
    bool fail_release = false, fail_compute = false;
    std::atomic<bool>* cancel_after_write = nullptr;
    std::string description() const override { return "test-only ownership spy"; }
    void allocate(const std::string& h, pl::Shape shape) override {
        if (buffers.count(h)) throw std::runtime_error("duplicate backend allocation");
        buffers.emplace(h, pl::Tensor{shape, std::vector<float>(shape.elements())});
    }
    void release(const std::string& h) override {
        if (fail_release) { fail_release=false; throw std::runtime_error("injected release failure"); }
        if (!buffers.erase(h)) throw std::runtime_error("duplicate backend release");
        ++releases[h];
    }
    void write(const std::string& h, const pl::Tensor& tensor) override {
        buffers.at(h)=tensor;
        if (cancel_after_write) cancel_after_write->store(true);
    }
    void read(const std::string& h, pl::Tensor& tensor) override { tensor=buffers.at(h); }
    void compute(const pl::Command& command) override {
        if (fail_compute) throw std::runtime_error("injected compute failure");
        std::vector<const pl::Tensor*> inputs;
        for (const auto& h : command.inputs) inputs.push_back(&buffers.at(h));
        pl::KernelStats stats;
        pl::compute_software(command.op, inputs, buffers.at(command.output), {}, stats);
    }
    void synchronize() override {}
    double last_elapsed_us() const override { return -1; }
};
void every_placement(const pl::Graph& graph, const pl::Config& config) {
    const auto plans=pl::enumerate(graph,config);
    check(plans.infeasible==0, "fixture all placements feasible");
    check(plans.plans.size()==(std::size_t{1}<<graph.nodes.size()), "all placement masks represented");
    for (const auto& plan : plans.plans) {
        const auto run=pl::execute(graph,plan,config);
        check(run.correct, "pooled placement numerical result");
        check(run.peak_device_reserved_bytes==plan.peak_device_bytes, "runtime pool equals planned reservation");
        check(run.software_buffer_reuses==plan.buffer_reuses, "runtime physically reuses planned slots");
        std::size_t syncs=0;
        double modeled=0;
        std::map<std::string,std::size_t> last_use;
        for (const auto& command:plan.commands) {
            for(const auto& h:command.inputs) last_use[h]=command.id;
            if(command.kind==pl::Kind::Compute || command.kind==pl::Kind::Copy) last_use[command.output]=command.id;
            if(command.kind==pl::Kind::Synchronize) ++syncs;
            modeled+=command.modeled_ms;
        }
        for(const auto& command:plan.commands) if(command.kind==pl::Kind::Free)
            check(last_use.at(command.output)<command.id,"shared input freed after its final use");
        check(syncs==plan.synchronizations,"sync count includes lifetime boundaries");
        check(std::abs(modeled-plan.modeled_ms)<1e-9,"all inserted sync costs charged");
        Spy spy;
        const auto external=pl::execute(graph,plan,config,&spy);
        check(external.correct && spy.buffers.empty(),"external backend freed exactly at end");
        check(external.software_buffer_reuses==0,"external release/reallocate is not reported as vector reuse");
        check(external.peak_device_reserved_bytes==plan.peak_live_device_bytes,"external logical live allocation peak");
        for(const auto& release:spy.releases) check(release.second==1,"each logical device handle released exactly once");
    }
}
}
int main() {
    try {
        const auto shared=parse("tensor A f32 2 2 1 2 3 4\ntensor B f32 2 2 4 3 2 1\n"
            "add first A B X\nadd second X A Y\nadd third X B Z\nadd fourth Y Z O\noutput O\n");
        const auto reduction=parse("tensor A f32 2 3 1 -2 3 4 5 -6\ntensor B f32 3 2 1 2 3 4 5 6\n"
            "matmul mm A B X\nrowsum reduce X Y\noutput Y\n");
        pl::Config config; config.reuse_buffers=true; config.relu_supported=true;
        every_placement(shared,config);
        every_placement(reduction,config);
        config.tile_size=2;
        every_placement(reduction,config);
        config.tile_size=0;

        std::string chain="tensor A f32 2 2 1 2 3 4\ntensor B f32 2 2 4 3 2 1\n";
        for(int i=0;i<6;++i) chain+="add n"+std::to_string(i)+" "+(i?"X"+std::to_string(i-1):"A")+" B X"+std::to_string(i)+"\n";
        chain+="output X5\n";
        const auto graph=parse(chain);
        const auto pooled=pl::choose(graph,config,"maximal");
        check(pooled.peak_device_bytes==48 && pooled.device_buffer_slots==3 && pooled.buffer_reuses==5,"chain uses three exact-size slots");
        check(pl::execute(graph,pooled,config).software_buffer_reuses==5,"five vector allocations actually reused");
        auto retained_config=config; retained_config.reuse_buffers=false;
        const auto retained=pl::choose(graph,retained_config,"maximal");
        check(retained.peak_device_bytes==128,"default retains eight device allocations");
        check(std::none_of(retained.commands.begin(),retained.commands.end(),[](const auto& c){return c.kind==pl::Kind::Free || c.buffer_slot!=pl::no_buffer_slot;}),"default command protocol remains unpooled");
        check(pl::execute(graph,retained,retained_config).correct,"default result remains correct");
        config.device_bytes=48;
        check(pl::execute(graph,pl::choose(graph,config,"maximal"),config).correct,"exact pool budget succeeds");
        config.device_bytes=49;
        check(pl::choose(graph,config,"maximal").peak_device_bytes==48,"budget plus one does not change slots");
        config.device_bytes=47;
        rejects([&]{(void)pl::lower(graph,std::vector<pl::Backend>(6,pl::Backend::Device),config);},"E_CAPABILITY");
        rejects([&]{(void)pl::execute(graph,pooled,config);},"E_MEMORY");

        config.device_bytes=72;
        const auto fragmented=pl::choose(reduction,config,"maximal");
        check(fragmented.peak_device_bytes==72 && fragmented.peak_live_device_bytes==64,"idle size classes count as fragmentation");
        check(pl::execute(reduction,fragmented,config).peak_device_reserved_bytes==72,"actual retained pool includes idle size classes");
        config.device_bytes=73;
        check(pl::choose(reduction,config,"maximal").peak_device_bytes==72,"fragmented pool plus-one budget");
        config.device_bytes=71;
        rejects([&]{(void)pl::lower(reduction,{pl::Backend::Device,pl::Backend::Device},config);},"E_MEMORY");
        rejects([&]{(void)pl::execute(reduction,fragmented,config);},"E_MEMORY");
        config.device_bytes=1024*1024;

        auto bad=pooled;
        auto free_index=first(bad,pl::Kind::Free);
        bad.commands[free_index].backend=pl::Backend::Cpu;
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_FREE");
        bad=pooled; bad.commands[free_index].shape={1,4};
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_FREE");
        bad=pooled; bad.commands[free_index].buffer_slot=pl::no_buffer_slot;
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_SLOT");
        bad=pooled; bad.commands[free_index].output="D:missing";
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_FREE");
        bad=pooled; bad.commands.insert(bad.commands.begin()+static_cast<std::ptrdiff_t>(free_index+1),bad.commands[free_index]); renumber(bad);
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_FREE");
        bad=pooled;
        auto second_alloc=std::find_if(bad.commands.begin()+1,bad.commands.end(),[](const auto& c){return c.kind==pl::Kind::Allocate;});
        second_alloc->buffer_slot=0;
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_SLOT");
        bad=pooled; bad.commands.erase(bad.commands.begin()+static_cast<std::ptrdiff_t>(free_index-1)); renumber(bad);
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_SYNC");
        bad=fragmented;
        for(auto& command:bad.commands) if(command.kind==pl::Kind::Allocate && command.output=="D:Y") command.buffer_slot=0;
        rejects([&]{(void)pl::execute(reduction,bad,config);},"E_SLOT");
        bad=pooled; bad.commands[first(bad,pl::Kind::Compute)].buffer_slot=0;
        rejects([&]{(void)pl::execute(graph,bad,config);},"E_SLOT");
        rejects([&]{(void)pl::execute(graph,pooled,retained_config);},"E_SLOT");

        // Dead device values and original host outputs have independent lifetimes.
        const auto original_output=parse("tensor A f32 1 1 2\ntensor B f32 1 1 3\nadd unused A B X\noutput A\n");
        const auto original_plan=pl::choose(original_output,config,"maximal");
        check(pl::execute(original_output,original_plan,config).output.values==std::vector<float>{2},"source output survives all device frees");

        Spy spy; spy.allocate("D:foreign",{1,1});
        for(int i=0;i<2;++i) check(pl::execute(graph,pooled,config,&spy).correct,"backend reused across full runs");
        check(spy.buffers.size()==1 && spy.buffers.count("D:foreign"),"foreign backend allocation untouched");
        for(const auto& release:spy.releases) check(release.second==2,"once per logical handle per run");
        spy.fail_compute=true;
        rejects([&]{(void)pl::execute(graph,pooled,config,&spy);},"injected compute failure");
        check(spy.buffers.size()==1,"compute failure cleanup preserves only foreign allocation");
        spy.fail_compute=false; spy.fail_release=true;
        rejects([&]{(void)pl::execute(graph,pooled,config,&spy);},"injected release failure");
        check(spy.buffers.size()==1,"release failure retries remaining ownership in destructor");
        std::atomic<bool> cancel{false}; spy.cancel_after_write=&cancel;
        rejects([&]{(void)pl::execute(graph,pooled,config,&spy,&cancel);},"E_CANCELLED");
        check(spy.buffers.size()==1,"cancel after upload releases current run allocations");
        spy.cancel_after_write=nullptr; cancel.store(false);
        check(pl::execute(graph,pooled,config,&spy,&cancel).correct,"same backend recovers after failure/cancel");
        std::cout<<"PASS: "<<checks<<" memory lifetime/pool assertions\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
