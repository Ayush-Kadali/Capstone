#include "placementlens/cnn.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
int checks=0;
void check(bool yes,const char* why) { ++checks;if(!yes)throw std::runtime_error(why); }
void rejects(const std::function<void()>& fn,const char* code) {
    bool rejected=false;
    try {fn();}catch(const std::exception& e){rejected=std::string(e.what()).find(code)!=std::string::npos;}
    check(rejected,code);
}
using P=std::vector<std::size_t>;
void values(pl::Op op,const std::vector<const pl::Tensor*>& inputs,const P& p,
            pl::Shape shape,const std::vector<float>& expected) {
    pl::Tensor out{shape,std::vector<float>(shape.elements(),-123)};
    const auto* data=out.values.data();
    pl::compute_cnn(op,inputs,out,p);
    check(out.values.data()==data,"CNN must preserve preallocated output storage");
    check(out.values.size()==expected.size(),"manual expected size");
    for(std::size_t i=0;i<expected.size();++i)
        check(std::abs(out.values[i]-expected[i])<1e-6,"CNN differs from manually computed fixture");
    const auto ref=pl::reference_cnn(op,inputs,p);
    check(ref.shape==shape,"reference shape");
    for(std::size_t i=0;i<expected.size();++i)
        check(std::abs(ref.values[i]-expected[i])<1e-6,"double reference differs from manual fixture");
}
void convolutions() {
    pl::Tensor x{{1,9},{1,2,3,4,5,6,7,8,9}},w{{1,4},{1,0,0,-1}};
    values(pl::Op::Conv2D,{&x,&w},{1,1,3,3,1,2,2,1,1,0,0,0,0,1,1,1},{1,4},{-4,-4,-4,-4});
    w.values={1,1,1,1};
    values(pl::Op::Conv2D,{&x,&w},{1,1,3,3,1,2,2,2,2,1,1,0,0,1,1,1},{1,4},{1,5,11,28});
    values(pl::Op::Conv2D,{&x,&w},{1,1,3,3,1,2,2,1,1,0,0,0,0,1,2,2},{1,1},{20});
    pl::Tensor grouped{{1,4},{1,2,10,20}},gw{{4,2},{1,0,0,1,1,1,1,-1}};
    values(pl::Op::Conv2D,{&grouped,&gw},{1,4,1,1,4,1,1,1,1,0,0,0,0,2,1,1},{1,4},{1,2,30,-10});
    pl::Tensor depth{{1,4},{1,2,10,20}},dw{{4,1},{2,3,-1,.5}};
    values(pl::Op::Conv2D,{&depth,&dw},{1,2,1,2,4,1,1,1,1,0,0,0,0,2,1,1},{1,8},{2,4,3,6,-10,-20,5,10});
    pl::Tensor batches{{2,4},{1,2,3,4,-1,-2,-3,-4}},bw{{1,4},{1,1,1,1}};
    values(pl::Op::Conv2D,{&batches,&bw},{2,1,2,2,1,2,2,1,1,0,0,0,0,1,1,1},{2,1},{10,-10});
}
void pooling_and_reshape() {
    pl::Tensor x{{1,4},{1,2,3,4}};
    P p{1,1,2,2,2,2,1,1,1,1,0,0,0};
    values(pl::Op::MaxPool2D,{&x},p,{1,4},{1,2,3,4});
    values(pl::Op::AveragePool2D,{&x},p,{1,4},{1,1.5,2,2.5});
    p[12]=1;
    values(pl::Op::AveragePool2D,{&x},p,{1,4},{.25,.75,1,2.5});
    pl::Tensor negative{{1,4},{-1,-2,-3,-4}};
    p[12]=0;
    values(pl::Op::MaxPool2D,{&negative},p,{1,4},{-1,-1,-1,-1});
    pl::Tensor multi{{2,8},{1,2,3,4,10,20,30,40,-1,-2,-3,-4,-10,-20,-30,-40}};
    values(pl::Op::GlobalAveragePool,{&multi},{2,2,2,2},{2,2},{2.5,25,-2.5,-25});
    values(pl::Op::Reshape,{&multi},{4,4},{4,4},multi.values);
}
void failures() {
    pl::Tensor x{{1,4},{1,2,3,4}},w{{1,4},{1,1,1,1}},out{{1,1},{0}};
    const P good{1,1,2,2,1,2,2,1,1,0,0,0,0,1,1,1};
    rejects([&]{pl::cnn_shape(pl::Op::Add,{{1,4}},{});},"E_OPCODE");
    rejects([&]{pl::cnn_shape(pl::Op::Conv2D,{{1,4}},good);},"E_ARITY");
    auto bad=good;bad.pop_back();
    rejects([&]{pl::cnn_shape(pl::Op::Conv2D,{{1,4},{1,4}},bad);},"E_ATTRIBUTE");
    for(auto index:{0U,1U,2U,3U,4U,5U,6U,7U,8U,13U,14U,15U}) {
        bad=good;bad[index]=0;
        rejects([&]{pl::cnn_shape(pl::Op::Conv2D,{{1,4},{1,4}},bad);},"E_SHAPE");
    }
    bad=good;bad[13]=2;
    rejects([&]{pl::cnn_shape(pl::Op::Conv2D,{{1,4},{1,4}},bad);},"E_SHAPE");
    bad=good;bad[9]=2;
    rejects([&]{pl::cnn_shape(pl::Op::Conv2D,{{1,4},{1,4}},bad);},"E_SHAPE");
    bad=good;bad[0]=pl::max_elements+1;
    rejects([&]{pl::cnn_shape(pl::Op::Conv2D,{{1,4},{1,4}},bad);},"E_SHAPE");
    rejects([&]{pl::cnn_shape(pl::Op::Reshape,{{1,4}},{3,2});},"E_SHAPE");
    rejects([&]{pl::cnn_shape(pl::Op::MaxPool2D,{{1,4}},{1,1,2,2,2,2,1,1,0,0,0,0,1});},"E_ATTRIBUTE");
    rejects([&]{pl::cnn_shape(pl::Op::AveragePool2D,{{1,1}},
        {1,1,1,1,65536,65536,1,1,65535,65535,0,0,1});},"E_SHAPE");
    bad=good;bad[14]=65536;
    rejects([&]{pl::cnn_shape(pl::Op::Conv2D,{{1,4},{1,4}},bad);},"E_SHAPE");
    rejects([&]{pl::compute_cnn(pl::Op::Conv2D,{nullptr,&w},out,good);},"E_BUFFER");
    rejects([&]{pl::compute_cnn(pl::Op::Reshape,{&x},x,{1,4});},"E_BUFFER");
    auto short_input=x;short_input.values.pop_back();
    rejects([&]{pl::compute_cnn(pl::Op::Conv2D,{&short_input,&w},out,good);},"E_VALUE");
    out.values.clear();
    rejects([&]{pl::compute_cnn(pl::Op::Conv2D,{&x,&w},out,good);},"E_VALUE");out.values={0};
    auto invalid=x;invalid.values[0]=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{pl::compute_cnn(pl::Op::Conv2D,{&invalid,&w},out,good);},"E_VALUE");
    invalid.values[0]=std::numeric_limits<float>::infinity();
    rejects([&]{pl::reference_cnn(pl::Op::Conv2D,{&invalid,&w},good);},"E_VALUE");
    auto huge=x;huge.values.assign(4,3e38f);
    rejects([&]{pl::compute_cnn(pl::Op::Conv2D,{&huge,&w},out,good);},"E_NUMERIC");
    std::atomic<bool> stop{true};
    rejects([&]{pl::compute_cnn(pl::Op::Conv2D,{&x,&w},out,good,&stop);},"E_CANCELLED");
    rejects([&]{pl::reference_cnn(pl::Op::Conv2D,{&x,&w},good,&stop);},"E_CANCELLED");
    pl::Tensor reshaped{{2,2},std::vector<float>(4)};
    rejects([&]{pl::compute_cnn(pl::Op::Reshape,{&x},reshaped,{2,2},&stop);},"E_CANCELLED");
}
void planner_and_command_validation() {
    std::istringstream stream("tensor X f32 1 4 1 2 3 4\ntensor W f32 1 4 1 1 1 1\n"
        "conv2d conv X W Y 1 1 2 2 1 2 2 1 1 0 0 0 0 1 1 1\noutput Y\n");
    const auto graph=pl::parse(stream);
    pl::Config config;config.reuse_buffers=true;
    check(!pl::capability(graph.nodes[0],graph,config).accepted,"CNN defaults to CPU fallback");
    config.cnn_supported=true;
    check(pl::capability(graph.nodes[0],graph,config).accepted,"CNN capability opt-in");
    for(const auto& plan:pl::enumerate(graph,config).plans) {
        const auto run=pl::execute(graph,plan,config);
        check(run.correct && run.output.values==std::vector<float>{10},"CPU/device CNN manual result");
    }
    const auto plan=pl::choose(graph,config,"maximal");
    for(std::size_t index=0;index<plan.commands.size();++index) {
        auto bad=plan;
        auto& command=bad.commands[index];
        if(command.kind==pl::Kind::Compute) command.parameters.pop_back();
        else command.parameters={1};
        rejects([&]{pl::execute(graph,bad,config);},"E_ATTRIBUTE");
    }
    config.mirrored_inputs={"X"};
    auto resident=pl::choose(graph,config,"maximal");
    resident.initial_buffers[0].parameters={1};
    rejects([&]{pl::execute(graph,resident,config);},"E_ATTRIBUTE");
    auto invalid_graph=graph;invalid_graph.nodes[0].parameters.clear();
    rejects([&]{pl::validate(invalid_graph);},"E_ATTRIBUTE");
    auto disabled=config;disabled.cnn_supported=false;disabled.mirrored_inputs.clear();
    rejects([&]{pl::execute(graph,plan,disabled);},"E_CAPABILITY");
}
}
int main() {
    try {convolutions();pooling_and_reshape();failures();planner_and_command_validation();
        std::cout<<"CNN independent contracts: "<<checks<<" checks passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<"CNN test failed after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}
}
