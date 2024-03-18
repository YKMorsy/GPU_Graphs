#include <CL/sycl.hpp>
using namespace cl::sycl;
class Add;

// work item = thread
// work group = block
// nd range = grid

struct add
{
    // alias to read and write data from 1-d global buffer
    using read_accessor_t = accessor<float, 1, access::mode::read, access::target::global_buffer>
    using write_accessor_t = accessor<float, 1, access::mode::write, access::target::global_buffer>

    read_accessor_t inA_, inB_;
    write_accessor_t outC_;

    void operator()(id<1> i)
    {
        outC_[i] = inA_[i] + inB_[i];
    }
};

int main(int argc, char *argv[])
{
    // intialize vectors for inputs and outputs
    std::vector<float> dA, dB, dC;

    // define and select device for queue
    queue gpuQueue{gpu_selector{}}; 

    {

    // create buffers for inputs and outputs to manage 
    // data accross host applciation and device
    buffer<float, 1> bufA(dA.data(), range<1>(dA.size()));
    buffer<float, 1> bufB(dB.data(), range<1>(dB.size()));
    buffer<float, 1> bufC(dC.data(), range<1>(dC.size()));

    // create command group
    gpuQueue.submit([&](handler &cgh)
    {
        // create accessors for the buffers
        auto inA = bufA.get_access<access::mode::read>(cgh);
        auto inB = bufB.get_access<access::mode::read>(cgh);
        auto outC = bufC.get_access<access::mode::write>(cgh);

        // define kernel function
        // cgh.parallel_for<add>(range<1>(dA.size()), // range to invoke kernel function over
        //                       [=](id<1> i)
        //                       {
        //                       outC[i] = inA[i] + inB[i];
        //                       } // lambda of kernel function
        //                       );

        cgh.parallel_for(range<1>(dA.size()), // range to invoke kernel function over
                              add{inA, inB, outC}
                              ); // id i is global id

    });

    }
}