#include <iostream>
#include <CL/sycl.hpp>

// work item = thread
// work group = block
// nd range = grid

int main(int argc, char *argv[])
{
    // define and select device for queue
    cl::sycl::queue gpuQueue{cl::sycl::gpu_selector_v}; 

    int size = 10000;
    int *host_distance = (int *)malloc(size * sizeof(int));
    // int *host_distance = sycl::malloc_host<int>(size, gpuQueue);

    int *device_distance = cl::sycl::malloc_device<int>(size, gpuQueue);

    int max_group_size = gpuQueue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

    std::cout << "Max group size: " << max_group_size << "\n";
   
    // // handler
    // gpuQueue.submit
    // (
    //     [&] (cl::sycl::handler& cgh)
    //     {
    //         cgh.parallel_for
    //         (
    //             cl::sycl::nd_range<1>(size, max_group_size),
    //             [=] (cl::sycl::nd_item<1> item)
    //             {
    //                 int i = item.get_global_id(0);
    //                 device_distance[i] = 2;
    //             }
    //         );
        
    //     }
    // ).wait();

    // // copy back to host
    // gpuQueue.memcpy(host_distance, device_distance, size * sizeof(int)).wait();

    // std::cout << host_distance[5] << " " << host_distance[size-1] << "\n";
}