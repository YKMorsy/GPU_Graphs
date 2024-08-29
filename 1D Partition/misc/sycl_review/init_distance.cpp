#include <iostream>
#include <CL/sycl.hpp>

int main() {
    // Define and select device for queue
    cl::sycl::queue gpuQueue{ cl::sycl::gpu_selector_v };

    // Define size
    const int size = 10000;

    // Allocate memory on host and device
    int *host_distance = static_cast<int*>(malloc(size * sizeof(int)));
    int *device_distance = cl::sycl::malloc_device<int>(size, gpuQueue);

    std::cout << "Local Memory Size: "
        << gpuQueue.get_device().get_info<cl::sycl::info::device::local_mem_size>()
        << std::endl;

    // Query max work group size
    int max_group_size = gpuQueue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int num_blocks = (size + max_group_size - 1) / max_group_size;
    std::cout << "Max group size: " << max_group_size << "\n";

    // Create nd_range with work-group size equal to max_group_size
    // cl::sycl::nd_range<1> range(cl::sycl::range<1>(size), cl::sycl::range<1>(max_group_size));

    gpuQueue.submit([&](cl::sycl::handler &cgh) 
    {
        // int *distance_ptr = device_distance;

        cgh.parallel_for
        (
            cl::sycl::nd_range<1>(num_blocks*max_group_size, max_group_size),
            [=] (cl::sycl::nd_item<1> item) 
            {
                int i = item.get_global_id(0);
                if (i < size)
                {
                    device_distance[i] = 2;
                }
            }
        );
    }).wait();

    // Copy back to host
    gpuQueue.memcpy(host_distance, device_distance, size * sizeof(int)).wait();

    std::cout << host_distance[5] << " " << host_distance[size - 1] << "\n";

    // Free memory
    free(host_distance);
    cl::sycl::free(device_distance, gpuQueue);
    
    return 0;
}
