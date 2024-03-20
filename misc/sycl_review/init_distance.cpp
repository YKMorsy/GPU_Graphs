#include <iostream>
#include <CL/sycl.hpp>

int main(int argc, char *argv[]) {
    // define and select device for queue
    cl::sycl::queue gpuQueue{ cl::sycl::gpu_selector{} };

    int size = 10000;
    int *host_distance = (int *)malloc(size * sizeof(int));
    int *device_distance = cl::sycl::malloc_device<int>(size, gpuQueue);

    // Query max work group size
    int max_group_size = gpuQueue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

    std::cout << "Max group size: " << max_group_size << "\n";

    // Determine appropriate work group size
    int group_size = std::min(max_group_size, size);

    // handler
    gpuQueue.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for(
            cl::sycl::range<1>(size),
            cl::sycl::id<1>(group_size),
            [=](cl::sycl::item<1> item) {
                int i = item.get_global_id(0);
                if (i < size)
                    device_distance[i] = 2;
            });
    }).wait();

    // copy back to host
    gpuQueue.memcpy(host_distance, device_distance, size * sizeof(int)).wait();

    std::cout << host_distance[5] << " " << host_distance[size - 1] << "\n";

    free(host_distance);
    return 0;
}
