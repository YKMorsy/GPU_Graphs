
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <stdio.h>

dpct::err0 addWithCuda(int *c, const int *a, const int *b, unsigned int size);

void addKernel(int *c, const int *a, const int *b,
               const sycl::nd_item<3> &item_ct1)
{
    int i = item_ct1.get_local_id(2);
    c[i] = a[i] + b[i];
}

int main() try {
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    dpct::err0 cudaStatus = addWithCuda(c, a, b, arraySize);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = DPCT_CHECK_ERROR(dpct::get_current_device().reset());

    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Helper function for using CUDA to add vectors in parallel.
dpct::err0 addWithCuda(int *c, const int *a, const int *b,
                       unsigned int size) try {
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    dpct::err0 cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    /*
    DPCT1093:36: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    cudaStatus = DPCT_CHECK_ERROR(dpct::select_device(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = DPCT_CHECK_ERROR(
        dev_c = sycl::malloc_device<int>(size, dpct::get_in_order_queue()));

    cudaStatus = DPCT_CHECK_ERROR(
        dev_a = sycl::malloc_device<int>(size, dpct::get_in_order_queue()));

    cudaStatus = DPCT_CHECK_ERROR(
        dev_b = sycl::malloc_device<int>(size, dpct::get_in_order_queue()));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(dev_a, a, size * sizeof(int)).wait());

    cudaStatus = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(dev_b, b, size * sizeof(int)).wait());

    // Launch a kernel on the GPU with one thread for each element.
    /*
    DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, size),
                          sycl::range<3>(1, 1, size)),
        [=](sycl::nd_item<3> item_ct1) {
            addKernel(dev_c, dev_a, dev_b, item_ct1);
        });

    // Check for any errors launching the kernel
    /*
    DPCT1010:37: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaStatus = 0;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus =
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(c, dev_c, size * sizeof(int)).wait());

Error:
    DPCT_CHECK_ERROR(sycl::free(dev_c, dpct::get_in_order_queue()));
    sycl::free(dev_a, dpct::get_in_order_queue());
    sycl::free(dev_b, dpct::get_in_order_queue());

    return cudaStatus;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
