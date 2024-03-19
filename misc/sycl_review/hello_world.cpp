#include <iostream>
#include <CL/sycl.hpp>

class vector_addition;

int main(int, char**) {

   // define host variables
   cl::sycl::float4 a = { 1.0, 2.0, 3.0, 4.0 };
   cl::sycl::float4 b = { 4.0, 3.0, 2.0, 1.0 };
   cl::sycl::float4 c = { 0.0, 0.0, 0.0, 0.0 };

   // select default device on system
   cl::sycl::default_selector device_selector;

   // define queue
   cl::sycl::queue queue(device_selector);
   std::cout << "Running on "
             << queue.get_device().get_info<cl::sycl::info::device::name>()
             << "\n";
   {
      // create sycle buffer for host to device variables
      cl::sycl::buffer<cl::sycl::float4, 1> a_sycl(&a, cl::sycl::range<1>(1));
      cl::sycl::buffer<cl::sycl::float4, 1> b_sycl(&b, cl::sycl::range<1>(1));
      cl::sycl::buffer<cl::sycl::float4, 1> c_sycl(&c, cl::sycl::range<1>(1));
   
      // submit to queue
      queue.submit([&] (cl::sycl::handler& cgh) {
         // defien accessors
         auto a_acc = a_sycl.get_access<cl::sycl::access::mode::read>(cgh);
         auto b_acc = b_sycl.get_access<cl::sycl::access::mode::read>(cgh);
         auto c_acc = c_sycl.get_access<cl::sycl::access::mode::discard_write>(cgh);

         // lambda to add in parallel
         cgh.single_task<class vector_addition>([=] () {
         c_acc[0] = a_acc[0] + b_acc[0];
         });
      });
   }

   // output
   std::cout << "  A { " << a.x() << ", " << a.y() << ", " << a.z() << ", " << a.w() << " }\n"
        << "+ B { " << b.x() << ", " << b.y() << ", " << b.z() << ", " << b.w() << " }\n"
        << "------------------\n"
        << "= C { " << c.x() << ", " << c.y() << ", " << c.z() << ", " << c.w() << " }"
        << std::endl;
		
   return 0;
}