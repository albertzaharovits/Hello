//
// include necessary system headers
//
#include <iostream>
#include <string>
#include <iterator>
#include <sys/time.h>
#include <assert.h>

//
// Activate OpenCL backend
//
#ifndef VIENNACL_WITH_OPENCL
  #define VIENNACL_WITH_OPENCL
#endif


//
// Helps with debugging OpenCL kernel build failures (CL_BUILD_PROGRAM_FAILURE)
//
#define VIENNACL_DEBUG_BUILD

#define GET_TIME_DELTA(t1, t2) (((t2).tv_sec - (t1).tv_sec) * 1000.0 + \
			((t2).tv_usec - (t1).tv_usec) / 1000.0)

//
// ViennaCL includes
//
#include "viennacl/ocl/backend.hpp"
#include "viennacl/backend/memory.hpp"
#include "viennacl/ocl/local_mem.hpp"

// Define the OpenCL compute program.
// Modify this program such that it returns the character with the highest ASCII code.
// The current kernel is for demonstration purposes only and happens to work only for the sample input.
//
// You can also write the OpenCL kernel sources into a file and load that into a string if you want
//
#define PROGRAM_FILE "reduce.cl"
#define KERNEL_FUNC "find_highest_ascii"
#define KERNEL_FUNC2 "find_highest_ascii2"

const char *my_compute_program = NULL;

char* read_file(const char* filename)
{
	FILE *handle;
	size_t size;
	char* buffer;

	handle = fopen(filename, "r");
	if(handle == NULL)
	{
		perror("Couldn't find the program file");
		return NULL;
	}
	fseek(handle, 0, SEEK_END);
	size = ftell(handle);
	rewind(handle);

	buffer = (char*)malloc(size*sizeof(char) + 1);
	fread( buffer, sizeof(char), size, handle);
	fclose(handle);

	buffer[size] = '\0';
	return buffer;
}

int main(int argc, char **argv)
{
  cl_uint size = 1<<28;
	struct timeval t1, t2;
	char result_char;

  // parse command line parameter
  if (argc > 1)
    size = atol(argv[1]);

  std::vector<char> host_input(size); // the array holding the character sequence on the host

  //
  // create some input on host. This could be arbitrarily random input.
  //
  for (cl_uint i=0; i<size; ++i)
    host_input[i] = 'A' + i % 20;

	host_input[3*(size/7)] = 'Z';
  //
  // Create OpenCL raw buffers:
  //
  viennacl::backend::mem_handle char_buffer;
  viennacl::backend::mem_handle result_buffer;
  viennacl::backend::mem_handle temp_buffer;

  viennacl::backend::mem_handle char_buffer2;
  viennacl::backend::mem_handle result_buffer2;
  viennacl::backend::mem_handle temp_buffer2;

	//initialize char_buffer with data from 'host_input'
  viennacl::backend::memory_create(char_buffer, size, &(host_input[0])); 
  viennacl::backend::memory_create(char_buffer2, size, &(host_input[0])); 
  viennacl::backend::memory_create(result_buffer, 1);
  viennacl::backend::memory_create(result_buffer2, 1);

  //
  // Set up the OpenCL program given in my_compute_kernel:
  // A program is one compilation unit and can hold many different compute kernels.
  //
	my_compute_program = read_file(PROGRAM_FILE);
	if(my_compute_program == NULL)
	{
		perror("Reading file");
		exit(1);
	}

  //std::cout << "Compiling OpenCL program..." << std::endl;
  viennacl::ocl::program & my_prog = viennacl::ocl::current_context().add_program(my_compute_program, PROGRAM_FILE);
  my_prog.add_kernel(KERNEL_FUNC);  //register our kernel
  my_prog.add_kernel(KERNEL_FUNC2);  //register our kernel
  
  //
  // After all kernels are registered, we can get the kernels from the program 'my_program'.
  //
  viennacl::ocl::kernel & my_ascii_kernel = my_prog.get_kernel(KERNEL_FUNC);
  viennacl::ocl::kernel & my_ascii_kernel2 = my_prog.get_kernel(KERNEL_FUNC2);

  //
  // Launch the kernel with 128 work groups, each with 64 threads
  //

  std::cout << "with/without loop unroll" << std::endl;
  my_ascii_kernel.local_work_size(0, 1<<10);
  my_ascii_kernel.global_work_size(0, 1<<24);
	// result buffer is used for work groups to communicate
  viennacl::backend::memory_create(temp_buffer, 1<<14);
	viennacl::ocl::local_mem d_data(1<<10);
  viennacl::backend::memory_create(temp_buffer2, 1<<14);
	viennacl::ocl::local_mem d_data2(1<<10);

	//**************WITH LOOP UNROLL*********************************
	size = 1<<28;
	gettimeofday(&t1, NULL);
  viennacl::ocl::enqueue( my_ascii_kernel(char_buffer.opencl_handle(),
			 size, d_data, temp_buffer.opencl_handle()) );

	// enqueue second kernel to get the final result
  my_ascii_kernel.local_work_size(0, 1<<10);
  my_ascii_kernel.global_work_size(0, 1<<10);
	 
	size = 1<<14;
  viennacl::ocl::enqueue( my_ascii_kernel(temp_buffer.opencl_handle(),
		 size, d_data, result_buffer.opencl_handle()) );
  result_char = 'a';
  viennacl::backend::memory_read(result_buffer, 0, 1, &result_char);
	gettimeofday(&t2, NULL);
	printf("%lf ", GET_TIME_DELTA(t1, t2));
	 
	assert(result_char == 'Z');

	// result buffer is used for work groups to communicate

	//**************WITHOUT LOOP UNROLL*********************************
  my_ascii_kernel2.local_work_size(0, 1<<10);
  my_ascii_kernel2.global_work_size(0, 1<<24);
	size = 1<<28;
	gettimeofday(&t1, NULL);
  viennacl::ocl::enqueue( my_ascii_kernel2(char_buffer2.opencl_handle(),
			 size, d_data2, temp_buffer2.opencl_handle()) );

	// enqueue second kernel to get the final result
  my_ascii_kernel2.local_work_size(0, 1<<10);
  my_ascii_kernel2.global_work_size(0, 1<<10);
	 
	size = 1<<14;
  viennacl::ocl::enqueue( my_ascii_kernel2(temp_buffer2.opencl_handle(),
		 size, d_data2, result_buffer2.opencl_handle()) );
  result_char = 'a';
  viennacl::backend::memory_read(result_buffer2, 0, 1, &result_char);
	gettimeofday(&t2, NULL);
	printf("%lf\n", GET_TIME_DELTA(t1, t2));
	 
	assert(result_char == 'Z');

  //std::cout << "Character with highest ASCII code in sequence: " << result_char << std::endl;
  // std::copy(host_input.begin(), host_input.end(), std::ostream_iterator<char>(std::cout, " ")); std::cout << std::endl;
  
  return EXIT_SUCCESS;
}

