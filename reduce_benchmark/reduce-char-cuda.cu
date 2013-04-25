// CUDA reduce kernel 
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <assert.h>
#include <sys/time.h>

#define SAFE_LOAD_GLOBAL(s, i, c) \
	((size_t)(i)) < ((size_t)(c)) ? \
			((const char*)(s))[(size_t)(i)] : SCHAR_MIN

#define GET_TIME_DELTA(t1, t2) (((t2).tv_sec - (t1).tv_sec) * 1000.0 + \
					((t2).tv_usec - (t1).tv_usec) / 1000.0)

typedef unsigned int uint;

__device__
void warpReduce(char *d_data, uint localIdx, uint groupDim)
{
		if(localIdx < 32)
		{
			if(groupDim >= 64)
				d_data[localIdx] = max(d_data[localIdx], d_data[localIdx+32]);
			if(groupDim >= 32)
				d_data[localIdx] = max(d_data[localIdx], d_data[localIdx+16]);
			if(groupDim >= 16)
				d_data[localIdx] = max(d_data[localIdx], d_data[localIdx+8]);
			if(groupDim >= 8)
				d_data[localIdx] = max(d_data[localIdx], d_data[localIdx+4]);
			if(groupDim >= 4)
				d_data[localIdx] = max(d_data[localIdx], d_data[localIdx+2]);
			if(groupDim >= 2)
				d_data[localIdx] = max(d_data[localIdx], d_data[localIdx+1]);
		}
}

__global__
void find_highest_ascii(const char *g_idata, uint count, char* g_odata)
{
	extern __shared__ char d_data[];
	char local_max = SCHAR_MIN;
	uint global_size = gridDim.x*blockDim.x;
	uint stride = (count + global_size-1)/global_size;
	uint globalIdx = threadIdx.x + blockIdx.x*(stride*blockDim.x);

	for(uint s=0;s<stride;s++) {
		local_max = max(local_max, (char)(SAFE_LOAD_GLOBAL(g_idata, 
			globalIdx, count)));
		globalIdx += blockDim.x;
	}

	d_data[threadIdx.x] = local_max;
	__syncthreads();

	for(uint s = blockDim.x >> 1;s>32;s>>=1)
	{
		if(threadIdx.x < s)
			d_data[threadIdx.x] = max(d_data[threadIdx.x],d_data[threadIdx.x+s]);

		__syncthreads();
	}

	warpReduce(d_data, threadIdx.x, blockDim.x);

	if(threadIdx.x == 0)
		g_odata[blockIdx.x] = d_data[0];

}

__global__
void find_highest_ascii2(const char *g_idata, uint count, char* g_odata)
{
	extern __shared__ char d_data[];
	char local_max = SCHAR_MIN;
	uint global_size = gridDim.x*blockDim.x;
	uint stride = (count + global_size-1)/global_size;
	uint globalIdx = threadIdx.x + blockIdx.x*(stride*blockDim.x);

	for(uint s=0;s<stride;s++) {
		local_max = max(local_max, (char)(SAFE_LOAD_GLOBAL(g_idata, 
			globalIdx, count)));
		globalIdx += blockDim.x;
	}

	d_data[threadIdx.x] = local_max;
	__syncthreads();

	for(uint s = blockDim.x >> 1;s>=1;s>>=1)
	{
		if(threadIdx.x < s)
			d_data[threadIdx.x] = max(d_data[threadIdx.x],d_data[threadIdx.x+s]);

		__syncthreads();
	}

	if(threadIdx.x == 0)
		g_odata[blockIdx.x] = d_data[0];
}

int main(int argc, char** argv)
{
	struct timeval t1, t2;
	uint size = 1<<28;
	uint temp_buffer_size = 1<<14;
	uint i;
	char *d_char_buffer, *d_temp_buffer, *d_result_buffer;
	char *d_char_buffer2, *d_temp_buffer2, *d_result_buffer2;
	char result_char;
	cudaError_t error;

	char *host_input = (char*)malloc(size*sizeof(char));
	for(i=0;i<size;++i)
		host_input[i] = 'A' + i%20;

	host_input[3*(size/7)] = 'Z';

	printf("with/without loop unroll\n");
	/*****************WITH LOOP UNROLL*********************************/
	size = 1<<28;
	error = cudaMalloc((void **)&d_char_buffer, size*sizeof(char));
	checkCudaErrors(error);
	error = cudaMalloc((void **)&d_temp_buffer, temp_buffer_size*sizeof(char));
	checkCudaErrors(error);
	error = cudaMalloc((void **)&d_result_buffer, sizeof(char));
	checkCudaErrors(error);
	error = cudaMemcpy(d_char_buffer, host_input, size*sizeof(char),
		cudaMemcpyHostToDevice);
	checkCudaErrors(error);

	gettimeofday(&t1, NULL);
	find_highest_ascii<<<(1<<14),(1<<10),(1<<10)*sizeof(char)>>>
		(d_char_buffer, size, d_temp_buffer);
	size = 1<<14;
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
	find_highest_ascii<<<1,(1<<10),(1<<10)*sizeof(char)>>>
		(d_temp_buffer, size, d_result_buffer);
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
	gettimeofday(&t2, NULL);

	error = cudaMemcpy(&result_char, d_result_buffer, sizeof(char),
		cudaMemcpyDeviceToHost);
	checkCudaErrors(error);
	assert(result_char == 'Z');
	printf("%lf ", GET_TIME_DELTA(t1,t2));
	cudaFree(d_char_buffer);
	cudaFree(d_temp_buffer);
	cudaFree(d_result_buffer);
	
	/*****************WITHOUT LOOP UNROLL*********************************/
	size = 1<<28;
	error = cudaMalloc((void **)&d_char_buffer2, size*sizeof(char));
	checkCudaErrors(error);
	error = cudaMalloc((void **)&d_temp_buffer2, temp_buffer_size*sizeof(char));
	checkCudaErrors(error);
	error = cudaMalloc((void **)&d_result_buffer2, sizeof(char));
	checkCudaErrors(error);
	error = cudaMemcpy(d_char_buffer2, host_input, size*sizeof(char),
		cudaMemcpyHostToDevice);
	checkCudaErrors(error);

	// no loop unrolled kernel
	gettimeofday(&t1, NULL);
	find_highest_ascii2<<<(1<<14),(1<<10),(1<<10)*sizeof(char)>>>
		(d_char_buffer2, size, d_temp_buffer2);
	size = 1<<14;
	find_highest_ascii2<<<1,(1<<10),(1<<10)*sizeof(char)>>>
		(d_temp_buffer2, size, d_result_buffer2);
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
	gettimeofday(&t2, NULL);

	error = cudaMemcpy(&result_char, d_result_buffer2, sizeof(char),
		cudaMemcpyDeviceToHost);
	checkCudaErrors(error);
	assert(result_char == 'Z');
	printf("%lf\n", GET_TIME_DELTA(t1,t2));

	cudaFree(d_char_buffer2);
	cudaFree(d_temp_buffer2);
	cudaFree(d_result_buffer2);
	return 0;
}

