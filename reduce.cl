/* CHAR REDUCE KERNEL (REVISIONED)
 * openCL Nvidia implementation
 * 24.4.2013
 * based on: 
 *	http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854?pgno=1
 *	http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *      nvidia sdk reduction sample
 *
 * Albert Zaharovits 
 * albert.zaharovits@gmail.com
 */

#define SAFE_LOAD_GLOBAL(s, i, c) \
	((size_t)(i)) < ((size_t)(c)) ? \
		((__global const char*)(s))[(size_t)(i)] : SCHAR_MIN

char operator(char a, char b)
{
	return max(a,b);
}

void warpReduce(__local char *d_data, uint localIdx, uint groupDim)
{
	if(localIdx < 32)
	{
		if(groupDim >= 64)
			d_data[localIdx] = operator(d_data[localIdx], 
					d_data[localIdx+32]);
		if(groupDim >= 32)
			d_data[localIdx] = operator(d_data[localIdx],
					d_data[localIdx+16]);
		if(groupDim >= 16)
			d_data[localIdx] = operator(d_data[localIdx],
					d_data[localIdx+8]);
		if(groupDim >= 8)
			d_data[localIdx] = operator(d_data[localIdx],
					d_data[localIdx+4]);
		if(groupDim >= 4)
			d_data[localIdx] = operator(d_data[localIdx],
					d_data[localIdx+2]);
		if(groupDim >= 2)
			d_data[localIdx] = operator(d_data[localIdx],
					d_data[localIdx+1]);
	}
}

/* CHAR MAX REDUCE KERNEL
 *
 * Features:
 *	 - there is no restriction on input data length
 * Restrictions:
 *	- size_of g_odata >= num_groups
 *	- size_of d_data >= group_dim
 *
 * Result is placed on g_odata[0]
 */

__kernel void
find_highest_ascii(__global const char *g_idata, const uint size,
	__local char *d_data, __global char *g_odata)
{	
	char local_max = SCHAR_MIN;
	uint localIdx = get_local_id(0);
	uint groupDim = get_local_size(0);
	uint groupIdx = get_group_id(0);
	uint globalDim = get_num_groups(0);
	// each thread will process stride elements (stride fitted to input size)
	uint stride = (size + get_global_size(0)-1)/get_global_size(0);
	uint globalIdx = localIdx + groupIdx * (stride*groupDim);
	uint count = size;

	for(uint s=0;s<stride;s++)
	{
		local_max = operator((char)local_max, 
				(char)(SAFE_LOAD_GLOBAL(g_idata, globalIdx, count)));
		globalIdx += groupDim;
	}
		
	d_data[localIdx] = local_max;

	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(uint s = groupDim >> 1;s>32;s>>=1)
	{
		if(localIdx < s)
			d_data[localIdx] = operator(d_data[localIdx],d_data[localIdx+s]);

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// warp/wavefront synchronous 
	warpReduce(d_data, localIdx, groupDim);

	if(localIdx == 0)
		g_odata[groupIdx] = d_data[0];
		
	barrier(CLK_GLOBAL_MEM_FENCE);

	// last work-group reduces per work-group results
	if(groupIdx == 0)
	{
		d_data[localIdx] = SAFE_LOAD_GLOBAL(g_odata, localIdx, globalDim);
		for(uint s = groupDim;s<globalDim;s+=groupDim)
		{
			d_data[localIdx] = operator(d_data[localIdx],
								SAFE_LOAD_GLOBAL(g_odata, localIdx+s, globalDim));
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		for(uint s = groupDim >> 1;s>32;s>>=1)
		{
			if(localIdx < s)
				d_data[localIdx] = operator(d_data[localIdx],d_data[localIdx+s]);

			barrier(CLK_LOCAL_MEM_FENCE);
		}
		
		// warp/wavefront synchronous 
		warpReduce(d_data, localIdx, groupDim);

		if(localIdx == 0)
			g_odata[0] = d_data[0];
	}

}
