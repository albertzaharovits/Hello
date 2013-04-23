/* CHAR REDUCE KERNEL
 * openCL Nvidia implementation
 * 21.4.2013
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
 * 	 - one group reduces 2 * get_local_size(0) elements
 * Restrictions:
 *	- number_of_groups * group_dim * 2 > input_data_size
 *	  [ get_num_groups(0) * get_local_size(0) * 2 > size ]
 *	- number_of_groups <= size_of g_odata
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
	uint globalIdx = localIdx + groupIdx * (2*groupDim);
	uint count = size;

	local_max = SAFE_LOAD_GLOBAL(g_idata, globalIdx, count);
	globalIdx += groupDim;
	local_max = operator((char)local_max, 
			(char)(SAFE_LOAD_GLOBAL(g_idata, globalIdx, count)));

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

	// first reduction step over. Do some more if necessary (num workgroups > 1)

	while(globalDim > 1)
	{
		barrier(CLK_GLOBAL_MEM_FENCE);

		count = globalDim;
		globalDim = globalDim/(groupDim*2) + 1;

		if(groupIdx < globalDim)
		{
			globalIdx = localIdx + groupIdx * (2*groupDim);
			local_max = SAFE_LOAD_GLOBAL(g_odata, globalIdx, count);
			globalIdx += groupDim;
			local_max = operator((char)local_max, 
					(char)(SAFE_LOAD_GLOBAL(g_odata, globalIdx, count)));

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
		}
	}
}
