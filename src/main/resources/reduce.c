// Copyright (c) Piotr Morgwai Kotarbinski, Licensed under the Apache License, Version 2.0
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable



void accumulateVolatile(uint sourceIndex, uint targetIndex, __local volatile double* localSlice) {
	localSlice[targetIndex] += localSlice[sourceIndex];
}



__kernel void reduceHybrid(
	__global const double *input,
	uint inputLength,
	__local double *localSlice,
	__global double *results
) {
	uint i = get_local_id(0);
	uint globalIndex = get_global_id(0);

	// copy input to local memory
	if (globalIndex < inputLength) localSlice[i] = input[globalIndex];
	barrier(CLK_LOCAL_MEM_FENCE);

	// main loop with barrier synchronization
	uint simdWidth = get_max_sub_group_size();
	uint activeThreadCount = get_local_size(0) >> 1;
	while (activeThreadCount > simdWidth) {
		if (
			i < activeThreadCount
			&& (globalIndex + activeThreadCount) < inputLength
		) {
			localSlice[i] += localSlice[i + activeThreadCount];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		activeThreadCount >>= 1;
	}

	// main loop with SIMD + volatile synchronization
	if (i >= simdWidth) return;
	while (activeThreadCount > 0) {
		if (globalIndex + activeThreadCount < inputLength) {
			accumulateVolatile(i + activeThreadCount, i, localSlice);
		}
		activeThreadCount >>= 1;
	}

	if (i == 0) results[get_group_id(0)] = localSlice[0];
}



__kernel void reduceBarrier(
	__global const double *input,
	uint inputLength,
	__local double *localSlice,
	__global double *results
) {
	uint i = get_local_id(0);
	uint globalIndex = get_global_id(0);

	// copy input to local memory
	if (globalIndex < inputLength) localSlice[i] = input[globalIndex];
	barrier(CLK_LOCAL_MEM_FENCE);

	// main loop with barrier synchronization
	uint activeThreadCount = get_local_size(0) >> 1;
	while (activeThreadCount > 0) {
		if (
			i < activeThreadCount
			&& (globalIndex + activeThreadCount) < inputLength
		) {
			localSlice[i] += localSlice[i + activeThreadCount];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		activeThreadCount >>= 1;
	}

	if (i == 0) results[get_group_id(0)] = localSlice[0];
}



// group size cannot be bigger than SIMD width
__kernel void reduceSimd(
	__global const double *input,
	uint inputLength,
	__local volatile double *localSlice,
	__global double *results
) {
	uint i = get_local_id(0);
	uint globalIndex = get_global_id(0);

	// copy input to local memory
	if (globalIndex < inputLength) localSlice[i] = input[globalIndex];

	// main loop with SIMD + volatile synchronization
	uint activeThreadCount = get_local_size(0) >> 1;
	while (activeThreadCount > 0) {
		if (globalIndex + activeThreadCount < inputLength) {
			localSlice[i] += localSlice[i + activeThreadCount];
		}
		activeThreadCount >>= 1;
	}

	results[get_group_id(0)] = localSlice[0];
}



// run with max work-group size
__kernel void getSimdWidth(__global uint *simdWidth) {
	if (get_local_id(0) == 0) simdWidth[0] = get_max_sub_group_size();
}
