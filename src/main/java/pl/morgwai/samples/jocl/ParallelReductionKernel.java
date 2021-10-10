// Copyright (c) Piotr Morgwai Kotarbinski, Licensed under the Apache License, Version 2.0
package pl.morgwai.samples.jocl;

import java.io.IOException;
import java.util.Random;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import static org.jocl.CL.*;



/**
 * Performs parallel reduction on an array using a GPU with jocl.
 * <p>Usage:</p>
 * <pre>
 * double[] myArray = getDoubleArray();
 * double sum = ParallelReductionKernel.calculateSum(myArray);</pre>
 */
public class ParallelReductionKernel implements AutoCloseable {



	public static enum SyncMode {

		/**
		 * Use local barriers to sync between threads. Uses maximum size work-groups.
		 */
		BARRIER,

		/**
		 * Use SIMD synchronous execution combined with annotating memory regions being written as
		 * volatile. Limits max work-group size to SIMD width. input size must be multiple of
		 * {@link #getSimdWidth() SIMD width}, otherwise {@link #HYBRID} will be used.
		 */
		SIMD,

		/**
		 * Use local barriers and maximum size work-groups first, later when the number of active
		 * threads goes down to SIMD width, stop using barriers but switch to accumulating function
		 * that marks memory as volatile.
		 */
		HYBRID
	}



	public static double calculateSum(double[] input) {
		return calculateSum(input, SyncMode.HYBRID);
	}

	public static double calculateSum(double[] input, SyncMode syncMode) {
		try (var kernel = new ParallelReductionKernel(syncMode)) {
			return kernel.reduceArray(input);
		}
	}



	static cl_context ctx;
	static cl_command_queue queue;
	static cl_program program;

	static int maxDimensionSize;
	public static int getMaxDimensionSize() { return maxDimensionSize; }

	static int simdWidth;
	public static int getSimdWidth() { return simdWidth; }

	SyncMode syncMode;
	cl_kernel kernel;
	int maxGroupSize;



	ParallelReductionKernel(SyncMode syncMode) {
		if (simdWidth == 0) init();
		this.syncMode = syncMode;
		String kernelName;
		switch (syncMode) {
			case BARRIER: kernelName = "reduceBarrier"; break;
			case SIMD: kernelName = "reduceSimd"; break;
			default: kernelName = "reduceHybrid";
		}
		kernel = clCreateKernel(program, kernelName, null);
		long[] kernelMaxGroupSizeBuffer = new long[1];
		clGetKernelWorkGroupInfo(kernel, null, CL_KERNEL_WORK_GROUP_SIZE, Sizeof.size_t,
				Pointer.to(kernelMaxGroupSizeBuffer), null);
		maxGroupSize = Math.min((int) kernelMaxGroupSizeBuffer[0], maxDimensionSize);
		maxGroupSize = maxDimensionSize;
	}



	@Override
	public final void close() {
		clReleaseKernel(kernel);
	}



	double reduceArray(double[] input) {
		var inputClBuffer = clCreateBuffer(
				ctx, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS,
				input.length * Sizeof.cl_double, Pointer.to(input), null);
		return reduceRecursively(inputClBuffer, input.length);
	}



	/**
	 * Dispatches to GPU {@link #reduceOnGpu(int, int) parallel reduction} of distinct slices of
	 * {@code input} in separate work-groups, after that recursively reduces array of results from
	 * all work-groups. Recursion stops when everything is accumulated into a single value.
	 * <p>
	 * work-group size approach:<br/>
	 * if {@code input.length <= maxGroupSize} then {@code groupSize = }({@code input.length}
	 * rounded up to the nearest power of 2)<br/>
	 * else {@code groupSize = maxGroupSize} and the last group gets rounded up to the full
	 * {@code maxGroupSize}.<br/>
	 * Either way, kernel code has safeguards to ignore elements beyond {@code input.lenght}.</p>
	 * @return value reduced from the whole input.
	 */
	double reduceRecursively(cl_mem input, int inputLength) {
		var syncMode = this.syncMode;
		if (syncMode == SyncMode.SIMD && inputLength > simdWidth && inputLength % simdWidth != 0) {
			syncMode = SyncMode.HYBRID;
			System.out.println("warning: inputLength % simdWidth != 0, falling back to HYBRID");
		}
		var groupSize = Math.min(inputLength, syncMode == SyncMode.SIMD ? simdWidth : maxGroupSize);
		var numberOfGroups = inputLength / groupSize;
		if (groupSize * numberOfGroups < inputLength) numberOfGroups++;  // group for uneven tail
		if (numberOfGroups == 1) groupSize = closest2Power(groupSize);  // rounding uneven input
		try {
			var results = reduceOnGpu(input, inputLength, numberOfGroups, groupSize);
			if (numberOfGroups > 1) return reduceRecursively(results, numberOfGroups);

			var resultBuffer = new double[1];
			clEnqueueReadBuffer(queue, results, CL_TRUE, 0, Sizeof.cl_double,
					Pointer.to(resultBuffer), 0, null, null);
			clReleaseMemObject(results);
			return resultBuffer[0];
		} finally {
			clReleaseMemObject(input);
		}
	}

	static int closest2Power(int x) {
		return (1 << (32 - Integer.numberOfLeadingZeros(x-1)));
	}



	/**
	 * Calls parallel reduction kernel 1 time.
	 * @return buffer with results from all work-groups.
	 */
	cl_mem reduceOnGpu(cl_mem input, int inputLength, int numberOfGroups, int groupSize) {

		// allocate results buffer
		var hostAccessMode = CL_MEM_HOST_NO_ACCESS;
		if (numberOfGroups == 1) hostAccessMode = CL_MEM_HOST_READ_ONLY;
		var results = clCreateBuffer(ctx, CL_MEM_READ_WRITE | hostAccessMode,
				numberOfGroups * Sizeof.cl_double, null, null);

		// set args and call kernel
		clSetKernelArg(kernel, 0/*input*/, Sizeof.cl_mem, Pointer.to(input));
		clSetKernelArg(kernel, 1/*inputLength*/, Sizeof.cl_int, Pointer.to(new int[]{inputLength}));
		clSetKernelArg(kernel, 2/*localSlice*/, Sizeof.cl_double * groupSize, null);
		clSetKernelArg(kernel, 3/*results*/, Sizeof.cl_mem, Pointer.to(results));
		clEnqueueNDRangeKernel(queue, kernel, 1, null, new long[]{numberOfGroups*groupSize},
				new long[]{groupSize}, 0, null,null);
		return results;
	}



	/**
	 * voodoo initiation. each time this function is called a puppy dies, so mind yourself ;-]
	 */
	static synchronized void init() {
		if (program != null) return;
		String programSource;
		try {
			programSource = new String(
					ParallelReductionKernel.class.getResourceAsStream("/reduce.c").readAllBytes());
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_GPU;
		final int deviceIndex = 0;
		CL.setExceptionsEnabled(true);
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];
		ctx = clCreateContext(contextProperties, 1, new cl_device_id[] { device }, null, null,null);
		cl_queue_properties properties = new cl_queue_properties();
		queue = clCreateCommandQueueWithProperties(ctx, device, properties, null);
		program = clCreateProgramWithSource(ctx, 1, new String[] { programSource }, null, null);
		clBuildProgram(program, 0, null, null, null, null);

		// get device maxDimensionSize
		int[] dimensionsBuffer = new int[1];
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, Sizeof.cl_uint,
				Pointer.to(dimensionsBuffer), null);
		long[] maxSizes = new long[dimensionsBuffer[0]];
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, Sizeof.size_t * dimensionsBuffer[0],
				Pointer.to(maxSizes), null);
		maxDimensionSize = (int) maxSizes[0];

		// get device SIMD width: jocl does not have binding for clGetKernelSubGroupInfo(...)
		cl_kernel simdWidthKernel = clCreateKernel(program, "getSimdWidth", null);
		long[] maxGroupSizeBuffer = new long[1];
		clGetKernelWorkGroupInfo(simdWidthKernel, null, CL_KERNEL_WORK_GROUP_SIZE, Sizeof.size_t,
				Pointer.to(maxGroupSizeBuffer), null);
		var maxGroupSize = Math.min((int) maxGroupSizeBuffer[0], maxDimensionSize);
		var simdWidthClBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
				Sizeof.cl_uint, null, null);
		clSetKernelArg(simdWidthKernel, 0, Sizeof.cl_mem, Pointer.to(simdWidthClBuffer));
		clEnqueueNDRangeKernel(queue, simdWidthKernel, 1, null, new long[]{maxGroupSize},
				new long[]{maxGroupSize}, 0, null,null);
		var simdWidthBuffer = new int[1];
		clEnqueueReadBuffer(queue, simdWidthClBuffer, CL_TRUE, 0, Sizeof.cl_uint,
				Pointer.to(simdWidthBuffer), 0, null, null);
		clReleaseMemObject(simdWidthClBuffer);
		clReleaseKernel(simdWidthKernel);
		simdWidth = simdWidthBuffer[0];
	}



	/**
	 * Runs all 3 {@link SyncMode}s and CPU reduction on random data and outputs evaluation times.
	 * @param args {@code args[0]}: optional data size, default 16M. {@code args[1]}: optional
	 * number of runs, default 20.
	 */
	public static void main(String[] args) {
		var size = 128*1024*1024;
		var numberOfRuns = 50;
		if (args.length > 0) size = Integer.parseInt(args[0]);
		if (args.length > 1) numberOfRuns = Integer.parseInt(args[1]);
		ParallelReductionKernel.init();
		var totalTimes = new long[SyncMode.values().length + 1];
		for (int i = 0; i < numberOfRuns; i++) {
			double[] input = new double[size];
			for (int j = 0; j < size; j++) input[j] = random.nextDouble() - 0.5;

			for (var syncMode: SyncMode.values()) measureExecutionTime(input, syncMode, totalTimes);

			var start = System.nanoTime();
			var result = 0.0;
			for (int j = 0; j < size; j++) result += input[j];
			var elapsed = System.nanoTime() - start;
			totalTimes[SyncMode.values().length] += elapsed;
			System.out.println(String.format(
					"%1$7s: %2$10d,  result: %3$20.12f", "CPU", elapsed, result));
			System.out.println();
		}
		for (var syncMode: SyncMode.values()) {
			System.out.println(String.format("%1$7s average: %2$10d",
					syncMode , totalTimes[syncMode.ordinal()] / numberOfRuns));
		}
		System.out.println(String.format("%1$7s average: %2$10d",
				"CPU" , totalTimes[SyncMode.values().length] / numberOfRuns));
	}

	static void measureExecutionTime(double[] input, SyncMode syncMode, long[] totalTimes) {
		var start = System.nanoTime();
		var result = ParallelReductionKernel.calculateSum(input, syncMode);
		var elapsed = System.nanoTime() - start;
		totalTimes[syncMode.ordinal()] += elapsed;
		System.out.println(String.format(
				"%1$7s: %2$10d,  result: %3$20.12f", syncMode, elapsed, result));
	}

	static Random random = new Random();
}
