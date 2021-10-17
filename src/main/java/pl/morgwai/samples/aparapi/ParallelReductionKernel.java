// Copyright (c) Piotr Morgwai Kotarbinski, Licensed under the Apache License, Version 2.0
package pl.morgwai.samples.aparapi;

import java.util.LinkedHashSet;
import java.util.Random;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.exception.CompileFailedException;
import com.aparapi.internal.kernel.KernelManager;



/**
 * Performs parallel reduction on an array using a GPU with Aparapi. By default sums values from the
 * array, but subclasses may override {@link #accumulateValue(int, int)} to do something else.
 * <p>Usage:</p>
 * <pre>
 * double[] myArray = getDoubleArray();
 * double sum = ParallelReductionKernel.calculateSum(myArray);</pre>
 */
public class ParallelReductionKernel extends Kernel implements AutoCloseable {



	@SuppressWarnings("deprecation")
	public static void compile() {
		var gpuDevice = Device.bestGPU();
		if (gpuDevice == null) throw new RuntimeException("No suitable GPU device found :(");
		try (var kernel = new ParallelReductionKernel()) {
			System.out.println("compiling for " + gpuDevice);
			System.out.println();
			kernel.compile(kernel.getTargetDevice());
			// need to execute the kernel to avoid segmentation fault:
			var preferredDevices = new LinkedHashSet<Device>();
			preferredDevices.add(gpuDevice);
			KernelManager.instance().setPreferredDevices(kernel, preferredDevices);
			kernel.reduceArray(new double[1]);
		} catch (CompileFailedException e) {
			throw new RuntimeException(e);
		}
	}



	@SuppressWarnings("deprecation")
	public static double calculateSum(double[] input) {
		try (var kernel = new ParallelReductionKernel()) {
			var preferredDevices = new LinkedHashSet<Device>();
			preferredDevices.add(Device.bestGPU());
			KernelManager.instance().setPreferredDevices(kernel, preferredDevices);
			return kernel.reduceArray(input);
		}
	}

	protected ParallelReductionKernel() {}



	double[] input;  // input values to accumulate
	double[] results; // results from all work-groups

	/**
	 * Group's local copy of its slice of the {@code input}.
	 */
	@Local protected double[] localSlice;



	/**
	 * Triggers GPU execution of {@link #run() parallel reduction} on distinct slices of
	 * {@code input} in separate work-groups, after that recursively reduces results from all
	 * work-groups. Recursion stops when everything is accumulated into a single value.
	 * <p>
	 * work-group size approach:<br/>
	 * if {@code input.length <= maxGroupSize} then {@code groupSize = }({@code input.length}
	 * rounded up to the nearest power of 2)<br/>
	 * else {@code groupSize = maxGroupSize} and the last group gets rounded up to the full
	 * {@code maxGroupSize}.<br/>
	 * Either way, kernel code has safeguards to ignore elements beyond {@code input.length}.</p>
	 * @return value reduced from the whole input.
	 */
	protected double reduceArray(double[] input) {
		this.input = input;
		int groupSize = Math.min(input.length, getTargetDevice().getMaxWorkGroupSize());
		int numberOfGroups = input.length / groupSize;
		if (groupSize * numberOfGroups < input.length) numberOfGroups++;  // group for uneven tail
		if (numberOfGroups == 1) groupSize = closest2Power(groupSize);  // rounding uneven input
		int paddedSize = groupSize * numberOfGroups;
		results = new double[numberOfGroups];
		localSlice = new double[groupSize];
		execute(Range.create(paddedSize, groupSize));
		if (numberOfGroups == 1) return results[0];
		return reduceArray(results);
	}

	int closest2Power(int x) {
		return (1 << (32 - Integer.numberOfLeadingZeros(x-1)));
	}



	/**
	 * Parallel reduction procedure executed by a single processing element. Each work-group first
	 * copies its slice of the {@code input} to {@link #localSlice} array in group's local memory.
	 * Next, the main parallel reduction loop is performed on the {@link #localSlice}.
	 * Finally, the 1st processing element writes group's reduced result to {@link #results}
	 * array.
	 */
	@Override
	public final void run() {
		int i = getLocalId();
		int globalIndex = getGlobalId();

		// copy group's slice into local memory
		if (globalIndex < input.length) localSlice[i] = input[globalIndex];
		localBarrier();

		// main reduction loop
		int activeThreadCount = getLocalSize() >> 1;  // threads 0..activeThreadCount-1 are active
		while (activeThreadCount > 0) {
			if (i < activeThreadCount && i + activeThreadCount < input.length) {
				accumulateValue(i + activeThreadCount, i);
			}
			localBarrier();
			activeThreadCount >>= 1;
		}
		if (i == 0) results[getGroupId()] = localSlice[0];
	}

	/**
	 * Accumulates value from {@code fromIndex} in {@link #localSlice} into {@code intoIndex}.
	 * Subclasses may override this method to do something else than summing.
	 * Subclasses should then provide a static method that creates a kernel and calls
	 * {@link #reduceArray(double[])} similarly to {@link #calculateSum(double[])}.
	 */
	protected void accumulateValue(int fromIndex, int intoIndex) {
		localSlice[intoIndex] += localSlice[fromIndex];
	}



	@Override
	public final void close() {
		dispose();
	}



	static Random random = new Random();

	public static long runReductionKernelExample(int size) {
		double[] input = new double[size];
		for (int i = 0; i < size; i++) {
			input[i] = random.nextDouble() - 0.5;
		}

		var start = System.nanoTime();
		double result = 0.0;
		for (int i = 0; i < size; i++) {
			result += input[i];
		}
		System.out.printf("cpu: %1$15d,  result: %2$20.12f%n", System.nanoTime() - start, result);

		start = System.nanoTime();
		result = ParallelReductionKernel.calculateSum(input);
		var stop = System.nanoTime() - start;
		System.out.printf("gpu: %1$15d,  result: %2$20.12f\n%n", stop, result);
		return stop;
	}

	public static void main(String[] args) {
		ParallelReductionKernel.compile();
		runReductionKernelExample(16*1024*1024);
		var totalTime = 0l;
		var numberOfRuns = 20;
		for (int i = 0; i < numberOfRuns; i++) {
			totalTime += runReductionKernelExample(16*1024*1024);
		}
		System.out.println("average: " + (totalTime/numberOfRuns));
	}
}
