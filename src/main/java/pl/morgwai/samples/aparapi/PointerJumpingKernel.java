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
 * Performs pointer-jumping on an array using a GPU with Aparapi. By default sums values from the
 * array, but subclasses may override {@link #accumulateValue(int, int)} to do something else.
 * <p>Usage:</p>
 * <pre>
 * double[] myArray = getDoubleArray();
 * double sum = PointerJumpingKernel.calculateSum(myArray);</pre>
 * <p>
 * On my integrated Intel GPU ({@code maxWorkWorkGroupSize=256, maxComputeUnits=48}), this works 4
 * times slower than sequential adding on the CPU.<br/>
 * This is probably because the algorithm is memory bound and spends most time on fetching values
 * from the memory. See
 * <a href='https://stackoverflow.com/questions/22866901/using-java-with-nvidia-gpus-cuda#22868938'>
 * this SO answer</a> for more info.<br/>
 * The other reason may be that <a
 * href='https://software.intel.com/content/www/us/en/develop/documentation/iocl-opg/top/optimizing-opencl-usage-with-intel-processor-graphics/work-group-size-recommendations-summary.html'>
 * intel has only 16 barrier registers (and only 64kB local memory <i>shared</i> among running
 * work-groups)</a>, so only up to 16 work-groups can run in parallel.</p>
 */
public class PointerJumpingKernel extends Kernel implements AutoCloseable {



	@SuppressWarnings("deprecation")
	public static void compile() {
		var gpuDevice = Device.bestGPU();
		if (gpuDevice == null) throw new RuntimeException("No suitable GPU device found :(");
		try (var kernel = new PointerJumpingKernel()) {
			System.out.println("compiling for " + gpuDevice);
			System.out.println();
			kernel.compile(kernel.getTargetDevice());
			// need to execute the kernel to avoid segmentation fault:
			var preferredDevices = new LinkedHashSet<Device>();
			preferredDevices.add(gpuDevice);
			KernelManager.instance().setPreferredDevices(kernel, preferredDevices);
			kernel.accumulateArray(new double[1]);
		} catch (CompileFailedException e) {
			throw new RuntimeException(e);
		}
	}



	@SuppressWarnings("deprecation")
	public static double calculateSum(double[] input) {
		try (var kernel = new PointerJumpingKernel()) {
			var preferredDevices = new LinkedHashSet<Device>();
			preferredDevices.add(Device.bestGPU());
			KernelManager.instance().setPreferredDevices(kernel, preferredDevices);
			return kernel.accumulateArray(input);
		}
	}

	protected PointerJumpingKernel() {}



	double[] input;  // input values to accumulate
	double[] results; // results from all work-groups

	/**
	 * Group's local copy of its slice of the {@code input}.
	 */
	@Local protected double[] localSlice;
	@Local int[] next; // next[i] is a pointer to the next value in localSlice not yet accumulated
			// by the processing element with localId == i, initially next[i] == i+1 for all i



	/**
	 * Triggers GPU execution of {@link #run() pointer-jumping} on distinct slices of {@code input}
	 * in separate work-groups, after that recursively accumulates results from all work-groups.
	 * Recursion stops when everything is accumulated into a single value.
	 * @return value accumulated from the whole input.
	 */
	protected double accumulateArray(double[] input) {
		this.input = input;
		int groupSize = Math.min(input.length, getTargetDevice().getMaxWorkGroupSize());
		int numberOfGroups = input.length / groupSize;
		if (groupSize * numberOfGroups < input.length) numberOfGroups++;  // padding the last group
		int paddedSize = groupSize * numberOfGroups;
		results = new double[numberOfGroups];
		localSlice = new double[groupSize];
		next = new int[groupSize];
		execute(Range.create(paddedSize, groupSize));
		if (numberOfGroups == 1) return results[0];
		return accumulateArray(results);
	}



	/**
	 * Pointer-jumping procedure executed by a single processing element. Each work-group first
	 * copies its slice of the {@code input} to {@link #localSlice} array in group's local memory.
	 * Next, the main pointer-jumping loop is performed on the {@link #localSlice}.
	 * Finally, the 1st processing element writes group's accumulated result to {@link #results}
	 * array.
	 */
	@Override
	public final void run() {
		int i = getLocalId();
		int globalIndex = getGlobalId();
		int groupSize = getLocalSize();
		int acivityIndicator = i;// Divided by 2 at each step of the main loop until odd.
				// When odd, the given processing element stays idle (just checks-in at the barrier)

		// copy group's slice into local memory and initialize next pointers
		if (globalIndex < input.length - 1) {
			next[i] = i + 1;
			localSlice[i] = input[globalIndex];
		} else {
			next[i] = getGlobalSize(); // padding in the last group: point beyond the array
			if (globalIndex == input.length - 1) localSlice[i] = input[globalIndex];
		}
		localBarrier();

		// main pointer-jumping loop
		while (next[0] < groupSize) { // run until the whole group is accumulated at index 0
			if ( (acivityIndicator & 1) == 0 && next[i] < groupSize) {
				accumulateValue(next[i], i);
				next[i] = next[next[i]];
				acivityIndicator >>= 1;
			}
			localBarrier();
		}
		if (i == 0) results[getGroupId()] = localSlice[0];
	}

	/**
	 * Accumulates value from {@code fromIndex} in {@link #localSlice} into {@code intoIndex}.
	 * Subclasses may override this method to do something else than summing.
	 * Subclasses should then provide a static method that creates a kernel and calls
	 * {@link #accumulateArray(double[])} similarly to {@link #calculateSum(double[])}.
	 */
	protected void accumulateValue(int fromIndex, int intoIndex) {
		localSlice[intoIndex] += localSlice[fromIndex];
	}



	@Override
	public final void close() {
		dispose();
	}



	static Random random = new Random();

	public static void runPointerJumpingExample(int size) {
		double[] values = new double[size];
		for (int i = 0; i < size; i++) {
			values[i] = random.nextDouble() - 0.5;
		}

		var start = System.nanoTime();
		double val = 0.0;
		for (int i = 0; i < size; i++) {
			val += values[i];
		}
		System.out.println(String.format(
				"cpu: %1$15d,  result: %2$20.12f", System.nanoTime() - start, val));

		start = System.nanoTime();
		double result = PointerJumpingKernel.calculateSum(values);
		System.out.println(String.format(
				"gpu: %1$15d,  result: %2$20.12f\n", System.nanoTime() - start, result));
	}

	public static void main(String[] args) {
		PointerJumpingKernel.compile();
		for (int i = 0; i < 10; i++)
			runPointerJumpingExample(16*1024*1024);
		System.out.println("bye bye!");
	}
}
