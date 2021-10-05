# Pointer-jumping on a GPU using Aparapi

Classic [pointer jumping](https://en.wikipedia.org/wiki/Pointer_jumping) algorithm summarizing values from an array adapted to run on a GPU instead of a PRAM using [Aparapi](https://aparapi.com/).

On my integrated Intel GPU (maxWorkWorkGroupSize=256, maxComputeUnits=48), this works 4 times slower than sequential adding on the CPU.

This is probably because the algorithm is memory bound and spends most time on fetching values from the memory. See [this SO answer](https://stackoverflow.com/questions/22866901/using-java-with-nvidia-gpus-cuda#22868938) for more info.

The other reason may be that [intel has only 16 barrier registers (and only 64kB local memory _shared_ among running work-groups)](https://software.intel.com/content/www/us/en/develop/documentation/iocl-opg/top/optimizing-opencl-usage-with-intel-processor-graphics/work-group-size-recommendations-summary.html), so only up to 16 work-groups can run in parallel.

To test this theory I need to run this on an Nvidia or AMD GPU, but I don't have any at hand. If someone who has, could run this code and send me back the results, I'd be very grateful :)

## building and running

First, make sure that you have an openCL driver for your GPU installed.

```bash
mvn clean package
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar
```

Thanks!
