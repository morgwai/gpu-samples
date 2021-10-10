# GPU samples

Parallel reduction and [pointer jumping](https://en.wikipedia.org/wiki/Pointer_jumping) algorithms summarizing values from an array adapted to run on a GPU using [Aparapi](https://aparapi.com/) and [JOCL](http://www.jocl.org/) (frontends to [openCL](https://www.khronos.org/opencl/)).


## building and running comparison of various sync methods in openCL parallel reduction

First, make sure that you have an openCL driver for your GPU installed: [Nvidia](https://developer.nvidia.com/cuda-downloads), [AMD Linux](https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-21-30) (AMD on windows should be available by default, hopefully).

```bash
mvn clean package
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[32*1024] 50
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[1024*1024] 50
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[32*1024*1024] 50
```

on my integrated intel GPU I get times similar to these:

32k element array:<pre>
BARRIER average:     444059
   SIMD average:     314828
 HYBRID average:     289607
    CPU average:      90483</pre>

1M:<pre>
BARRIER average:    1787720
   SIMD average:    1611933
 HYBRID average:    1336646
    CPU average:    1267648</pre>

32M:<pre>
BARRIER average:  101806901
   SIMD average:  102234318
 HYBRID average:   95539077
    CPU average:   41322452</pre>
