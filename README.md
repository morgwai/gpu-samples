# GPU samples

Parallel reduction and [pointer jumping](https://en.wikipedia.org/wiki/Pointer_jumping) algorithms summarizing values from an array adapted to run on a GPU using [Aparapi](https://aparapi.com/) and [JOCL](http://www.jocl.org/) (frontends to [openCL](https://www.khronos.org/opencl/)).


## building and running comparison of various sync methods in openCL parallel reduction

First, make sure that you have an openCL driver for your GPU installed: [Nvidia](https://developer.nvidia.com/cuda-downloads), [AMD Linux](https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-21-30) (AMD on windows should be available by default, hopefully).

```bash
mvn clean package
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[128*1024] 50
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[512*1024] 50
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[2*1024*1024] 50
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[4*1024*1024] 50
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar $[128*1024*1024] 50
```

on my integrated intel GPU I get times similar to these:

32k element array:<pre>
BARRIER average:     403076
   SIMD average:     295953
 HYBRID average:     269073
    CPU average:      62924</pre>

128k:<pre>
BARRIER average:     768170
   SIMD average:     483343
 HYBRID average:     433704
    CPU average:     175977</pre>

256::<pre>
BARRIER average:    1018578
   SIMD average:     793267
 HYBRID average:     738423
    CPU average:     367999</pre>

512k:<pre>
BARRIER average:    1191166
   SIMD average:    1019678
 HYBRID average:     828609
    CPU average:     780270</pre>

1M:<pre>
BARRIER average:    1759843
   SIMD average:    1580668
 HYBRID average:    1366559
    CPU average:    1288948</pre>

2M:<pre>
BARRIER average:    3406786
   SIMD average:    3070155
 HYBRID average:    2398054
    CPU average:    2674748</pre>

4M-4k:<pre>
BARRIER average:    6573353 (1 recursive step on HYBRID)
   SIMD average:    6758205
 HYBRID average:    5653419
    CPU average:    5582159</pre>

4M:<pre>
BARRIER average:   13797841
   SIMD average:   13367851
 HYBRID average:   12600975
    CPU average:    5427631</pre>

32M:<pre>
BARRIER average:  102840013
   SIMD average:  103991061
 HYBRID average:   95481061
    CPU average:   41226782</pre>

255M:<pre>
BARRIER average:  878887550 (1 recursive step on HYBRID)
   SIMD average:  819652415
 HYBRID average:  730983353
    CPU average:  323803437</pre>
