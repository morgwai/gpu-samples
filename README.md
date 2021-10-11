# GPU samples

Parallel reduction and [pointer jumping](https://en.wikipedia.org/wiki/Pointer_jumping) algorithms summarizing values from an array adapted to run on a GPU using [Aparapi](https://aparapi.com/) and [JOCL](http://www.jocl.org/) (frontends to [openCL](https://www.khronos.org/opencl/)).


## building and running comparison of various sync methods in openCL parallel reduction

First, make sure that you have an openCL driver for your GPU installed: [Nvidia](https://developer.nvidia.com/cuda-downloads), [AMD Linux](https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-21-30) (AMD on windows should be available by default, hopefully).

```bash
mvn clean package
java -jar target/pointer-jumping-gpu-1.0-SNAPSHOT-jar-with-dependencies.jar
```
This will run parallel reduction kernels using 3 different approaches to synchronization on
arrays of various sizes from 32k to 128M elements, 50 times for each size. On my machine it
takes about 5 minutes. For each size it will output average time for each sync method.

These are times I got on my integrated Intel GPU:

32k element array:

```
BARRIER average:     403076
   SIMD average:     295953
 HYBRID average:     269073
    CPU average:      62924
```
128k:

```
BARRIER average:     768170
   SIMD average:     483343
 HYBRID average:     433704
    CPU average:     175977
```
256k:

```
BARRIER average:    1018578
   SIMD average:     793267
 HYBRID average:     738423
    CPU average:     367999
```
512k:

```
BARRIER average:    1191166
   SIMD average:    1019678
 HYBRID average:     828609
    CPU average:     780270
```
1M:

```
BARRIER average:    1759843
   SIMD average:    1580668
 HYBRID average:    1366559
    CPU average:    1288948
```
2M:

```
BARRIER average:    3406786
   SIMD average:    3070155
 HYBRID average:    2398054
    CPU average:    2674748
```
3M:

```
BARRIER average:    4166284
   SIMD average:    4192948
 HYBRID average:    3480526
    CPU average:    3575055
```
4M-4k:

```
BARRIER average:    6573353 (1 recursive step on HYBRID)
   SIMD average:    6758205
 HYBRID average:    5653419
    CPU average:    5582159
```
4M:

```
BARRIER average:   13797841
   SIMD average:   13367851
 HYBRID average:   12600975
    CPU average:    5427631
```
32M:

```
BARRIER average:  102840013
   SIMD average:  103991061
 HYBRID average:   95481061
    CPU average:   41226782
```
128M:

```
BARRIER average:  363563970
   SIMD average:  387534517
 HYBRID average:  344870087
    CPU average:  160136923
```
255M:

```
BARRIER average:  878887550 (1 recursive step on HYBRID)
   SIMD average:  819652415
 HYBRID average:  730983353
    CPU average:  323803437
```