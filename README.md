# Steps to Build and Run

1. git clone the repo
2. git submodule update --init --recursive 
3. Run .\tools\premake5\win\premake5.exe vs2022
4. Build directory will be created which will have the solution file.

# HIP-BVH-Construction

This repo is implementation of different GPU BVH build methods and optimizations. Following methods are on the roadmap.

1. [LBVH](https://github.com/Niravaana/HIP-BVH-Construction/blob/main/src/TwoPassLbvhKernel.h) - Two pass method based on the research paper [Maximizing Parallelism in the Construction of BVHs,Octrees, and k-d Trees](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf) **[DONE]**
2. [LBVH](https://github.com/Niravaana/HIP-BVH-Construction/blob/main/src/SinglePassLbvhKernel.h) - Single pass method based on the research paper  [Fast and Simple Agglomerative LBVH Construction](https://diglib.eg.org/server/api/core/bitstreams/ad092db2-6aec-4f2c-941d-8687de258f00/content) **[DONE]**
3. [Inplace LBVH](https://github.com/Niravaana/HIP-BVH-Construction/blob/b0073927b0e1e8b202bd4c0985ec3ef626a43e86/src/BatchedBuildKernel.h#L202) - Based on the [HIPRT paper](https://dl.acm.org/doi/10.1145/3675378).Refer to HIPRT source for more details. **[DONE]**
4. [Inplace PLOC++](https://github.com/Niravaana/HIP-BVH-Construction/blob/b0073927b0e1e8b202bd4c0985ec3ef626a43e86/src/Ploc%2B%2BKernel.h#L77) - When nClusters drop below block size we can optimize the PLOC++ further. After this ploc bvh builder takes ~1ms less. Based on the [HIPRT paper](https://dl.acm.org/doi/10.1145/3675378). Refer to HIPRT source for more details. **[DONE]**
5. [HPLOC](https://github.com/Niravaana/HIP-BVH-Construction/blob/b0073927b0e1e8b202bd4c0985ec3ef626a43e86/src/HplocKernel.h#L236) - Based on the [paper](https://gpuopen.com/download/publications/HPLOC.pdf). **[DONE]**
6. [Ploc++](https://github.com/Niravaana/HIP-BVH-Construction/blob/1d589886d355db23693f122b7f6d06bcb601bb0a/src/Ploc%2B%2BKernel.h#L211) - Based on the [PLOC++ paper](https://www.intel.com/content/www/us/en/developer/articles/technical/ploc-for-bounding-volume.html). You can find the kernel is heavily commented to help understand each step more in detail. **[DONE]**
7. [Collapse LBVH to nwide BVH](https://github.com/Niravaana/HIP-BVH-Construction/blob/ddbb471d69318f0cbabd2ba29e7d3b5dd68805d7/src/TwoPassLbvhKernel.h#L215) - Based on the research paper [Getting Rid of Packets Efficient SIMD Single-Ray Traversal using Multi-branching BVHs](https://typeset.io/pdf/getting-rid-of-packets-efficient-simd-single-ray-traversal-29rwgbmwv3.pdf) **[Cpu/Gpu DONE]**
   The BVH SAH cost for bunny model drops from ~46 to ~22 while for sponza it drops to ~59 from ~131.
8. [Binned SAH Builder](https://github.com/Niravaana/HIP-BVH-Construction/blob/main/src/BinnedSahBvh.cpp) - Based on research paper [On fast Construction of SAH-based Bounding Volume Hierarchies, by I. Wald](https://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf) **[DONE]**
   Currently this is CPU only build, though implemented based on task model so should be easy to port to GPU.
9. [Early Split Clipping](https://github.com/Niravaana/HIP-BVH-Construction/blob/ddbb471d69318f0cbabd2ba29e7d3b5dd68805d7/src/TwoPassLbvh.cpp#L29) - Based on the research paper [Early Split Clipping for Bounding Volume Hierarchies](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ddfac027fa516d63fa705c52155ea9313543cf3a)
   Basic idea is to split the AABB of the primitive along the maximum extent midpoint. The problem is leaf node explosion and the criteria to decide to split is user defined parameter. It is hard to find a generic value for this heuristic. Though, I dont see any performance boost in traversal at least for sponza.(Hard to verify my implementation as of now).
10. [If-If/while-while traversal shaders](https://github.com/Niravaana/HIP-BVH-Construction/blob/main/src/TraversalKernel.h) - Based on the research paper [Understanding the Efficiency of Ray Traversal on GPUs – Kepler and Fermi Addendum](https://research.nvidia.com/sites/default/files/pubs/2012-06_Understanding-the-Efficiency/nvr-2012-02.pdf)
11. [Restart Trail Traversal](https://github.com/Niravaana/HIP-BVH-Construction/blob/afd326510b0107387fb8ae2c3d70e1b016224bec/src/TraversalKernel.h#L28) : Based on the research [paper](https://research.nvidia.com/sites/default/files/pubs/2010-06_Restart-Trail-for/laine2010hpg_paper.pdf).

Note : You might find function duplication in the kernel files. I did not get time to refactor this much though it helps to understand code as we have all code in one file.

# Reference Images

![test](https://github.com/user-attachments/assets/59203a5b-fa09-4afb-a696-ad854371f037)

![test](https://github.com/user-attachments/assets/52f37b52-7c81-44e6-b890-e07489f82386)

![test](https://github.com/user-attachments/assets/7b371357-7ff3-40ba-a214-b410f3bd3fb2)

Leaf Node Visits 

![colorMap](https://github.com/user-attachments/assets/929753e0-11e8-4150-8020-054ac80c24f4)

![colorMap](https://github.com/user-attachments/assets/f5234849-bd3d-4af3-aba7-c054f14bed08)


# Performance Numbers With Two Pass LBVH

**Timings for Bunny Box(150K triangles) on RX6800 AMD**

-----------------------Perf Times----------------------------

CalculateCentroidExtentsTime :0.247701ms

CalculateMortonCodesTime :0.037101ms

SortingTime : 0.1635ms

BvhBuildTime : 0.693201ms

CollapseTime : 2.9558ms

Bvh Cost : 22.6397

Total Time : 1.1415ms

------------------------------------------------------------

**Timings for Sponza(260K triangles) on RX6800 AMD**

-------------------------Perf Times------------------------

CalculateCentroidExtentsTime :0.223ms

CalculateMortonCodesTime :0.082599ms

SortingTime : 0.249ms

BvhBuildTime : 0.929ms

CollapseTime : 3.6383ms

Bvh Cost : 59.4779

Total Time : 1.4836ms

---------------------------------------------------------------

# Performance Numbers with Single Pass LBVH 

**Timings for Bunny Box(150K triangles) on RX6800 AMD**

------------------------Perf Times------------------------------

CalculateCentroidExtentsTime :0.2209ms

CalculateMortonCodesTime :0.0465ms

SortingTime : 0.173501ms

BvhBuildTime : 0.4865ms

CollapseTime : 3.3596ms

Bvh Cost : 22.6397

Total Time : 0.927401ms

------------------------------------------------------------

**Timings for Sponza(260K triangles) on RX6800 AMD**

-----------------------Perf Times---------------------------

CalculateCentroidExtentsTime :0.224899ms

CalculateMortonCodesTime :0.0853ms

SortingTime : 0.2496ms

BvhBuildTime : 0.428799ms

CollapseTime : 3.316ms

Bvh Cost : 59.4779

Total Time : 0.988598ms

------------------------------------------------------------

# Performance Numbers with HPLOC 

**Timings for Bunny Box(150K triangles) on RX6800 AMD**

-----------------------Perf Times----------------------------

CalculateCentroidExtentsTime :0.2593ms

CalculateMortonCodesTime :0.070599ms

SortingTime : 0.177599ms

BvhBuildTime : 0.5147ms

CollapseTime : 3.3104ms

Bvh Cost : 21.9676

Total Time : 1.0222ms

------------------------------------------------------------

**Timings for Sponza(260K triangles) on RX6800 AMD**

-----------------------Perf Times---------------------------

CalculateCentroidExtentsTime :0.279299ms

CalculateMortonCodesTime :0.204699ms

SortingTime : 0.2535ms

BvhBuildTime : 0.6133ms

CollapseTime : 4.0426ms

Bvh Cost : 48.2362

Total Time : 1.3508ms

------------------------------------------------------------

# Performance Numbers with PLOC++ 

**Timings for Bunny Box(150K triangles) on RX6800 AMD**

-----------------------Perf Times---------------------------

CalculateCentroidExtentsTime :0.1981ms

CalculateMortonCodesTime :0.0776ms

SortingTime : 0.1944ms

BvhBuildTime : 0.688ms

CollapseTime : 3.2174ms

Bvh Cost : 21.9248

Total Time : 1.1581ms

------------------------------------------------------------

**Timings for Sponza(260K triangles) on RX6800 AMD**

-----------------------Perf Times---------------------------

CalculateCentroidExtentsTime :0.2975ms

CalculateMortonCodesTime :0.1004ms

SortingTime : 0.2598ms

BvhBuildTime : 0.983297ms

CollapseTime : 4.3278ms

Bvh Cost : 48.842

Total Time : 1.641ms

------------------------------------------------------------

Note : Timings are on AMD 6800W Pro

