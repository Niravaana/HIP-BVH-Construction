# HIP-BVH-Construction

This repo is implementation of different GPU BVH build methods and optimizations. Following methods are on the roadmap.

1. LBVH - Based on research paper [Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf) **[DONE]**

   This method builds the tree in top down fashion using above paper. We build a hierarchy in top down approach as a first pass and in second pass doing bottom up traversal we fit the tree.
   [Single Pass LBVH](https://diglib.eg.org/server/api/core/bitstreams/ad092db2-6aec-4f2c-941d-8687de258f00/content) method improves on vanila method and build and fits heirarchy in one pass.
   You can enable single pass build using define SINGLE_PASS_LBVH.

   Note : Currently the build time for single pass is much higher than two pass, my analysis is, it is due to the way findparent generates the parent node idx. These node indices are very
   far from each other for neighbouring threads. Because of this global load and store instruction timings are very bad in the trace.I am not sure is it because of my node size (64 bytes) currently.
   Have to try if node size reduced to 32 bytes improves performance if building in single pass.
             
3. HPLOC - Based on the research paper [HPLOC](https://meistdan.github.io/publications/hploc/paper.pdf).
   
   This method is more suited to the GPU programming framework. So instead of trying PLOC we will implement HPLOC. They claim they are faster than PLOC and yet give almost similar tracing performance.

4. Binned SAH Builder - Based on research paper "On fast Construction of SAH-based Bounding Volume Hierarchies, by I. Wald" [CPU implementation DONE]
   
   The GPU implementation of this method is not on the plan immediately. The CPU implementation was done so as to understand binning and SAH more.


# Reference Images

![test](https://github.com/user-attachments/assets/59203a5b-fa09-4afb-a696-ad854371f037)

![test](https://github.com/user-attachments/assets/52f37b52-7c81-44e6-b890-e07489f82386)

![test](https://github.com/user-attachments/assets/7b371357-7ff3-40ba-a214-b410f3bd3fb2)

Leaf Node Visits 

![colorMap](https://github.com/user-attachments/assets/929753e0-11e8-4150-8020-054ac80c24f4)

![colorMap](https://github.com/user-attachments/assets/f5234849-bd3d-4af3-aba7-c054f14bed08)


# Performance Numbers With LBVH

**Timings for Bunny Box(150K triangles) on RX6800 AMD**

-----------------------Perf Times----------------------------

CalculateCentroidExtentsTime :0.0213ms

CalculateMortonCodesTime :0.0032ms

SortingTime : 0.247ms

BvhBuildTime : 0.3238ms

Total Time : 0.5953ms

-----------------------------------------------------------

**Timings for Sponza(260K triangles) on RX6800 AMD**

-----------------------Perf Times----------------------------

CalculateCentroidExtentsTime :0.023ms

CalculateMortonCodesTime :0.0241ms

SortingTime : 0.3357ms

BvhBuildTime : 0.3868ms

Total Time : 0.7696ms

-------------------------------------------------------------

**Todo :** Still need to explore collapsing for LBVH and see its impact on performance, take timings with Bistro like scene and write blog post explaining the method.

Note : on AMD APU 780M total time is 6.494ms
