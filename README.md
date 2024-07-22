# HIP-BVH-Construction

This repo is implementation of different GPU BVH build methods and optimizations. Following methods are on the roadmap.

1. LBVH - Based on research paper [Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf) **[DONE]**

   This method builds the tree in top down fashion using above paper. We build a hierarchy in top down approach as a first pass and in second pass doing bottom up traversal we fit the tree.
   [Single Pass LBVH](https://diglib.eg.org/server/api/core/bitstreams/ad092db2-6aec-4f2c-941d-8687de258f00/content) method improves on vanila method and build and fits heirarchy in one pass.
   You can enable single pass build using define SINGLE_PASS_LBVH.

   Note : Currently the build time for single pass is much higher than two pass, my analysis is, it is due to the way leaf nodes are stored. Currently, I am storing bvh nodes in one array.
          I store internal nodes followed by the leaf nodes. If we have n number of leafs then we know we will have 2 * n - 1 internal nodes. So we to access leaf node we always
          offset by nInternalNodes. When traversing the tree as nodes are in global memory this breaks memory coelascing causing the performance degradation. I think If I store leaf and
          internal nodes in two different array and not access with this offset method the perf will improve but, this is to be done!.
             
2. HPLOC - Based on the research paper [HPLOC](https://meistdan.github.io/publications/hploc/paper.pdf).
   
   This method is more suited to the GPU programming framework. So instead of trying PLOC we will implement HPLOC. They claim they are faster than PLOC and yet give almost similar tracing performance.

4. Binned SAH Builder - Based on research paper "On fast Construction of SAH-based Bounding Volume Hierarchies, by I. Wald" [CPU implementation DONE]
   
   The GPU implementation of this method is not on the plan immediately. The CPU implementation was done so as to understand binning and SAH more.


# Reference Images

![test](https://github.com/user-attachments/assets/59203a5b-fa09-4afb-a696-ad854371f037)

![test](https://github.com/user-attachments/assets/52f37b52-7c81-44e6-b890-e07489f82386)

![test](https://github.com/user-attachments/assets/7b371357-7ff3-40ba-a214-b410f3bd3fb2)

# Performance Numbers With LBVH

**Timings for cornell Box(32 triangles) on RX6800 AMD**

-----------------------Perf Times----------------------------

CalculateSceneExtentsTime :0.0138ms

CalculateMortonCodesTime :0.0094ms

SortingTime : 0.3442ms

BvhBuildTime : 0.6043ms

Total Time : 0.9ms

-------------------------------------------------------------

**Timings for Bunny Box(150K triangles) on RX6800 AMD**

-----------------------Perf Times----------------------------

CalculateCentroidExtentsTime :0.0642ms

CalculateMortonCodesTime :0.0307ms

SortingTime : 0.6024ms

BvhBuildTime : 1.1857ms

Total Time : 1.883ms

-----------------------------------------------------------

**Timings for Sponza(260K triangles) on RX6800 AMD**

-----------------------Perf Times----------------------------

CalculateCentroidExtentsTime :0.1093ms

CalculateMortonCodesTime :0.0487ms

SortingTime : 0.9192ms

BvhBuildTime : 1.569ms

Total Time : 2.6462ms

-------------------------------------------------------------

**Todo :** Still need to explore collapsing for LBVH and see its impact on performance, take timings with Bistro like scene and write blog post explaining the method.
