# HIP-BVH-Construction

This repo is implementation of different GPU BVH build methods and optimizations. Following methods are on the roadmap.

1. LBVH - Based on research paper [Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf) **[DONE]**
2. Binned SAH Builder - Based on research paper "On fast Construction of SAH-based Bounding Volume Hierarchies, by I. Wald" [CPU implementation DONE]
3. PLOC - Based on research paper "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction, by D. Meister and J. Bittner" 

# Details on LBVH Implementation 

**Reference Images**

![test](https://github.com/user-attachments/assets/59203a5b-fa09-4afb-a696-ad854371f037)

![test](https://github.com/user-attachments/assets/52f37b52-7c81-44e6-b890-e07489f82386)

![test](https://github.com/user-attachments/assets/7b371357-7ff3-40ba-a214-b410f3bd3fb2)

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
