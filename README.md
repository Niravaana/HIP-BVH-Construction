# HIP-BVH-Construction

This repo is implementation of different GPU build methods. Following methods are on the roadmap.

1. LBVH - Based on research paper [Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf) **[DONE]**
2. Binned SAH Builder - Based on research paper "On fast Construction of SAH-based Bounding Volume Hierarchies, by I. Wald"
3. PLOC - Based on research paper "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction, by D. Meister and J. Bittner" 

#Details on LBVH Implementation 

Reference Images 
![depth](https://github.com/user-attachments/assets/af76dac1-f4ae-41a6-a5b7-2e90e0062dc6)

![test](https://github.com/user-attachments/assets/7b371357-7ff3-40ba-a214-b410f3bd3fb2)

Timings for cornell Box(32 triangles) on RX6800 AMD

-----------------------Perf Times----------------------------

CalculateSceneExtentsTime :0.0138ms

CalculateMortonCodesTime :0.0094ms

SortingTime : 0.3442ms

BvhBuildTime : 0.6043ms

TraversalTime : 0.3493ms

Total Time : 1.321ms

-------------------------------------------------------------

Todo : Still need to explore collapsing for LBVH and see its impact on performance and take timings with Bistro like scene.
