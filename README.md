# HIP-LBVH-Construction-
This is reference implementation for research paper [Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf)

![depth](https://github.com/user-attachments/assets/af76dac1-f4ae-41a6-a5b7-2e90e0062dc6)
![test](https://github.com/user-attachments/assets/7b371357-7ff3-40ba-a214-b410f3bd3fb2)

Timings for cornell Box(32 triangles) on RX6800 AMD

-----------------------Perf Times----------------------------

CalculateCentroidExtentsTime :0.0072ms
CalculateMortonCodesTime :0.009ms
SortingTime : 0.3453ms
BvhBuildTime : 0.5497ms
TraversalTime : 0.6379ms
Total Time : 1.5491ms

-------------------------------------------------------------
