# HIP-BVH-Construction

This repo is implementation of different GPU BVH build methods and optimizations. Following methods are on the roadmap.

1. LBVH - Two pass method based on the research paper [Maximizing Parallelism in the Construction of BVHs,Octrees, and k-d Trees](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf) **[DONE]**
2. LBVH - Single pass method based on the research paper  [Fast and Simple Agglomerative LBVH Construction](https://diglib.eg.org/server/api/core/bitstreams/ad092db2-6aec-4f2c-941d-8687de258f00/content) **[DONE]**
3. Collapse LBVH to nwide BVH - Based on the research paper [Getting Rid of Packets Efficient SIMD Single-Ray Traversal using Multi-branching BVHs](https://typeset.io/pdf/getting-rid-of-packets-efficient-simd-single-ray-traversal-29rwgbmwv3.pdf) **[DONE]**

   Currently CPU implementation is done though it is done based on task model so easy to port it to GPU. Soon it will be ported to GPU.
   The BVH SAH cost for bunny model drops from ~46 to ~22 while for sponza it drops to ~59 from ~131.
5. Binned SAH Builder - Based on research paper "On fast Construction of SAH-based Bounding Volume Hierarchies, by I. Wald" **[DONE]**
   Currently this is CPU only build, though implemented based on task model so should be easy to port to GPU.
7. If-If/while-while traversal shaders - Based on the research paper [Understanding the Efficiency of Ray Traversal on GPUs – Kepler and Fermi Addendum](https://research.nvidia.com/sites/default/files/pubs/2012-06_Understanding-the-Efficiency/nvr-2012-02.pdf)
 
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

Note : on AMD APU 780M total time is 5.3ms for sponza and 3.22ms for bunny.
