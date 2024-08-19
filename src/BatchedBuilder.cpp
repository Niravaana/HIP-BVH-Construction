#include "BatchedBuilder.h"
#include <src/Utility.h>
#include <dependencies/stbi/stbi_image_write.h>
#include <dependencies/stbi/stb_image.h>
#include <ParallelPrimitives/RadixSort.h>
#include <iostream>
#include <queue>
#include <assert.h>


using namespace BvhConstruction;

#define IFIF 1
//#define USE_LBVH 1

void BatchedBvhBuilder::build(Context& context, std::vector<BatchedBuildInput>& batch)
{
	
    //We will accept D_BatchedBuildInputs from user that will have device pointers
	std::vector<D_BatchedBuildInputs> h_batchedBuildInputs(batch.size());
	u32 nTotalInternalNodes = 0; 
	u32 nTotalLeafNodes = 0;
	Oro::GpuMemory<Triangle> d_triangles;

	for (size_t i = 0; i < batch.size(); i++)
	{
		d_triangles.reset();
		d_triangles.resize(batch[i].m_primitives.size());
		OrochiUtils::copyHtoD(d_triangles.ptr(), batch[i].m_primitives.data(), batch[i].m_primitives.size());
		h_batchedBuildInputs[i].m_prims = d_triangles.ptr();
		h_batchedBuildInputs[i].m_nPrimtives = batch[i].m_primitives.size();
		nTotalLeafNodes += h_batchedBuildInputs[i].m_nPrimtives;
		nTotalInternalNodes += divideRoundUp(2 * h_batchedBuildInputs[i].m_nPrimtives, 3);
	}

	Oro::GpuMemory<D_BatchedBuildInputs> d_batchedBuildInputs(h_batchedBuildInputs.size());
	OrochiUtils::copyHtoD(d_batchedBuildInputs.ptr(), h_batchedBuildInputs.data(), h_batchedBuildInputs.size());

	//allocate memory for all the batched bvh geoms
	d_bvhNodes.resize(nTotalInternalNodes);
	d_primRefs.resize(nTotalLeafNodes);
	d_rootNodes.resize(batch.size());
	Oro::GpuMemory<uint2> d_spans(nTotalLeafNodes); d_spans.reset();
	std::string functionName = "BatchedBuildKernelLbvh";

	const u32 gridDim = context.getMaxGridSize();
	const u32 gridDimY = std::max(1u, divideRoundUp((u32)batch.size(), gridDim));
	const u32 gridDimX = divideRoundUp((u32)batch.size(), gridDimY);
	Oro::GpuMemory<Aabb> d_sceneExtent(batch.size());

	Kernel batchedBuildKernel;

	buildKernelFromSrc(
		batchedBuildKernel,
		context.m_orochiDevice,
		"../src/BatchedBuildKernel.h",
		functionName.c_str(),
		std::nullopt);

	batchedBuildKernel.setArgs({ d_batchedBuildInputs.ptr(), d_bvhNodes.ptr(), d_primRefs.ptr(), d_spans.ptr(), d_rootNodes.ptr(), batch.size(), d_sceneExtent.ptr()});
	m_timer.measure(TimerCodes::BvhBuildTime, [&]() { batchedBuildKernel.launch(gridDimX, gridDimY, 1, MaxBatchedBlockSize, 1, 1, 0, 0 ); });
	
	//Validate bvhs 
	const auto h_bvhNodes = d_bvhNodes.getData();
	const auto h_leafNodes = d_primRefs.getData();
	const auto h_roots = d_rootNodes.getData();

	for (size_t i = 0; i < batch.size(); i++)
	{
		assert(Utility::checkPlocBvh2Correctness(h_bvhNodes.data(), h_leafNodes.data(), h_roots[i], batch[i].m_primitives.size(), batch[i].m_primitives.size() - 1) == true);
	}

	std::cout << "==========================Perf Times==========================" << std::endl;
	std::cout << "BatchSize : "  << batch.size() << std::endl;
	std::cout << "BvhBuildTime : " << m_timer.getTimeRecord(BvhBuildTime) << "ms" << std::endl;
	std::cout << "==============================================================" << std::endl;
}

void BatchedBvhBuilder::traverseBvh(Context& context)
{

}
