#pragma once
#include "TensorGraph.h"
#include "Tensor.h"

namespace Aoba::Core::Math
{
	void TensorGraph::merge(std::shared_ptr<TensorGraph>& graphL, std::shared_ptr<TensorGraph> graphR)
	{
		for (auto iter = graphR->mTensorPtrTbl.begin(); iter != graphR->mTensorPtrTbl.end(); iter++)
		{
			graphL->mTensorPtrTbl[iter->first] = iter->second;

			iter->second->mTensorGraph = graphL;
		}
	}
}