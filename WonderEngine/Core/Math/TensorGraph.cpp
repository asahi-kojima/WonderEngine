#pragma once
#include "TensorGraph.h"
#include "Tensor.h"

namespace Aoba::Core::Math
{
	void TensorGraph::merge(std::shared_ptr<TensorGraph>& graphL, std::shared_ptr<TensorGraph> graphR)
	{
		for (auto iter = graphR->mGraph.begin(); iter != graphR->mGraph.end(); iter++)
		{
			u32 originIndex = iter->first;
			const std::vector<u32>& targetIndexTbl = iter->second;

			for (const auto& targetIndex : targetIndexTbl)
			{
				graphL->insert(originIndex, targetIndex);
			}
		}


		for (auto iter = graphR->mTensorPtrTbl.begin(); iter != graphR->mTensorPtrTbl.end(); iter++)
		{
			graphL->mTensorPtrTbl[iter->first] = iter->second;

			iter->second->mTensorGraph = graphL;
		}
	}

	void TensorGraph::sortGraph()
	{
		std::map<u32, bool> seen;
		for (auto tensorPtr : mTensorPtrTbl)
		{
			seen[tensorPtr.first] = false;
		}

		auto rec = [](auto self, const Graph& graph, u32 originIndex, std::map<u32, bool>& isSeen, std::vector<u32>& sortedList)->void
		{
			isSeen[originIndex] = true;

			bool isExist = graph.count(originIndex);

			if (isExist)
			for (const u32& targetIndex : graph.at(originIndex))
			{
				if (isSeen[targetIndex])
					continue;

				self(self, graph, targetIndex, isSeen, sortedList);
			}

			sortedList.push_back(originIndex);
		};

		mSortedList.clear();
		mSortedBackwardList.clear();

		for (auto tensorPtr : mTensorPtrTbl)
		{
			if (seen[tensorPtr.first])
				continue;
			rec(rec, mGraph, tensorPtr.first, seen, mSortedList);
		}


		std::reverse(mSortedList.begin(), mSortedList.end());
		for (u32 i = 0; i < mSortedList.size(); i++)
		{
			mSortedBackwardList.push_back(mSortedList[mSortedList.size() - i - 1]);
		}

	}

}