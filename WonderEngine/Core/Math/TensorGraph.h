#pragma once
#include <map>
#include <vector>
#include <memory>
#include <iostream>

#include "typeinfo.h"

namespace Aoba::Core::Math
{
	class Tensor;

	class TensorGraph
	{
		friend class Tensor;

		using Graph = std::map<u32, std::vector<u32> >;
		Graph mGraph;
		void insert(u32 origin, u32 target) { mGraph[origin].push_back(target); }
		std::vector<u32> mSortedList;

		std::map<u32, Tensor*> mTensorPtrTbl;

		void sortGraph();

		//計算グラフをマージする
		static void merge(std::shared_ptr<TensorGraph>&, std::shared_ptr<TensorGraph>);


#if _DEBUG
	public:
		~TensorGraph() { std::cout << "destruct\n"; }
	};
#endif
}