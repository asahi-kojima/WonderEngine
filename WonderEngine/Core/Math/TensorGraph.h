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
		std::vector<std::vector<u32> > mGraph;
		std::map<u32, Tensor*> mTensorPtrTbl;

		static void merge(std::shared_ptr<TensorGraph>&, std::shared_ptr<TensorGraph>);

#if _DEBUG
	public:
		~TensorGraph() { std::cout << "destruct\n"; }
	};
#endif
}