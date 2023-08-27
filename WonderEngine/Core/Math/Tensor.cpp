#include "Tensor.h"

namespace Aoba::Core::Math
{
	/////////////////////////////////////////////////////////////
	// コンストラクタ
	/////////////////////////////////////////////////////////////
	Tensor::Tensor()
		:mInstanceID(InstanceID++)
	{
		mTensorPtr = nullptr;

		if (!(InstancePtrTbl.size() == mInstanceID))
		{
			assert(0);
		}
		InstancePtrTbl.push_back(this);
	}

	Tensor::Tensor(const Tensor& tensorVariable)
		:mInstanceID(InstanceID++)
	{
		mTensorPtr = new TensorCore(*(tensorVariable.mTensorPtr));

		if (!(InstancePtrTbl.size() == mInstanceID))
		{
			assert(0);
		}
		InstancePtrTbl.push_back(this);
	}

	Tensor::Tensor(Tensor&& tensor)
		:mInstanceID(tensor.mInstanceID)
	{
		mTensorPtr = tensor.mTensorPtr;
		tensor.mTensorPtr = nullptr;

		InstancePtrTbl[mInstanceID] = this;


		if (tensor.mTensorGraph)
		{
			mTensorGraph = tensor.mTensorGraph;
			tensor.mTensorGraph = nullptr;
			mTensorGraph->mTensorPtrTbl[mInstanceID] = this;
		}
	}

	Tensor::~Tensor()
	{
		//mTensorPtrがnullptrの時はムーブされた時のみで、
		//その時はムーブ先が残り続けるのでInstancePtrTblは何も触らないようにする。
		if (mTensorPtr == nullptr)
		{
			return;
		}
		InstancePtrTbl[mInstanceID] = nullptr;
		delete mTensorPtr;
	}

	void Tensor::forward()
	{
		const auto& sortedList = mTensorGraph->mSortedList;
		const u32 index = mInstanceID;

		auto iter = std::find(sortedList.begin(), sortedList.end(), index);
		if (iter == sortedList.end())
		{
			assert(0);
		}

		for (; iter != sortedList.end(); iter++)
		{
			const u32 index = (*iter);
			mTensorGraph->mTensorPtrTbl[index]->mTensorPtr->forward();

		}
	}

	void Tensor::backward()
	{
		const auto& sortedList = mTensorGraph->mSortedBackwardList;
		const u32 index = mInstanceID;

		auto iter = std::find(sortedList.begin(), sortedList.end(), index);
		if (iter == sortedList.end())
		{
			assert(0);
		}

		for (; iter != sortedList.end(); iter++)
		{
			const u32 index = (*iter);
			mTensorGraph->mTensorPtrTbl[index]->mTensorPtr->backward();

		}
	}


	/////////////////////////////////////////////////////////////
	// 引数に与えられたテンソルと同じ形状のテンソルを生成する。
	/////////////////////////////////////////////////////////////
	Tensor Tensor::makeTensorLike(const Tensor& tensorVariable)
	{
		Tensor newTensorVariable{};
		newTensorVariable.mTensorPtr = TensorCore::createTensorPtrLike(*(tensorVariable.mTensorPtr));
		return newTensorVariable;
	}

	/////////////////////////////////////////////////////////////
	// staticメンバ関数
	// ２つのテンソルの形状を比較
	/////////////////////////////////////////////////////////////
	bool Tensor::isSameShape(const Tensor& tensorL, const Tensor& tensorR)
	{
		return TensorCore::isSameShape(*tensorL.mTensorPtr, *tensorR.mTensorPtr);
	}

	std::vector<Tensor*> Tensor::InstancePtrTbl = std::vector<Tensor*>();
}