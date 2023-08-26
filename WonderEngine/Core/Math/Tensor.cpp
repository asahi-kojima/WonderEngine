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



	/////////////////////////////////////////////////////////////
	// 引数に与えられたテンソルと同じ形状のテンソルを生成する。
	/////////////////////////////////////////////////////////////
	Tensor Tensor::makeTensorVariableLike(const Tensor& tensorVariable)
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