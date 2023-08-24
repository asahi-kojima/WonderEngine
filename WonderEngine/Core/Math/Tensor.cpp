#include <cassert>

#include "Tensor.h"

namespace Aoba::Core::Math
{
	//Tensor::Tensor(Tensor&&)
	//	: mTensorDataSize(0)
	//	, mInstanceID(InstanceID)
	//{
	//}
	//Tensor& Tensor::operator=(Tensor&&)
	//{
	//	// TODO: return ステートメントをここに挿入します
	//	return *this;
	//}

	Tensor::Tensor(const Tensor& tensor)
		: mTensorDataSize(tensor.mTensorDataSize)
		, mInstanceID(InstanceID)
	{
		mTensorDimension = tensor.mTensorDimension;
		if (mTensorDimension == 0)
		{
			assert(0);
		}

		//初期状態の各次元のサイズを決定する。
		mEachAxisSize.resize(mTensorDimension);
		for (u32 i = 0; i < mTensorDimension; i++)
		{
			mEachAxisSize[i] = tensor.mEachAxisSize[i];
		}

		//データサイズを定める。
		mTensorData.resize(mTensorDataSize);
		mDeltaTensorData.resize(mTensorDataSize);

		InstanceID++;
	}

	//OK
	f32 Tensor::operator[](u32 index) const
	{
#if _DEBUG
		if (index < 0 || index >= mTensorDataSize)
		{
			assert(0);
		}
#endif
		return mTensorData[index];
	}

	//OK
	f32& Tensor::operator[](u32 index)
	{
#if _DEBUG
		if (index < 0 || index >= mTensorDataSize)
		{
			assert(0);
		}
#endif
		return mTensorData[index];
	}

	//OK
	u32 Tensor::getTensorDataSize() const
	{
		return mTensorDataSize;
	}



	void Tensor::forward()
	{
		mForwardFunction(*this);
	}

	void Tensor::backward()
	{
		const u32 branchNum = mBackwardFunctionTbl.size();

		for (u32 i = 0; i < branchNum; i++)
		{
			mBackwardFunctionTbl[i](*this, mFollowingTensorTbl[i]);
		}
	}

	bool Tensor::isSameShape(const Tensor& tensorL, const Tensor& tensorR)
	{
		if (tensorL.mTensorDimension != tensorR.mTensorDimension)
		{
			return false;
		}

		if (tensorL.mTensorDataSize != tensorR.mTensorDataSize)
		{
			return false;
		}

		for (u32 i = 0; i < tensorL.mTensorDimension; i++)
		{
			if (tensorL.mEachAxisSize[i] != tensorR.mEachAxisSize[i])
			{
				return false;
			}
		}

		return true;
	}

}