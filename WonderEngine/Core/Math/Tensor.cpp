#include <cassert>

#include "Tensor.h"

namespace Aoba::Core::Math
{
	Tensor::Tensor()
		:mTensorDataSize(0)
	{
	}

	Tensor::Tensor(const Tensor& tensor)
		: mTensorDataSize(tensor.mTensorDataSize)
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


		for (u32 i = 0; i < mTensorDataSize; i++)
		{
			mTensorData[i] = tensor[i];
		}

	}

	//未実装
	Tensor::Tensor(Tensor&&)
		: mTensorDataSize(0)
	{
		//未実装
	}


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

	f32& Tensor::getDeltaTensorData(u32 index)
	{
#if _DEBUG
		assert(index < mTensorDataSize);
#endif
		return mDeltaTensorData[index];
	}

	f32 Tensor::getDeltaTensorData(u32 index) const
	{
#if _DEBUG
		assert(index < mTensorDataSize);
#endif
		return mDeltaTensorData[index];
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

	void Tensor::transpose(u32 axis0, u32 axis1)
	{
		if (mTensorDimension == 1)
		{
			std::cout << "current tensor is 1 dimension!" << std::endl;
			assert(0);
		}

		if (!(axis0 < mTensorDimension && axis1 < mTensorDimension))
		{
			std::cout << "designated axis (" << axis0 << " or " << axis1 << ") is over!" << std::endl;
			assert(0);
		}

		if (axis0 == axis1)
		{
			std::cout << "designated 2 axis is same." << std::endl;
			return;
		}
	}

	//あくまで同じ形状を作るだけで、計算グラフなどは独立
	Tensor* Tensor::createTensorPtrLike(const Tensor& tensor)
	{
		Tensor* newTensor = new Tensor();

		//newTensor->mTensorDataSize = tensor.mTensorDataSize;
		u32* p = const_cast<u32*>(&(newTensor->mTensorDataSize));
		*p = tensor.mTensorDataSize;

		newTensor->mTensorDimension = tensor.mTensorDimension;
		if (newTensor->mTensorDimension == 0)
		{
			assert(0);
		}

		//初期状態の各次元のサイズを決定する。
		newTensor->mEachAxisSize.resize(tensor.mTensorDimension);
		for (u32 i = 0; i < tensor.mTensorDimension; i++)
		{
			newTensor->mEachAxisSize[i] = tensor.mEachAxisSize[i];
		}

		//データサイズを定める。
		newTensor->mTensorData.resize(tensor.mTensorDataSize);
		newTensor->mDeltaTensorData.resize(tensor.mTensorDataSize);

		return newTensor;
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