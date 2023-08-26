#include <cassert>

#include "TensorCore.h"

namespace Aoba::Core::Math
{
	TensorCore::TensorCore()
		:mTensorDataSize(0)
	{
	}

	TensorCore::TensorCore(const TensorCore& tensor)
		: mTensorDataSize(tensor.mTensorDataSize)
	{
		mTensorDimension = tensor.mTensorDimension;
		if (mTensorDimension == 0)
		{
			assert(0);
		}

		//������Ԃ̊e�����̃T�C�Y�����肷��B
		mEachAxisSize.resize(mTensorDimension);
		for (u32 i = 0; i < mTensorDimension; i++)
		{
			mEachAxisSize[i] = tensor.mEachAxisSize[i];
		}

		//�f�[�^�T�C�Y���߂�B
		mTensorData.resize(mTensorDataSize);
		mDeltaTensorData.resize(mTensorDataSize);


		for (u32 i = 0; i < mTensorDataSize; i++)
		{
			mTensorData[i] = tensor[i];
		}

	}

	//������
	TensorCore::TensorCore(TensorCore&&)
		: mTensorDataSize(0)
	{
		//������
	}


	f32 TensorCore::operator[](u32 index) const
	{
#if _DEBUG
		if (index < 0 || index >= mTensorDataSize)
		{
			assert(0);
		}
#endif
		return mTensorData[index];
	}
	f32& TensorCore::operator[](u32 index)
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
	u32 TensorCore::getTensorDataSize() const
	{
		return mTensorDataSize;
	}

	f32& TensorCore::getDeltaTensorData(u32 index)
	{
#if _DEBUG
		assert(index < mTensorDataSize);
#endif
		return mDeltaTensorData[index];
	}

	f32 TensorCore::getDeltaTensorData(u32 index) const
	{
#if _DEBUG
		assert(index < mTensorDataSize);
#endif
		return mDeltaTensorData[index];
	}

	void TensorCore::forward()
	{
		mForwardFunction(*this);
	}

	void TensorCore::backward()
	{
		const u32 branchNum = mBackwardFunctionTbl.size();

		for (u32 i = 0; i < branchNum; i++)
		{
			mBackwardFunctionTbl[i](*this, mFollowingTensorTbl[i]);
		}
	}

	void TensorCore::transpose(u32 axis0, u32 axis1)
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

	//�����܂œ����`�����邾���ŁA�v�Z�O���t�Ȃǂ͓Ɨ�
	TensorCore* TensorCore::createTensorPtrLike(const TensorCore& tensor)
	{
		TensorCore* newTensor = new TensorCore();

		//newTensor->mTensorDataSize = tensor.mTensorDataSize;
		u32* p = const_cast<u32*>(&(newTensor->mTensorDataSize));
		*p = tensor.mTensorDataSize;

		newTensor->mTensorDimension = tensor.mTensorDimension;
		if (newTensor->mTensorDimension == 0)
		{
			assert(0);
		}

		//������Ԃ̊e�����̃T�C�Y�����肷��B
		newTensor->mEachAxisSize.resize(tensor.mTensorDimension);
		for (u32 i = 0; i < tensor.mTensorDimension; i++)
		{
			newTensor->mEachAxisSize[i] = tensor.mEachAxisSize[i];
		}

		//�f�[�^�T�C�Y���߂�B
		newTensor->mTensorData.resize(tensor.mTensorDataSize);
		newTensor->mDeltaTensorData.resize(tensor.mTensorDataSize);

		return newTensor;
	}

	bool TensorCore::isSameShape(const TensorCore& tensorL, const TensorCore& tensorR)
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