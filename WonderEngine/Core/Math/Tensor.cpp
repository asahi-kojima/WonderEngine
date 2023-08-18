#include <cassert>

#include "Tensor.h"

namespace Aoba::Core::Math
{


	Tensor& Tensor::operator=(Tensor&&)
	{
		// TODO: return ステートメントをここに挿入します
		return *this;
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

	u32 Tensor::getTensorSize() const
	{
		return mTensorDataSize;
	}

}