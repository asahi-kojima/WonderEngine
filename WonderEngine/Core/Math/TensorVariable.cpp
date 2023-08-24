#include "TensorVariable.h"

namespace Aoba::Core::Math
{
	TensorVariable::TensorVariable(const TensorVariable& tensorVariable)
	{
		mTensorPtr = new Tensor(*(tensorVariable.mTensorPtr));
	}

	TensorVariable::TensorVariable(TensorVariable&& tensor)
	{
		mTensorPtr = tensor.mTensorPtr;
		tensor.mTensorPtr = nullptr;
	}

}