#include "Tensor.h"

/////////////////////////////////////////////////////////////
// テンソル間の演算子はここで定義する
/////////////////////////////////////////////////////////////
namespace Aoba::Core::Math
{

	Tensor Tensor::operator+(Tensor& tensorVariableR)
	{
		if (!isSameShape(*(this), tensorVariableR))
		{
			assert(0);
		}

		//ここは修正すべきかも
		Tensor newTensorVariable = makeTensorVariableLike(tensorVariableR);
		//Tensor newTensorVariable = tensorVariableR;

		//順伝搬用の情報の保存
		newTensorVariable.mTensorPtr->mRootTensor.push_back(this->mTensorPtr);
		newTensorVariable.mTensorPtr->mRootTensor.push_back(tensorVariableR.mTensorPtr);
		newTensorVariable.mTensorPtr->mForwardFunction = [](TensorCore& tensor)
		{
			const u32 tensorSize = tensor.getTensorDataSize();
			for (u32 i = 0; i < tensorSize; i++)
			{
				tensor[i] = (*tensor.mRootTensor[0])[i] + (*tensor.mRootTensor[1])[i];
			}
		};

		newTensorVariable.mTensorPtr->forward();


		//逆伝搬用の情報の保存
		//左辺用
		{
			std::vector<TensorCore*> tmpTensorTbl;
			tmpTensorTbl.push_back(newTensorVariable.mTensorPtr);
			tmpTensorTbl.push_back(tensorVariableR.mTensorPtr);
			this->mTensorPtr->mFollowingTensorTbl.push_back(tmpTensorTbl);
			this->mTensorPtr->mBackwardFunctionTbl.push_back(
				[](TensorCore& tensor, std::vector<TensorCore*> tensorTbl)
				{
					for (u32 i = 0; i < tensor.getTensorDataSize(); i++)
					{
						tensor.getDeltaTensorData(i) += (*tensorTbl[0]).getDeltaTensorData(i);
					}
				});
		}
		//右辺用
		{
			std::vector<TensorCore*> tmpTensorTbl;
			tmpTensorTbl.push_back(newTensorVariable.mTensorPtr);
			tmpTensorTbl.push_back(this->mTensorPtr);
			tensorVariableR.mTensorPtr->mFollowingTensorTbl.push_back(tmpTensorTbl);
			tensorVariableR.mTensorPtr->mBackwardFunctionTbl.push_back(
				[](TensorCore& tensor, std::vector<TensorCore*> tensorTbl)
				{
					for (u32 i = 0; i < tensor.getTensorDataSize(); i++)
					{
						tensor.getDeltaTensorData(i) += (*tensorTbl[0]).getDeltaTensorData(i);
					}
				});
		}

		return newTensorVariable;
	}


}