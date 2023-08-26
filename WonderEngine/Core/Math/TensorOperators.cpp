#include "Tensor.h"

/////////////////////////////////////////////////////////////
// �e���\���Ԃ̉��Z�q�͂����Œ�`����
/////////////////////////////////////////////////////////////
namespace Aoba::Core::Math
{

	Tensor Tensor::operator+(Tensor& tensorVariableR)
	{
		if (!isSameShape(*(this), tensorVariableR))
		{
			assert(0);
		}

		//�����͏C�����ׂ�����
		Tensor newTensorVariable = makeTensorVariableLike(tensorVariableR);
		//Tensor newTensorVariable = tensorVariableR;

		//���`���p�̏��̕ۑ�
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


		//�t�`���p�̏��̕ۑ�
		//���ӗp
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
		//�E�ӗp
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