#include "TensorVariable.h"

namespace Aoba::Core::Math
{
	/////////////////////////////////////////////////////////////
	// �R���X�g���N�^
	/////////////////////////////////////////////////////////////

	TensorVariable::TensorVariable(const TensorVariable& tensorVariable)
		:mInstanceID(InstanceID++)
	{
		mTensorPtr = new Tensor(*(tensorVariable.mTensorPtr));
	}

	TensorVariable::TensorVariable(TensorVariable&& tensor)
		:mInstanceID(tensor.mInstanceID)
	{
		mTensorPtr = tensor.mTensorPtr;
		tensor.mTensorPtr = nullptr;
	}

	TensorVariable::~TensorVariable()
	{
		delete mTensorPtr;
	}


	/////////////////////////////////////////////////////////////
	// �����ɗ^����ꂽ�e���\���Ɠ����`��̃e���\���𐶐�����B
	/////////////////////////////////////////////////////////////


	TensorVariable TensorVariable::makeTensorVariableLike(const TensorVariable& tensorVariable)
	{
		TensorVariable newTensorVariable = Tensor::createTensorLike(*(tensorVariable.mTensorPtr));
		return newTensorVariable;
	}



	/////////////////////////////////////////////////////////////
	//���Z�q
	/////////////////////////////////////////////////////////////

	TensorVariable TensorVariable::operator+(TensorVariable& tensorVariableR)
	{
		if (!isSameShape(*(this), tensorVariableR))
		{
			assert(0);
		}

		//�����͏C�����ׂ�����
		//TensorVariable newTensorVariable = createLike(tensorVariableR);
		TensorVariable newTensorVariable = tensorVariableR;

		//���`���p�̏��̕ۑ�
		newTensorVariable.mTensorPtr->mRootTensor.push_back(this->mTensorPtr);
		newTensorVariable.mTensorPtr->mRootTensor.push_back(tensorVariableR.mTensorPtr);
		newTensorVariable.mTensorPtr->mForwardFunction = [](Tensor& tensor)
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
			std::vector<Tensor*> tmpTensorTbl;
			tmpTensorTbl.push_back(newTensorVariable.mTensorPtr);
			tmpTensorTbl.push_back(tensorVariableR.mTensorPtr);
			this->mTensorPtr->mFollowingTensorTbl.push_back(tmpTensorTbl);
			this->mTensorPtr->mBackwardFunctionTbl.push_back(
				[](Tensor& tensor, std::vector<Tensor*> tensorTbl)
				{
					for (u32 i = 0; i < tensor.getTensorDataSize(); i++)
					{
						tensor.getDeltaTensorData(i) += (*tensorTbl[0]).getDeltaTensorData(i);
					}
				});
		}
		//�E�ӗp
		{
			std::vector<Tensor*> tmpTensorTbl;
			tmpTensorTbl.push_back(newTensorVariable.mTensorPtr);
			tmpTensorTbl.push_back(this->mTensorPtr);
			tensorVariableR.mTensorPtr->mFollowingTensorTbl.push_back(tmpTensorTbl);
			tensorVariableR.mTensorPtr->mBackwardFunctionTbl.push_back(
				[](Tensor& tensor, std::vector<Tensor*> tensorTbl)
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