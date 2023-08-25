#include "TensorVariable.h"

namespace Aoba::Core::Math
{
	/////////////////////////////////////////////////////////////
	// コンストラクタ
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
	// 引数に与えられたテンソルと同じ形状のテンソルを生成する。
	/////////////////////////////////////////////////////////////


	TensorVariable TensorVariable::makeTensorVariableLike(const TensorVariable& tensorVariable)
	{
		TensorVariable newTensorVariable = Tensor::createTensorLike(*(tensorVariable.mTensorPtr));
		return newTensorVariable;
	}



	/////////////////////////////////////////////////////////////
	//演算子
	/////////////////////////////////////////////////////////////

	TensorVariable TensorVariable::operator+(TensorVariable& tensorVariableR)
	{
		if (!isSameShape(*(this), tensorVariableR))
		{
			assert(0);
		}

		//ここは修正すべきかも
		//TensorVariable newTensorVariable = createLike(tensorVariableR);
		TensorVariable newTensorVariable = tensorVariableR;

		//順伝搬用の情報の保存
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


		//逆伝搬用の情報の保存
		//左辺用
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
		//右辺用
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