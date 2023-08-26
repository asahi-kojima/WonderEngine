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
	}

	Tensor::Tensor(const Tensor& tensorVariable)
		:mInstanceID(InstanceID++)
	{
		mTensorPtr = new TensorCore(*(tensorVariable.mTensorPtr));
	}

	Tensor::Tensor(Tensor&& tensor)
		:mInstanceID(tensor.mInstanceID)
	{
		mTensorPtr = tensor.mTensorPtr;
		tensor.mTensorPtr = nullptr;
	}

	Tensor::~Tensor()
	{
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
	//演算子
	/////////////////////////////////////////////////////////////

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