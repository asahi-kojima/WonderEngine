#pragma once

#include "Tensor.h"

namespace Aoba::Core::Math
{

	class TensorVariable
	{
	public:
		template <typename ... Args>
		TensorVariable(Args ... args)
			: mTensorPtr(new Tensor(args...))
		{

		}

		TensorVariable(const TensorVariable&);
		TensorVariable(TensorVariable&&);

		TensorVariable operator+(TensorVariable& tensorVariableR)
		{
			if (!isSameShape(*(this), tensorVariableR))
			{
				assert(0);
			}

			TensorVariable newTensorVariable = tensorVariableR;

			//èáì`î¿ópÇÃèÓïÒÇÃï€ë∂
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


			//ãtì`î¿ópÇÃèÓïÒÇÃï€ë∂
			//ç∂ï”óp
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
			//âEï”óp
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

		void forward() { mTensorPtr->forward(); }
		void backward() { mTensorPtr->backward(); }

		inline static bool isSameShape(const TensorVariable& tensorL, const TensorVariable& tensorR)
		{
			return Tensor::isSameShape(*tensorL.mTensorPtr, *tensorR.mTensorPtr);
		}

		u32 getTensorDataSize() const { return mTensorPtr->getTensorDataSize(); }

		f32 operator[](u32 index) const { return (*mTensorPtr)[index]; }
		f32& operator[](u32 index) { return (*mTensorPtr)[index]; }

#if _DEBUG
		Tensor* getTensor()
		{
			return mTensorPtr;
		}
#endif
	private:
		Tensor* mTensorPtr;
	};
}