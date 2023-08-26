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
			, mInstanceID(InstanceID++)
		{

		}
		TensorVariable();
		TensorVariable(const TensorVariable&);
		TensorVariable(TensorVariable&&);
		~TensorVariable();

		TensorVariable operator+(TensorVariable& tensorVariableR);

		void forward() { mTensorPtr->forward(); }
		void backward() { mTensorPtr->backward(); }


		u32 getTensorDataSize() const { return mTensorPtr->getTensorDataSize(); }

		f32 operator[](u32 index) const { return (*mTensorPtr)[index]; }
		f32& operator[](u32 index) { return (*mTensorPtr)[index]; }

		inline static bool isSameShape(const TensorVariable& tensorL, const TensorVariable& tensorR)
		{
			return Tensor::isSameShape(*tensorL.mTensorPtr, *tensorR.mTensorPtr);
		}

#if _DEBUG
		Tensor* getTensor()
		{
			return mTensorPtr;
		}
#endif


	private:
		TensorVariable makeTensorVariableLike(const TensorVariable&);


	private:
		Tensor* mTensorPtr;
		const u32 mInstanceID;
		inline static u32 InstanceID = 0;
	};
}