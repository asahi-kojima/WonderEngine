#pragma once

#include "TensorCore.h"

namespace Aoba::Core::Math
{

	class Tensor
	{
	public:
		template <typename ... Args>
		Tensor(Args ... args)
			: mTensorPtr(new TensorCore(args...))
			, mInstanceID(InstanceID++)
		{

		}
		Tensor();
		Tensor(const Tensor&);
		Tensor(Tensor&&);
		~Tensor();

		Tensor operator+(Tensor& tensorVariableR);

		void forward() { mTensorPtr->forward(); }
		void backward() { mTensorPtr->backward(); }


		u32 getTensorDataSize() const { return mTensorPtr->getTensorDataSize(); }

		f32 operator[](u32 index) const { return (*mTensorPtr)[index]; }
		f32& operator[](u32 index) { return (*mTensorPtr)[index]; }

		inline static bool isSameShape(const Tensor& tensorL, const Tensor& tensorR)
		{
			return TensorCore::isSameShape(*tensorL.mTensorPtr, *tensorR.mTensorPtr);
		}

#if _DEBUG
		TensorCore* getTensor()
		{
			return mTensorPtr;
		}
#endif


	private:
		Tensor makeTensorVariableLike(const Tensor&);


	private:
		TensorCore* mTensorPtr;
		const u32 mInstanceID;
		inline static u32 InstanceID = 0;
	};
}