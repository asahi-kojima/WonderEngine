#pragma once

#include "TensorCore.h"

namespace Aoba::Core::Math
{

	class Tensor
	{
	public://ƒƒ“ƒoŠÖ”
		template <typename ... Args>
		Tensor(Args ... args)
			: mTensorPtr(new TensorCore(args...))
			, mInstanceID(InstanceID++)
		{
			if (!(InstancePtrTbl.size() == mInstanceID))
			{
				assert(0);
			}
			InstancePtrTbl.push_back(this);
		}
		Tensor(const Tensor&);
		Tensor(Tensor&&);
		~Tensor();

		Tensor operator+(Tensor& tensorVariableR);

		void forward() { mTensorPtr->forward(); }
		void backward() { mTensorPtr->backward(); }


		u32 getTensorDataSize() const { return mTensorPtr->getTensorDataSize(); }

		f32 operator[](u32 index) const { return (*mTensorPtr)[index]; }
		f32& operator[](u32 index) { return (*mTensorPtr)[index]; }

		f32& grad(u32 index) { return mTensorPtr->mDeltaTensorData[index]; }



		static bool isSameShape(const Tensor& tensorL, const Tensor& tensorR);
		static std::vector<Tensor*> InstancePtrTbl;

	private://ƒƒ“ƒoŠÖ”
		Tensor();
		Tensor makeTensorVariableLike(const Tensor&);


	private://ƒƒ“ƒo•Ï”
		TensorCore* mTensorPtr;

		const u32 mInstanceID;
		inline static u32 InstanceID = 0;
	};
}