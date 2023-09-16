#pragma once

#include "TensorCore.h"
#include "TensorGraph.h"

namespace Aoba::Core::Math
{

	class Tensor
	{
		friend class TensorGraph;
	public://ƒƒ“ƒoŠÖ”
		template <typename ... Args>
		Tensor(Args ... args)
			: mTensorPtr(new TensorCore(args...))
			, mInstanceID(InstanceID++)
			, mTensorGraphPtr(std::make_unique<TensorGraphWrapper>())
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


		void forward();
		void backward();


		u32 getTensorDataSize() const { return mTensorPtr->getTensorDataSize(); }

		f32 operator[](u32 index) const { return (*mTensorPtr)[index]; }
		f32& operator[](u32 index) { return (*mTensorPtr)[index]; }

		f32& grad(u32 index) { return mTensorPtr->mDeltaTensorData[index]; }



		Tensor defineBinaryOperator(const Tensor&, const Tensor&,
			const std::function<void(TensorCore&)>&, const std::function<void(TensorCore&, std::vector<TensorCore*>)>&);
		Tensor operator+(const Tensor& tensorVariableR);
		//Tensor operator*(Tensor& tensorVariableR);
		//Tensor operator-(Tensor& tensorVariableR);
		void constructCalculationGraph2(const Tensor&, const Tensor&, const Tensor&);



		static bool isSameShape(const Tensor& tensorL, const Tensor& tensorR);
		static std::vector<Tensor*> InstancePtrTbl;

	private://ƒƒ“ƒoŠÖ”
		Tensor();
		Tensor makeTensorLike(const Tensor&);


	private://ƒƒ“ƒo•Ï”
		TensorCore* mTensorPtr;

		class TensorGraphWrapper
		{
		public:
			TensorGraphWrapper() : mTensorGraph(nullptr) {}
			std::shared_ptr<TensorGraph> mTensorGraph;
		};
		std::unique_ptr<TensorGraphWrapper> mTensorGraphPtr;

		const u32 mInstanceID;
		inline static u32 InstanceID = 0;
	};
}