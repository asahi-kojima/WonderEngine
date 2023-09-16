#include "Tensor.h"

/////////////////////////////////////////////////////////////
// テンソル間の演算子はここで定義する
/////////////////////////////////////////////////////////////
namespace Aoba::Core::Math
{
	Tensor Tensor::defineBinaryOperator(const Tensor& tensorL, const Tensor& tensorR,
		const std::function<void(TensorCore&)>& forwardRule, const std::function<void(TensorCore&, std::vector<TensorCore*>)>& backwardRule)
	{
		Tensor targetTensor = makeTensorLike(tensorR);

		if (!isSameShape(tensorL, tensorR))
		{
			assert(0);
		}

		//順伝搬用の情報の保存
		targetTensor.mTensorPtr->mRootTensor.push_back(tensorL.mTensorPtr);
		targetTensor.mTensorPtr->mRootTensor.push_back(tensorR.mTensorPtr);
		targetTensor.mTensorPtr->mForwardFunction = forwardRule;

		targetTensor.mTensorPtr->forward();

		//逆伝搬用の情報の保存
		//左辺用
		{
			std::vector<TensorCore*> tmpTensorTbl;
			tmpTensorTbl.push_back(targetTensor.mTensorPtr);
			tmpTensorTbl.push_back(tensorR.mTensorPtr);
			tensorL.mTensorPtr->mFollowingTensorTbl.push_back(std::move(tmpTensorTbl));
			tensorL.mTensorPtr->mBackwardFunctionTbl.push_back(backwardRule);
		}
		//右辺用
		{
			std::vector<TensorCore*> tmpTensorTbl;
			tmpTensorTbl.push_back(targetTensor.mTensorPtr);
			tmpTensorTbl.push_back(tensorL.mTensorPtr);
			tensorR.mTensorPtr->mFollowingTensorTbl.push_back(std::move(tmpTensorTbl));
			tensorR.mTensorPtr->mBackwardFunctionTbl.push_back(backwardRule);
		}


		//グラフの作成
		constructCalculationGraph2(tensorL, tensorR, targetTensor);

		return targetTensor;
	}


	Tensor Tensor::operator+(const Tensor& tensorR)
	{
		Tensor& tensorL = *this;

		auto forwardRule = [](TensorCore& tensor)
		{
			const u32 tensorSize = tensor.getTensorDataSize();
			for (u32 i = 0; i < tensorSize; i++)
			{
				tensor[i] = (*tensor.mRootTensor[0])[i] + (*tensor.mRootTensor[1])[i];
			}
		};

		auto backwardRule = [](TensorCore& tensor, std::vector<TensorCore*> tensorTbl)
		{
			for (u32 i = 0; i < tensor.getTensorDataSize(); i++)
			{
				tensor.getDeltaTensorData(i) += (*tensorTbl[0]).getDeltaTensorData(i);
			}
		};



		return defineBinaryOperator(tensorL, tensorR, forwardRule, backwardRule);
	}


	//Tensor Tensor::operator*(Tensor& tensorR)
	//{
	//	Tensor& tensorL = *this;

	//	auto forwardRule = [](TensorCore& tensor)
	//	{
	//		const u32 tensorSize = tensor.getTensorDataSize();
	//		for (u32 i = 0; i < tensorSize; i++)
	//		{
	//			tensor[i] = (*tensor.mRootTensor[0])[i] * (*tensor.mRootTensor[1])[i];
	//		}
	//	};
	//	 
	//	auto backwardRule = [](TensorCore& tensor, std::vector<TensorCore*> tensorTbl)
	//	{
	//		for (u32 i = 0; i < tensor.getTensorDataSize(); i++)
	//		{
	//			tensor.getDeltaTensorData(i) += (*tensorTbl[0]).getDeltaTensorData(i) * (*tensorTbl[1])[i];
	//		}
	//	};

	//	return defineBinaryOperator(tensorL, tensorR, forwardRule, backwardRule);
	//}


	void Tensor::constructCalculationGraph2(const Tensor& tensorL, const Tensor& tensorR, const Tensor& newTensor)
	{
		bool hasGraphL = (tensorL.mTensorGraphPtr->mTensorGraph ? true : false);
		bool hasGraphR = (tensorR.mTensorGraphPtr->mTensorGraph ? true : false);

		if (hasGraphL && hasGraphR)
		{
			//同じグラフに属する
			if (tensorL.mTensorGraphPtr->mTensorGraph.get() == tensorR.mTensorGraphPtr->mTensorGraph.get())
			{
				//この時は何もしなくていい。
			}
			//別のグラフに属する
			else
			{
				TensorGraph::merge(tensorL.mTensorGraphPtr->mTensorGraph, tensorR.mTensorGraphPtr->mTensorGraph);
			}
		}
		else if (hasGraphL && !hasGraphR)
		{
			tensorR.mTensorGraphPtr->mTensorGraph = tensorL.mTensorGraphPtr->mTensorGraph;

			tensorL.mTensorGraphPtr->mTensorGraph->mTensorPtrTbl[tensorR.mInstanceID] = &tensorR;
		}
		else if (!hasGraphL && hasGraphR)
		{
			tensorL.mTensorGraphPtr->mTensorGraph = tensorR.mTensorGraphPtr->mTensorGraph;

			tensorR.mTensorGraphPtr->mTensorGraph->mTensorPtrTbl[tensorL.mInstanceID] = &tensorL;
		}
		else
		{
			tensorL.mTensorGraphPtr->mTensorGraph = std::make_shared<TensorGraph>();
			tensorR.mTensorGraphPtr->mTensorGraph = tensorL.mTensorGraphPtr->mTensorGraph;

			tensorL.mTensorGraphPtr->mTensorGraph->mTensorPtrTbl[tensorL.mInstanceID] = &tensorL;
			tensorL.mTensorGraphPtr->mTensorGraph->mTensorPtrTbl[tensorR.mInstanceID] = &tensorR;
		}

		newTensor.mTensorGraphPtr->mTensorGraph = tensorL.mTensorGraphPtr->mTensorGraph;
		tensorL.mTensorGraphPtr->mTensorGraph->mTensorPtrTbl[newTensor.mInstanceID] = &newTensor;


		tensorL.mTensorGraphPtr->mTensorGraph->insert(tensorL.mInstanceID, newTensor.mInstanceID);
		tensorR.mTensorGraphPtr->mTensorGraph->insert(tensorR.mInstanceID, newTensor.mInstanceID);

		//毎回ソートする必要はないかも
		//backward()呼ぶ時に掛ければOK
		tensorL.mTensorGraphPtr->mTensorGraph->sortGraph();
	}
}