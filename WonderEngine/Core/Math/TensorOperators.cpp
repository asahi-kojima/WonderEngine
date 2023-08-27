#include "Tensor.h"

/////////////////////////////////////////////////////////////
// テンソル間の演算子はここで定義する
/////////////////////////////////////////////////////////////
namespace Aoba::Core::Math
{

	Tensor Tensor::operator+(Tensor& tensorR)
	{
		Tensor& tensorL = *this;

		if (!isSameShape(tensorL, tensorR))
		{
			assert(0);
		}

		Tensor newTensor = makeTensorLike(tensorR);

		//順伝搬用の情報の保存
		newTensor.mTensorPtr->mRootTensor.push_back(tensorL.mTensorPtr);
		newTensor.mTensorPtr->mRootTensor.push_back(tensorR.mTensorPtr);
		newTensor.mTensorPtr->mForwardFunction = [](TensorCore& tensor)
		{
			const u32 tensorSize = tensor.getTensorDataSize();
			for (u32 i = 0; i < tensorSize; i++)
			{
				tensor[i] = (*tensor.mRootTensor[0])[i] + (*tensor.mRootTensor[1])[i];
			}
		};

		newTensor.mTensorPtr->forward();


		//逆伝搬用の情報の保存
		//左辺用
		{
			std::vector<TensorCore*> tmpTensorTbl;
			tmpTensorTbl.push_back(newTensor.mTensorPtr);
			tmpTensorTbl.push_back(tensorR.mTensorPtr);
			tensorL.mTensorPtr->mFollowingTensorTbl.push_back(tmpTensorTbl);
			tensorL.mTensorPtr->mBackwardFunctionTbl.push_back(
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
			tmpTensorTbl.push_back(newTensor.mTensorPtr);
			tmpTensorTbl.push_back(tensorL.mTensorPtr);
			tensorR.mTensorPtr->mFollowingTensorTbl.push_back(tmpTensorTbl);
			tensorR.mTensorPtr->mBackwardFunctionTbl.push_back(
				[](TensorCore& tensor, std::vector<TensorCore*> tensorTbl)
				{
					for (u32 i = 0; i < tensor.getTensorDataSize(); i++)
					{
						tensor.getDeltaTensorData(i) += (*tensorTbl[0]).getDeltaTensorData(i);
					}
				});
		}


		//グラフの作成
		constructComutationalGraph2(tensorL, tensorR, newTensor);


		////毎回ソートする必要はないかも
		////backward()呼ぶ時に掛ければOK
		//tensorL.mTensorGraph->sortGraph();

		return newTensor;
	}

	void Tensor::constructComutationalGraph2(Tensor& tensorL, Tensor& tensorR, Tensor& newTensor)
	{
		bool hasGraphL = (tensorL.mTensorGraph ? true : false);
		bool hasGraphR = (tensorR.mTensorGraph ? true : false);

		if (hasGraphL && hasGraphR)
		{
			//同じグラフに属する
			if (tensorL.mTensorGraph.get() == tensorR.mTensorGraph.get())
			{
				//この時は何もしなくていい。
			}
			//別のグラフに属する
			else
			{
				TensorGraph::merge(tensorL.mTensorGraph, tensorR.mTensorGraph);
			}
		}
		else if (hasGraphL && !hasGraphR)
		{
			tensorR.mTensorGraph = tensorL.mTensorGraph;

			tensorL.mTensorGraph->mTensorPtrTbl[tensorR.mInstanceID] = &tensorR;
		}
		else if (!hasGraphL && hasGraphR)
		{
			tensorL.mTensorGraph = tensorR.mTensorGraph;

			tensorR.mTensorGraph->mTensorPtrTbl[tensorL.mInstanceID] = &tensorL;
		}
		else
		{
			tensorL.mTensorGraph = std::make_shared<TensorGraph>();
			tensorR.mTensorGraph = tensorL.mTensorGraph;

			tensorL.mTensorGraph->mTensorPtrTbl[tensorL.mInstanceID] = &tensorL;
			tensorL.mTensorGraph->mTensorPtrTbl[tensorR.mInstanceID] = &tensorR;
		}

		newTensor.mTensorGraph = tensorL.mTensorGraph;
		tensorL.mTensorGraph->mTensorPtrTbl[newTensor.mInstanceID] = &newTensor;


		tensorL.mTensorGraph->insert(tensorL.mInstanceID, newTensor.mInstanceID);
		tensorR.mTensorGraph->insert(tensorR.mInstanceID, newTensor.mInstanceID);

		//毎回ソートする必要はないかも
		//backward()呼ぶ時に掛ければOK
		tensorL.mTensorGraph->sortGraph();
	}
}