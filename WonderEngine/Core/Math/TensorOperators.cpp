#include "Tensor.h"

/////////////////////////////////////////////////////////////
// �e���\���Ԃ̉��Z�q�͂����Œ�`����
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

		//���`���p�̏��̕ۑ�
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


		//�t�`���p�̏��̕ۑ�
		//���ӗp
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
		//�E�ӗp
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


		//�O���t�̍쐬
		constructComutationalGraph2(tensorL, tensorR, newTensor);


		////����\�[�g����K�v�͂Ȃ�����
		////backward()�ĂԎ��Ɋ|�����OK
		//tensorL.mTensorGraph->sortGraph();

		return newTensor;
	}

	void Tensor::constructComutationalGraph2(Tensor& tensorL, Tensor& tensorR, Tensor& newTensor)
	{
		bool hasGraphL = (tensorL.mTensorGraph ? true : false);
		bool hasGraphR = (tensorR.mTensorGraph ? true : false);

		if (hasGraphL && hasGraphR)
		{
			//�����O���t�ɑ�����
			if (tensorL.mTensorGraph.get() == tensorR.mTensorGraph.get())
			{
				//���̎��͉������Ȃ��Ă����B
			}
			//�ʂ̃O���t�ɑ�����
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

		//����\�[�g����K�v�͂Ȃ�����
		//backward()�ĂԎ��Ɋ|�����OK
		tensorL.mTensorGraph->sortGraph();
	}
}