#include "Tensor.h"

namespace Aoba::Core::Math
{
	/////////////////////////////////////////////////////////////
	// �R���X�g���N�^
	/////////////////////////////////////////////////////////////
	Tensor::Tensor()
		:mInstanceID(InstanceID++)
	{
		mTensorPtr = nullptr;

		if (!(InstancePtrTbl.size() == mInstanceID))
		{
			assert(0);
		}
		InstancePtrTbl.push_back(this);
	}

	Tensor::Tensor(const Tensor& tensorVariable)
		:mInstanceID(InstanceID++)
	{
		mTensorPtr = new TensorCore(*(tensorVariable.mTensorPtr));

		if (!(InstancePtrTbl.size() == mInstanceID))
		{
			assert(0);
		}
		InstancePtrTbl.push_back(this);
	}

	Tensor::Tensor(Tensor&& tensor)
		:mInstanceID(tensor.mInstanceID)
	{
		mTensorPtr = tensor.mTensorPtr;
		tensor.mTensorPtr = nullptr;

		InstancePtrTbl[mInstanceID] = this;


		if (tensor.mTensorGraph)
		{
			mTensorGraph = tensor.mTensorGraph;
			tensor.mTensorGraph = nullptr;
			mTensorGraph->mTensorPtrTbl[mInstanceID] = this;
		}
	}

	Tensor::~Tensor()
	{
		//mTensorPtr��nullptr�̎��̓��[�u���ꂽ���݂̂ŁA
		//���̎��̓��[�u�悪�c�葱����̂�InstancePtrTbl�͉����G��Ȃ��悤�ɂ���B
		if (mTensorPtr == nullptr)
		{
			return;
		}
		InstancePtrTbl[mInstanceID] = nullptr;
		delete mTensorPtr;
	}

	void Tensor::forward()
	{
		const auto& sortedList = mTensorGraph->mSortedList;
		const u32 index = mInstanceID;

		auto iter = std::find(sortedList.begin(), sortedList.end(), index);
		if (iter == sortedList.end())
		{
			assert(0);
		}

		for (; iter != sortedList.end(); iter++)
		{
			const u32 index = (*iter);
			mTensorGraph->mTensorPtrTbl[index]->mTensorPtr->forward();

		}
	}

	void Tensor::backward()
	{
		const auto& sortedList = mTensorGraph->mSortedBackwardList;
		const u32 index = mInstanceID;

		auto iter = std::find(sortedList.begin(), sortedList.end(), index);
		if (iter == sortedList.end())
		{
			assert(0);
		}

		for (; iter != sortedList.end(); iter++)
		{
			const u32 index = (*iter);
			mTensorGraph->mTensorPtrTbl[index]->mTensorPtr->backward();

		}
	}


	/////////////////////////////////////////////////////////////
	// �����ɗ^����ꂽ�e���\���Ɠ����`��̃e���\���𐶐�����B
	/////////////////////////////////////////////////////////////
	Tensor Tensor::makeTensorLike(const Tensor& tensorVariable)
	{
		Tensor newTensorVariable{};
		newTensorVariable.mTensorPtr = TensorCore::createTensorPtrLike(*(tensorVariable.mTensorPtr));
		return newTensorVariable;
	}

	/////////////////////////////////////////////////////////////
	// static�����o�֐�
	// �Q�̃e���\���̌`����r
	/////////////////////////////////////////////////////////////
	bool Tensor::isSameShape(const Tensor& tensorL, const Tensor& tensorR)
	{
		return TensorCore::isSameShape(*tensorL.mTensorPtr, *tensorR.mTensorPtr);
	}

	std::vector<Tensor*> Tensor::InstancePtrTbl = std::vector<Tensor*>();
}