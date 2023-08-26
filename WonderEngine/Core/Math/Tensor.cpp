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



	/////////////////////////////////////////////////////////////
	// �����ɗ^����ꂽ�e���\���Ɠ����`��̃e���\���𐶐�����B
	/////////////////////////////////////////////////////////////
	Tensor Tensor::makeTensorVariableLike(const Tensor& tensorVariable)
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