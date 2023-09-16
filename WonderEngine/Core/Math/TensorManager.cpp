#include "TensorManager.h"


namespace Aoba::Core::Math
{
	void TensorManager::regist(u32& instanceNo, std::weak_ptr<TensorCore>& tensorCorePtr)
	{
		//�C���X�^���X�ԍ���o�^
		instanceNo = mCreatedTensorNum++;

		//���̃e���\���̑S�Ă̏���ێ�����C���X�^���X�B
		std::shared_ptr<TensorInformation> tensorInformation = std::make_shared<TensorInformation>(instanceNo);
		mTensorMap[instanceNo] = tensorInformation;
		tensorInformation->mTensorID = instanceNo;

		mTensorMap[instanceNo] = 
		TensorInformation info;
		mTensorMap[mCreatedTensorNum] = std::make_shared<TensorCore2>();
		tensorCorePtr = std::weak_ptr<TensorCore2>(mTensorMap[mCreatedTensorNum]);

		mCreatedTensorNum++;


	}
}