#pragma once
#include <vector>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <functional>
#include <map>

#include "typeinfo.h"
#include "TensorCore.h"


namespace Aoba::Core::Math
{
	//Tensor���Ǘ�����singleton�N���X
	class TensorManager
	{
	public:
		TensorManager& getInstance()
		{
			static TensorManager manager;
			return manager;
		}

		void regist(u32& instanceNo, std::weak_ptr<TensorCore>& tensorCorePtr);

	private:
		TensorManager()
		{}
		~TensorManager() {}

		class TensorInformation
		{
		public:
			template <typename ... Args>
			TensorInformation(u32 id, Args ... args) : mTensorID(id) {}

			u32 mTensorID;
			std::shared_ptr<TensorCore> mTensorCore;

			std::vector<std::weak_ptr<TensorCore> > mForwardTensorTbl;//���������O���̃e���\��
			std::vector<std::weak_ptr<TensorCore> > mBackwardTensorTbl;//������������̃e���\��
		};


		/// <summary>
		/// ����܂łɐ������ꂽ�e���\���̐����L�^�i�ʎZ�j
		/// �����̊֌W�ŁA�e���\�����j������Ă����̐��͌���Ȃ��B
		/// </summary>
		inline static u32 mCreatedTensorNum = 0;

		/// <summary>
		/// �e�e���\���ɂ̓C���X�^���X�ԍ�������U���Ă���A
		/// ���̔ԍ��Ƀ|�C���^��Ή������Ă���B
		/// </summary>
		std::map<u32, std::shared_ptr<TensorInformation> > mTensorMap;


	};
}