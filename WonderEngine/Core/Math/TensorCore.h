#pragma once
#include <vector>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <functional>

#include "typeinfo.h"


namespace Aoba::Core::Math
{
	class Tensor;

	class TensorCore
	{
		friend class Tensor;

	private://��X������private�ɕύX����B
		//�ϒ��R���X�g���N�^
		template <typename ...Args>
		TensorCore(Args ... args)
			: mTensorDataSize(0)
			, mWhetherToLearn(false)
		{
			mTensorDimension = sizeof...(args);
			if (mTensorDimension == 0)
			{
				assert(0);
			}

			//������Ԃ̊e�����̃T�C�Y�����肷��B
			mEachAxisSize.resize(mTensorDimension);
			constructorArgsDevider(0, args...);

			//�f�[�^�T�C�Y���߂�B
			u64 size = 1;
			for (const auto& eachSize : mEachAxisSize)
			{
				size *= eachSize;
			}

			mTensorData.resize(size);
			mDeltaTensorData.resize(size);

			//������菉�������邽�߂ɋ����I��const���O���Ă���B
			u32* p2mTensorDataSize = const_cast<u32*>(&mTensorDataSize);
			*p2mTensorDataSize = size;
		}

		//�f�t�H���g�R���X�g���N�^
		TensorCore();
		~TensorCore() = default;

		//�R�s�[�R���X�g���N�^�Ƒ�����Z�q
		TensorCore(const TensorCore&);
		TensorCore& operator=(const TensorCore&) = default;

		//���[�u�R���X�g���N�^�Ƒ�����Z�q
		TensorCore(TensorCore&&);
		TensorCore& operator=(TensorCore&&) = default;

		//�v�f�A�N�Z�X[index]���Z�q
		f32 operator[](u32 index) const;
		f32& operator[](u32 index);


		u32 getTensorDataSize() const;

		//�e���\���̒l�ւ̃A�N�Z�X
		template <typename ...Args>
		f32& operator()(Args ... args)
		{
			const u32 argsSize = sizeof...(args);
			if (argsSize != mTensorDimension)
			{
				std::cout << "Error : argument nums(=" << argsSize << ") are contradict with current tensor dimension(=" << mTensorDimension << ")." << std::endl;
				assert(0);
			}

			u32 eachAxisIndexTbl[argsSize];
			genericArgsDevider(0, eachAxisIndexTbl, args...);

			u32 prod = 1;
			u32 index = 0;

			for (s32 id = argsSize - 1; id >= 0; id--)
			{
#if _DEBUG
				assert(eachAxisIndexTbl[id] < mEachAxisSize[id]);
#endif
				index += eachAxisIndexTbl[id] * prod;
				prod *= mEachAxisSize[id];
			}

#if _DEBUG
			assert(index < mTensorDataSize);
#endif
			return mTensorData[index];
		}
		template <typename ...Args>
		f32 operator()(Args ... args) const
		{
			const u32 argsSize = sizeof...(args);
			if (argsSize != mTensorDimension)
			{
				std::cout << "Error : argument nums(=" << argsSize << ") are contradict with current tensor dimension(=" << std::endl;
				assert(0);
			}

			u32 eachAxisIndexTbl[argsSize];
			genericArgsDevider(0, eachAxisIndexTbl, args...);

			u32 prod = 1;
			u32 index = 0;

			for (s32 id = argsSize - 1; id >= 0; id--)
			{
#if _DEBUG
				assert(eachAxisIndexTbl[id] < mEachAxisSize[id]);
#endif
				index += eachAxisIndexTbl[id] * prod;
				prod *= mEachAxisSize[id];
			}

#if _DEBUG
			assert(index < mTensorDataSize);
#endif
			return mTensorData[index];
		}

		//�e���\���̔����l�ւ̃A�N�Z�X�B�@�f�o�b�O�p�̊֐��ȋC�����邩��A���Ԃ��ŏ���
		f32& getDeltaTensorData(u32 index);
		f32 getDeltaTensorData(u32 index) const;


		template <typename ...Args>
		void reshape(Args ... args)
		{
			const u32 reshapedTensorDim = sizeof...(args);

			//�ϒ��ϐ��𕪊����Ĕz��Ɋi�[
			s32 reshapedTensorEachAxisSizeTbl[reshapedTensorDim];
			reshapeArgsDevider(0, reshapedTensorEachAxisSizeTbl, args...);

			//�ϒ������̒���-1���Q�ȏ�Ȃ����Ƃ��m�F
			u32 minus1Counter = 0;
			s32 minus1Index = -1;
			for (u32 id = 0; id < reshapedTensorDim; id++)
			{
				const u32 arg = reshapedTensorEachAxisSizeTbl[id];
				if (arg == -1)
				{
					minus1Counter++;
					if (minus1Counter >= 2)
					{
						std::cout << "more than one -1 argument in \"reshape\" function" << std::endl;
						assert(0);
					}
					minus1Index = id;
				}
			}

			//-1���������ꍇ�ɁA�����K�؂Ȓl�ɒu���ł��邩�`�F�b�N
			if (minus1Counter == 1)
			{
				u32 prod = 1;
				for (u32 id = 0; id < reshapedTensorDim; id++)
				{
					if (id == minus1Index)
					{
						continue;
					}
					const u32 arg = reshapedTensorEachAxisSizeTbl[id];
					prod *= arg;
				}

				if (mTensorData.size() % prod == 0)
				{
					reshapedTensorEachAxisSizeTbl[minus1Index] = mTensorData.size() / prod;
				}
				else
				{
					std::cout << "-1 in reshape function mismatch with other args" << std::endl;
					assert(0);
				}
			}
			else//-1���Ȃ��ꍇ�ɁA���ꂪreshape�O�̃e���\���̃T�C�Y�Ɛ����I���`�F�b�N
			{
				u32 prod = 1;
				for (u32 id = 0; id < reshapedTensorDim; id++)
				{
					const u32 arg = reshapedTensorEachAxisSizeTbl[id];
					prod *= arg;
				}

				if (!(mTensorData.size() == prod))
				{
					std::cout << "can't reshape with these args" << std::endl;
					assert(0);
				}
			}


			mTensorDimension = reshapedTensorDim;
			mEachAxisSize.resize(mTensorDimension);
			for (u32 id = 0; id < mTensorDimension; id++)
			{
				mEachAxisSize[id] = reshapedTensorEachAxisSizeTbl[id];
			}
		}

		void transpose(u32 axis0, u32 axis1);

		static TensorCore* createTensorPtrLike(const TensorCore&);


		static bool isSameShape(const TensorCore& tensorL, const TensorCore& tensorR);

	private:
		bool mWhetherToLearn;

		//���݂̃e���\���̎���
		u32 mTensorDimension;
		//���݂̃e���\���̊e�������̃T�C�Y
		std::vector<u32> mEachAxisSize;

		//�e���\���̃f�[�^�ʁi����͕s�ρj
		const u32 mTensorDataSize;
		//�e���\���̃f�[�^�i���̃T�C�Y���s�� =====> resize�͈�񂵂��Ă΂Ȃ�����!!!�j
		std::vector<f32> mTensorData;
		std::vector<f32> mDeltaTensorData;



		//���`���p
		std::function<void (TensorCore&)> mForwardFunction;
		std::vector<TensorCore*> mRootTensor;
		void forward();

		//�t�`���p
		std::vector<std::vector<TensorCore*> > mFollowingTensorTbl;
		std::vector<std::function<void(TensorCore&, std::vector<TensorCore*>)> > mBackwardFunctionTbl;
		void backward();


		//�R���X�g���N�^�̉ϒ��������������邽�߂̊֐�
		//�e�������̃T�C�Y�������Ŋi�[����B
		template <typename Head, typename ... Tail>
		void constructorArgsDevider(u32 id, Head&& head, Tail&& ... tail)
		{
			const auto& headID = typeid(Head);
			const auto& s32ID = typeid(s32);
			const auto& u32ID = typeid(u32);

			if (!(headID == u32ID || headID == s32ID))
			{
				std::cout << "TensorCore " << id << "th component type error" << std::endl;
				assert(0);
			}

			if (head <= 0)
			{
				std::cout << "TensorCore " << id << "th component size error" << std::endl;
				assert(0);
			}

			mEachAxisSize[id] = head;
			constructorArgsDevider(id + 1, tail...);
		}
		void constructorArgsDevider(u32 id) {}

		//reshape�̉ϒ��������������邽�߂̊֐�
		template <typename Head, typename ... Tail>
		void reshapeArgsDevider(u32 id, s32 reshapedTensorEachAxisSizeTbl[], Head&& head, Tail&& ... tail)
		{
			const auto& headID = typeid(Head);
			const auto& s32ID = typeid(s32);
			const auto& u32ID = typeid(u32);

			if (!(headID == u32ID || headID == s32ID))
			{
				std::cout << "ReshapedTensor " << id << "th component type error" << std::endl;
				assert(0);
			}

			if (head <= -2 || head == 0)
			{
				std::cout << "TensorCore " << id << "th component size error" << std::endl;
				assert(0);
			}

			reshapedTensorEachAxisSizeTbl[id] = head;
			reshapeArgsDevider(id + 1, reshapedTensorEachAxisSizeTbl, tail...);
		}
		void reshapeArgsDevider(u32 id, s32 reshapedTensorEachAxisSizeTbl[]) {}

		//�ėp�̉ϒ������̏����֐�
		//�����ň�����0�ȏ�ł��邱�Ƃ��`�F�b�N
		template <typename Head, typename ... Tail>
		void genericArgsDevider(u32 id, u32 tensorEachAxisSizeTbl[], Head&& head, Tail&& ... tail)
		{
			const auto& headID = typeid(Head);
			const auto& s32ID = typeid(s32);
			const auto& u32ID = typeid(u32);

			if (!(headID == u32ID || headID == s32ID))
			{
				std::cout << "TensorCore " << id << "th Index type error" << std::endl;
				assert(0);
			}

			if (head < 0)
			{
				std::cout << "TensorCore " << id << "th Index range error" << std::endl;
				assert(0);
			}

			tensorEachAxisSizeTbl[id] = head;
			genericArgsDevider(id + 1, tensorEachAxisSizeTbl, tail...);
		}
		void genericArgsDevider(u32 id, u32 tensorEachAxisSizeTbl[]) {}



	};

	

}