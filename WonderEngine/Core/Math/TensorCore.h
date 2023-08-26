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

	private://後々ここはprivateに変更する。
		//可変長コンストラクタ
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

			//初期状態の各次元のサイズを決定する。
			mEachAxisSize.resize(mTensorDimension);
			constructorArgsDevider(0, args...);

			//データサイズを定める。
			u64 size = 1;
			for (const auto& eachSize : mEachAxisSize)
			{
				size *= eachSize;
			}

			mTensorData.resize(size);
			mDeltaTensorData.resize(size);

			//無理やり初期化するために強制的にconstを外している。
			u32* p2mTensorDataSize = const_cast<u32*>(&mTensorDataSize);
			*p2mTensorDataSize = size;
		}

		//デフォルトコンストラクタ
		TensorCore();
		~TensorCore() = default;

		//コピーコンストラクタと代入演算子
		TensorCore(const TensorCore&);
		TensorCore& operator=(const TensorCore&) = default;

		//ムーブコンストラクタと代入演算子
		TensorCore(TensorCore&&);
		TensorCore& operator=(TensorCore&&) = default;

		//要素アクセス[index]演算子
		f32 operator[](u32 index) const;
		f32& operator[](u32 index);


		u32 getTensorDataSize() const;

		//テンソルの値へのアクセス
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

		//テンソルの微分値へのアクセス。　デバッグ用の関数な気がするから、たぶん後で消す
		f32& getDeltaTensorData(u32 index);
		f32 getDeltaTensorData(u32 index) const;


		template <typename ...Args>
		void reshape(Args ... args)
		{
			const u32 reshapedTensorDim = sizeof...(args);

			//可変長変数を分割して配列に格納
			s32 reshapedTensorEachAxisSizeTbl[reshapedTensorDim];
			reshapeArgsDevider(0, reshapedTensorEachAxisSizeTbl, args...);

			//可変長引数の中に-1が２つ以上ないことを確認
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

			//-1があった場合に、それを適切な値に置換できるかチェック
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
			else//-1がない場合に、それがreshape前のテンソルのサイズと整合的かチェック
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

		//現在のテンソルの次元
		u32 mTensorDimension;
		//現在のテンソルの各次元毎のサイズ
		std::vector<u32> mEachAxisSize;

		//テンソルのデータ量（これは不変）
		const u32 mTensorDataSize;
		//テンソルのデータ（このサイズも不変 =====> resizeは一回しか呼ばないこと!!!）
		std::vector<f32> mTensorData;
		std::vector<f32> mDeltaTensorData;



		//順伝搬用
		std::function<void (TensorCore&)> mForwardFunction;
		std::vector<TensorCore*> mRootTensor;
		void forward();

		//逆伝搬用
		std::vector<std::vector<TensorCore*> > mFollowingTensorTbl;
		std::vector<std::function<void(TensorCore&, std::vector<TensorCore*>)> > mBackwardFunctionTbl;
		void backward();


		//コンストラクタの可変長引数を処理するための関数
		//各軸方向のサイズもここで格納する。
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

		//reshapeの可変長引数を処理するための関数
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

		//汎用の可変長引数の処理関数
		//内部で引数が0以上であることをチェック
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