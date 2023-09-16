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
	//Tensorを管理するsingletonクラス
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

			std::vector<std::weak_ptr<TensorCore> > mForwardTensorTbl;//自分よりも前方のテンソル
			std::vector<std::weak_ptr<TensorCore> > mBackwardTensorTbl;//自分よりも後方のテンソル
		};


		/// <summary>
		/// これまでに生成されたテンソルの数を記録（通算）
		/// 実装の関係で、テンソルが破棄されてもこの数は減らない。
		/// </summary>
		inline static u32 mCreatedTensorNum = 0;

		/// <summary>
		/// 各テンソルにはインスタンス番号が割り振られており、
		/// その番号にポインタを対応させている。
		/// </summary>
		std::map<u32, std::shared_ptr<TensorInformation> > mTensorMap;


	};
}