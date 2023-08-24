#include <iostream>
#include "Math/Tensor.h"

int main()
{
	//Aoba::Core::Math::Tensor tensor(1, 10, 1000);
	//tensor.reshape(10, 2, -1, 10, 10, 1);
	// 
	// 
	// 
	
	using namespace Aoba::Core::Math;
	//Tensor tensor0(2,4,8);
	//Tensor tensor1(2,4,8);
	//for (u32 i = 0; i < tensor0.getTensorDataSize(); i++)
	//{
	//	tensor0[i] = i;
	//	tensor1[i] = 2 * i;
	//}

	//Tensor newTensor = tensor0 + tensor1;
	//Tensor newTensor1 = newTensor + tensor0;

	TensorValiable v0(2,4,8);
	TensorValiable v1(2,4,8);
	TensorValiable v2(2,4,8);
	for (u32 i = 0; i < v0.getTensorDataSize(); i++)
	{
		v0[i] = i;
		v1[i] = 2 * i;
		v2[i] = 4 * i;

		/*v0.mTensorPtr->getDeltaTensorData(i)= i;
		v1.mTensorPtr->getDeltaTensorData(i)= 2 * i;*/

	}

	TensorValiable V0 = v0 + v1;
	TensorValiable V1 = v2 + v1;
	for (u32 i = 0; i < v0.getTensorDataSize(); i++)
	{
		V0.getTensor()->getDeltaTensorData(i) = i;
		V1.getTensor()->getDeltaTensorData(i) = 2 * i;
	}
	v1.backward();

	int x = 1 + 1;
	//tensor[0];
	//tensor.transpose(0, 10);
	//tensor.getTensorSize();
	//f32 value = tensor(1,0,0,0,0,0);
	//f32 value = tensor(1, 1, 1, 1, 1, 0);
	//const Aoba::Core::Math::Tensor tensor0(1, 90, 1);
}
