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

	Tensor v0(2,4,8);
	Tensor v1(2,4,8);
	Tensor v2(2,4,8);
	for (u32 i = 0; i < v0.getTensorDataSize(); i++)
	{
		v0[i] = i;
		v1[i] = 2 * i;
		v2[i] = 4 * i;

		/*v0.mTensorPtr->getDeltaTensorData(i)= i;
		v1.mTensorPtr->getDeltaTensorData(i)= 2 * i;*/

	}
	{
		Tensor vTest(2, 4, 8);
	}

	Tensor V0 = v0 + v1;
	Tensor V1 = v2 + v1;
	for (u32 i = 0; i < v0.getTensorDataSize(); i++)
	{
		V0.grad(i) = i;
		V1.grad(i) = 2 * i;
	}
	v1.backward();
	v0.backward();
	v2.backward();

	int x = 1 + 1;
}
