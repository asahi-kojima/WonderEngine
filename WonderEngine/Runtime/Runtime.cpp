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


	Tensor v3 = v0 + v1;
	Tensor v4 = v2 + v1;


	Tensor v5(2, 4, 8);
	Tensor v6(2, 4, 8);
	Tensor v7(2, 4, 8);

	Tensor v8 = v5 + v6;
	Tensor v9 = v7 + v6;

	Tensor v10 = v0 + v5;
	//for (u32 i = 0; i < v0.getTensorDataSize(); i++)
	//{
	//	V0.grad(i) = i;
	//	V1.grad(i) = 2 * i;
	//}
	//v1.backward();
	//v0.backward();
	//v2.backward();


	Tensor w0(2, 4, 8);//11
	Tensor w1(2, 4, 8);//12
	Tensor w2(2, 4, 8);//13

	Tensor w3 = w0 + w1;//14
	Tensor w4 = w1 + w2;//15

	Tensor w5 = w3 + w4;//16

	Tensor vw = v10 + w5;

	int x = 1 + 1;


}
