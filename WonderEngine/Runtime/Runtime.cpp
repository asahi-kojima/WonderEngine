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

	Tensor v0(100, 3, 28, 28);
	Tensor v1(100, 3, 28, 28);
	Tensor v2(100, 3, 28, 28);
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


	Tensor v5(100, 3, 28, 28);
	Tensor v6(100, 3, 28, 28);
	Tensor v7(100, 3, 28, 28);

	Tensor v8 = v5 + v6;
	Tensor v9 = v7 + v6;

	Tensor v10 = v3 + v5;
	//for (u32 i = 0; i < v0.getTensorDataSize(); i++)
	//{
	//	V0.grad(i) = i;
	//	V1.grad(i) = 2 * i;
	//}
	//v1.backward();
	//v0.backward();
	//v2.backward();


	Tensor w0(100, 3, 28, 28);//11
	Tensor w1(100, 3, 28, 28);//12
	Tensor w2(100, 3, 28, 28);//13

	for (u32 i = 0; i < w0.getTensorDataSize(); i++)
	{
		w0[i] = i % 3 + 1;
		w1[i] = i % 2 + 1;
		w2[i] = 2;
	}

	Tensor w3 = w0 * w1;//14
	Tensor w4 = w1 * w2;//15

	Tensor w5 = w3 * w4;//16

	Tensor vw = v10 + w5;
	for (u32 i = 0; i < vw.getTensorDataSize(); i++)
	{
		vw.grad(i) = i;
	}
	vw.backward();
	int x = 1 + 1;


}
