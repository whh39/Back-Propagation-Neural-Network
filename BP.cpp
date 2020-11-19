#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <fstream>
#include <cassert>
#include <iostream>

#include "BP.h"

/*	任何一个闭区间内的连续函数都可以由一个含有隐含层的BP网络逼近，
	也称“万能逼近定理”；
	隐含层的节点数目h=sqrt（输入层节点数+输出层节点数）+ a，const a=1~10 */

using namespace std;

//载入训练集
void BP::GetData(const Vector<Data> _data)
{
	data = _data;
}

//开始训练
void BP::Train()
{
	ofstream outfile;
	outfile.open("outdata.txt");

	printf("Begin to train BP NetWork.\n");
	GetNums();
	InitNetWork();
	int num = data.size();

	for (int iter = 0; iter <= ITERS; iter++)
	{
		for (int cnt = 0; cnt < num; cnt++)
		{
			//第一层输入节点赋值
			for (int i = 0; i < in_num; i++)
				x[0][i] = data.at(cnt).x[i];

			while (1)
			{
				ForwardTransfer();
				//cout << GetError(cnt) << endl;
				if (GetError(cnt) < ERROR)	//如果误差比较小，则针对单个样本跳出循环
					break;
				ReverseTransfer(cnt);
			}
		}
		printf("This is the %d th trainning NetWork.\n", iter);

		Type accu = GetAccu();

		outfile << accu << "\t";
		printf("All Samples Accuracy is %lf\n", accu);

		if (accu < ACCU) break;
	}
	outfile.close();

	outfile.open("w1.txt");
	for (int i = 0; i < in_num; i++)
	{
		for (int j = 0; j < hd_num; j++)
			outfile << w[1][i][j] << "\t";
		outfile << endl;
	}
	outfile.close();

	outfile.open("w2.txt");
	for (int i = 0; i < hd_num; i++)
	{
		for (int j = 0; j < ou_num; j++)
			outfile << w[2][i][j] << "\t";
		outfile << endl;
	}
	outfile.close();

	outfile.open("b1.txt");
	for (int i = 0; i < hd_num; i++)
	{
		outfile << b[1][i] << "\t";
	}
	outfile.close();

	outfile.open("b2.txt");
	for (int i = 0; i < ou_num; i++)
	{
		outfile << b[1][i] << "\t";
	}

	printf("The BP NetWork train End.\n");
}

//预测
Vector<Type> BP::ForeCast(const Vector<Type> data)
{
	int n = data.size();
	assert(n == in_num);
	for (int i = 0; i < in_num; i++)
		x[0][i] = data[i];

	ForwardTransfer();
	Vector<Type> v;
	for (int i = 0; i < ou_num; i++)
		v.push_back(x[2][i]);
	return v;
}

//获取网络节点数
void BP::GetNums()
{
	in_num = data[0].x.size();		//获取输入层节点数
	ou_num = data[0].y.size();		//获取输出层节点数
	hd_num = (int)sqrt((in_num + ou_num) * 1.0) + 7;	//获取隐含层节点数
	if (hd_num > NUM) hd_num = NUM;
}

//初始化网络
void BP::InitNetWork()
{
	memset(w, 0.1, sizeof(w));		//初始化权值和阀值为0（也可以随机）
	memset(b, 0.1, sizeof(b));
}

//工作信号的正向传递子过程
void BP::ForwardTransfer()
{
	//计算隐含层各个节点的输出值
	for (int j = 0; j < hd_num; j++)
	{
		Type t = 0;
		for (int i = 0; i < in_num; i++)
			t += w[1][i][j] * x[0][i];
		t += b[1][j];
		x[1][j] = Sigmoid(t);
	}

	//计算输出层各节点的输出值
	for (int j = 0; j < ou_num; j++)
	{
		Type t = 0;
		for (int i = 0; i < hd_num; i++)
			t += w[2][i][j] * x[1][i];
		t += b[2][j];
		x[2][j] = Sigmoid(t);
	}
}

//计算单个样本的误差
Type BP::GetError(int cnt)
{
	Type ans = 0;
	for (int i = 0; i < ou_num; i++)
		ans += 0.5 * (x[2][i] - data.at(cnt).y[i]) * (x[2][i] - data.at(cnt).y[i]);
	return ans;
}

//误差的反向传递子过程
void BP::ReverseTransfer(int cnt)
{
	CalcDelta(cnt);
	UpdateNetWork();
}

//计算所有样本的精度
Type BP::GetAccu()
{
	Type ans = 0;
	int num = data.size();
	for (int i = 0; i < num; i++)
	{
		int m = data.at(i).x.size();
		for (int j = 0; j < m; j++)
			x[0][j] = data.at(i).x[j];
		ForwardTransfer();
		int n = data.at(i).y.size();
		for (int j = 0; j < n; j++)
			ans += 0.5 * (x[2][j] - data.at(i).y[j]) * (x[2][j] - data.at(i).y[j]);
	}
	return ans / num;
}

//计算调整量
void BP::CalcDelta(int cnt)
{
	//计算输出层的delta值
	for (int i = 0; i < ou_num; i++)
		d[2][i] = (x[2][i] - data.at(cnt).y[i]) * x[2][i] * (A - x[2][i]) / (A * B);
	//计算隐含层的delta值
	for (int i = 0; i < hd_num; i++)
	{
		Type t = 0;
		for (int j = 0; j < ou_num; j++)
			t += w[2][i][j] * d[2][j];
		d[1][i] = t * x[1][i] * (A - x[1][i]) / (A * B);
	}
}

//根据调整量调整BP网络
void BP::UpdateNetWork()
{
	//隐含层和输出层之间权值和阀值调整
	for (int i = 0; i < hd_num; i++)
	{
		for (int j = 0; j < ou_num; j++)
			w[2][i][j] -= ETA_W * d[2][j] * x[1][i];
	}
	for (int i = 0; i < ou_num; i++)
		b[2][i] -= ETA_B * d[2][i];

	//输入层和隐含层之间的权值和阀值调整
	for (int i = 0; i < in_num; i++)
	{
		for (int j = 0; j < hd_num; j++)
			w[1][i][j] -= ETA_W * d[1][j] * x[0][i];
	}
	for (int i = 0; i < hd_num; i++)
		b[1][i] -= ETA_B * d[1][i];
}

//计算Sigmoid函数的值
Type BP::Sigmoid(const Type x)
{
	return A / (1 + exp(-x / B));
}

//随机数
Type BP::randnum(int x)
{
	return random(x);
}