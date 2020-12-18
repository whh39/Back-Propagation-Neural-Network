#pragma once
#ifndef _BP_H_
#define _BP_H_

#define random(x) (rand()%x)

#include <vector>

#define LAYER	3		//����������
#define NUM		10		//ÿ�����ڵ���

#define A		1.0
#define B		1.0    //A��B��S�ͺ����Ĳ���
#define ITERS	1000	//���ѵ������
#define ETA_W	0.0035	//Ȩֵ������
#define ETA_B	0.002	//��ֵ������
#define ERROR	0.0005	//����������������
#define ACCU	0.00002	//ÿ�ε�����������
#define ALPHA   0.3		//������
#define E		1		//����Ӧѧϰ��

#define Type double
#define Vector std::vector



struct Data
{
	Vector<Type> x;	//��������
	Vector<Type> y;	//�������
};

class BP {

public:

	void GetData(const Vector<Data>);
	void Train();
	Vector<Type> ForeCast(const Vector<Type>);

private:

	void InitNetWork();			//��ʼ������
	void GetNums();				//��ȡ���롢�����������Ľڵ���
	void ForwardTransfer();		//���򴫲��ӹ���
	void ReverseTransfer(int);		//���򴫲��ӹ���
	void CalcDelta(int);		//����w��b�ĵ�����
	void UpdateNetWork();		//����Ȩֵ�ͷ�ֵ

	Type randnum(int);			//�����

	Type GetError(int);			//���㵥�����������
	Type GetAccu();				//�������������ľ���
	Type Sigmoid(const Type);	//����Sigmoid������ֵ

private:

	int in_num;		//�����ڵ���
	int ou_num;		//�����ڵ���
	int hd_num;		//������ڵ���
	double dsum1 = 0;	//ѧϰ���ʵĵ���
	double dsum2 = 0;	//ѧϰ���ʵĵ���

	Vector<Data> data;		//�����������

	Type w[LAYER][NUM][NUM];	//BP�����Ȩֵ
	Type wp[LAYER][NUM][NUM];	//BP����Ķ�����
	Type b[LAYER][NUM];			//BP����ڵ�ķ�ֵ

	Type x[LAYER][NUM];			//ÿ����Ԫ��ֵ��S�ͺ���ת��������ֵ��������Ϊԭֵ
	Type d[LAYER][NUM];			//��¼deltaѧϰ������delta��ֵ

};

#endif		//_BP_H_