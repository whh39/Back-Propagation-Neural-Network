#include <iostream>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>

#define A 1.0
#define B 1.0

using namespace std;
double sigmoid(const double);

int main()
{
	//初始化
	double x1[7];
	double x2;
	double testdata[10][4] = 
	{
		{	4.8	,	3	,	1.4	,	0.3	}	,
		{	5.1	,	3.8	,	1.6	,	0.2	}	,
		{	4.6	,	3.2	,	1.4	,	0.2	}	,
		{	5.3	,	3.7	,	1.5	,	0.2	}	,
		{	5	,	3.3	,	1.4	,	0.2	}	,
		{	5.7	,	3	,	4.2	,	1.2	}	,
		{	5.7	,	2.9	,	4.2	,	1.3	}	,
		{	6.2	,	2.9	,	4.3	,	1.3	}	,
		{	5.1	,	2.5	,	3	,	1.1	}	,
		{	5.7	,	2.8	,	4.1	,	1.3	}	
	};

	int lines = sizeof(testdata) / sizeof(testdata[0][0]);
	int row = sizeof(testdata) / sizeof(testdata[0]);
	int column = lines / row;

	double w1[4][7] =
	{
		{	-0.439234	,	-0.439234	,	-0.439234	,	-0.439234	,	-0.439234, -0.439234	,	-0.439234	}	,
{	-1.0159	,	-1.0159	,	-1.0159	,	-1.0159	,	-1.0159, -1.0159	,	-1.0159	}	,
{	1.49609	,	1.49609	,	1.49609	,	1.49609	,	1.49609	,1.49609	,	1.49609	}	,
{	0.651349	,	0.651349	,	0.651349	,	0.651349	,	0.651349	,0.651349	,	0.651349	}
	};
	double w2[7] = { -1.8411	,	-1.8411	,	-1.8411	,	-1.8411	,	-1.8411	,	-1.8411	,	-1.8411 };
	double b1[7] = { -0.111601	,	-0.111601	,	-0.111601	,	-0.111601	,	-0.111601	,	-0.111601	,	-0.111601 };
	double b2 = -0.111601;

	//带入数值
	ofstream outfile;
	outfile.open("outdata.txt");

	for (int k = 0; k < row; k++)
	{
		for (int j = 0; j < 5; j++)
		{
			double t = 0;
			for (int i = 0; i < 4; i++)
				t += w1[i][j] * testdata[k][i];
			t += b1[j];
			x1[j] = sigmoid(t);
		}

		double t = 0;
		for (int i = 0; i < 5; i++)
			t += w2[i] * x1[i];
		t += b2;
		x2 = sigmoid(t);
		outfile << x2 << "\t";
	}
	cout << "OUTPUT is over" << endl;
	return 0;
}

double sigmoid(const double x)
{
	return A / (1 + exp(-x / B));
}
