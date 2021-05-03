// OpenCVApplication.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
//#include "common.h"
#include <iostream>
#include <fstream>

using namespace std;

void  least_Mean_Square_kx2()
{

	float points[1000][2] = { {1.f,0.2f},
							  {2.f,1.f},
							  {3.f,3.f},
							  {4.f,5.f},
							  {5.f,7.3f} };


	float x4 = 0.f, x2y = 0.f;

	for (int i = 0; i < 5; i++)
	{	// X => points[i][0] 
		// Y => points[i][1] 
		x4 += points[i][0] * points[i][0] * points[i][0] * points[i][0];
		x2y += points[i][0] * points[i][0] * points[i][1];

	}

	cout << x2y / x4;
}

void mean_square_ax()
{
	float p[10][10] = { {60,160},
					   {70,160},
					   {80,170},
					   {90,180} };

	int Sx = 0;
	int Sxy = 0;
	for (int i = 0; i < 4; i++)
	{
		Sx += p[i][0] * p[i][0];
		Sxy += p[i][0] * p[i][1];

	}

	cout << Sx << "  " << Sxy << endl;
	cout << 1.0 * Sxy / Sx << endl;
}

void covariance()
{
	const int nr_sample = 4;
	const int nr_features = 3;

	float mean_feature[nr_features] = { 0.f };
	float covariance[nr_features][nr_features] = { 0.f };
	float correlation[nr_features][nr_features] = { 0.f };

	float points[nr_sample][nr_features] = { {4,2,6},
						   {1,8,2},
						   {5,1,9},
						   {6,9,11}
	};

	for (int i = 0; i < nr_sample; i++)
	{
		mean_feature[0] += points[i][0];
		mean_feature[1] += points[i][1];
		mean_feature[2] += points[i][2];
	}

	for (int i = 0; i < nr_features; i++)
	{
		mean_feature[i] /= nr_sample;

		cout << "miu[" << i << "]= " << mean_feature[i] << endl;
	}

	cout << "\ncovariance \n\n";

	for (int i = 0; i < nr_features; i++)
	{
		for (int j = 0; j < nr_features; j++)
		{
			for (int k = 0; k < nr_sample; k++)
			{
				covariance[i][j] += (points[k][i] - mean_feature[i]) * (points[k][j] - mean_feature[j]);
			}
			cout << covariance[i][j] / nr_sample << " ";

		}
		cout << endl;
	}

	cout << "\n\ncorrelation\n\n";

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{

			correlation[i][j] = covariance[i][j] / (sqrt(covariance[i][i]) * sqrt(covariance[j][j]));

			cout << correlation[i][j] << " ";

		}
		cout << endl;
	}

}

void perceptron_fixed()
{
	const int nr_points = 3;
	const int nr_features = 2;
	float learning_rate = 1.f;

	int points[nr_points][nr_features + 1] = { {-1,-1,-3},
                                               {1,7,1},
                                               {1,5,3} };

	float weights[nr_features + 1] = { 1.f,1.f,1.f };

	bool isWeightConstant = false;

	while (!isWeightConstant)
	{
		isWeightConstant = true;
		for (int i = 0; i < nr_points; i++)
		{
			int f = points[i][0] * weights[0] + points[i][1] * weights[1] + points[i][2] * weights[2];
			if (f < 0)
			{
				isWeightConstant = false;

				for (int j = 0; j < nr_features + 1; j++)
				{
					weights[j] += learning_rate * points[i][j];
				}
			}
		}
	}
	for (int j = 0; j < nr_features + 1; j++)
	{
		cout << weights[j] << " ";
	}
}


void perceptron_Batch()
{
	const int nr_points = 3;
	const int nr_features = 2;
	float learning_rate = 1.f;

	int points[nr_points][nr_features + 1] = { {1,2,4},
											   {1,4,5},
											   {-1,-4,-2} };

	float weights[nr_features + 1] = { 1.f,1.f,1.f };
	bool isWeightConstant = false;

	while (!isWeightConstant)
	{
		isWeightConstant = true;
		int isSampleWrongClassified[nr_points] = { 0 };

		for (int i = 0; i < nr_points; i++)
		{
			int f = points[i][0] * weights[0] + points[i][1] * weights[1] + points[i][2] * weights[2];
			if (f < 0)
			{
				isSampleWrongClassified[i] = 1;
				cout << "yes " << f << endl;
			}
			else
			{
				cout << "no " << f << endl;
			}
		
		}

		for (int i = 0; i < nr_points; i++)
		{
			if (isSampleWrongClassified[i] == 1)
			{
				isWeightConstant = false;

				for (int j = 0; j < nr_features + 1; j++)
				{
					weights[j] += learning_rate * points[i][j];

					cout << weights[j] << " ";
				}
				cout << endl;
			}
		}
	}
	for (int j = 0; j < nr_features + 1; j++)
	{
		cout << weights[j] << " ";
	}
}


int gx(int x3, int x2, int x1)
{
	return  2 * x3 + 3 * x2 - 4 * x1 - 5;


}

void liniar_discriminator()
{
	int x3 = 10;
	int x2 = 2;
	int x1 = 2;

	int p1[3] = { 1,1,1 };
	int p2[3] = { 4,0,0 };
	int p3[3] = { 0,2,5 };
	int p4[3] = { 1,10,2 };
	int p5[3] = { -1,1,-1 };


	cout << gx(x3, x2, x1) << endl;

	cout << gx(p1[0], p1[1], p1[2]) << endl;
	cout << gx(p2[0], p2[1], p2[2]) << endl;
	cout << gx(p3[0], p3[1], p3[2]) << endl;
	cout << gx(p4[0], p4[1], p4[2]) << endl;
	cout << gx(p5[0], p5[1], p5[2]) << endl;

}


int main()
{

//	perceptron_Batch();
	perceptron_fixed();

	//liniar_discriminator();

	//covariance();

	//least_Mean_Square_kx2();
	//mean_square_ax();

	//system("pause");
	return 0;
}