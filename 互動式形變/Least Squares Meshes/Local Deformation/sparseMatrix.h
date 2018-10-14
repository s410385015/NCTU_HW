#include<map>
#include<math.h>
using namespace std;

class sparseMatrix{


public:

	int row;
	int col;
	map<int, float> m;

	sparseMatrix()
	{

	}

	sparseMatrix(int i, int j)
	{
		row = i;
		col = j;
	}

	float operator[](int i){
		if (m.find(i) == m.end())
			return 0;

		return m[i];
	}


	void SetValue(int i, int j, float f)
	{
		m.insert(pair<int, float>(i*row + j, f));
	}

	sparseMatrix Transpose()
	{
		sparseMatrix tmp(col, row);

		for (map<int, float>::iterator it = m.begin(); it != m.end(); it++)
		{
			int n = it->first;
			int a = n / row;
			int b = n%row;
			tmp.SetValue(b, a, it->second);
		}
	}


	void printMatrix()
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				printf("%f", this[(i*col + j)]);
			}
			printf("\n");
		}

	}
};