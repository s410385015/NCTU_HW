#include "matrix.h"
#include <stdio.h>

matrix::matrix(int _h, int _w)
{
	m = new float[_h*_w];
	

	w = _w;
	h = _h;
	for (int i = 0; i < _h;i++)
		for (int j = 0; j < _w; j++)
			m[i*_h+j] = 0;

}


matrix matrix::identityMatrix(int s)
{
	matrix iMatrix(s, s);
	for (int i = 0; i < s; i++)
		iMatrix.m[(s+1)*i] = 1.0f;
	return iMatrix;
}

matrix matrix::Mul(matrix a, matrix b)
{
	matrix r(a.h, b.w);
	float value;
	for (int i = 0; i < a.h; i++){
		for (int j = 0; j < b.w; j++){
			value = 0;
			for (int k = 0; k < a.w; k++){
				
				value += a.m[i*a.w+k] * b.m[k*b.w+j];
				//printf("%d %d %d %f %f %f\n", i, j, k, a.m[i][k], b.m[k][j],value);
				//printf("-------\n");
			}
			
			r.m[i*r.w+j] = value;
		}
	}

	return r;
}

matrix matrix::Mul(float f)
{
	for (int i = 0; i < h;i++)
		for (int j = 0; j < w; j++)
			m[i*w+j] *= f;

	return *this;
}


void matrix::setMatrix(vector<vector<float>> f)
{
	h = f.size();
	w = f[0].size();
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			m[i*w+j] = f[i][j];
}
void matrix::setMatrix(float *f,int _h,int _w)
{
	
	h = _h;
	w = _w;
	m = new float[h*w];
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			m[i*w+j] = f[i*w+j];
}

matrix matrix::Invert()
{
	
	
	
	matrix root = identityMatrix(w);
	matrix L = identityMatrix(w);
	matrix U;
	U.setMatrix(m,h,w);
	
	for (int i = 0; i < h; i++)
	{
		matrix tmp = identityMatrix(w);
		matrix tmp1 = identityMatrix(w);
		for (int j = i + 1; j < h; j++)
		{
			tmp.m[j*w+i] = -U.m[j*w+i] / U.m[i*w+i];
			tmp1.m[j*w + i] = U.m[j*w + i] / U.m[i*w+i];
		}
		
		L = Mul(L, tmp1);
		U = Mul(tmp, U);
		
	}
	
	matrix Inv(L.h, L.h);
	for (int k = 0; k < L.h; k++)
	{
		matrix base(1,L.h);
		base.m[k] = 1.0f;
		matrix X(1,L.h);
		matrix Y(1, L.h);
		for (int i = 0; i < L.h;i++)
		{
			for (int j = 0; j < i; j++)
			{
				Y.m[i] -= L.m[i*L.w+j] * Y.m[j];
			}
			Y.m[i] += L.m[i*L.w+i] * base.m[i];
		}
		for (int i = L.h - 1; i>-1; i--)
		{
			for (int j = i + 1; j < L.h; j++)
			{
				X.m[i] -= U.m [i*U.w+j] * X.m[j];
			}
			X.m[i] += Y.m[i];
			X.m[i] /= U.m[i*(U.w+1)];
		}
		for (int i = 0; i < X.w; i++)
			Inv.m[i*Inv.w+k] = X.m[i];

		
		
	}




	
	
	return Inv;
	
}

void matrix::print()
{
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++)
		{
			printf("%.2f ", m[i*w+j]);
		}
		printf("\n");
	}
}