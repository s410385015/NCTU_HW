#include<math.h>
#include<vector>
using namespace std;

class matrix
{

	public:
		float *m;
		int w;
		int h;
		matrix(){};
		matrix(int _w,int _h);
		matrix(int _w, int _h,bool flag);
		matrix identityMatrix(int s);
		matrix Mul(matrix a,matrix b);
		matrix Mul(float f);
		void setMatrix(vector<vector<float>> f);
		void setMatrix(float *f,int _h,int _w);
		matrix Invert();
		void print();
};


