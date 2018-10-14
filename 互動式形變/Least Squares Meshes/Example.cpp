#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <Windows.h>
#include <gl/GL.h>
#include <glut.h>
#include "glm.h"
#include "mtxlib.h"
#include "trackball.h"
#include "matrix.h"
#include <algorithm>
#include <random> 
#include <chrono>    
#include <time.h>
#include <Eigen/Sparse>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

_GLMmodel *mesh;
_GLMmodel *mesh_new;
GLfloat *rawVertex;
int WindWidth, WindHeight;
int WinId;
int last_x, last_y;
int selectedFeature = -1;
const int RandomSize = 1000;
vector4 cur_vec;
vector<int> featureList;
vector<vector<int>> connectivityData;
double Start, End;

//vertex is start from 1
void RandomSelectFeature(int n)
{
	vector<int> v;
	for (int i = 0; i < mesh->numvertices; i++)
		v.push_back(i);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	shuffle(v.begin(), v.end(), default_random_engine(seed));

	for (int i = 0; i < n; i++)
		featureList.push_back(v[i]);
}




void FindConnectivity()
{
	connectivityData.resize(mesh->numvertices);
	for (int i = 0; i < mesh->numtriangles; i++){
		for (int j = 0; j < 3; j++){
			for (int k = 0; k < 3; k++){
				
				if (j != k
					&&find(connectivityData[mesh->triangles[i].vindices[j] - 1].begin(),
					connectivityData[mesh->triangles[i].vindices[j] - 1].end(),
					mesh->triangles[i].vindices[k] - 1)
					== connectivityData[mesh->triangles[i].vindices[j] - 1].end())
				{

					connectivityData[(mesh->triangles[i].vindices[j]) - 1].push_back((mesh->triangles[i].vindices[k]) - 1);
				}
				
			}
		}
	}
}

SparseMatrix<double> GenerateConstraints()
{
	
	vector<Triplet<double>> coff;
	
	//Fairness constraint 
	for (int i = 0; i < mesh->numvertices; i++)
	{
		coff.push_back(Triplet<double>(i, i, 1));
		

		for (int j = 0; j < connectivityData[i].size(); j++)
		{
			coff.push_back(Triplet<double>(i, connectivityData[i][j], (-1 / (double)(connectivityData[i].size()))));
			
		}
	}

	
	for (int i = 0; i < RandomSize; i++)
	{
		coff.push_back(Triplet<double>(mesh->numvertices + i, featureList[i] , 1));

	}

	SparseMatrix<double> A(mesh->numvertices + RandomSize, mesh->numvertices);
	A.setFromTriplets(coff.begin(), coff.end());
	

	return A;
	
}

SparseMatrix<double> GenerateTargerPos()
{

	vector<Triplet<double>> coff;
	
	
	for (int i = 0; i < RandomSize; i++)
		for (int j = 0; j < 3; j++)
			coff.push_back(Triplet<double>(i + mesh->numvertices, j , mesh->vertices[(featureList[i]+1) * 3 + j]));
			
	
	SparseMatrix<double> B(mesh->numvertices + RandomSize, 3);
	B.setFromTriplets(coff.begin(), coff.end());

	
	return B;
}

void SolveLinearLeastSquares()
{

	SparseMatrix<double> A = GenerateConstraints();
	SparseMatrix<double> ATA = (A.transpose())*A;
	SparseMatrix<double> B = (A.transpose())*GenerateTargerPos();
	
	
	SimplicialCholesky<SparseMatrix<double>> chol(ATA);
	MatrixXd  result = chol.solve(B);
	

	for (int i = 0; i < mesh->numvertices; i++)
		for (int j = 0; j < 3; j++)
			mesh_new->vertices[(i+1) * 3 + j] = result(i, j);

}



float GetFittingError()
{
	float error = 0;
	for (int i = 0; i < mesh->numvertices; i++)
	{
		float x = rawVertex[i * 3] - mesh_new->vertices[i * 3];
		float y = rawVertex[i * 3+1] - mesh_new->vertices[i * 3 + 1];
		float z = rawVertex[i * 3+2] - mesh_new->vertices[i * 3 + 2];
		
		error += (x*x + y*y + z*z);
	}

	return sqrt((error / (float)mesh->numvertices));
}


float RadialBasisFunc(float r)
{

	float sigma = 0.1f;
	return exp(-(r*r) / (2 * (sigma*sigma)));
}

void initMesh()
{
	for (int i = 0; i < mesh->numvertices * 3; i++)
		mesh->vertices[i] = rawVertex[i];
	selectedFeature = -1;
	featureList.clear();
	tbInit(NULL);

}

matrix GeneratePsiMatrix()
{
	matrix PSI = matrix().identityMatrix(featureList.size());
	for (int i = 0; i < featureList.size(); i++){
		for (int j = 0; j < i; j++){
			int idx = featureList[i];
			int _idx = featureList[j];
			vector3 p1(mesh->vertices[3 * idx + 0], mesh->vertices[3 * idx + 1], mesh->vertices[3 * idx + 2]);
			vector3 p2(mesh->vertices[3 * _idx + 0], mesh->vertices[3 * _idx + 1], mesh->vertices[3 * _idx + 2]);
			float dis = (p1 - p2).length();
			PSI.m[i*PSI.w + j] = RadialBasisFunc(dis);
			PSI.m[j*PSI.w + i] = RadialBasisFunc(dis);
		}
	}

	PSI.print();
	PSI = PSI.Invert();



	return PSI;
}

matrix GenerateVMatrix()
{
	vector<vector<float>> f;

	for (int i = 0; i < featureList.size(); i++)
	{
		int idx = featureList[i];

		vector<float> vi;


		vector3 p1(mesh->vertices[3 * idx + 0], mesh->vertices[3 * idx + 1], mesh->vertices[3 * idx + 2]);
		vector3 p2(rawVertex[3 * idx + 0], rawVertex[3 * idx + 1], rawVertex[3 * idx + 2]);
		vi.push_back(p1.x - p2.x);
		vi.push_back(p1.y - p2.y);
		vi.push_back(p1.z - p2.z);

		f.push_back(vi);
	}

	matrix V(featureList.size(), 3);
	if (f.size() >= 1)
		V.setMatrix(f);
	return V;
}
void FeatureBasedDeformation()
{
	matrix psi = GeneratePsiMatrix();
	matrix v = GenerateVMatrix();


	matrix mat = matrix().Mul(psi, v);


	for (int i = 0; i < mesh->numvertices; i++)
	{
		bool flag = true;
		vector3 p1(mesh->vertices[3 * i + 0], mesh->vertices[3 * i + 1], mesh->vertices[3 * i + 2]);
		float x, y, z;
		x = y = z = 0;
		for (int j = 0; j < featureList.size(); j++)
		{
			int idx = featureList[j];
			vector3 p2(mesh->vertices[3 * idx + 0], mesh->vertices[3 * idx + 1], mesh->vertices[3 * idx + 2]);
			float dis = (p1 - p2).length();
			float r = RadialBasisFunc(dis);
			x += mat.m[j*mat.w] * r;
			y += mat.m[j*mat.w + 1] * r;
			z += mat.m[j*mat.w + 2] * r;


		}



		matrix44 m;
		vector4 vec = vector4((float)(x), (float)(y), (float)(z), 1.0);


		mesh->vertices[3 * i + 0] = rawVertex[3 * i + 0] + vec.x;
		mesh->vertices[3 * i + 1] = rawVertex[3 * i + 1] + vec.y;
		mesh->vertices[3 * i + 2] = rawVertex[3 * i + 2] + vec.z;



	}
}


void Reshape(int width, int height)
{
	int base = min(width, height);

	tbReshape(width, height);
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLdouble)width / (GLdouble)height, 1.0, 128.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -3.5);

	WindWidth = width;
	WindHeight = height;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	
	glLoadIdentity();
	glTranslatef(0.8, 0.0, -3.5);

	glPushMatrix();
	tbMatrix();

	
	// render solid model
	glEnable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glmDraw(mesh, GLM_SMOOTH);

	
	// render wire model
	glPolygonOffset(1.0, 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glLineWidth(1.0f);
	glColor3f(0.6, 0.0, 0.8);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glmDraw(mesh, GLM_SMOOTH);

	// render features
	glPointSize(10.0);
	glColor3f(1.0, 0.0, 0.0);
	glDisable(GL_LIGHTING);
	//glBegin(GL_POINTS);
	
	glPopMatrix();



	glLoadIdentity();
	glTranslatef(-0.8, 0.0, -3.5);

	glPushMatrix();
	tbMatrix();

	// render solid model
	glEnable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glmDraw(mesh_new, GLM_SMOOTH);

	
	// render wire model
	//glPolygonOffset(1.0, 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glLineWidth(1.0f);
	glColor3f(0.6, 0.0, 0.8);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glmDraw(mesh_new, GLM_SMOOTH);

	// render features
	glPointSize(10.0);
	glColor3f(1.0, 0.0, 0.0);
	glDisable(GL_LIGHTING);
	//glBegin(GL_POINTS);
	
	for (int i = 0; i < featureList.size(); i++)
	{
		int idx = featureList[i];

		glVertex3fv((float *)&mesh->vertices[3 * idx]);
	}
	glEnd();

	glPopMatrix();

	glFlush();
	glutSwapBuffers();




}

vector3 Unprojection(vector2 _2Dpos)
{
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];				//Model_view matrix
	double ProjectionMatrix[16];			//Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x, viewport[3] - (int)_2Dpos.y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = { 0.0, 0.0, 0.0 };

	gluUnProject(X, ((double)viewport[3] - Y), (double)Depth, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0], &wpos[1], &wpos[2]);

	return vector3(wpos[0], wpos[1], wpos[2]);
}



void key(unsigned char key, int x, int y)
{

	switch (key)
	{
		// ascii code 27 is escape key
	case 27:
		glutDestroyWindow(WinId);
		exit(0);
	case 114:
	case 82:
		initMesh();
		break;
	default:
		break;
	}
}




void mouse(int button, int state, int x, int y)
{
	tbMouse(button, state, x, y);

	// add feature
	if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN)
	{
		int minIdx = 0;
		float minDis = 9999999.0f;

		vector3 pos = Unprojection(vector2((float)x, (float)y));

		for (int i = 0; i < mesh->numvertices; i++)
		{
			vector3 pt(mesh->vertices[3 * i + 0], mesh->vertices[3 * i + 1], mesh->vertices[3 * i + 2]);
			float dis = (pos - pt).length();

			if (minDis > dis)
			{
				minDis = dis;
				minIdx = i;
			}
		}

		featureList.push_back(minIdx);
	}

	// manipulate feature
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
	{
		int minIdx = 0;
		float minDis = 9999999.0f;

		vector3 pos = Unprojection(vector2((float)x, (float)y));

		for (int i = 0; i < featureList.size(); i++)
		{
			int idx = featureList[i];
			vector3 pt(mesh->vertices[3 * idx + 0], mesh->vertices[3 * idx + 1], mesh->vertices[3 * idx + 2]);
			float dis = (pos - pt).length();

			if (minDis > dis)
			{
				minDis = dis;
				minIdx = featureList[i];
			}
		}

		selectedFeature = minIdx;

	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
		selectedFeature = -1;

	last_x = x;
	last_y = y;
}

void motion(int x, int y)
{

	tbMotion(x, y);

	if (selectedFeature != -1)
	{
		matrix44 m;
		vector4 vec = vector4((float)(x - last_x) / 100.0f, (float)(y - last_y) / 100.0f, 0.0, 1.0);

		gettbMatrix((float *)&m);
		vec = m * vec;

		mesh->vertices[3 * selectedFeature + 0] += vec.x;
		mesh->vertices[3 * selectedFeature + 1] -= vec.y;
		mesh->vertices[3 * selectedFeature + 2] += vec.z;

		if (abs(x - last_x) + abs(y - last_y)>0)
			FeatureBasedDeformation();
	}

	last_x = x;
	last_y = y;
}



void timf(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1, timf, 0);
}

int main(int argc, char *argv[])
{
	//sMatrix s(2, 3);
	//s.SetValue(1, 1, 2);
	//s.Transpose().printMatrix();
	


	//return 0;
	WindWidth = 800;
	WindHeight = 800;

	GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat light_diffuse[] = { 0.8, 0.8, 0.8, 1.0 };
	GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat light_position[] = { 0.0, 0.0, 1.0, 0.0 };

	glutInit(&argc, argv);
	glutInitWindowSize(WindWidth, WindHeight);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	WinId = glutCreateWindow("Trackball Example");

	glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glClearColor(0, 0, 0, 0);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHT0);
	glDepthFunc(GL_LESS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	tbInit(GLUT_LEFT_BUTTON);
	tbAnimate(GL_TRUE);

	glutTimerFunc(40, timf, 0); // Set up timer for 40ms, about 25 fps

	// load 3D model
	mesh = glmReadOBJ("../data/Dino.obj");
	mesh_new = glmReadOBJ("../data/Dino.obj");
	glmUnitize(mesh);
	glmFacetNormals(mesh);
	glmVertexNormals(mesh, 90.0);

	glmUnitize(mesh_new);
	glmFacetNormals(mesh_new);
	glmVertexNormals(mesh_new, 90.0);


	rawVertex = (GLfloat*)malloc(sizeof(mesh->vertices)*mesh->numvertices * 3);
	for (int i = 0; i < mesh->numvertices * 3; i++)
		rawVertex[i] = mesh->vertices[i];


	
	
		
	RandomSelectFeature(RandomSize);
	FindConnectivity();

	Start = clock();
	connectivityData.resize(mesh->numvertices);
	SolveLinearLeastSquares();
	End = clock();

	printf("reconstruction time=%f\n", (End-Start)/CLOCKS_PER_SEC);
	printf("Fitting error=%f\n", GetFittingError());
	featureList.clear();


	glutMainLoop();

	free(rawVertex);
	return 0;

}

