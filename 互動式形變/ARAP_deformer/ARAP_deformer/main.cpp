#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <limits>
#include <Windows.h>
#include <gl/GL.h>
#include <glut.h>
#include <map>
#include "glm.h"
#include "mtxlib.h"
#include "trackball.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;
// ----------------------------------------------------------------------------------------------------
// global variables

_GLMmodel *mesh;
GLfloat *rawVertex;
int WindWidth, WindHeight;
int last_x , last_y;
int select_x, select_y;

typedef enum { SELECT_MODE, DEFORM_MODE } ControlMode;
ControlMode current_mode = SELECT_MODE;

vector<float*> colors;
vector<vector<int> > handles;
int selected_handle_id = -1;
bool deform_mesh_flag = false;
vector<vector<int>> connectivityData;

float dis_tmp;

MatrixX3d d;

bool it_flag;


SparseMatrix<double> L;

vector<MatrixXd> Rm;

int handle_size=0;

vector<vector <vector3>> E;
vector<vector <vector3>> _E;

void initMesh()
{
	for (int i = 0; i < mesh->numvertices * 3; i++)
		mesh->vertices[i] = rawVertex[i];


	
	handles.clear();
	selected_handle_id = -1;
    deform_mesh_flag = false;
	current_mode = SELECT_MODE;
	tbInit(NULL);

}
bool UpdateE()
{
	float dis = 1, buff;
	for (int i = 0; i < mesh->numvertices; i++)
	{
		for (int j = 0; j < connectivityData[i].size(); j++)
		{
			int idx = connectivityData[i][j];
			vector3 tmp(mesh->vertices[3 * (i + 1) + 0], mesh->vertices[3 * (i + 1) + 1], mesh->vertices[3 * (i + 1) + 2]);
			vector3 _tmp(mesh->vertices[3 * (idx+1) + 0], mesh->vertices[3 * (idx+1) + 1], mesh->vertices[3 * (idx+1) + 2]);
			_E[i][j] = tmp-_tmp;
			buff = abs(_E[i][j].x) + abs(_E[i][j].y) + abs(_E[i][j].z);
			if (buff < dis)
				dis = buff;
		}
	}
	
	cout << abs(dis_tmp - dis) << endl;
	if (abs(dis_tmp - dis) > 0.00003) {
		dis_tmp = dis;
	}
	else {
		return  false;
	}
	
	return true;
}



void FindR()
{
	
	
	for (int i = 0; i < mesh->numvertices; i++)
	{
		Matrix3d Si = Matrix3d::Zero();
		vector3 ei, _ei;
		for (int j = 0; j < connectivityData[i].size(); j++){
			ei = vector3(E[i][j].x, E[i][j].y, E[i][j].z);
			_ei = vector3(_E[i][j].x, _E[i][j].y, _E[i][j].z);
			for (int k = 0; k < 3; k++) {
				for (int m = 0; m < 3; m++) {
					Si(k, m) += ei[k] * _ei[m];
				}
			}
		}
		JacobiSVD<MatrixXd> svd(Si, ComputeFullU | ComputeFullV);
		const MatrixXd U = svd.matrixU();
		const MatrixXd V = svd.matrixV();	
		const MatrixXd S = svd.singularValues();

		Matrix3d Ri;

		Ri = V*(U.transpose());
		
		Rm[i] = Ri;

	}
	
	
}


void FindWeight()
{
	vector<Triplet<double>> coff;
	
	SparseMatrix<double> _weight;
	
	for (int i = 0; i < mesh->numvertices; i++)
	{
		coff.push_back(Triplet<double>(i, i, connectivityData[i].size()));
		for (int j = 0; j < connectivityData[i].size(); j++)
		{
			int idx = connectivityData[i][j];
			coff.push_back(Triplet<double>(i, idx, -1.0f));
		}
		
	}
	

	int offset = 0;
	for (int handleIter = 0; handleIter < handles.size(); handleIter++)
	{
		for (int vertIter = 0; vertIter < handles[handleIter].size(); vertIter++)
		{
			int idx = handles[handleIter][vertIter] - 1;
			coff.push_back(Triplet<double>(mesh->numvertices+offset, idx, 1.0f));
			offset++;
		}

	}

	_weight.resize(mesh->numvertices+offset, mesh->numvertices);
	//weight.resize(mesh->numvertices*3, mesh->numvertices*3);
	_weight.setFromTriplets(coff.begin(), coff.end());
	//L = _weight;
	L.resize(mesh->numvertices + offset, mesh->numvertices);
	L.setFromTriplets(coff.begin(), coff.end());
	
}

void Update()
{
	
	FindR();
	d = MatrixX3d::Zero(mesh->numvertices+handle_size, 3);
	Matrix3Xd e = Matrix3Xd::Zero(3, 1);
	for (int i = 0; i < mesh->numvertices; i++)
	{
		for (int j = 0; j < connectivityData[i].size(); j++)
		{
			int idx = connectivityData[i][j];
			e(0, 0) = E[i][j].x;
			e(1, 0) = E[i][j].y;
			e(2, 0) = E[i][j].z;
			
			d.row(i) += (((Rm[i] + Rm[idx])*(e))).transpose();
			
		}
		d.row(i) *= 0.5;
	}

	int offset = 0;
	for (int handleIter = 0; handleIter < handles.size(); handleIter++)
	{
		for (int vertIter = 0; vertIter < handles[handleIter].size(); vertIter++)
		{

			int idx = handles[handleIter][vertIter] ;
			Vector3d tmp(mesh->vertices[3 * idx + 0], mesh->vertices[3 * idx + 1], mesh->vertices[3 * idx + 2]);
			
			d.row(mesh->numvertices + offset) = tmp;
	
			offset++;
		}

	}
	
	
	SimplicialCholesky<SparseMatrix<double>> chol(L.transpose()*L);
	
	MatrixXd result = chol.solve(L.transpose()*d);


	

	for (int i = 0; i < mesh->numvertices; i++)
		for (int j = 0; j < 3; j++)
			mesh->vertices[(i + 1) * 3 + j] = result(i, j);

}



void FindE()
{
	//eij.resize(3,mesh->numvertices);
	
	
	for (int i = 0; i < mesh->numvertices; i++){
		for (int j = 0; j < connectivityData[i].size(); j++)
		{
			int idx = connectivityData[i][j];
			vector3 tmp(mesh->vertices[3 * (i + 1) + 0], mesh->vertices[3 * (i + 1) + 1], mesh->vertices[3 * (i + 1) + 2]);
			vector3 _tmp(mesh->vertices[3 * (idx + 1) + 0], mesh->vertices[3 * (idx + 1) + 1], mesh->vertices[3 * (idx + 1) + 2]);
			
			_E[i][j] = tmp-_tmp;
			E[i][j] = tmp-_tmp;
		}
		
		//eij.col(i) << mesh->vertices[3 * (i + 1)], mesh->vertices[3 * (i + 1) + 1], mesh->vertices[3 * (i + 1) + 2];
	}
	//_eij = eij;

}



void FindCotangent()
{
	/*
	wij.resize(mesh->numvertices);
	
	for (int i = 0; i < mesh->numtriangles; i++)
	{
		
		
		int indexA = mesh->triangles[i].vindices[0] - 1;
		int indexB = mesh->triangles[i].vindices[1] - 1;
		int indexC = mesh->triangles[i].vindices[2] - 1;
		vector3 pA(mesh->vertices[3 * (indexA + 1) + 0], mesh->vertices[3 * (indexA + 1) + 1], mesh->vertices[3 * (indexA + 1) + 2]);
		vector3 pB(mesh->vertices[3 * (indexB + 1) + 0], mesh->vertices[3 * (indexB + 1) + 1], mesh->vertices[3 * (indexB + 1) + 2]);
		vector3 pC(mesh->vertices[3 * (indexC + 1) + 0], mesh->vertices[3 * (indexC + 1) + 1], mesh->vertices[3 * (indexC + 1) + 2]);
		float eAB = (pA - pB).length();
		float eAC = (pA - pC).length();
		float eBC = (pB - pC).length();


		float cosA = (eAB*eAB + eAC*eAC - eBC*eBC) / (2 * eAB*eAC);
		float cosB = (eAB*eAB + eBC*eBC - eAC*eAC) / (2 * eAB*eBC);
		float cosC = (eAC*eAC + eBC*eBC - eAB*eAB) / (2 * eAC*eBC);

		float cotA = cosA / sqrt(1 - cosA*cosA);
		float cotB = cosB / sqrt(1 - cosB*cosB);
		float cotC = cosC / sqrt(1 - cosC*cosC);
		//cout << cotA << endl;
		//cout << cotB << endl;
		//cout << cotC << endl;
		vector<int> index;
		index.push_back(indexA);
		index.push_back(indexB);
		index.push_back(indexC);
		vector<float> cot;
		cot.push_back(cotA);
		cot.push_back(cotB);
		cot.push_back(cotC);


		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (i == j){
					wij[index[i]].insert(pair<int, float>(index[j], 1));
					continue;
				}
				int idx=0;
				while (idx == i || idx == j) idx++;

				if (wij[index[i]].find(index[j]) != wij[index[i]].end())
					wij[index[i]][index[j]] += (0.5*cot[idx]);
				else
					wij[index[i]].insert(pair<int,float>(index[j], 0.5*cot[idx]));
				
			}
		}
		
	}
	*/
	
	
}





void FindConnectivity()
{
	connectivityData.resize(mesh->numvertices);
	


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

	E.resize(mesh->numvertices);
	_E.resize(mesh->numvertices);
	for (int i = 0; i < mesh->numvertices; i++) {
		E[i].resize(connectivityData[i].size());
		_E[i].resize(connectivityData[i].size());
	}
}




// ----------------------------------------------------------------------------------------------------
// render related functions



void Reshape(int width, int height)
{
	int base = min(width , height);

	tbReshape(width, height);
	glViewport(0 , 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,(GLdouble)width / (GLdouble)height , 1.0, 128.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -3.5);

	WindWidth = width;
	WindHeight = height;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
	tbMatrix();

	// render solid model
	glEnable(GL_LIGHTING);
	glColor3f(1.0 , 1.0 , 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK , GL_FILL);
	glmDraw(mesh , GLM_SMOOTH);

	// render wire model
	glPolygonOffset(1.0 , 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glLineWidth(1.0f);
	glColor3f(0.6 , 0.0 , 0.8);
	glPolygonMode(GL_FRONT_AND_BACK , GL_LINE);
	glmDraw(mesh , GLM_SMOOTH);

	// render handle points
	glPointSize(10.0);
	glEnable(GL_POINT_SMOOTH);
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	for(int handleIter=0; handleIter<handles.size(); handleIter++)
	{
		glColor3fv(colors[ handleIter%colors.size() ]);
		for(int vertIter=0; vertIter<handles[handleIter].size(); vertIter++)
		{
			int idx = handles[handleIter][vertIter];
			glVertex3fv((float *)&mesh->vertices[3 * idx]);
		}
	}
	glEnd();

	glPopMatrix();

	glFlush();  
	glutSwapBuffers();
}

// ----------------------------------------------------------------------------------------------------
// mouse related functions

vector3 Unprojection(vector2 _2Dpos)
{
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x , viewport[3] - (int)_2Dpos.y , 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = {0.0 , 0.0 , 0.0};

	gluUnProject(X , ((double)viewport[3] - Y) , (double)Depth , ModelViewMatrix , ProjectionMatrix , viewport, &wpos[0] , &wpos[1] , &wpos[2]);

	return vector3(wpos[0] , wpos[1] , wpos[2]);
}

vector2 projection_helper(vector3 _3Dpos)
{
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	double wpos[3] = {0.0 , 0.0 , 0.0};
	gluProject(_3Dpos.x, _3Dpos.y, _3Dpos.z, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0] , &wpos[1] , &wpos[2]);

	return vector2(wpos[0], (double)viewport[3]-wpos[1]);
}

void mouse(int button, int state, int x, int y)
{
	tbMouse(button, state, x, y);

	if( current_mode==SELECT_MODE && button==GLUT_RIGHT_BUTTON )
	{
		if(state==GLUT_DOWN)
		{
			select_x = x;
			select_y = y;
		}
		else
		{
			handle_size = 0;
			vector<int> this_handle;

			// project all mesh vertices to current viewport
			for(int vertIter=0; vertIter<mesh->numvertices; vertIter++)
			{
				vector3 pt(mesh->vertices[3 * vertIter + 0] , mesh->vertices[3 * vertIter + 1] , mesh->vertices[3 * vertIter + 2]);
				vector2 pos = projection_helper(pt);
				handle_size++;
				// if the projection is inside the box specified by mouse click&drag, add it to current handle
				if(pos.x>=select_x && pos.y>=select_y && pos.x<=x && pos.y<=y)
				{
					this_handle.push_back(vertIter);
					
					
				}
			}
			handles.push_back(this_handle);
		}
	}
	// select handle
	else if( current_mode==DEFORM_MODE && button==GLUT_RIGHT_BUTTON && state==GLUT_DOWN && handles.empty()==false )
	{
		// project all handle vertices to current viewport
		// see which is closest to selection point
		double min_dist = 999999;
		int handle_id = -1;
		
		handle_size = 0;
		for(int handleIter=0; handleIter<handles.size(); handleIter++)
		{
			for(int vertIter=0; vertIter<handles[handleIter].size(); vertIter++)
			{
				
				int idx = handles[handleIter][vertIter];
	
				vector3 pt(mesh->vertices[3 * idx + 0] , mesh->vertices[3 * idx + 1] , mesh->vertices[3 * idx + 2]);
				vector2 pos = projection_helper(pt);
				handle_size++;
				double this_dist = sqrt((double)(pos.x-x)*(pos.x-x) + (double)(pos.y-y)*(pos.y-y));
				if(this_dist<min_dist)
				{
					min_dist = this_dist;
					handle_id = handleIter;
					
				}
				
			}
		}

		selected_handle_id = handle_id;
		deform_mesh_flag = true;
		
		
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP&&current_mode == DEFORM_MODE){
		
		it_flag = true;
		int iters = 0;
		FindWeight();
		dis_tmp = 1;
		
		while (it_flag&&iters<10)
		{
			
			it_flag=UpdateE();
			Update();
			iters++;
			Display();
			
		}
		deform_mesh_flag = false;
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP){
		deform_mesh_flag = false;
		
	}

	last_x = x;
	last_y = y;
}

void motion(int x, int y)
{
	tbMotion(x, y);

	// if in deform mode and a handle is selected, deform the mesh
	if( current_mode==DEFORM_MODE && deform_mesh_flag==true )
	{
		matrix44 m;
		vector4 vec = vector4((float)(x - last_x) / 1000.0f , (float)(y - last_y) / 1000.0f , 0.0 , 1.0);

		gettbMatrix((float *)&m);
		vec = m * vec;

		// deform handle points
		for(int vertIter=0; vertIter<handles[selected_handle_id].size(); vertIter++)
		{
			int idx = handles[selected_handle_id][vertIter];
			vector3 pt(mesh->vertices[3*idx+0]+vec.x, mesh->vertices[3*idx+1]-vec.y, mesh->vertices[3*idx+2]-vec.z);
			mesh->vertices[3 * idx + 0] = pt[0];
			mesh->vertices[3 * idx + 1] = pt[1];
			mesh->vertices[3 * idx + 2] = pt[2];
		}

		
		
		
	}

	last_x = x;
	last_y = y;
}

// ----------------------------------------------------------------------------------------------------
// keyboard related functions

void keyboard(unsigned char key, int x, int y )
{
	switch(key)
	{
		
		case 'd':
			current_mode = DEFORM_MODE;
			break;
		case 's':
			current_mode = SELECT_MODE;
			break;
		case 114:
		case 82:
			initMesh();
			break;
		default:
			break;
	}
}

// ----------------------------------------------------------------------------------------------------
// main function

void timf(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1, timf, 0);
}



int main(int argc, char *argv[])
{
	


	WindWidth = 800;
	WindHeight = 800;

	GLfloat light_ambient[] = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[] = {0.8, 0.8, 0.8, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_position[] = {0.0, 0.0, 1.0, 0.0};

	// color list for rendering handles
	float red[] = {1.0, 0.0, 0.0};
	colors.push_back(red);
	float yellow[] = {1.0, 1.0, 0.0};
	colors.push_back(yellow);
	float blue[] = {0.0, 1.0, 1.0};
	colors.push_back(blue);
	float green[] = {0.0, 1.0, 0.0};
	colors.push_back(green);

	glutInit(&argc, argv);
	glutInitWindowSize(WindWidth, WindHeight);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutCreateWindow("ARAP");

	glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);
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
	mesh = glmReadOBJ("../data/man.obj");

	glmUnitize(mesh);
	glmFacetNormals(mesh);
	glmVertexNormals(mesh , 90.0);
	rawVertex = (GLfloat*)malloc(sizeof(mesh->vertices)*mesh->numvertices * 3);
	for (int i = 0; i < mesh->numvertices * 3; i++)
		rawVertex[i] = mesh->vertices[i];

	Rm.resize(mesh->numvertices);


	FindConnectivity();
	FindE();

	glutMainLoop();

	return 0;

}