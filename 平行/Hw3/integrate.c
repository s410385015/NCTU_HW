#include <stdio.h>
#include <math.h>
#include <mpi.h>
#define PI 3.1415926535

int main(int argc, char **argv) 
{

  MPI_Init(&argc,&argv);
  

  long long i, num_intervals;
  double rect_width, area, sum, x_middle; 
  long start,end;
  double total;  

  sscanf(argv[1],"%llu",&num_intervals);

  rect_width = PI / num_intervals;

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 


  start=(num_intervals/size)*rank;
  end=start+num_intervals/size;

  if(rank==size-1)
      end+=num_intervals%size;

  sum = 0;
  for(i=start+1; i <= end; i++) {

    /* find the middle of the interval on the X-axis. */ 

    x_middle = (i - 0.5) * rect_width;
    area = sin(x_middle) * rect_width; 
    sum = sum + area;
  } 

  MPI_Reduce(&sum,&total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  if(rank==0)
    printf("The total area is: %f\n", (float)total);

  MPI_Finalize();
}   



