#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
int isprime(int n) {
  int i,squareroot;
  if (n>10) {
    squareroot = (int) sqrt(n);
    for (i=3; i<=squareroot; i=i+2)
      if ((n%i)==0)
        return 0;
    return 1;
  }
  else
    return 0;
}

int main(int argc, char *argv[])
{

  MPI_Init(&argc,&argv);
  


  int pc,       /* prime counter */
      foundone; /* most recent prime found */
  long long int n, limit;
  long long int stride,start,sum,_max;

  sscanf(argv[1],"%llu",&limit);

  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  start=11+rank*2; /* Assume (2,3,5,7) are counted here */
  stride=size*2;
  	
  if(rank==0)
      printf("Starting. Numbers to be scanned= %lld\n",limit);

      
  pc=0;     

  for (n=start; n<=limit; n=n+stride) {
    if (isprime(n)) {
      pc++;
      foundone = n;
    }			
  }
  MPI_Reduce(&pc,&sum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&foundone,&_max,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
  

  sum+=4; /* Assume (2,3,5,7) are counted here */
  if(rank==0)
    printf("Done. Largest prime is %lld Total primes %lld\n",_max,sum);


  MPI_Finalize();

} 
