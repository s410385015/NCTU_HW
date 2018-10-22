#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<pthread.h>
#define RAND_MAX_32 ((1U << 31) - 1)

long long hits=0;
long long *box;
__thread int rseed=0;

struct info
{
   long long r;
   long long count;
};



inline int rand()
{
    return (rseed = (rseed * 214013 + 2531011) & RAND_MAX_32) ;
}

void *toss( void *ptr )
{
     struct info* in=(struct info*)ptr;
     long long round=in->r;     
     long long count=0;
     rseed=time(NULL);

     for(long long i=0;i< round;i++)
     {  
        
	double x=((double)(rand())/(double)INT_MAX);
	double y=((double)(rand())/(double)INT_MAX);
        
	if(x*x+y*y<=1)
	   count++;
     }
      
     in->count=count;
 
}


int main(int argc, char *argv[]) {

    
    size_t num_thread=1,round=10000;

    if(argc>1)
        num_thread=atoi(argv[1]);
    if (argc>2)
        round=atoi(argv[2]);

    pthread_t *thread;
    thread =malloc(sizeof(pthread_t)*num_thread);
    box=malloc(sizeof(long long)*num_thread);
    
    long long each=(long long)(round/num_thread);
    long long offset=(long long)(round%num_thread);
    
    struct info *in;
    in=malloc(sizeof(struct info)*num_thread);

    for(int i=0;i<num_thread;i++){
        in[i].r=each;
        if(i<offset)
           in[i].r++;
	
    	pthread_create(&thread[i], NULL , toss , &in[i]);
    }
    

    long long hit=0;	   
    for(int i=0;i<num_thread;i++)
    {
	pthread_join(thread[i],NULL);
        
    }
 
    for(int i=0;i<num_thread;i++)
        hits+=in[i].count;
    
    
    printf("%f\n",(double)(hits*4)/round);
}
