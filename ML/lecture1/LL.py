import numpy as np 
import math


def FindL(A):
    L=np.zeros((len(A),len(A)))
    LT=np.zeros((len(A),len(A)))
    for i in range(0,len(A)):
        for j in range(0,i+1):
            if i==j :
                tmp=0
                for k in range(0,j):
                    tmp+=L[j][k]*L[j][k]
                L[j][j]=math.sqrt(A[j][j]-tmp)
            else :
                tmp=0
                for k in range(0,j):
                    tmp+=L[i][k]*L[j][k]
                L[i][j]=(A[i][j]-tmp)/L[j][j]
            LT[j][i]=L[i][j]

    print("L:")
    print(L)
    print("LT:")
    print(LT)
    return L,LT


def main():
    A=[[4,1],[1,1]]
    B=[7,2]
    L,LT=FindL(A)
    Y=np.zeros(len(B))
    X=Y.copy()

    for i in range(0,len(B)):
        for j in range(0,i):
            Y[i]-=L[i][j]*Y[j]
        Y[i]+=B[i]
        Y[i]/=L[i][i]
  
    for i in range (len(B)-1,-1,-1):
        for j in range (i+1,len(B)):
            X[i]-=LT[i][j]*X[j]
        X[i]+=Y[i]
        X[i]/=LT[i][i]
    
    print("solution:")
    print(X)


if __name__ =="__main__":
    main()

