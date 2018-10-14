import numpy as np 


def MatrixMul(a,b): 
    r=np.zeros((len(a),len(a)))
    for i in range(0,len(a)):
        for j in range(0,len(b)):
            n=0
            for k in range(0,len(a)):
                n+=a[i][k]*b[k][j]  
            r[i][j]=n

    return r 


def FindLU(A):
   
    size=len(A)
    root=np.zeros((size,size))
    for i in range(0,size):
        root[i][i]=1
    L=root.copy()
    for i in range(0,size):
        tmp=root.copy()
        tmp1=root.copy()
        for j in range(i+1,size):
            tmp[j][i]=-A[j][i]/A[i][i]
            tmp1[j][i]=A[j][i]/A[i][i]
        L=MatrixMul(L,tmp1) 
        A=MatrixMul(tmp,A)       
    return L,A
        


def main():
    A=[[4,1],[1,1]]
    B=[7,2]
    #A=[[3,-1,2],[6,-1,5],[-9,7,3]]
    #B=[10,22,-7]
    Y=np.zeros(len(B))
    X=Y.copy()
    L,U=FindLU(A)

    print("L")
    print(L)
    print("U")
    print(U)
    
    for i in range(0,len(B)):
        for j in range(0,i):
            Y[i]-=L[i][j]*Y[j]
        Y[i]+=L[i][i]*B[i]
    
    for i in range (len(B)-1,-1,-1):
        for j in range (i+1,len(B)):
            X[i]-=U[i][j]*X[j]
        X[i]+=Y[i]
        X[i]/=U[i][i]
    print("solution:")
    print(X)
            

if __name__ =="__main__":
    main()