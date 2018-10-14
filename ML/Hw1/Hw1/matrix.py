import numpy as np

"""
Find the inverse matix from L and U matrix 
referenece from https://www.youtube.com/watch?v=dza5JTvMpzk
substitute method:
Ax=b
(LU)x=b -> Ly=b ->Ux=y
"""
def FindInverseFromLU(L,U):
    Inv=[[0 for i in range (len(L))]for i in range(len(L))]
    for k in range (0,len(L)):
        base=[0 for j in range(len(L))]
        base[k]=1
        Y=[0 for j in range(len(base))]
        X=[0 for j in range(len(base))]
        for i in range(len(base)):
            for j in range(i):
                Y[i]-=L[i][j]*Y[j]
            Y[i]+=L[i][i]*base[i]
    
        for i in range (len(base)-1,-1,-1):
            for j in range (i+1,len(base)):
                X[i]-=U[i][j]*X[j]
            X[i]+=Y[i]
            X[i]/=U[i][i]
        for i in range (len(X)):
            Inv[i][k]=X[i]

     
    Inv=np.array(Inv)
    return Inv



"""
Generate L and U matrix from A matrix
reference from https://zh.wikipedia.org/wiki/LU%E5%88%86%E8%A7%A3
"""
def FindLU(A):
    root=eyeMatrix(len(A))
    L=root.copy()
    U=A.copy()
    for i in range(0,len(A)):
        tmp=root.copy()
        tmp1=root.copy()
        for j in range(i+1,len(A)):
            tmp[j][i]=-U[j][i]/U[i][i]
            tmp1[j][i]=U[j][i]/U[i][i]
        L=Mul(L,tmp1) 
        U=Mul(tmp,U)       
    return L,U


"""
Generate identity matrix with size s and factor by value f, 
f default value as 1
eg.
size = 3 
factor = 0.5
identity matix:
               | 0.5   0   0 |
               |   0 0.5   0 |
               |   0   0 0.5 |
"""
def eyeMatrix(size,factor=1.):
    eye=[[0 for x in range(size)] for y in range(size)]
    for i in range (size):
        eye[i][i]=1.*factor
    eye=np.array(eye)
    return eye


"""
multiply two matrix
eg.               | 1 2 |      
    | 1 3 5 7 | * | 3 4 | = |  84 100 |
    | 2 4 6 8 |   | 5 6 |   | 100 120 |   
                  | 7 8 |
        [2,4]   *  [4,2]  =   [2,2]  
"""
def Mul(a,b):
    b=b if isinstance(b[0],np.ndarray) else [b]
    a=a if isinstance(a[0],np.ndarray) else [a]
    r=[[0 for x in range(len(b[0]))] for y in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            n=0
            for k in range(len(a[i])):
                n+=a[i][k]*b[k][j]  
            r[i][j]=n
    r=np.array(r)
    return r 

"""
add two matrix
eg.                     
    |  84 100 | + | 0.5   0 | = |  84.5 100 |
    | 100 120 |   |   0 0.5 |   | 100   120.5 |   
"""
def Add(a,b):
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j]+=b[i][j]
    a=np.array(a)
    return a


"""
subtract two matrix 
eg.                     
    |  84 100 | - | 1  1 | = |  83  99 |
    | 100 120 |   | 1  1 |   |  99 101 |   
"""
def Sub(a,b):
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j]-=b[i][j]
    a=np.array(a)
    return a


"""
factor a matrix by value
eg.
    value=2
2 * |  84 100 | = |  168  200 |
    | 100 120 |   |  200  240 |  
"""
def Factor(a,f=1):
    a=a if isinstance(a[0],np.ndarray) else [a]
    a=[[a[x][y]*f for x in range(len(a[0]))] for y in range (len(a))]
    a=np.array(a)
    return a

"""
Transpose the matrix
eg. | 1 2 |      
    | 3 4 |  ->  | 1 3 5 7 |
    | 5 6 |      | 2 4 6 8 |
    | 7 8 |      
"""
def TransposeMatrix(m):
    m=m if isinstance(m[0],np.ndarray) else [m]
    mt=[[m[x][y] for x in range(len(m))] for y in range(len(m[0]))]
    mt=np.array(mt)
    return mt


"""
print function of format:
A:
| 1 2 |
| 3 4 |
"""
def printMatrix(matrix,title):
    print(title+":")
    print(matrix)