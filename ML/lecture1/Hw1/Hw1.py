import numpy as np 
import matrix
import random
import matplotlib.pyplot as plt
import numpy as np

"""
Read data points from the file,
and push it into 2d list array as float type
"""
def ReadFile(path):
    file=open(path,'r')
    tmp=file.read().split('\n')
    data=[]
    for i in range (0,len(tmp)):
        data.append(tmp[i].split(','))
    
    data=np.array(data)
    data=data.astype(np.float)
    
    return data


"""
For the data points set,generate the design matix A and the target matrix B in Ax-b
eg.
    data point (1,2) (3,4) (5,6) (7,8) base= 3

design matrix A:                        target matrix B:
    | x0^2 x0 1 |     |  1  1  1 |          | y0 |      | 2 |  
    | X1^2 X1 1 |  =  |  9  3  1 |          | y1 |   =  | 4 |          
    | X2^2 X2 1 |     | 25  5  1 |          | y2 |      | 6 | 
    | X3^2 X3 1 |     | 49  7  1 |          | y3 |      | 8 |
"""
def FindAandB(data,base):
    A=[]
    for i in range(0,len(data)):
        tmp=[data[i][0]**j for j in range(base-1,-1,-1)]
        A.append(tmp)
    A=np.array(A)

    B=[[data[i][1]] for i in range(0,len(data))]
    B=np.array(B)

    return A,B

"""
Generate the Hessian function for the coefficient _X
reference from https://www.zhihu.com/question/19723347
eq.
    base = 3                | X^4 X^3 X^2 |    x will be the summarize _X
    function = ax^2+bx+c    | X^3 X^2 X   |  (0,0) = a^4+b^4+c^4
    _X=[a,b,c]              | X^2 X   1   |
"""
def HessianMatrix(data,base):
    h=[[0 for i in range(base)]for j in range(base)]
    for k in range (len(data)):
        for i in range(base):
            for j in range(base):
                if i==j :
                    h[i][i]+=data[k][0]**((base-i-1)*2)
                else :
                    h[i][j]+=data[k][0]**((base-i-1)+(base-j-1))
    h=np.array(h)
    return h


"""
Generate the gradient ( derivate of _x)
eq.
    base = 3                | x^2(ax^2+bx+c-y) |    
    function = ax^2+bx+c    | x(ax^2+bx+c-y)   |  
    _x=[a,b,c]              | (ax^2+bx+c-y)    |  
"""
def GenerateGradientMatrix(_x,data,base):
    g=[[0]for i in range(base)]
    for k in range(len(data)):
        value=0
        for i in range(base):
            value+=_x[i][0]*(data[k][0]**(base-i-1))
        for i in range(base):
            g[i][0]+=(data[k][0]**(base-i-1))*(value-data[k][1])
    g=np.array(g)
    return g


"""
print the equation in format ax^2+bx^1+c=y
"""
def printEquation(_x):
    eq="equation = "
    for i in range (len(_x)-1):
        eq+="{:.2}".format(_x[i][0])+"x^"+str(len(_x)-i-1)+" + "
    eq+="{:.2}".format(_x[len(_x)-1][0])
    eq+=" = y"
    print(eq)


def main():

    
    path=input("Enter file name with path(eg. D:\lecture1\data.txt):")
    base=input("Enter the number of polynomial bases:")
    l=input("Enter lambda:")

    d=ReadFile(path)
    base=int(base)
    l=float(l)
    
    """
    d=ReadFile('D:\桌面用\MeachinLearning\MLclass\lecture1\data.txt')
    #d=ReadFile('data.txt')
    base=3
    l=0.5
    """
    print("Problem A.")
    A,B=FindAandB(d,base)
    #matrix.printMatrix(A,"A")
    #matrix.printMatrix(B,"B")


    AT=matrix.TransposeMatrix(A)
    #matrix.printMatrix(AT,"AT")

    ATA=matrix.Mul(AT,A)
    #matrix.printMatrix(ATA,"ATA")

    eye=matrix.eyeMatrix(base,l)
    _ATA=matrix.Add(ATA,eye)
   
    matrix.printMatrix(_ATA,"ATA+λi")

    L,U=matrix.FindLU(_ATA)
    #matrix.printMatrix(L,"L")
    #matrix.printMatrix(U,"U")

    ATAi=matrix.FindInverseFromLU(L,U)
    print("1.")
    matrix.printMatrix(ATAi,"Inverse of ATA+λi")

    print("2.")
    _x=matrix.Mul(matrix.Mul(ATAi,AT),B)
    #matrix.printMatrix(_x,"_x")
    printEquation(_x)

    error=matrix.Sub(matrix.Mul(A,_x),B)
    
    w_error=matrix.Mul(matrix.TransposeMatrix(_x),_x)
    w_error=matrix.Factor(w_error,l)
    #matrix.printMatrix(w_error,"weight error")
    value_error=matrix.Mul(matrix.TransposeMatrix(error),error)
    #matrix.printMatrix(value_error,"value error")
    matrix.printMatrix(matrix.Add(value_error,w_error),"a_error")
    
    print("------------------------")
    #------------------------------------------------------------------
    print("Problem b.")
    h=HessianMatrix(d,base)
    #matrix.printMatrix(h,"Hessian Matrix")
    hL,hU=matrix.FindLU(h)
    hi=matrix.FindInverseFromLU(hL,hU)

    # Random generate the coefficient
    _x0=[[random.random()]for i in range(base)]
    #matrix.printMatrix(_x0,"_x0")
    g=GenerateGradientMatrix(_x0,d,base)
    #matrix.printMatrix(g,"Gradient")
    
    print("1.")
    _x0=matrix.Sub(_x0,matrix.Mul(hi,g))
    #matrix.printMatrix(_x0,"solution")
    printEquation(_x0)
    error=matrix.Sub(matrix.Mul(A,_x0),B)
    value_error=matrix.Mul(matrix.TransposeMatrix(error),error)
    matrix.printMatrix(value_error,"b_error")


    x_plot=np.arange(0,10,0.1)
    y_plot=[0 for i in range (len(x_plot))]
    for i in range(len(x_plot)):
        for j in range (len(_x)):
            y_plot[i]+=_x[j]*(x_plot[i]**(len(_x)-j-1))

  
    plt.title("a-rLSE")
    plt.plot(x_plot, y_plot)
    plt.show()


    x_plot=np.arange(0,10,0.1)
    y_plot=[0 for i in range (len(x_plot))]
    for i in range(len(x_plot)):
        for j in range (len(_x0)):
            y_plot[i]+=_x0[j]*(x_plot[i]**(len(_x0)-j-1))

  
    plt.title("b-Newton")
    plt.plot(x_plot, y_plot)
    plt.show()



if __name__ =="__main__":
    main()