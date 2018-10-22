from math import *
import random
class RandomDataGenerator:
    '''
    a.univariate gaussian data generator
      value (m), variance (s)
    b. polynomial basis linear model data generator
      basis number(n) , a, w
      >> y = WTPhi(x)+e ; e ~ N(0, a)
      >> (ex. n=2 -> y = w0x0 +w1x1)
    '''
    def __init__(self,value,variance,basis,aVariance,Weight):
        self.m=value
        self.s=variance
        self.n=basis
        self.a=aVariance
        self.w=Weight

    '''
    Generate the normal distributiom
    '''
    def NormalDistribution(self,x,mean,variance):
        return  (1/sqrt(2*pi*variance))*(e**(-(x-mean)**2/(2*variance)))
    
    '''
    Apply Box-Muller transform methed:
    reference https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    U1 & U2 are independent samples chosen from the uniform distribution on the unit interval (0, 1) 
    '''
    def UnivariateGaussian(self):
        U1=random.random()
        U2=random.random()
        Z0=sqrt(-2*log(U1))*cos(2*pi*U2)
        return U1,U2,(Z0*sqrt(self.s)+self.m)
    

    '''
    y = W*phi(x)+e 
    where e ~ N(0, a)
    '''
    def PolynomialBasisLinearMode(self):
        x=random.uniform(-10,10)
        y=0
        _x=self.GeneratePhi(x)
        for i in range(len(_x)):
            y+=_x[i]*self.w[i]
        
        y+=self.NormalDistribution(x,0,self.a)

        return x,y
    

    '''
    Eg. 
       basis=3
       phi(x)=[x^2,x^1,1]
    '''
    def GeneratePhi(self,x):
        _x=[x**i for i in range(self.n-1,0,-1)]
        return _x

'''
Sequential estimate the mean & variance
reference from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
'''
class SequentialEstimate:
    def __init__(self):
        self.k=0
        self.n=0
        self.Ex=0
        self.Ex2=0

    def add_variable(self,x):
        if self.n==0:
            self.k=x
        self.n=self.n+1
        self.Ex+=x-self.k
        self.Ex2+=(x-self.k)*(x-self.k) 
     
    def get_mean(self):
        return self.k+self.Ex/self.n
    
    def get_variance(self):
        if self.n-1 == 0:
            return 0
        return  (self.Ex2-(self.Ex*self.Ex)/self.n)/(self.n-1)


class BaysianLinearRegression:
    def __init__(self,a,b,w):
        self.a=a
        self.b=b
        self.var=0xFFFF
        self.mean=0
        self.w=w


    '''
    if mean = 0 
    var  = axTx+bI
    mean = a(var^-1)(xT)y

    else
    var  = axTx+bI
    mean = (var^-1)(axTy+Sm)

    S=prior's var^-1
    m=prior's mean
    '''
    def update(self,x,y):
        if self.mean == 0:
            self.var=self.a*x*x+self.b
            self.mean=self.a*(1/self.var)*x*y
        else:
            S=1/self.var
            self.var=self.a*x*x+self.b
            self.mean=(1/self.var)*(self.a*x*y+S*self.mean)
    
    '''
    Predictive Distribution:
    N(mT*phi,(1/a)+phiT(var^-1)phi)
    '''
    def GetPredictiveDistribution(self):
        _m=0
        _var=0
        for i in range(len(self.w)):
            _m+=self.w[i]*self.mean
        
        _var+=(1/self.a)

        for i in range(len(self.w)):
            _var+=(self.w[i]*self.w[i]*(1/self.var))
        
        return _m,_var


def main():
    m,s=input("Enter the expectation value (m), variance (s): ").split(' ')
    basis,a=input("Enter the basis number, and a of ~N(0,a) : ").split(' ')

    m=float(m)
    s=float(s)
    basis=int(basis)
    a=float(a)

    print("Enter the weight:")
    w=[0 for i in range(basis)]
    #eg basis=3 [w2,w1,w0]
    for i in range(basis):
        w[basis-i-1]=float(input("w"+str(i)+": "))
    
    b=input("Enter the precision (b) for initial prior w ~ N(0, b-1I): ")
    b=float(b)

    gen=RandomDataGenerator(m,s,basis,a,w)
    #u1,u2,y=gen.UnivariateGaussian()
    se=SequentialEstimate()
    #se.add_variable(y)
    blr=BaysianLinearRegression((1/a),b,w)

    print("Enter the command: ")
    print("-- [1a] to generate data from univariate gaussian data generator")
    print("-- [1b] to generate data polynomial basis linear model")
    print("-- [2] to sequential estimate the mean and variance from univariate gaussian data generator(1a)")
    print("-- [3] to do Baysian Linear regression")
    print("-- [quit] to exit the program")
    while(True):
        command=input("command :")

        if command.find("1a") != -1:
            u1,u2,y=gen.UnivariateGaussian()
            print("From Univariate Gaussian with m:"+str(gen.m)+" and s:"+str(gen.s))
            print("data: "+str(y))

        if command.find("1b") != -1:
            x,y=gen.PolynomialBasisLinearMode()
            print("From Univariate Gaussian with polynomial basis linear model with basis:"+str(gen.n)+ ", a:"+str(gen.a))
            print("and w: "+str(gen.w))
            print("x: "+str(x))
            print("y: "+str(y))
        if command.find('2') != -1:
            u1,u2,y=gen.UnivariateGaussian()
            print("From Univariate Gaussian with m:"+str(gen.m)+" and s:"+str(gen.s))
            print("data: "+str(y))
            se.add_variable(y)
            print("Estimate m:"+str(se.get_mean())+" s:"+str(se.get_variance()))
        if command.find('3') != -1:
             x,y=gen.PolynomialBasisLinearMode()
             print("Generate data point: (x,y) : ("+str(x)+" , "+str(y)+")")
             blr.update(x,y)
             print("Posterior mean & variance: "+" "+str(blr.mean)+" , "+str(blr.var))
             _m,_var=blr.GetPredictiveDistribution()
             print("Predictive distribution ~N("+str(_m)+","+str(_var)+")")
        print('--------------------------')
        if command.find('q') != -1:
            break
        

if __name__ =="__main__":
    main()