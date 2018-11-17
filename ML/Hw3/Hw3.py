from math import *
import random
import numpy as np
import matplotlib.pyplot as plt

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
    def UnivariateGaussianA(self):
        U1=random.random()
        U2=random.random()
        Z0=sqrt(-2*log(U1))*cos(2*pi*U2)
        return U1,U2,(Z0*sqrt(self.s)+self.m)
    
    def UnivariateGaussian(self,m,a):
        U1=random.random()
        U2=random.random()
        Z0=sqrt(-2*log(U1))*cos(2*pi*U2)
        return U1,U2,(Z0*sqrt(a)+m)

    '''
    y = W*phi(x)+e 
    where e ~ N(0, a)
    '''
    def PolynomialBasisLinearMode(self,x,flag=False):
        
        y=0
        _x=self.GeneratePhi(x)
        for i in range(len(_x)):
            y+=_x[i]*self.w[i]
        
        if flag:
            y+=self.NormalDistribution(x,0,self.a)
            

        
        _x=(np.array(_x)[np.newaxis])
        

        return _x,y
    
    def GenerateFromSin(self,x,flag=False):
        
        _x=self.GeneratePhi(x)
        y=sin(2*pi*x)
        _x=(np.array(_x)[np.newaxis])
        if flag:
            u1,u2,v=self.UnivariateGaussian(0,self.a)
            print(v)
            y+=v
        return _x,y

    '''
    Eg. 
       basis=3
       phi(x)=[1,x,x^2]
    '''
    def GeneratePhi(self,x):
        _x=[x**i for i in range(self.n)]
        
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
    def __init__(self,a,b,w,base):
        self.a=a
        self.b=b
        self.var=0xFFFF
        self.mean=np.zeros((base))[np.newaxis]
        self.mean=np.matrix(np.transpose(self.mean))
        print(self.mean)
        self.w=w
        self.base=base

    '''
    if mean = 0 
    var  = axTx+bI
    mean = a(var^-1)(xT)y

    else
    var  = axTx+
    mean = (var^-1)(axTy+Sm)

    S=prior's var^-1
    m=prior's mean
    '''
    def update(self,x,y):
        x=np.array(x)
            
        _x=np.matrix(x)
  
        if  not np.any(self.mean):
          
            self.var=(self.a*(np.transpose(_x)*_x))+(self.b*np.identity(self.base))
           
            self.mean=self.a*(np.linalg.inv(self.var))*np.transpose(_x)*y
        else:
            #S=np.linalg.inv(self.var)
            #S=np.linalg.inv(self.var)
            S=self.var
            self.var=(self.a*np.transpose(_x)*_x)+(self.var)
            
            self.mean=np.linalg.inv(self.var)*((self.a*np.transpose(_x)*y)+(S*self.mean)) 
            #self.mean=np.linalg.inv(self.var)*(self.a*np.transpose(_x)*y+S*self.mean)

   
    '''
    Predictive Distribution:
    N(mT*phi,(1/a)+phiT(var^-1)phi)
    '''
    def GetPredictiveDistribution(self,x):
        
        _x=np.matrix(x)
        _m=_x*self.mean
        
        _var=(1/self.a)+_x*np.linalg.inv(self.var)*np.transpose(_x)

        _m=_m.item(0,0)
        _var=_var.item(0,0)
 
        return _m,_var

    def GetValue(self,x):
        x=np.matrix(x)

        
        value=x*self.mean
    
        
        value=value.item(0,0)
        return value



def main():

    '''
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
        w[i]=float(input("w"+str(i)+": "))
    
    b=input("Enter the precision (b) for initial prior w ~ N(0, b-1I): ")
    b=float(b)
    '''
  
    m=1
    s=2
    basis=3
    a=0.1
    w=[1,1,1]
    b=0.1

    gen=RandomDataGenerator(m,s,basis,a,w)
    #u1,u2,y=gen.UnivariateGaussian()
    se=SequentialEstimate()
    #se.add_variable(y)
    blr=BaysianLinearRegression((1/a),b,w,basis)

    
   
    print("Enter the command: ")
    print("-- [1a] to generate data from univariate gaussian data generator")
    print("-- [1b] to generate data polynomial basis linear model")
    print("-- [2] to sequential estimate the mean and variance from univariate gaussian data generator(1a)")
    print("-- [3] to do Baysian Linear regression")
    print("-- [quit] to exit the program")
    while(True):
        command=input("command :")

        if command.find("1a") != -1:
            u1,u2,y=gen.UnivariateGaussianA()
            print("From Univariate Gaussian with m:"+str(gen.m)+" and s:"+str(gen.s))
            print("data: "+str(y))

        if command.find("1b") != -1:
            x,y=gen.PolynomialBasisLinearMode(random.uniform(-10,10))
            print("From Univariate Gaussian with polynomial basis linear model with basis:"+str(gen.n)+ ", a:"+str(gen.a))
            print("and w: "+str(gen.w))
            print("x: "+str(x))
            print("y: "+str(y))
        if command.find('2') != -1:
            
            for i in range(1000):
                u1,u2,y=gen.UnivariateGaussianA()
                print("From Univariate Gaussian with m:"+str(gen.m)+" and s:"+str(gen.s))
                print("data: "+str(y))
                se.add_variable(y)
                print("Estimate m:"+str(se.get_mean())+" s:"+str(se.get_variance()))
   
        if command.find('3') != -1:

            data_gen_x=[]
            data_gen_y=[]
            X=[]
            Y=[]
            data_x=np.linspace(-10,10,100)
            for k in range(50):
                data_y=[]
                data_new=[]
                for i in range(100):
                    x,y=gen.PolynomialBasisLinearMode(data_x[i])
                    #x,y=gen.GenerateFromSin(data_x[i])
                    data_y.append(y)
                    #d=(x*0).item((0,0))
                    #data_new.append(d)
              
                data=random.uniform(-10,10)
                #data=gen.UnivariateGaussianA()
                
                #x,y=gen.GenerateFromSin(data,True)
                x,y=gen.PolynomialBasisLinearMode(data)
              
            
                blr.update(x,y)
                


                data_gen_x.append(data)
                data_gen_y.append(y)
                _m,_var=blr.GetPredictiveDistribution(gen.GeneratePhi(data))
                print("-----------")
                
                print("Y:"+str(y))
                print("_mean:"+str(_m))
                print("_var:"+str(_var))

                #plt.plot(data_x,data_new)

              
                #print(blr.GetValue(x))
                
                #_m,_var=blr.GetPredictiveDistribution(x,y)
                print("mean")
                print(blr.mean)
                #print("variance")
                #print(blr.var)
                data_new=[]
                upper=[]
                lower=[]
                for i in range(100): 
                    new_x=gen.GeneratePhi(data_x[i])
                    new_x=blr.GetValue(new_x)
                    _m,_var=blr.GetPredictiveDistribution(gen.GeneratePhi(data_x[i]))
                    upper.append(_m+sqrt(_var))
                    lower.append(_m-sqrt(_var))
                    data_new.append(new_x)
                #data_x.append(i)
                #print("Generate data point: (x,y) : ("+str(x)+" , "+str(y)+")")

                
                #plt.plot(data_x,data_new)
                if k==0 or k==1 or k==3 or k==25:
                    plt.plot(data_gen_x,data_gen_y,'ro',color='blue')
                    plt.plot(data_x,data_y,color='green')
                    plt.fill_between(data_x,upper,lower,facecolor='pink')
                    plt.plot(data_x,data_new,color='red')
                   #plt.ylim((-1.5,1.5))
                    plt.show()
                #
              
            plt.plot(data_gen_x,data_gen_y,'ro',color='blue')
            plt.plot(data_x,data_y,color='green')
            plt.fill_between(data_x,upper,lower,facecolor='pink')
            plt.plot(data_x,data_new,color='red')
            #plt.ylim((-1,1))
            #plt.show()
             
            #print("Posterior mean : \n"+" "+str(blr.mean)+" \n variance : \n"+str(blr.var))
            
            
            #print("Predictive distribution ~N("+str(_m)+","+str(_var)+")")
        print('--------------------------')
        if command.find('q') != -1:
            break
        

if __name__ =="__main__":
    main()