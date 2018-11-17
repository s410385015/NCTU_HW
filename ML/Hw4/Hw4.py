from math import *
import random
import numpy as np
from scipy.special import *
import matplotlib
import matplotlib.pyplot as plt
import time
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def UnivariateGaussian(mean,variance):
    U1=random.random()
    U2=random.random()
    Z0=sqrt(-2*log(U1))*cos(2*pi*U2)
    return (Z0*sqrt(variance)+mean)


def Sigmoid(x):

    return 1.0/(1.0+e**(-x))


def CrossEntropy(w,x,target):
    error=0
    
 
    z=Sigmoid(np.dot(w,x.T))

    e1=xlogy(target,z.T)
    e2=xlogy(1-target,(1-z).T)
    error=np.sum(e1)+np.sum(e2)
    error=-error/len(x)
    #error=(np.dot(-target,np.log(z).T)-np.dot(1-target,np.log(1-z).T))/len(x)

    
    #error+= (max(_x,0)-_x*y[i]+log(1+e**(-abs(_x))))
    return error

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def main():
    n=50
    mx1=1
    vx1=2
    my1=3
    vy1=4
    mx2=5
    vx2=6
    my2=7
    vy2=8
    data1=[]
    data2=[]
    data=[]
    w=np.array([0,0,0])
    iteration=0
    feature=[]
    for i in range(n):
        x1=UnivariateGaussian(mx1,vx1)
        y1=UnivariateGaussian(my1,vy1)
        data1.append((x1,y1,0))
        feature.append((1,x1,y1))
    
    for i in range(n):
        x2=UnivariateGaussian(mx2,vx2)
        y2=UnivariateGaussian(my2,vy2)
        data2.append((x2,y2,1))
        feature.append((1,x2,y2))

    data1=np.array(data1)
    data2=np.array(data2)
    feature=np.array(feature)
    data=np.concatenate((data1, data2), axis=0)
    
    lim_x=(min(data[:,0])-1,max(data[:,0])+1)
    lim_y=(min(data[:,1])-1,max(data[:,1])+1)
    #print(data)
    X=[[data[i][0],1]for i in range(n*2)]
    XT=np.transpose(np.array(X))
 
    target=data[:,2]
    y=data[:,1]
    x=data[:,0]
    lr=1

    error=0
    pre_error=-1

    flag=False

  
    while True:
        
        
        #
        #z=[(1/(1+e**-(x[i]*w[0]+w[1])))for i in range(n*2)]
        
        #z=[(1/(1+e**-(x[i]*w[1]+w[])))for i in range(n*2)]

        if flag:
            z=Sigmoid(np.dot(w,feature.T))

            gradient=np.dot(feature.T,lr*(target-z))
            gradient/=(n*2)
            w=w+gradient
        else:
            z=Sigmoid(np.dot(w,feature.T))
            gradient=np.dot(feature.T,(z-target))
            gradient/=(n*2) 

            H=np.dot(np.dot(feature.T,(np.diag(z)*np.diag(1-z))),feature) 
            H/=(n*2)
        
            if is_invertible(H):
                _H=np.linalg.inv(H)
                w=w-np.dot(_H,gradient)
            else:
                print("Singalur")
                flag=True 
                w=np.array([0,0,0])
                continue
                gradient=np.dot(feature.T,lr*(target-z))
                print(gradient)
                gradient/=(n*2)
                w=w+gradient

            #print(np.dot(_H,gradient))
            #print(_H)
            print(w)
        
        iteration+=1
      
        error=CrossEntropy(w,feature,target)
    

        
        if pre_error!=-1:
            if error<pre_error:
                if error<0.1 or iteration>300 or pre_error-error<0.000001:
                    break
                lr*=1.05
            else:
                lr*=0.5

        pre_error=error
        print(str(iteration)+" "+str(error)+" "+str(lr))
 


        tmpW=w
        plt.clf()
        plt.plot(data1[:,0],data1[:,1],'ro',color='blue')
        plt.plot(data2[:,0],data2[:,1],'ro',color='red')
        line_x=np.linspace(lim_x[0],lim_x[1])
        plt.plot(line_x,(line_x*w[1]+w[0])/-w[2])
        plt.xlim(lim_x)
        plt.ylim(lim_y)
        #plt.show()
        plt.pause(0.0001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    
    plt.clf()
    plt.ioff()
    plt.plot(data1[:,0],data1[:,1],'ro',color='blue')
    plt.plot(data2[:,0],data2[:,1],'ro',color='red')
    line_x=np.linspace(lim_x[0],lim_x[1])
    plt.plot(line_x,(line_x*w[1]+w[0])/-w[2])
    plt.xlim(lim_x)
    plt.ylim(lim_y)
    plt.show()
    print(iteration)



    TP=0
    FP=0
    FN=0
    TN=0
    z=Sigmoid(np.dot(w,feature.T))
    for i in range(len(z)):
        if target[i]==1:
            if z[i]>=0.5:
                TP+=1
            else:
                FN+=1
        else:
            if z[i]>=0.5:
                FP+=1
            else:
                TN+=1 

    print(np.array([[TP,FP],[FN,TN]]))
    print("sensitivity"+str(TP/(TP+FN)))
    print("specificity"+str(FP/(FP+TN)))
    '''
    plt.plot(data1[:,0],data1[:,1],'ro',color='blue')
    plt.plot(data2[:,0],data2[:,1],'ro',color='red')
    line_x=np.linspace(lim_x[0],lim_x[1])
    plt.plot(line_x,line_x*w[0]+w[1])
    plt.xlim(lim_x)
    plt.ylim(lim_y)
    plt.show()
    '''

if __name__ =="__main__":
    main()
        