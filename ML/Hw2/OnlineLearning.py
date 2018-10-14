import os
import numpy as np
from math import *
'''
The file contains servel line with 0 & 1
eg. 0101 
    0110101 
    01 

the return value is two array

the first array contain the orginal string of each raw:
    raw=["0101" ,"0110101","01"]

the sccond array will look like this:
    array size = n lines * 2
    p[3,2]=([4,2],[7,4],[2,1])  
    first number indicate the number of trails
    second number indicate how many ones 
'''
def ReadFromFile(path):
    path=os.path.split(os.path.realpath(__file__))[0]+'\\'+path
    file=open(path,'r')

    orginal=[]
    p=[]

    for line in file.readlines():
        line=line.strip()
        var=[]
        orginal.append(line)
        var.append(len(line))
        var.append(line.count('1'))
        p.append(var)
    
    file.close()
    p=np.array(p)
    p.astype(float)
    return orginal,p


def main():
    o,p = ReadFromFile("trails.txt")
    a=input(" parameter a for the initial beta prior : ")
    b=input(" parameter b for the initial beta prior : ")
    a=float(a)
    b=float(b)
    for i in range(len(o)):
        print("-------------------------")
        print(o[i])
        print("Binomial likelihood(MLE):")
        var=(p[i][1]/p[i][0])
        print(round(var,3))
        print("Prior:")
        print(a/(a+b))
        print("Posterior:")
        a+=p[i][1]
        b+=(p[i][0]-p[i][1])
        print(a/(a+b))


if __name__ =="__main__":
    main()