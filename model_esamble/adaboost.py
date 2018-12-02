# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:00:04 2018

@author: masha
"""
#计算可能的划分点
import numpy as np
def generateGxList(x):
    gxlist=[]
    for i in range(len(x)-1):
        gx=(x[i]+x[i+1])/2
        gxlist.append(gx)
    return gxlist
#计算弱分类器的权重
def calAlpha(min_err):
    alpha=0.5*np.log((1-min_err)/min_err)
    return alpha

def calcNew_weight(alpha,ygx,weight,gx,y):
    new_weight=[]
    sum_weight=0
    for i in range(len(weight)):
        flag=1
        if i<gx and y[i]!=ygx:
            flag=-1
        if i>gx and y[i]!=-ygx:
            flag=-1
        weight_i=weight[i]*np.exp(-alpha*flag)
        new_weight.append(weight_i)
        sum_weight+=weight_i
    new_weight=new_weight/sum_weight
    return new_weight

def Calc_err_num(gx,x,y,weight):
    err_1=0
    err_neg1=0
    ygx=1
    for i in range(len(x)):
        if i<gx and y[i]!=1:
            err_1+=weight[i]
        if i>gx and y[i]!=-1:
            err_1+=weight[i]
        if i<gx and y[i]!=-1:
            err_neg1+=weight[i]
        if i>gx and y[i]!=1:
            err_neg1+=weight[i]
    if err_neg1<err_1:
        return err_neg1,-1
    return err_1,1

def train(fx,i,x,y,weight):
    min_err=np.inf
    bestgx=0.5
    gxlist=generateGxList(x)
    bestygx=1
    for x_i in gxlist:
        err,ygx=Calc_err_num(x_i,x,y,weight)
        if err<min_err:
            min_err=err
            bestgx=x_i
            bestygx=ygx
    fx[i]["gx"]=bestgx
    alpha=calAlpha(min_err)
    fx[i]["alpha"]=alpha
    fx[i]["ygx"]=bestygx
    new_w=calcNew_weight(alpha,bestygx,weight,bestgx,y)
    return new_w
def calc_err(fx,n,x,y):
    err_num=0
    for i in range(len(x)):
        f_i=0
        for j in range(n):
            fx_i_alpha=fx[j]["alpha"]
            fx_i_gx=fx[j]["gx"]
            ygx=fx[j]["ygx"]
            if i<fx_i_gx:
                fgx=ygx
            else:
                fgx=-ygx
            f_i+=fx_i_alpha*fgx
        if np.sign(f_i)!=y[i]:
            err_num+=1
    return err_num/len(x)

def adaboost(x,y,err_threshold,max_iter):
    fx={}
    weight=[]
    x_num=len(x)
    weight=[1/x_num]*x_num    
    for i in range(max_iter):
        fx[i]={}
        new_w=train(fx,i,x,y,weight)
        weight=new_w
        fx_err=calc_err(fx,(i+1),x,y)
        if fx_err<err_threshold:
            break
    
    return fx

def load_dataset():
    x=[0,1,2,3,4,5]
    y=[1,1,-1,-1,1,-1]
    return x,y

if __name__=="main":
    x,y=load_dataset()
    err_threshold=0.01
    max_iter=10
    fx=adaboost(x,y,err_threshold,max_iter)
    print(fx)
        