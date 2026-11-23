"""Scalar signed interval operations; no full-grid values or conditional division."""
from fractions import Fraction as F
from math import isqrt


def need(ok,why):
    if not ok:raise ValueError(why)

def number(x):return F(int(x[0]),int(x[1])) if isinstance(x,list) else F(x)
def interval(x):return tuple(number(v) for v in x)
def encode(x):return [str(v) for v in x]
def plus(x,y):return x[0]+y[0],x[1]+y[1]
def times(x,y):
    products=(x[0]*y[0],x[0]*y[1],x[1]*y[0],x[1]*y[1]);return min(products),max(products)
def total(rows):
    lo=hi=F()
    for a,b in rows:lo+=a;hi+=b
    return lo,hi
def dot(weights,values):
    need(len(weights)==len(values)==100 and all(0<=a<=b for a,b in weights),'Complete nonnegative weighting law')
    return total(times(p,v) for p,v in zip(weights,values))

def utility(budget,j):
    need(type(budget)is int and budget>=0 and type(j)is int and 0<=j<=1000,'Exact physical budget/index')
    out=[];scale=1<<80
    for n in range(1,101):
        x=budget*j-n*1_000_000;k=isqrt(abs(x)*scale*scale//1_000_000)
        h=k+(k*k*1_000_000!=abs(x)*scale*scale)
        out.append((-F(2*h,scale),-F(2*k,scale)) if x<0 else (F(k,scale),F(h,scale)))
    return out

def proper(row):
    need(len(row)==100 and all(0<=a<=b<=1 for a,b in row),'Complete probability support')
    box=total(row);need(box[0]<=1<=box[1],'Proper law total enclosure');return box

def report_joint(law,channel,y):
    need(type(y)is int and 1<=y<=100,'Full report integer')
    q=[times(p,channel[n][y-1]) for n,p in enumerate(law)]
    m=total(q);m=m[0],min(F(1),m[1]);need(0<=m[0]<=m[1] and m[1]>0,'Positive-support upper mass; zero lower retained')
    return q,m

def report_value(task,joint,mass,other,queried_u,other_u):
    primary=dot(joint,queried_u);opposite=times(mass,dot(other,other_u))
    base=2*F(task['gpa_baseline']);scale=F(task['gpa_scale']);need(scale>0,'Positive GPA scale')
    return plus(times((base,base),mass),times((scale,scale),plus(primary,opposite)))
