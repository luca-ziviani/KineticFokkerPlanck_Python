# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:49:01 2025

FINITE DIFFERENCE SCHEME

    This is the branch RoyBorzi

@author: lucaz
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import pickle
#import os
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps


#script_dir = os.path.abspath(os.path.dirname(__file__))
#os.chdir(script_dir)

def minmod(*args):
    signs = [x > 0 for x in args if x != 0]

    if all(signs) or not any(signs):  # tutti positivi o tutti negativi (escludendo zeri)
        return min(args, key=abs) if args else 0
    else:
        return 0

def H_build(f, theta=1):    
    
    Df=np.zeros((Nv,Nx))
    
    for n in range(1,Nx-1):
        for m in range(Nv): # probably range(1, Nv-1)
            Df[m,n] = minmod(theta*(f[m,n] - f[m,n-1])/dx , (f[m,n+1]-f[m,n-1])/(2*dx) , theta*(f[m,n+1]-f[m,n])/dx)
    
    #f_plus = f[:, 1:] - dx/2 * Df[:, 1:]        # size [Nv , Nx-1]
    #f_minus = f[:, :-1] + dx/2 * Df[:, :-1]     # size [Nv , Nx-1]

    f_plus = np.zeros((Nv, Nx-1))
    f_minus = np.zeros((Nv, Nx-1))
    
    f_plus[:,:-1] = f[:, 1:-1] - dx/2 * Df[:, 1:-1]        # size [Nv , Nx-1]
    f_minus[:,1:] = f[:, 1:-1] + dx/2 * Df[:, 1:-1]     # size [Nv , Nx-1]
    
    f_plus[:,-1] = f_minus[::-1,-1]
    f_minus[:,0] = f_plus[::-1,0]
    
    H=np.zeros((Nv,Nx-1))
    
    for n in range(0,Nx-1):
        for m in range(Nv): # probably range(1, Nv-1)
            H[m,n] = V[m]*(f_plus[m,n] + f_minus[m,n])/2 - np.abs(V[m]) * (f_plus[m,n] - f_minus[m,n]) /2
    
    return H


def F_build(f):
    V_c = np.linspace(-Lv+dv/2,Lv-dv/2,Nv-1) 
    
    alpha = 2
    beta=2
    B=np.zeros((Nv-1,Nx))
    
    for n in range(Nx):
        for m in range(Nv-1):
            B[m,n]= (1+V_c[m]**2)**( (beta-2)/2 )*V_c[m] + (1+X[n]**2)**( (alpha-2)/2 )*X[n]
            
    w = dv * B
    delta = 1/w - 1/(np.exp(w)-1)
    F=np.zeros((Nv-1,Nx))
    
    for n in range(Nx):
        for m in range(Nv-1):
            F[m,n] = (1/dv + (1-delta[m,n])*B[m,n]) * f[m+1,n] - (1/dv - B[m,n]*delta[m,n])*f[m,n]
    
    F[0,:] = 0
    F[-1,:] = 0



    return F


def FD_Advection(f):
    """
    Finite difference step in time for
    
    d_t f = Ux d_x f + Uv d_v f 

    CFL:
    DeltaT * u/DeltaX + DeltaT* v /DeltaV eq 1

    f , U , V have the same shape

    """
    # Pad df with zeros around
    df = np.zeros((np.shape(f)[0]+2,np.shape(f)[1]+2))
    
    df[U_p[0]+1,U_p[1]+1] -= dt / dx * (f[U_p[0],U_p[1]] - f[U_p[0],U_p[1]-1])* Ux[U_p[0],U_p[1]]
    df[U_n[0]+1,U_n[1]+1] -= dt / dx * (f[U_n[0],U_n[1]+1] - f[U_n[0],U_n[1]]) * Ux[U_n[0],U_n[1]]
    df[V_p[0]+1,V_p[1]+1] -= dt / dv * (f[V_p[0],V_p[1]] - f[V_p[0]-1,V_p[1]])* Uv[V_p[0],V_p[1]]
    df[V_n[0]+1,V_n[1]+1] -= dt / dv * (f[V_n[0]+1,V_n[1]] - f[V_n[0],V_n[1]]) * Uv[V_n[0],V_n[1]]

    return f + df[1:-1,1:-1]

def FV_Advection(f):
    return
    
def Chang_Cooper(u):
    """
    Always stable because is implicit.
    If explicit, make sure Stabilty if :

        dv < 1/(2 max(B) )

    """
     
    u[1:-1] = lu.solve(u[1:-1])

    return u


class Grid:
    def __init__(self, rows, columns, Xmax, Vmax):
        # grid with ghost points
        self.rows = rows
        self.columns = columns
        self.values = np.zeros((rows+2,columns+2))
        
        
        # grid size
        self.Xmax = Xmax
        self.Vmax = Vmax

        self.dx = 2 * self.Xmax / self.columns
        self.dv = 2 * self.Vmax / self.rows
        
        self.x = np.linspace(-self.Xmax +self.dx/2 , self.Xmax - self.dx/2 , self.columns)
        self.v = np.linspace(-self.Vmax +self.dv/2 , self.Vmax - self.dv/2 , self.rows)
        #self.v=self.v[::-1]

        [self.xx,self.vv] = np.meshgrid(self.x ,self.v)
        
        # Tools
        self.GradX = np.ones((rows, columns))
        self.f_plus = np.zeros((rows, columns+1))
        self.f_minus = np.zeros((rows, columns+1))
        self.H = np.zeros(( rows, columns +1))
        
        
        
    def specular_BD(self):
        """
        Update ghost points in x according to
        specular boundary conditions

        """
        
        for i in range(self.rows):
            self.values[i+1 , -1] = self.values[self.rows-i-1 , -2]
            self.values[i+1 , 0] = self.values[self.rows-i-1,1 ]

    #    for i in range(self.rows):
     #       self.values[i+1 , -2] = self.values[self.rows-i-1 , -3]
      #      self.values[i+1 , 1] = self.values[self.rows-i-1, 2]
            

    def H_update(self, theta=1.5):    
        
        for n in range(1,self.columns+1):
            for m in range(1, self.rows+1): 
                #self.GradX[m-1,n-1] = minmod(theta*(self.values[m,n] - self.values[m,n-1])/self.dx ,
                #                 (self.values[m,n+1]-self.values[m,n-1])/(2*self.dx) , 
                #                 theta*(self.values[m,n+1]-self.values[m,n])/self.dx)
                
                # UPWIND for v>0
                if self.v[m-1]>0:
                    self.GradX[m-1,n-1 ] =(self.values[m,n]-self.values[m,n-1])/(self.dx)
                else:
                    self.GradX[m-1,n-1 ] =(self.values[m,n+1]-self.values[m,n])/(self.dx)
        
        self.f_plus[:,:-1] = self.values[1:-1,1:-1] - self.dx/2 * self.GradX
        self.f_minus[:,1:] = self.values[1:-1,1:-1] + self.dx/2 * self.GradX
        
        for n in range(self.columns+1):
            for m in range(self.rows):
                if self.v[m]>=0:
                    self.H[m,n] = self.v[m]*self.f_minus[m,n]
                else:
                    self.H[m,n] = self.v[m]*self.f_plus[m,n]
                #self.H[m,n] = self.v[m]*(self.f_plus[m,n] + self.f_minus[m,n])/2 
                #- np.abs(self.v[m]) * (self.f_plus[m,n] - self.f_minus[m,n]) /2

        # Specular flux in bounary conditions
        for i in range(int(self.rows /2)):
            self.H[self.rows - 1 -i, 0] = -self.H[i,0]
            self.H[i, -1] = -self.H[self.rows - 1 -i,-1]
        
        return 

    def mass(self):
        return sum(sum(self.values[1:-1,1:-1]))*self.dx*self.dv

    def plot_grid(self):
        ax = plt.axes(projection='3d')
        #ax.plot_surface(self.vv, self.xx, self.H[:,1:] , cmap=cm.jet)
        ax.plot_surface(self.xx, self.vv, self.values[1:-1,1:-1] , cmap=cm.jet) #[1:-1][::-1, 1:-1]
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title("Plot of f")


grid = Grid(80,80,10,10)

grid.values[1:-1,1:-1] = np.exp(-(grid.xx-5)**2/2 - (grid.vv-3)**2/2)/(2*np.pi)
#grid.values[1:-1,-2] = np.ones(grid.rows)
grid.specular_BD()



print("Mass:", grid.mass())

dt = 0.004
Nt = 500

for k in range(Nt):
    grid.H_update()
    grid.values[1:-1,1:-1] = grid.values[1:-1,1:-1] - dt/grid.dx *(grid.H[:,1:] - grid.H[:,:-1])
    grid.specular_BD()

print("Mass:", grid.mass())

grid.plot_grid()



##########################
#      MAIN PROGRAM      #
##########################
"""

# DOMAIN: (x,v) in [-Lx, Lx] x [-Lv, Lv]
Lx = 5
Lv = 5

# NUMBER OV CELLS
Nx = 22
Nv = 20

# SIZE OF CELLS
dx = 2 * Lx / Nx 
dv = 2 * Lv / Nv 

X = np.linspace(-Lx +dx/2 , Lx - dx/2 , Nx)
#X = np.linspace(-Lx, Lx, Nx)
V= np.linspace(-Lv +dv/2 , Lv - dv/2 , Nv)
#V = np.linspace(-Lv, Lv, Nv)
[xx,vv] = np.meshgrid(X,V)

# Initialization
f= np.zeros([Nv,Nx]) # Ghost points set to 0
#f[1:-1,1:-1] = np.exp(-(xx[1:-1,1:-1])**2 - (vv[1:-1,1:-1])**2)
f = np.exp(-(xx-1)**2 - (vv)**2)
f = f /(dx*dv*sum(sum(f)))





#H = H(f)
#F = F(f)

T = 1
Nt=10

dt = T/Nt

k=0

f1= np.zeros(np.shape(f))

while k<Nt :
    for n in range(1,Nx-1):
        for m in range(1,Nv-1):
            H = H_build(f)
            F = F_build(f)
            f[m,n] = f[m,n] + dt * (H[m, n]-H[m,n-1 ])/dx +dt* (F[m,n]-F[m-1,n])/dv
    
    k+=1
    



#ax = plt.axes(projection='3d')
#ax.plot_surface(xx[:,1:], vv[:,1:], H, cmap=cm.jet)

#ax = plt.axes(projection='3d')
#ax.plot_surface(xx[1:,:], vv[1:,:], F, cmap=cm.jet)

ax = plt.axes(projection='3d')
ax.plot_surface(xx, vv, f, cmap=cm.jet)

alpha = 2
beta = 1.
#gamma = 2 # gamma > 1

#   TOOLS FOR ADVECTION
#----------------------------------------------------------
Ux = -vv  
Uv = (1+xx**2)**((alpha -2)/2.) * xx

U_p = np.where(Ux > 0) 
U_n = np.where(Ux <= 0)
V_p = np.where(Uv > 0)
V_n = np.where(Uv <= 0)

dt = dx*dv/(dx * np.abs(Uv[-1,-1]) + dv * np.abs(Ux[-1,-1]) )
T = 300
Nt = int(T/dt) + 1
T_points = np.linspace(0, T, Nt)


#   TOOLS FOR CHANG-COOPER
#----------------------------------------------------------
# Centers of cells in v
V_c = np.linspace(-Lv+dv/2,Lv-dv/2,Nv-1) #(-LX+dx/2:dx:LX-dx/2);
B = (1+V_c**2)**((beta -2)/2.) * V_c
#B = (1 + gamma) * (1+V_c**2)**(-2) * V_c
w = dv * B
delta = 1/w - 1/(np.exp(w)-1)

d_upper = - dt/dv * ( 1/dv + (1-delta[1:]) * B[1:] )
d_main = 1 + dt/dv * ( 2/dv + (1 - delta[:-1]) * B[:-1] - delta[1:] * B[1:])
d_lower = - dt/dv * ( 1/dv - delta[:-1] * B[:-1])

# Dirichlet boundary condition
d_upper = np.insert(d_upper, 0, 0.) #prepend d_upper with 0
d_main = np.insert(d_main, 0, 1.) # prepend and append d_main with 1
d_main = np.append(d_main, 1)
d_lower = np.append(d_lower, 0) #append d_lower with 0

A = diags([d_lower, d_main, d_upper], [-1, 0, 1])
lu = splu(A)

Nx += 2 # Ghost points
Nv += 2 # Ghost points

# Initialization
f= np.zeros([Nv,Nx]) # Ghost points set to 0
f[1:-1,1:-1] = np.exp(-(xx)**2 - (vv)**2)
f = f /(dx*dv*sum(sum(f)))
#with open('KFP_alpha2_beta0.5_Lx50Lv50.pkl','rb') as file:
#    ( f, X, V, T )= pickle.load(file)



#print("sum(f(0)) = " + str(dx*dv*sum(sum(f))))

# Equilibrium for beta = 2
eq = np.zeros([Nv,Nx]) # Ghost points set to 0
eq[1:-1,1:-1] = np.exp(-(1+xx**2)**((alpha-2)/2 ) - 0.5*(vv)**2)
eq = eq /(dx*dv*sum(sum(eq)))


for k in range(Nt):
    f = FD_Advection(f)
    for n in range(Nx):
        f[:,n] = Chang_Cooper(f[:,n])
    

#print("sum(f(T)) = " + str(dx*dv*sum(sum(f))))
  

try:
    gamma
except NameError:
    with open('KFP_alpha'+str(alpha)+'_beta'+str(beta)+'_Lx'+str(Lx)+'Lv'+str(Lv)+'.pkl', 'wb') as file:
        pickle.dump(( f, X, V, T ),file)
else:
    with open('KFP_alpha'+str(alpha)+'_gamma'+str(gamma)+'_Lx'+str(Lx)+'Lv'+str(Lv)+'.pkl', 'wb') as file:
        pickle.dump(( f, X, V, T ),file)


"""

m=0







