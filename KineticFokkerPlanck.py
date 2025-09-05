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
import os

script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

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
    
def Chang_Cooper(u):
    """
    Always stable because is implicit.
    If explicit, make sure Stabilty if :

        dv < 1/(2 max(B) )

    """
     
    u[1:-1] = lu.solve(u[1:-1])

    return u

##########################
#      MAIN PROGRAM      #
##########################

# DOMAIN: (x,v) in [-Lx, Lx] x [-Lv, Lv]
Lx = 50
Lv = 50

# GRID OF POINTS
Nx = 101 
Nv = 101 

dx = 2 * Lx / (Nx-1) 
dv = 2 * Lv / (Nv-1) # Stability for Chang-Cooper

X = np.linspace(-Lx, Lx, Nx)
V = np.linspace(-Lv, Lv, Nv)
[xx,vv] = np.meshgrid(X,V)

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












