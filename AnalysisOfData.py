# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:15:47 2025

@author: lucaz

ANALYSIS OF DATA FROM CLUSTER

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
import matplotlib.colors as colors
import pickle
import os

import os

script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

# Tools > Preferences > IPython Console > Graphics > Backend: Qt5
def plot_f(x, v, f):
    """

    Parameters
    ----------
    x : Array of float64 of lenght Nx
        Example:    x = np.linspace(-3, 3, Nx)
    
    v : Array of float64 of lenght Nv
        Example:    v = np.linspace(-3, 3, Nv)

    f : Array of float64 of size [Nv,Mx]
        Example:    [xx,vv] = np.meshgrid(x,v)
                    f= np.exp(-xx**2 - vv**2).

    Returns
    -------
    Make a plot of f

    """
    [xx,vv] = np.meshgrid(x,v)
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, vv, f[1:-1,1:-1], cmap=cm.jet)
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title("Plot of f")
    ax.set_zlim(0, np.max(f)) 
    plt.show()
    return ax

def contour_f(x,v,f,alpha=None,beta=None):
    plt.contour(x, v, f[1:-1,1:-1], norm=colors.LogNorm() , cmap=cm.jet)
    plt.colorbar(label='Values of f')
    plt.xlabel(r'x, $\alpha =$ '+str(alpha))
    plt.ylabel(r'v, $\beta =$ '+str(beta))
    plt.title('Contour plot of f')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def mass(f,dx,dv):
    return dx*dv*sum(sum(f))

def rho(f,dv):
    return dv * np.sum(f,axis=0)

T=40
alpha=2
beta = 2

with open('f_T'+str(T)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
    grid = pickle.load(file)
    
    
grid.build_rho()

fig1=plt.figure(1)
plt.semilogy(grid.x, grid.rho,label ="rho")
Z=sum(np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha))*grid.dx
plt.semilogy(grid.x , np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha)/Z, label = "analytical")
plt.legend()
plt.title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))
#grid.plot_grid()

fig2=plt.figure(2)
DF = grid.build_Vdensity()
#plt.plot(grid.v, DF)
#plt.plot(grid.v, np.exp(-grid.v**2 /2) / np.sqrt(2*np.pi))
plt.semilogy(grid.v, np.exp(-grid.v**2 /2) / np.sqrt(2*np.pi), label = "analytical")
plt.semilogy(grid.v, DF, label = "numeric")
plt.legend()
plt.title(r"Plot of v-density with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))

fig2 =plt.figure(3)
grid.plot_grid()