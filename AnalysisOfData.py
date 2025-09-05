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

Lv = 100


with open('KFP_alpha2_beta1.0_Lx100Lv'+str(Lv)+ '.pkl','rb') as file:
    ( f, x, v, T )= pickle.load(file)
    
dx=x[1]-x[0]
dv=v[1]-v[0]
#contour_f(X,V,f,2,1)


plt.semilogy(x,rho(f,dv)[1:-1], label=r"$v_{max} = $"+str(Lv))
#plt.semilogy(x, 0.2*np.exp(-x**0.47))
#plt.semilogy(X, np.exp(-X**2/50))
#plt.title(r'$\rho_G$ with $\alpha=2$ and $\beta = 0.5$')
#plt.show()
plt.legend()
#plt.ylim(10**(-6), 1)


# f0=f[:, 101]
# plt.semilogy(v, f0[1:-1])
# plt.semilogy(v, 0.01*np.exp(-1.1*v**0.5))

