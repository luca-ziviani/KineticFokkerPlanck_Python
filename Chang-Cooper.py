# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 01:15:12 2025

@author: lucaz

    Implicit scheme for Chang-Cooper method
    for Fokker-planck equation in dimension 1

Stabilty: unconditioned


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import splu


class Chang_Cooper:
    def __init__(self, Xmax, Nx, beta):
        self.Xmax = Xmax
        self.Nx = Nx
        self.dx = 2 * Xmax/(Nx-1)
        
        # List of points (centers of cells)
        self.x = np.linspace(-Xmax, Xmax, Nx)
        
        # Centers of mid-points (sides of cells)
        self.x_s = np.linspace(-Xmax+self.dx/2, Xmax-self.dx/2, Nx-1) 
        
        # Drift term on the mid-points
        self.beta = beta
        self.B = (1+self.x_s**2)**((self.beta -2)/2.) * self.x_s
        
        self.u = np.zeros(Nx)
        self.delta = np.zeros(np.shape(self.B))
        self.A = None
        
    def B_build(self):
        """
        Tools for Chang-Cooper algorith
        """
        for i in range(self.Nx-1):
            w = self.dx * self.B[i]
            if w!=0:
                self.delta[i] = 1/w - 1/(np.exp(w)-1)
            else:
                self.delta[i]=0.5
        return
        
    def matrix_build(self, dt):
        """
        Construction of the matrix
        """

        d_upper = - dt/self.dx * ( 1/self.dx + (1-self.delta[1:]) * self.B[1:] )
        d_main = 1 + dt/self.dx * ( 2/self.dx + (1 - self.delta[:-1]) * self.B[:-1] - self.delta[1:] * self.B[1:])
        d_lower = - dt/self.dx * ( 1/self.dx - self.delta[:-1] * self.B[:-1])

        # Dirichlet boundary condition
        d_upper = np.insert(d_upper, 0, 0.) #prepend d_upper with 0
        d_main = np.insert(d_main, 0, 1.) # prepend and append d_main with 1
        d_main = np.append(d_main, 1)
        d_lower = np.append(d_lower, 0) #append d_lower with 0
    
        self.A = diags([d_lower, d_main, d_upper], [-1, 0, 1])
        
        return

beta = 0.4

# My simulation
MyS = Chang_Cooper(10, 100, beta)

# INITIAL CONDITIONS AND EQUILIBRIUM
MyS.u = np.exp(-(MyS.x-2)**2)
#MyS.u = np.zeros(np.shape(MyS.x)); MyS.u[np.where(np.abs(MyS.x+2)<2)] = 1

# Normalisation
MyS.u = MyS.u / (sum(MyS.u)*MyS.dx)
MyS.u = np.insert(MyS.u, 0, 0) # prepend and append d_main with 1
MyS.u = np.append(MyS.u, 0)

# equilibrium
eq = np.exp((-(1+MyS.x**2)**(MyS.beta/2))/MyS.beta)
eq = eq / (sum(eq)*MyS.dx)

# TIME PARAMETERS
dt=0.1
T = 10
Nt = int(T/dt) + 1
#T_points = np.linspace(0, T, Nt)

MyS.B_build()
MyS.matrix_build(dt)
lu = splu(MyS.A)
        
        
###############
#  MAIN LOOP  #
###############

for k in range(Nt):
    MyS.u[1:-1] = lu.solve(MyS.u[1:-1])
    
    # PLOT
    plt.clf()               # clean the figure
    plt.plot(MyS.x, eq)
    plt.plot(MyS.x, MyS.u[1:-1])
    plt.ylim((-0.2, 0.6))    
    plt.title("Plot of the solution at t = "+ str(round(k*dt,2)))
    plt.draw()              
    plt.pause(0.005)

