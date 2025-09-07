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

"""
def minmod(*args):
    signs = [x > 0 for x in args if x != 0]

    if all(signs) or not any(signs):  # tutti positivi o tutti negativi (escludendo zeri)
        return min(args, key=abs) if args else 0
    else:
        return 0
"""




class Grid:
    def __init__(self, rows, columns, Xmax, Vmax):
        self.alpha = 0
        self.beta = 0
        
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

        [self.xx,self.vv] = np.meshgrid(self.x ,self.v)
        
        # Tools
        self.GradX = np.ones((rows, columns))
        self.f_plus = np.zeros((rows, columns+1))
        self.f_minus = np.zeros((rows, columns+1))
        self.H = np.zeros(( rows, columns +1))          # Flux in the x direction
        self.F = np.zeros(( rows+1, columns ))          # Flux in the v direction
        
        self.B = np.zeros(( rows-1, columns ))
        #self.w = self.B * self.dv
        self.delta = np.zeros(( rows-1, columns ))       
        
    def B_delta_build(self):
        V_edges = np.linspace(-self.Vmax +self.dv , self.Vmax - self.dv , self.rows-1)
        for n in range(self.columns):
            for m in range(self.rows-1):
                self.B[m,n] = (1+ V_edges[m]**2)**((self.beta - 2)/2)*V_edges[m] \
                    + (1+ self.x[n]**2)**((self.alpha - 2)/2) *self.x[n]
                  
                w = self.B[m,n] * self.dv
                if w!=0:
                    self.delta[m,n] = 1/w - 1/(np.exp(w)-1)
                else:
                    self.delta[m,n] = 0.5
        
               
        
        return
        
    def specular_BD(self):
        """
        Update ghost points in x according to
        specular boundary conditions

        """
        
        for i in range(self.rows):
            self.values[i+1 , -1] = self.values[self.rows-i-1 , -2]
            self.values[i+1 , 0] = self.values[self.rows-i-1,1 ]
            

    def H_update(self, theta=1.5):    
        """
        
        Compute the flux H along the x direction
        
        """
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

    def F_update(self):
        
        for n in range(self.columns):
            for m in range(self.rows-1):
                self.F[m+1,n] = (1/self.dv + (1-self.delta[m,n])*self.B[m,n]) * self.values[m+2,n+1] \
                    - (1/self.dv - self.B[m,n]*self.delta[m,n])* self.values[m+1,n+1]
        
        # F is already zero on the boundary!

        return             

    def mass(self):
        return sum(sum(self.values[1:-1,1:-1]))*self.dx*self.dv

    def plot_grid(self):
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.xx, self.vv, self.values[1:-1,1:-1] , cmap=cm.jet) #[1:-1][::-1, 1:-1]
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title("Plot of f")


grid = Grid(40,40,10,10)
#grid = Grid(80,80,10,10)

grid.alpha = 2
grid.beta = 2

grid.values[1:-1,1:-1] = np.exp(-(grid.xx)**2/2 - (grid.vv-3)**2/2)/(2*np.pi)
#grid.values[1:-1,-2] = np.ones(grid.rows)
grid.specular_BD()

grid.B_delta_build()

#grid.F_update()

print("Mass:", grid.mass())



dt = 0.004
Nt = 1000

for k in range(Nt):
    grid.H_update()
    grid.F_update()
    # RK: if modify the sign in front of H, modify the Upwind too!
    grid.values[1:-1,1:-1] = grid.values[1:-1,1:-1] - dt/grid.dx *(grid.H[:,1:] - grid.H[:,:-1]) + dt/grid.dv *(grid.F[1:,:] - grid.F[:-1,:])
    
    grid.specular_BD()

print("Mass:", grid.mass())

grid.plot_grid()






