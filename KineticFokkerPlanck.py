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




class Grid:
    def __init__(self, rows, columns, Xmax, Vmax):
        self.alpha = 0
        self.beta = 0
        
        # grid with ghost points
        self.rows = rows
        self.columns = columns
        self.values = np.zeros((rows+2,columns+2))
        self.rho = None  
        
        # grid size
        self.Xmax = Xmax
        self.Vmax = Vmax

        self.dx = 2 * self.Xmax / self.columns
        self.dv = 2 * self.Vmax / self.rows
        self.dt = self.dx /(4*Vmax) 
        
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
        M1 = 0 # maximum needed to compute CFL condition on dt
        M2 = 0
        for n in range(self.columns):
            for m in range(self.rows-1):
                self.B[m,n] = (1+ V_edges[m]**2)**((self.beta - 2)/2)*V_edges[m] \
                    + (1+ self.x[n]**2)**((self.alpha - 2)/2) *self.x[n]
                  
                w = self.B[m,n] * self.dv
                if w!=0:
                    self.delta[m,n] = 1/w - 1/(np.exp(w)-1)
                    if M1 <= w/(np.exp(w)-1):
                        M1 = w/(np.exp(w)-1)
                    if M2 <= w*np.exp(w)/(np.exp(w)-1):
                        M2 = w*np.exp(w)/(np.exp(w)-1)
                else:
                    self.delta[m,n] = 0.5
                
                if self.dt >= 0.5 * (self.dv)**2 /(M1+M2):  
                    self.dt = 0.5 * (self.dv)**2 /(M1+M2)

        self.dt = self.dt/2
        
        return
        
    def specular_BD(self):
        """
        Update ghost points in x according to
        specular boundary conditions

        """
        
        for i in range(self.rows):
            self.values[i+1 , -1] = self.values[self.rows-i-1 , -2]
            self.values[i+1 , 0] = self.values[self.rows-i-1,1 ]
    
    def H_update(self):
        """
        
        Compute the flux H along the x direction
        
        theta is a parameter for resolution and precision:
            - theta -> 1 : high gradient
            - theta -> 2 : flat regions
        
        """
        for n in range(1,self.columns+1):
            for m in range(1, self.rows+1):
                
                theta = 1.8
                                
                self.GradX[m-1,n-1] = minmod(theta*(self.values[m,n] - self.values[m,n-1])/self.dx ,
                                 (self.values[m,n+1]-self.values[m,n-1])/(2*self.dx) , 
                                 theta*(self.values[m,n+1]-self.values[m,n])/self.dx)
                
                # UPWIND for v>0
#                if self.v[m-1]>0:
 #                   self.GradX[m-1,n-1 ] =(self.values[m,n]-self.values[m,n-1])/(self.dx)
  #              else:
   #                 self.GradX[m-1,n-1 ] =(self.values[m,n+1]-self.values[m,n])/(self.dx)
        
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

        # Specular flux in boundary conditions
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

    def build_rho(self):
        self.rho = sum(self.values[1:-1,1:-1])*self.dv
        return
    
    def build_Vdensity(self):
        DF = np.zeros(self.rows)
        for i in range(self.rows):
            DF[i] = sum(self.values[i+1,1:-1])
        return DF

    def plot_grid(self):
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.xx, self.vv, self.values[1:-1,1:-1] , cmap=cm.jet) #[1:-1][::-1, 1:-1]
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title("Plot of f")

alpha = 1
beta  = 0.5
T = 50
modulo = 10

CONTINUE = False


if CONTINUE:
    T_old = 3
    with open('f_T'+str(round( T_old ))+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl', 'rb') as filef:
        grid = pickle.load(filef)
    
    grid1 = Grid(grid.rows, grid.columns, grid.Xmax, grid.Vmax)
    grid1.alpha = alpha
    grid1.beta = beta
    grid1.B_delta_build()
    
    
    Nt = int(T/grid.dt)

    
else:
    grid  = Grid(100,100,100,100)
    grid1 = Grid(100,100,100,100) # For Runghe-Kutta
    
    grid.alpha = alpha
    grid.beta = beta
    grid1.alpha = alpha
    grid1.beta = beta
    
    print("")
    print("alpha = ", grid.alpha)
    print("beta  = ", grid.beta)
    print("")
    
    # Initialisation:
    grid.values[1:-1,1:-1] = np.exp(-(np.abs(grid.xx)**2)/2 - (grid.vv)**2/2)/(2*np.pi)
    grid.B_delta_build()
    grid1.B_delta_build()
    
    T_old = 0
    Nt = int(T/grid.dt)
    
    print("dt = " , grid.dt)
    print("T = ", T)
    print("Nt = ", Nt)
    print(" ")
    
    print("Initial mass:", grid.mass())
    print(" ")



for k in range(Nt):
    grid.specular_BD()
    grid.H_update()
    grid.F_update()
    # RK: if modify the sign in front of H, modify the Upwind too!
    grid1.values[1:-1,1:-1] = grid.values[1:-1,1:-1] - grid.dt/grid.dx *(grid.H[:,1:] - grid.H[:,:-1]) \
                                                    + grid.dt/grid.dv *(grid.F[1:,:] - grid.F[:-1,:])
    
    grid1.specular_BD()
    grid1.H_update()
    grid1.F_update()
    
    grid.values[1:-1,1:-1] = 0.5 *  grid.values[1:-1,1:-1] + \
                             0.5 * (grid1.values[1:-1,1:-1] - grid1.dt/grid1.dx *(grid1.H[:,1:] - grid1.H[:,:-1])  \
                                                            + grid1.dt/grid1.dv *(grid1.F[1:,:] - grid1.F[:-1,:]) )
    
    

    if k % int(Nt/modulo) == 0:
        print(f"Iteration: {k} / {Nt}")
        
        #with open('f_T'+str(round(T_old + k*grid.dt))+'_alpha'+str(grid.alpha)+'_beta'+str(grid.beta)+'.pkl', 'wb') as filef:
        #    pickle.dump(grid,filef)
        
        grid.build_rho()
        plt.semilogy(grid.x, grid.rho,label =r"$\rho$")
        
        # beta < 2
        #Z=sum(np.exp(-((1+grid.x**2 )**(grid.alpha/2) / grid.alpha )**(grid.beta/2)))*grid.dx
        #plt.semilogy(grid.x , np.exp(-((1+grid.x**2 )**(grid.alpha/2) / grid.alpha )**(grid.beta/2))/Z, label = "analytical")
        #beta > 2
        #Z=sum(np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha))*grid.dx
        #plt.semilogy(grid.x , np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha)/Z, label = "analytical")

        plt.legend()
        plt.title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(round(k*grid.dt , 1)))
        plt.show()
        

print(" ")        
print("Final mass:", grid.mass())




fig1=plt.figure(1)
grid.build_rho()
plt.semilogy(grid.x, grid.rho,label ="rho")
Z=sum(np.exp(-((1+grid.x**2 )**(grid.alpha/2) / grid.alpha )**(grid.beta/2)))*grid.dx
plt.semilogy(grid.x , np.exp(-((1+grid.x**2 )**(grid.alpha/2) / grid.alpha )**(grid.beta/2))/Z, label = "analytical")
#Z=sum(np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha))*grid.dx
#plt.semilogy(grid.x , np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha)/Z, label = "analytical")
plt.legend()
plt.title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))

fig1=plt.figure(2)
DF = grid.build_Vdensity()
plt.semilogy(grid.v, np.exp(-grid.v**2 /2) / np.sqrt(2*np.pi), label = "analytical")
plt.semilogy(grid.v, DF, label = "numeric")
plt.legend()
plt.title(r"Plot of v-density with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))

fig2 = plt.figure(3)
grid.plot_grid()



    

