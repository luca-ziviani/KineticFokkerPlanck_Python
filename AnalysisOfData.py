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
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

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
    
    def H_update(self, theta=1.8):
        """
        
        Compute the flux H along the x direction
        
        """
        for n in range(1,self.columns+1):
            for m in range(1, self.rows+1): 
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

def color_f(grid, log = False):
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
    Make a density plot of f

    """
    [xx,vv] = np.meshgrid(grid.x,grid.v)  
    plt.figure()
    if log:
        pcm = plt.pcolor(xx, vv, grid.values[1:-1,1:-1],  norm=LogNorm(), cmap=cm.jet, shading='auto')
    else:
        pcm = plt.pcolor(xx, vv, grid.values[1:-1,1:-1], cmap=cm.jet, shading='auto')
    plt.colorbar(pcm, label="f(t,x,v)")
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title("t = " + str(T) + r", $\alpha=$" + str(grid.alpha) + r", $\beta = $"+str(grid.beta))
    #plt.clim(0,0.15)
    plt.show()
    
    return

def animate_f(Time):
    """

    Parameters
    ----------
    Time : int
        number of frames (files .pkl to open).

    Returns
    -------
    animation
    
    """
    f_values = []
    
    for i in range(int(Time/period)+1):
        with open(folder + 'f_T'+str(i*period)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
            grid = pickle.load(file)
        f_values.append(grid.values[1:-1,1:-1])
        print("i = ", i)
    
    xx, vv = np.meshgrid(grid.x, grid.v)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xx, vv, f_values[0], norm=LogNorm(), cmap=cm.jet, shading="auto")
    fig.colorbar(pcm, ax=ax, label="f(x,v)")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    title = ax.set_title(r"Plot of f at t = 0, $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta))
    # funzione di aggiornamento dell’animazione
    def update(frame):
        pcm.set_array(f_values[frame].ravel())
        pcm.autoscale()
        title.set_text("Plot of f at t = " + str( frame*period ) +r", $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta))
        return pcm, title
    
    ani = FuncAnimation(fig, update, frames=len(f_values), interval=750, blit=False)
    
    plt.show()

    return ani

# Tools > Preferences > IPython Console > Graphics > Backend: Qt5
def plot_f(grid, log=False):
    """
    Returns
    -------
    If log = False :    Make a plot of f
    If log = True :    Make a log-plot of f

    """
    fig, ax = plt.subplots()
    [xx,vv] = np.meshgrid(grid.x,grid.v)
    ax = plt.axes(projection='3d')
    if log:
        ax.plot_surface(xx, vv, grid.values[1:-1,1:-1], norm=LogNorm(), cmap=cm.jet)
    else:
        ax.plot_surface(xx, vv, grid.values[1:-1,1:-1], cmap=cm.jet)
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title("Plot of f")
    ax.set_zlim(0, np.max(grid.values)) 
    plt.show()
    return ax

def contour_f(grid,log = True):
    """
    Returns
    -------
    If log = False :    Make a contour-plot of f
    If log = True :    Make a log-contour-plot of f

    """
    fig, ax = plt.subplots()
    if log:
        cs = ax.contour(grid.x, grid.v, grid.values[1:-1,1:-1], norm=colors.LogNorm() , cmap=cm.jet)
    else:
        cs = ax.contour(grid.x, grid.v, grid.values[1:-1,1:-1], cmap=cm.jet)
    cbar = fig.colorbar(cs, ax=ax)  # add colorbar
    cbar.set_label("f(t,x,v)")        # label colorbar
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'v')
    ax.set_title(r'Contour plot of f at $t=$'+str(T) + r", $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta))

def mass(f,dx,dv):
    return dx*dv*sum(sum(f))

def rho(f,dv):
    return dv * np.sum(f,axis=0)


alpha = 1.5
beta = 0.5
T = 30
period = 2
folder = "New/"

delta = 4


with open(folder + 'f_T'+str(T)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
    grid = pickle.load(file)
    
    
grid.build_rho()

grid.rho = grid.rho/grid.mass()




#ani = animate_f(T)
#fig=plt.figure()
#color_f(grid,True)

plot_f(grid,True)

"""
fig1=plt.figure(1)
plt.clf()
plt.semilogy(grid.x, grid.rho,label =r"$\rho$ numeric")
#Z=sum(np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha))*grid.dx
#plt.semilogy(grid.x , np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha)/Z, label = "analytical")
Z=sum(np.exp(-delta*((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))*grid.dx
analytical = r"$\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
plt.semilogy(grid.x , np.exp(-delta*((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2))/Z, label = analytical)
plt.legend()
plt.title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T)+ r", $\delta=$"+str(delta))
#grid.plot_grid()
"""
"""
fig2 = plt.figure(2)
plt.semilogy(grid.x, grid.rho,label ="rho")
Z=sum(np.exp(-(1+grid.x**2 )**(grid.beta/4)  ))*grid.dx
analytical = r"$\exp(- |x|^{\beta/2} )$"
plt.semilogy(grid.x , np.exp(-(1+grid.x**2 )**(grid.beta/4)) /Z, label = analytical)
plt.legend()
plt.title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))
"""

#fig2=plt.figure(2)
#DF = grid.build_Vdensity()
#plt.plot(grid.v, DF)
#plt.plot(grid.v, np.exp(-grid.v**2 /2) / np.sqrt(2*np.pi))
#plt.semilogy(grid.v, np.exp(-grid.v**2 /2) / np.sqrt(2*np.pi), label = "analytical")
#plt.semilogy(grid.v, DF, label = "numeric")
#plt.legend()
#plt.title(r"Plot of v-density with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))

#fig2 =plt.figure(3)
#plt.clf()
#grid.plot_grid()
#plt.clf()
#plt.figure(1)
#plt.clf()
#contour_f(grid.x,grid.v,grid.values,grid.alpha,grid.beta)