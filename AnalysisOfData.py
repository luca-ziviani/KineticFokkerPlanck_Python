# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:15:47 2025

@author: lucaz

ANALYSIS OF DATA .PKL OF THE SIMULATIONS

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # for colormaps
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import pickle
import os


script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

from KineticFokkerPlanck import Grid

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

#---------------------------------------------------------------------

def plot_rho(grid, log=True):
    """
    Returns
    -------
    If log = False :    Make a plot of rho
    If log = True :    Make a semilogy-plot of rho

    """
    fig, ax = plt.subplots()
    if log:
        ax.semilogy(grid.x, grid.rho,label =r"$\rho$ numeric")
    else:
        ax.semilogy(grid.x, grid.rho,label =r"$\rho$ numeric")
    ax.set_xlabel('x')
    #plt.xlabel('x')
    ax.set_ylabel(r'$\rho(x)$')
    ax.set_title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T)+ r", $\delta=$"+str(delta))
    #ax.set_ylim(0, np.max(grid.values)) 
    
    if grid.beta < 2:
        analytical = r"$\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
        y= np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))
        #y=(1+grid.x**2 )**(grid.alpha/4 * (1-grid.beta/2)) * np.exp(-delta*((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) )
        Z=sum(y)*grid.dx
        plt.semilogy(grid.x , y/Z, label = analytical)
    else:
        Z=sum(np.exp(- (1+grid.x**2 )**(grid.alpha/2) / grid.alpha ))*grid.dx
        analytical = r"$\exp(- \frac{|x|^\alpha}{\alpha})$"
        plt.semilogy(grid.x , np.exp(- (1+grid.x**2 )**(grid.alpha/2) / grid.alpha)/Z, label = analytical)
    
    plt.legend()
    plt.show()

    return ax

def plot_exponent(grid):
    """
    Returns
    -------
    Plot of ln( - ln(rho) ) / ln(x)

    """
    fig, ax = plt.subplots()
    ax.plot(grid.x, np.log(-np.log(grid.rho))/np.log(np.abs(grid.x)) ,label =r"numeric $\frac{\ln(-\ln(\rho_f))}{\ln(x)}$")
    if grid.beta < 2:
        ax.set_ylim((0,1))
        ax.set_yticks([0.1*i for i in range(11)])
        ax.plot(grid.x, grid.alpha*grid.beta/2 * np.ones(len(grid.x)), label= r"$\frac{\alpha\beta}{2} \approx $"+ str(round(grid.alpha*grid.beta / 2 , 2)))
    else:
        ax.set_ylim((0,2))
        ax.set_yticks([0.2*i for i in range(11)])
        ax.plot(grid.x, grid.alpha * np.ones(len(grid.x)), label= r"$\alpha $")
    plt.legend()
    ax.set_title(r"Plot of $\rho_f$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))
    
    return ax

#---------------------------------------------------------------------
        
def animate_f(Time):
    """

    Parameters
    ----------
    Time : int
        final time of the animation
    Returns
    -------
    Animation from time 0 to Time
    
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
    
    def update(frame):
        pcm.set_array(f_values[frame].ravel())
        pcm.autoscale()
        title.set_text("Plot of f at t = " + str( frame*period ) +r", $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta))
        return pcm, title
    
    ani = FuncAnimation(fig, update, frames=len(f_values), interval=750, blit=False)
    
    plt.show()

    return ani

def animate_rho(T):
    """

    Parameters
    ----------
    num : int
        number of frames (files .pkl to open).

    Returns
    -------
    animation
    
    """
    rho_values = []
    num = int(T/period)+1
    for i in range(num):
        with open(folder + 'f_T'+str(i*period)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
            grid = pickle.load(file)
        #with open(folder + 'f_T'+str(i*period)+'_alpha'+str(alpha)+'_gamma'+str(beta)+'.pkl','rb') as file:
        #    grid = pickle.load(file)
        grid.build_rho()

        rho_values.append(grid.rho)
    
    fig, ax = plt.subplots()
    y = rho_values[0]
    line, = ax.semilogy(grid.x,y, label = r'Numeric $\rho_f$')
    ax.set_xlabel("x")
    if beta<2:
        y = np.exp(-delta*((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) )
        Z=sum(y)*grid.dx
        analytical = r"$\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
        plt.semilogy(grid.x , y/Z, label = analytical)
        ax.set_ylim(np.min(y[0]*10**(-5)) , 0.2)
    else:
        y = np.exp(-(1+grid.x**2 )**(grid.alpha/2) / grid.alpha )
        Z=sum(y)*grid.dx
        analytical = r"$\exp(- (\frac{|x|^\alpha}{\alpha} ) )$"
        plt.semilogy(grid.x , y/Z, label = analytical)
        ax.set_ylim(np.min(y[0]) , 0.2)
        
    plt.legend()

    title = ax.set_title(r"Plot of $\rho_f$ at t = 0$"+ r", $\alpha = $" + str(alpha)+ r", $\delta = $" + str(beta)+ r", $\beta = $" + str(delta))
    
    def update(frame):
        y = rho_values[frame]
        line.set_data(grid.x,y)
        #if beta<2:
            #ax.autoscale()
        title.set_text(r"Plot of $\rho_f$ at t = " + str( round(frame*period, 1) )+ r", $\alpha = $" + str(alpha)+ r", $\beta = $" + str(beta) + r", $\delta = $" + str(delta))
        return line, title
    
    ani = FuncAnimation(fig, update, frames=len(rho_values), interval=200, blit=False)
    
    plt.show()

    return ani


alpha = 2
beta = 0.25
T = 115
period = 5
folder = "New/"

delta = 1 #0.5

with open(folder + 'f_T'+str(T)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
    grid = pickle.load(file)
        
grid.build_rho()
grid.rho = grid.rho/grid.mass()


# Uncomment the plots you want to see 
#--------------------------------------------------

# ANIMATIONS
#ani = animate_f(T)
#ani = animate_rho(T)
#ani.save("animation_name.mp4", writer="ffmpeg", fps=5)

# SINGLE PLOT OF F
#color_f(grid,True)
#contour_f(grid,True)

# (ANSATZ) EXPECTED STEADY STATE
#SS = Grid(grid.rows, grid.columns, grid.Xmax, grid.Vmax)
#SS.values = SS.values/SS.mass()
#SS.alpha = alpha
#SS.beta = beta
#SS.values[1:-1,1:-1] = np.exp(- delta * ( (np.abs(SS.vv)**2)/2 + (np.abs(SS.xx)**alpha)/alpha )**(beta/2) )
#contour_f(SS,True)

# PLOT OF RHO 
#plot_rho(grid)

# EXPONENT OF SUB-DECAY OF RHO
#plot_exponent(grid)

# COMPARISON CONTOUR F AND SS
"""
def profile(E):
    y = np.exp(-delta*(E**(grid.beta/2)))
    return y
    #return np.exp(-0.3*(s**0.5))
[xx,vv] = np.meshgrid(grid.x,grid.v)
Energy = (grid.vv**2) / 2 +  ((1+grid.xx**2)**(grid.alpha/2) ) / grid.alpha
Prof_energy = profile(Energy)

Prof_energy= Prof_energy/(sum(sum(Prof_energy))*grid.dx*grid.dv)

levels = []
for i in range(20):
    levels.append(profile(Energy[0 , 10*i ])) #profile(reversed(np.linspace(0,565 , 15)))  # stessi livelli del tuo f

plt.figure()
plt.clf()
plt.contour(grid.x, grid.v, grid.values[1:-1,1:-1], norm=colors.LogNorm(), levels=20, colors='blue', label = "f")
plt.contour(grid.x, grid.v, Prof_energy,norm=colors.LogNorm(), levels=20 ,colors='red', linestyles='dashed', label = r'$\exp(- E^{\beta/2})$')
plt.xlabel(r'x')
plt.ylabel(r'v')
plt.legend()
plt.title(r'Contour plot of f at $t=$'+str(T) + r", $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta))

legend_elements = [
    Line2D([0], [0], color="blue", label="f(x,y)"),
    Line2D([0], [0], color="red", linestyle="--", label=r'$\exp(- E^{\beta/2})$')
]

plt.legend(handles=legend_elements, loc="upper right")
plt.show()


plt.figure()
plt.clf()
plt.scatter(Energy.ravel(), grid.values[1:-1,1:-1].ravel(), s=1, alpha=0.2, color='black', label = 'f')
plt.scatter(Energy.ravel(), Prof_energy.ravel(), s=1, alpha=0.2, color='blue', label = r'$\Gamma(E)$')
plt.yscale("log")
plt.xlabel("E(x,v)")
plt.ylabel("f(x,v)")
plt.legend()
plt.title("Profile of f with same energy at T="+str(T))

"""

# PLOT OF V-DENSITY
#fig2=plt.figure(2)
#DF = grid.build_Vdensity()
#plt.plot(grid.v, DF)
#plt.plot(grid.v, np.exp(-grid.v**2 /2) / np.sqrt(2*np.pi))
#plt.semilogy(grid.v, np.exp(-np.abs(grid.v)**grid.beta ) / np.sqrt(2*np.pi), label = "analytical")
#plt.semilogy(grid.v, DF, label = "numeric")
#plt.legend()
#plt.title(r"Plot of v-density with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T))

