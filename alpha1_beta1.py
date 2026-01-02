# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 12:39:36 2025

@author: lucaz


    HERE I NEED TO CUT THE ENERGY

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

def rho(grid):
    return sum(grid.values[int(grid.rows/4): int(3*grid.rows/4), 1:-1])*grid.dv
    


alpha = 1
beta = 1
T = 200
period = 10
folder = "NewSmall/"

delta = 2 #1.55

with open(folder + 'f_T'+str(T)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
    grid = pickle.load(file)
    
        
grid.build_rho()
grid.rho = grid.rho/grid.mass()

Vzoom = 2
def profile(E):
    y = np.exp(-delta*(E**(grid.beta/2))) 
    return y

[xx,vv] = np.meshgrid(grid.x,grid.v)

# PROF_ENERGY IS EXP( - DELTA E^( BETA/2 )) NORMALISED
#Energy = (grid.vv**2) / 2 +  ((1+grid.xx**2)**(grid.alpha/2) ) / grid.alpha
Energy = (grid.vv[int((0.5-1/Vzoom)*grid.rows): int((0.5+1/Vzoom)*grid.rows), :]**2) / 2 +  ((1+grid.xx[int((0.5-1/Vzoom)*grid.rows): int((0.5+1/Vzoom)*grid.rows), :]**2)**(grid.alpha/2) ) / grid.alpha
Prof_energy = profile(Energy)
Prof_energy= Prof_energy / (sum(sum(Prof_energy))*grid.dx*grid.dv)

# THE CORRECT ASYMPTOTIC OF RHO IS THE FOLLOWING
y= (1+grid.x**2 )**((grid.alpha/4)*(1-grid.beta/2) )* np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))

# THE NORMALISATION CONSTANT IS
Z = sum(y)*grid.dx 
y = y/Z


levels = []
for i in range(10):
    shp = np.shape(Prof_energy)
    levels.append(Prof_energy[int(shp[0]*i/20), int(shp[1]*i/20 )]) #profile(reversed(np.linspace(0,565 , 15)))  # stessi livelli del tuo f


# CONTOUR PLOT 
#---------------------------------------------------



plt.figure()
plt.clf()
plt.contour(grid.x, grid.v[int((0.5-1/Vzoom)*grid.rows): int((0.5+1/Vzoom)*grid.rows)], grid.values[int((0.5-1/Vzoom)*grid.rows)+1: int((0.5+1/Vzoom)*grid.rows)+1, 1:-1], norm=colors.LogNorm(), levels=levels, colors='blue', label = "f")
plt.contour(grid.x, grid.v[int((0.5-1/Vzoom)*grid.rows): int((0.5+1/Vzoom)*grid.rows)], Prof_energy,norm=colors.LogNorm(), levels=levels ,colors='red', linestyles='dashed', label = r'$\exp(-\delta E^{\beta/2})$')
#plt.contour(grid.x, grid.v, grid.values[1:-1,1:-1], norm=colors.LogNorm(), levels=levels, colors='blue', label = "f")
#plt.contour(grid.x, grid.v, Prof_energy, norm=colors.LogNorm(), levels=levels ,colors='red', linestyles='dashed', label = r'$\exp(-\delta E^{\beta/2})$')
plt.xlabel(r'x')
plt.ylabel(r'v')
plt.xlim( (-grid.Xmax,grid.Xmax) )
plt.ylim( (-2*grid.Vmax/Vzoom,2*grid.Vmax/Vzoom) )
plt.legend()
#plt.title(r'Contour plot of f at $t=$'+str(T) + r", $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta)+r', $\delta = $'+str(delta))

#legend_elements = [
#    Line2D([0], [0], color="blue", label="f(x,y)"),
#    Line2D([0], [0], color="red", linestyle="--", label=r'$\exp(-\delta E^{\beta/2})$')
#]

#plt.legend(handles=legend_elements, loc="upper right")
plt.show()

# ENERGY PROFILE PLOT 
#---------------------------------------------------

plt.figure()
plt.clf()
plt.scatter(Energy.ravel(), grid.values[int((0.5-1/Vzoom)*grid.rows)+1: int((0.5+1/Vzoom)*grid.rows+1), 1:-1].ravel(), s=1, alpha=0.2, color='black', label = 'f')
#plt.scatter(Energy.ravel(), grid.values[1:-1, 1:-1].ravel(), s=1, alpha=0.2, color='black', label = 'f')
plt.scatter(Energy.ravel(), Prof_energy.ravel(), s=1, alpha=0.2, color='blue', label = r'$\exp(-\delta E^{\beta/2})$')
plt.yscale("log")
plt.xlabel("E(x,v)")
plt.ylabel("f(x,v)")
legend = plt.legend(markerscale=1)
for handle in legend.legend_handles:
    handle.set_sizes([10]) 
    handle.set_alpha(1) 
#plt.title("Profile of f with same energy at T="+str(T))


#PLOT OF RHO
#---------------------------------------------------
fig, ax = plt.subplots()
ax.semilogy(grid.x, grid.rho,label =r"$\rho$ numeric")
ax.set_xlabel('x')
ax.set_ylabel(r'$\rho(x)$')
ax.set_title(r"Plot of $\rho_f$ at $T=$"+ str(T))
analytical = r"$|x|^{\frac{\alpha}{2}(1-\frac{\beta}{2})}\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
plt.semilogy(grid.x , y, label = analytical, linestyle='dashed',color='black')
#plt.legend()
plt.show()