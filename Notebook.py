# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 11:05:05 2025

@author: lucaz
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
    


alpha = 0.5
beta = 1
T = 300
period = 2
folder = "NewSmall/"

delta = 2

with open(folder + 'f_T'+str(T)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
    grid = pickle.load(file)
    
        
grid.build_rho()
grid.rho = grid.rho/grid.mass()


def profile(E):
    y = np.exp(-delta*(E**(grid.beta/2))) # TRUE!!!
    #y =E**(grid.beta/2-1)* np.exp(-delta*(E**(grid.beta/2)))
    return y

[xx,vv] = np.meshgrid(grid.x,grid.v)
#Energy = (grid.vv[int(grid.rows/4): int(3*grid.rows /4), :]**2) / 2 +  ((1+grid.xx[int(grid.rows/4): int(3*grid.rows /4), :]**2)**(grid.alpha/2) ) / grid.alpha
Energy = (grid.vv**2) / 2 +  ((1+grid.xx**2)**(grid.alpha/2) ) / grid.alpha
Prof_energy = profile(Energy)

# PROF_ENERGY IS EXP( - DELTA E^( BETA/2 )) 


# THE CORRECT ASYMPTOTIC OF RHO IS THE FOLLOWING
y= (1+grid.x**2 )**((grid.alpha/4)*(1-grid.beta/2) )* np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))

# THE NORMALISATION CONSTANT IS
#Z = 2*np.sqrt(np.pi) /( np.sqrt(grid.beta*delta)* grid.alpha**(0.5-grid.beta/4) )
Z = sum(y)*grid.dx 
y = y/Z

# COMPUTE THE NORMALISATION CONSTANT OF THE ENERGY PROFILE AND USE
# IT TO NORMALISE RHO AS WELL!
Prof_energy= Prof_energy / (sum(sum(Prof_energy))*grid.dx*grid.dv)
#y= y / (sum(sum(Prof_energy))*grid.dx*grid.dv)


levels = []
for i in range(10):
    levels.append(Prof_energy[10*i,20*i]) #profile(reversed(np.linspace(0,565 , 15)))  # stessi livelli del tuo f

#levels.sort()
levels = 20

plt.figure()
plt.clf()
#plt.contour(grid.x, grid.v[int(grid.rows/4): int(3*grid.rows /4)], grid.values[int(grid.rows/4)+1: int(3*grid.rows/4)+1, 1:-1], norm=colors.LogNorm(), levels=levels, colors='blue', label = "f")
#plt.contour(grid.x, grid.v[int(grid.rows/4): int(3*grid.rows /4)], Prof_energy,norm=colors.LogNorm(), levels=levels ,colors='red', linestyles='dashed', label = r'$\exp(-\delta E^{\beta/2})$')
plt.contour(grid.x, grid.v, grid.values[1:-1,1:-1], norm=colors.LogNorm(), levels=levels, colors='blue', label = "f")
plt.contour(grid.x, grid.v, Prof_energy, norm=colors.LogNorm(), levels=levels ,colors='red', linestyles='dashed', label = r'$\exp(-\delta E^{\beta/2})$')
plt.xlabel(r'x')
plt.ylabel(r'v')
plt.xlim( (-400,400) )
plt.ylim( (-400,400) )
plt.legend()
plt.title(r'Contour plot of f at $t=$'+str(T) + r", $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta)+r', $\delta = $'+str(delta))

legend_elements = [
    Line2D([0], [0], color="blue", label="f(x,y)"),
    Line2D([0], [0], color="red", linestyle="--", label=r'$\exp(-\delta E^{\beta/2})$')
]

plt.legend(handles=legend_elements, loc="upper right")
plt.show()


plt.figure()
plt.clf()
#plt.scatter(Energy.ravel(), grid.values[int(grid.rows/4+1): int(3*grid.rows/4+1), 1:-1].ravel(), s=1, alpha=0.2, color='black', label = 'f')
plt.scatter(Energy.ravel(), grid.values[1:-1, 1:-1].ravel(), s=1, alpha=0.2, color='black', label = 'f')
plt.scatter(Energy.ravel(), Prof_energy.ravel(), s=1, alpha=0.2, color='blue', label = r'$\exp(-\delta E^{\beta/2})$')
plt.yscale("log")
plt.xlabel("E(x,v)")
plt.ylabel("f(x,v)")
legend = plt.legend(markerscale=1)
for handle in legend.legend_handles:
    handle.set_sizes([10]) 
    handle.set_alpha(1) 
plt.title("Profile of f with same energy at T="+str(T))


rhoG = rho(grid)

fig, ax = plt.subplots()
ax.semilogy(grid.x, grid.rho,label =r"$\rho$ numeric")
ax.set_xlabel('x')
ax.set_ylabel(r'$\rho(x)$')
ax.set_title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T)+ r", $\delta=$"+str(delta))
#analytical = r"$\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
#y= np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))
analytical = r"$|x|^{\frac{\alpha}{2}(1-\frac{\beta}{2})}\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
#y= (1+grid.x**2 )**((grid.alpha/4)*(1-grid.beta/2) )* np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))
#Z=sum(y)*grid.dx*1.5
plt.semilogy(grid.x , y, label = analytical)
plt.legend()
plt.show()



