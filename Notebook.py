# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 11:05:05 2025

----------------------------------------------

    In this file we open and analyse the saved data from a simlation
    We are interested in the profile in the energy of the steady state
    and its tails in x and v

    Here we obtain the images for the paper https://arxiv.org/pdf/2510.12331

---------------------------------------------- 

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

# Ensure the script runs from its own folder so relative paths work correctly
script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

from KineticFokkerPlanck import Grid

# Compute a reduced density rho by integrating grid values over velocity
def rho(grid):
    return sum(grid.values[int(grid.rows/4): int(3*grid.rows/4), 1:-1])*grid.dv
    
# Set the simulation parameters and the data folder to analyse
alpha = 1
beta = 1
T = 30
period = 50
folder = "Example/"

delta = 2 # for the energy profile, we use exp(- delta E^(beta/2) ) as a reference, and we need to specify delta to compare with the numerical results

# Load the saved grid data from the pickle file
with open(folder + 'f_T'+str(T)+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
    grid = pickle.load(file)
    
# Build the spatial density from the loaded grid and normalize by mass
grid.build_rho()
grid.rho = grid.rho/grid.mass()


#-------- profiles to compare to the numerical solution ---------------

# ENRGY PROFILE: EXP( - DELTA E^( BETA/2 )) 

# Define the theoretical energy profile used for comparison
def profile(E):
    y = np.exp(-delta*(E**(grid.beta/2))) # Correct profile
    #y =E**(grid.beta/2-1)* np.exp(-delta*(E**(grid.beta/2)))
    return y

# Build coordinate grids for x and v, and compute the energy function E(x,v)
[xx,vv] = np.meshgrid(grid.x,grid.v)
#Energy = (grid.vv[int(grid.rows/4): int(3*grid.rows /4), :]**2) / 2 +  ((1+grid.xx[int(grid.rows/4): int(3*grid.rows /4), :]**2)**(grid.alpha/2) ) / grid.alpha
Energy = (grid.vv**2) / 2 +  ((1+grid.xx**2)**(grid.alpha/2) ) / grid.alpha
Prof_energy = profile(Energy)
Prof_energy= Prof_energy / (sum(sum(Prof_energy))*grid.dx*grid.dv)


# RHO PROFILE: | x |^{ (ALPHA/4)(1-BETA/2) } * EXP( - DELTA ( |x|^ALPHA / ALPHA )^(BETA/2) )

# THE CORRECT ASYMPTOTIC OF RHO IS THE FOLLOWING
y= (1+grid.x**2 )**((grid.alpha/4)*(1-grid.beta/2) )* np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))

# THE NORMALISATION CONSTANT IS
#Z = 2*np.sqrt(np.pi) /( np.sqrt(grid.beta*delta)* grid.alpha**(0.5-grid.beta/4) )
Z = sum(y)*grid.dx 
y = y/Z

# COMPUTE THE NORMALISATION CONSTANT OF THE ENERGY PROFILE AND USE
# IT TO NORMALISE RHO AS WELL!

#y= y / (sum(sum(Prof_energy))*grid.dx*grid.dv)

#----------- contour plots --------------------------------------------

#levels = []
#for i in range(10):
#    levels.append(Prof_energy[int(grid.rows*i/10),int(grid.columns*i/10)]) #profile(reversed(np.linspace(0,565 , 15)))  # stessi livelli di f

#levels.sort()
levels = 20

plt.figure()
plt.clf()
# Plot contours of the numerical solution f and the theoretical energy profile
plt.contour(grid.x, grid.v, grid.values[1:-1,1:-1], norm=colors.LogNorm(), levels=levels, colors='blue', label = "f")
plt.contour(grid.x, grid.v, Prof_energy, norm=colors.LogNorm(), levels=levels ,colors='red', linestyles='dashed', label = r'$\exp(-\delta E^{\beta/2})$')
plt.xlabel(r'x')
plt.ylabel(r'v')
plt.xlim( (-grid.Xmax,grid.Xmax) )
plt.ylim( (-grid.Vmax,grid.Vmax) )
plt.legend()
plt.title(r'Contour plot of f at $t=$'+str(T) + r", $\alpha=$"+str(alpha)+r', $\beta= $'+str(beta)+r', $\delta = $'+str(delta))

legend_elements = [
    Line2D([0], [0], color="blue", label="f(x,y)"),
    Line2D([0], [0], color="red", linestyle="--", label=r'$\exp(-\delta E^{\beta/2})$')
]

plt.legend(handles=legend_elements, loc="upper right")
plt.show()

#------------ scatter plots -------------------------------------------

# Scatter plot comparing numerical values against same-energy theoretical profile
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

#------------ plot of rho ---------------------------------------------

# Compute reduced density rho and compare with analytical tail asymptotic
rhoG = rho(grid)

fig, ax = plt.subplots()
ax.semilogy(grid.x, grid.rho,label =r"$\rho_G$ numeric")
ax.set_xlabel('x')
ax.set_ylabel(r'$\rho_G(x)$')
ax.set_title(r"Plot of $\rho_G$ with $\alpha=$" + str(grid.alpha) + r", $\beta=$" +str(grid.beta)+r", $T=$"+ str(T)+ r", $\delta=$"+str(delta))
#analytical = r"$\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
#y= np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))
analytical = r"$|x|^{\frac{\alpha}{2}(1-\frac{\beta}{2})}\exp(- \delta(\frac{|x|^\alpha}{\alpha} )^{\beta/2} )$"
#y= (1+grid.x**2 )**((grid.alpha/4)*(1-grid.beta/2) )* np.exp(-delta*(((1+grid.x**2 )**(grid.alpha/2) / grid.alpha)**(grid.beta/2) ))
#Z=sum(y)*grid.dx*1.5
plt.semilogy(grid.x , y, label = analytical, linestyle='dashed', color = 'black')
plt.legend()
plt.show()

#------------ animation ---------------------------------------------

def animate_f(period):
    """

    Parameters
    ----------
    num : int
        number of frames (files .pkl to open).

    Returns
    -------
    animation
    
    """
    f_values = []
    
    for i in range(period):
        with open(folder + 'f_T'+str(round(i*T/period))+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl','rb') as file:
            grid = pickle.load(file)
        f_values.append(grid.values[1:-1,1:-1])
    
    xx, vv = np.meshgrid(grid.x, grid.v)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xx, vv, f_values[0],  cmap=cm.jet, shading="auto")
    fig.colorbar(pcm, ax=ax, label="f(x,v)")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    title = ax.set_title(r"Plot of f at t = 0")
    
    # funzione di aggiornamento dell’animazione
    def update(frame):
        pcm.set_array(f_values[frame].ravel())
        title.set_text("Plot of f at t = " + str( round(frame*T/period, 1) ))
        return pcm, title
    
    ani = FuncAnimation(fig, update, frames=len(f_values), interval=200, blit=False)
    
    plt.show()

    return ani

ani = animate_f(period)