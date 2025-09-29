# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 17:49:01 2025

FINITE DIFFERENCE SCHEME

    This is the branch RoyBorzi

@author: lucaz
"""

import numpy as np
import pickle
import os

script_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(script_dir)

from KineticFokkerPlanck import Grid, minmod


alpha = 1
beta  = 1
T = 10
modulo = 2

folder = '' # Specify the folder of the backups (files .pkl)

#----------------------------------------
# Automate the resume of a simulation:
#    CONTINUE = True 
#       ->  Open the last state ( set manually T_old as the last time )
#           and resume the simulation with the same parameters.
#   CONTINUE = False
#       ->  Start a new simulation with the given parameters

CONTINUE = False

if CONTINUE:
    T_old = 98
    with open(folder + 'f_T'+str(round( T_old ))+'_alpha'+str(alpha)+'_beta'+str(beta)+'.pkl', 'rb') as filef:
        grid = pickle.load(filef)
    
    grid1 = Grid(grid.rows, grid.columns, grid.Xmax, grid.Vmax)
    grid1.alpha = alpha
    grid1.beta = beta
    grid1.B_delta_build()
    
    
    Nt = int(T/grid.dt)

else:
    grid  = Grid(10,10,10,10)
    grid1 = Grid(10,10,10,10) # For Runghe-Kutta 2
    
    grid.alpha = alpha
    grid.beta = beta
    grid1.alpha = alpha
    grid1.beta = beta
    
    # Initialisation:
    grid.values[1:-1,1:-1] = np.exp(-np.abs(grid.xx)/2 - np.abs(grid.vv)/2)/16
    grid.B_delta_build()
    grid1.B_delta_build()
    
    T_old = 0
    Nt = int(T/grid.dt)

# Start the simulation    
#---------------------------------------------------------------------------------

for k in range(Nt):
    grid.specular_BD()
    grid.H_update()
    grid.F_update()
    grid1.values[1:-1,1:-1] = grid.values[1:-1,1:-1] - grid.dt/grid.dx *(grid.H[:,1:] - grid.H[:,:-1]) \
                                                    + grid.dt/grid.dv *(grid.F[1:,:] - grid.F[:-1,:])
    
    grid1.specular_BD()
    grid1.H_update()
    grid1.F_update()
    
    grid.values[1:-1,1:-1] = 0.5 *  grid.values[1:-1,1:-1] + \
                             0.5 * (grid1.values[1:-1,1:-1] - grid1.dt/grid1.dx *(grid1.H[:,1:] - grid1.H[:,:-1]) \
                                                            + grid1.dt/grid1.dv *(grid1.F[1:,:] - grid1.F[:-1,:]) )
    
    # Save date on file
    if k % int(Nt/modulo) == 0:
        
        with open(folder + 'f_T'+str(round(T_old + k*grid.dt))+'_alpha'+str(grid.alpha)+'_beta'+str(grid.beta)+'.pkl', 'wb') as filef:
            pickle.dump(grid,filef)
        


