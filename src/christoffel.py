#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:04:34 2022

Tools for solving the chrsitfoffel equation for a complex elastic tensor C

@author: ja17375
"""

import numpy as np

def calc_velocity_and_attenuation(cmplx_c, rho, incs, azis):
    '''
    Solves christoffel equation for rays propagating at theta degrees from the crack normal.
    '''
    nincs = len(incs)
    mazis = len(azis)
    velocity = np.zeros((3, nincs, mazis))
    attenuation = np.zeros((3, nincs, mazis))
    for i in range(0,len(incs)):
        for j in range(0,len(azis)):
            velo, attn = christoffel_solver(cmplx_c, rho, incs[i], azis[j])
            velocity[:,i,j] = velo
            attenuation[:,i,j] = attn
    
    if (mazis == 1) and (nincs == 1):
        velocity = velocity.reshape((3,))
        attenuation = attenuation.reshape((3,))
    elif (mazis == 1):
        velocity = velocity.reshape((3, nincs))
        attenuation = attenuation.reshape((3, nincs))
    elif nincs == 1:
        velocity = velocity.reshape((3, mazis))
        attenuation = attenuation.reshape((3, mazis))
        
    return velocity, attenuation

def sphe2cart(inc, azi):
    '''
    Converts from spherical to cartesian co-ordinates where:
    North, x = [100]. West, y = [010]. Vertical, z = [001]
    '''
    ir = np.deg2rad(inc)
    ar = np.deg2rad(azi)
    
    X = np.array([np.cos(ar)*np.cos(ir),
                  -1*np.sin(ar)*np.cos(ir),
                  np.sin(ir)
                 ])
    # normalise X vector
    r = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    
    return X/r

def christoffel_solver(C, rho, inc, azi):
    '''
    Solves the christoffel equation for a complex (or real) elastic tensor C

    Returns sorted phase velocities and dissipation coefficiants (Q)
    '''
    X = sphe2cart(inc, azi)
    # Form 3x3 Christoffel Tensor using Winterstein method (pg 1076, Winterstein, 1999)
    # This is the approach used in MSAT
    gamma = np.array([
                     [X[0],    0,    0,    0, X[2], X[1]],
                     [   0, X[1],    0, X[2],    0, X[0]],
                     [   0,    0, X[2], X[1], X[0],    0]
                     ])
    T = gamma @ C @ gamma.T # @ symbol does matrix multiplication as of Python 3.5
    eigvals, eigvecs = np.linalg.eig(T)
    velo_raw = np.sqrt(np.real(eigvals)/rho) # Check unit control to see if this factor of 10 is needed
    q_raw = np.imag(eigvals)/np.real(eigvals)
    idx = np.argsort(velo_raw)[::-1] # [::-1] flips indicies so sort is descending
    return velo_raw[idx], q_raw[idx]



