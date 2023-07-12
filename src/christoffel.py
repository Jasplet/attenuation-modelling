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
    Solves christoffel equation for rays propagating at given inclination and azimuth.

    Parameters
    -----------
    cmplc_c : 6 x 6 array
        complex elastic tensor (in 6x6 Voight notation form)
    rho : float
        density of medium [kg/m3]
    incs : 1-d numpy array
        inclination angles of interest in range 0-90. inc = 0 is horizontal propoagation, inc = 90 is vertical
    azis : 1-d numpy array
        azimuths of interest in range 0-360.
    
    Returns:
    ----------
    velocity : nd-array
        seismic velocities for P, S1, S2 return in shape (3,nincs, nazis)
    attenuation : nd-array
        1/Q values for P, S1, S2 return in shape (3,nincs, nazis)
    fast_polarisations : nd-array
        S1 polarisation vector for each inclination and azimuth
    '''
    nincs = len(incs)
    mazis = len(azis)
    velocity = np.zeros((3, nincs, mazis))
    attenuation = np.zeros((3, nincs, mazis))
    fast_polarisations = np.zeros((1, nincs, mazis))
    for i in range(0,len(incs)):
        for j in range(0,len(azis)):
            velo, attn, fpol = christoffel_solver(cmplx_c, rho, incs[i], azis[j])
            velocity[:,i,j] = velo
            attenuation[:,i,j] = attn
            fast_polarisations[:,i,j] = fpol

    if (mazis == 1) and (nincs == 1):
        velocity = velocity.reshape((3,))
        attenuation = attenuation.reshape((3,))
        fast_polarisations = fast_polarisations.reshape((1,))
    elif (mazis == 1):
        velocity = velocity.reshape((3, nincs))
        attenuation = attenuation.reshape((3, nincs))
        fast_polarisations = fast_polarisations.reshape((1,nincs,))
    elif nincs == 1:
        velocity = velocity.reshape((3, mazis))
        attenuation = attenuation.reshape((3, mazis))
        fast_polarisations = fast_polarisations.reshape((1,mazis))

    return velocity, attenuation, fast_polarisations

def sphe2cart(inc, azi):
    '''
    Converts from spherical to cartesian co-ordinates where:
    North, x = [100]. West, y = [010]. Vertical, z = [001]

    Parameters:
    ----------
    inc : float
        inclination angle
    azi : float
        azimuth angle

    Returns:
    ----------
    X/r : array
        normalised vector of inc/azi in cartesian co-ordinates
    '''
    ir = np.deg2rad(inc)
    ar = np.deg2rad(azi)
    caz = np.cos(ar)
    saz = np.sin(ar)
    cinc = np.cos(ir)
    sinc = np.sin(ir)
    X = np.array([caz*cinc,
                  -1*saz*cinc,
                  sinc
                 ])
    # normalise X vector
    r = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    
    return X/r

def christoffel_solver(C, rho, inc, azi):
    '''
    Solves the christoffel equation for a seismic ray propagating through a medium 
    described by a complex elastic tensor C

    Returns sorted phase velocities and dissipation coefficiants (1/Q)
    
    Parameters:
    ----------
    C : array shape (6, 6)
        complex elastic tensor describing medium
    rho : float
        density of medium [kg/m3]
    inc : float
        incidence angle of seismic ray [deg]
    azi : float
        azimuth angle of seismic ray [deg]
    
    Returns:
    ----------
    velocity : array
        seismic velocities for [P, S1, S2]
    attenuation : array
        attenuation (1/Q) for [P, S1, S2]
    fpol : float 
        angle in plane normal to raypath of fast shear-wave (deg, zero is x3 direction, +ve c'wise looking along raypath at origin).
        for incidence = 90 this corresponds to fast polarisation in geographic reference frame.
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
    q_raw = np.imag(eigvals)/np.real(eigvals) # Equation 8.7 in crampin (1981). this is 1/Q
    idx = np.argsort(velo_raw)[::-1] # [::-1] flips indicies so sort is descending
    eigvec_sort = np.real(eigvecs[:,idx])
    # Sort out polarisations
    P = eigvec_sort[:,0]
    S1 = eigvec_sort[:,1]
    S2 = eigvec_sort[:,2]
    # Project vectors for S1,2 on
    #following MSAT notation
    S1N = np.cross(X, S1)
    S1P = np.cross(X, S1N)
    #S2N = np.cross(X, S2)
    #S2P = np.cross(X, S2N)
    # Rotation into y-z planeto calculate angles
    # Using ported msat functions to make sure i do the right rotations
    S1PR = v_rot_gamma(S1P, azi)
    S1PRR = v_rot_beta(S1PR, inc)
    fpol = np.rad2deg(np.arctan2(S1PRR[1], S1PRR[2]))
    if fpol < -90:
        fpol = fpol + 180
    elif fpol > 90:
        fpol = fpol - 180 
    velocity = velo_raw[idx]
    attenuation = q_raw[idx]
    return velocity, attenuation, fpol

def v_rot_gamma(vec, gamma):
    '''
    Rotates input vector about X3 axis, borrowed (ported) from MSAT to ensure consitency 
    
    Parameters:
    ----------
    vec : array 
        input (3-component) vector to rotate
    gamma : float
        rotation angle about X3 axis

    Returns:
    ----------
    vector rotated about X3 axis
    '''
    g_rad = np.deg2rad(gamma)
    rotmat = np.array([
                        [np.cos(g_rad), np.sin(g_rad), 0],
                        [-1*np.sin(g_rad), np.cos(g_rad), 0],
                        [0, 0, 1]])
    return vec @ rotmat

def v_rot_beta(vec, beta):
    '''
    Rotates about X2 axis, borrowed (ported) from MSAT to ensure consitency 
    
    Parameters:
    ----------
    vec : array 
        input (3-component) vector to rotate
    beta : float
        rotation angle about X2 axis

    Returns:
    ----------
    vector rotated about X2 axis
    '''
    b_rad = np.deg2rad(beta)
    rotmat = np.array([
                        [np.cos(b_rad), 0, -1*np.sin(b_rad)],
                        [0, 1, 0],
                        [np.sin(b_rad), 0, np.cos(b_rad)]])
    return vec @ rotmat

