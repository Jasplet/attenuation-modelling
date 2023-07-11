#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:50:54 2022

@author: ja17375
"""

import numpy as np

def chapman_ani(f, lam, mu, bulk_f, visc_f, aspect, cden, fden,
                frac_length, por, tau, grainsize):
    '''
    Calculates a hexagonal, frequency dependent, complex elastic tensor C using 
    Mark Chapman's 2003 squirt flow model. This implementation assumes a fully saturated
    cracked solid.

    Following the cartiesian co-ordinates in Chapman (2003) where x3 (vertical) axis is aligned with fracture normal
    i.e., base model is horizontally oriented fractures. 
    
    Parameters
    ----------
    f : float
        frequency [Hz]
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    bulk_f : float
        bulk modulus of the crack fill [Pa]
    visc_f : float 
        viscosity of the crack fill fluid [Pa.s]
    aspect : float
        aspect ratio of the crack [-]
    cden : float
        micro-scale crack density [-]
    fden : float 
        meso-scale fracture density [-]
    frac_length : float
        meso-scale fracture length [m]
    por : float
        total (measured/modelled) porosity of the medium
    tau : float
        grain-scale relaxation time (inverse of squirt flow frequency) at full saturation

    Returns
    -------
    c : array-like
        the frequency dependent complex elastic tensor for the medium described
        by the input parameters. c is returned as a 6x6 array.
    '''
    omega = 2*np.pi*f
    cpor, ppor, fpor = split_porosity(por, cden, fden, aspect)
    sigma_c, bulk_c = calc_crack_parameters(lam, mu, bulk_f, aspect)
    tau_m, tau_f = calc_taus(bulk_c, visc_f, tau, frac_length, grainsize)
    gamma, gamma_prime = calc_gammas(lam, mu, bulk_f, bulk_c)
    gamma = 10 
    gamma_prime = 1
    d1, d2 = calc_d_terms(gamma, gamma_prime, bulk_c, tau_m, tau_f, cpor, ppor, fpor, aspect, omega)
    f1, f2 = calc_f_terms(d1, d2, gamma, gamma_prime, bulk_c, tau_m, tau_f, cpor, ppor, aspect, omega)
    g1, g2, g3 = calc_g_terms(d1, d2, gamma, gamma_prime, bulk_c, tau_m, omega)
    c = calc_chapman_tensor(lam, mu, aspect, sigma_c, cpor, ppor, fpor, d1, d2, f1, f2, g1, g2, g3) 
    return c

def calc_chapman_tensor(lam, mu, aspect, sigma_c, cpor, ppor, fpor, d1, d2, f1, f2, g1, g2, g3):
    '''
    Uses Chapaman's equations to caclualte the elastic constants and form them into 
    a 6x6 elastic tenor with hexagonal symettry.
    
    Parameters
    ----------
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    aspect : float
        aspect ratio of the crack [-]
    sigma_c : float
        normal stress acting on crack faces [Pa]
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    f1 : float
        Chapman's F1 parameter
    f2 : float
        Chapman's F2 parameter
    g1 : float
        Chapman's G1 parameter
    g2 : float 
        Chapman's G2 parameter
    g3 : float
        Chapman's G3 parameter
        
    Returns
    -------
    c : array-like
        the frequency dependent complex elastic tensor for the medium described
        by the input parameters. c is returned as a 6x6 array. 
    '''
    c11 = calc_c11(lam, mu, aspect, sigma_c, cpor, ppor, fpor,
                   d1, d2, f1, f2, g1, g2, g3)
    c12 = calc_c12(c11, lam, mu, aspect, sigma_c, cpor, ppor, fpor,
                   d1, d2, f1, f2, g1, g2, g3)
    c33 = calc_c33(lam, mu, aspect, sigma_c, cpor, ppor, fpor,
                   d1, d2, f1, f2, g1, g2, g3)
    c13 = calc_c13(c11, c33, lam, mu, aspect, sigma_c, cpor, ppor, fpor,
                   d1, d2, f1, f2, g1, g2, g3)
    c44 = calc_c44(lam, mu, aspect, sigma_c, cpor, ppor, fpor, g1)
    c66 = (1/2)*(c11 - c12)
    C = np.array([[c11, c12, c13,   0,   0,   0],
                  [c12, c11, c13,   0,   0,   0],
                  [c13, c13, c33,   0,   0,   0],
                  [  0,   0,   0, c44,   0,   0],
                  [  0,   0,   0,   0, c44,   0],
                  [  0,   0,   0,   0,   0, c66]])
    return C

# Calculate complex elastic constants 

def calc_c11(lam, mu, aspect, sigma_c, cpor, ppor, fpor, d1, d2, f1, f2, g1, g2, g3):
    '''
    Calculates the elastic constant c11 (or c1111 in ijkl notation) using
    equation 51 of Chapman (2003)
    
    Parameters
    ----------
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    aspect : float
        aspect ratio of the crack [-]
    sigma_c : float
        normal stress acting on crack faces [Pa]
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    f1 : float
        Chapman's F1 parameter
    f2 : float
        Chapman's F2 parameter
    g1 : float
        Chapman's G1 parameter
    g2 : float 
        Chapman's G2 parameter
    g3 : float
        Chapman's G3 parameter

    Returns 
    ------- 
    c11 : float
        complex elastic constant c11 (or c1111) [Pa]
    '''
    l2 = lam**2 + (4/3)*(lam*mu) + (4/5)*mu**2
    v = calc_poisson_ratio(lam, mu)
    kappa = lam +(2/3)*mu
    ccor_a = l2/sigma_c + mu*(32/15)*(1 - v)/((2 -v)*np.pi*aspect)
    ccor_b = (l2/sigma_c + kappa)*g1 
    ccor_c = ((3*kappa**2)/sigma_c + kappa)*g2
    ccor_d = (kappa/sigma_c + 1)*lam*g3
    ccor = ccor_a - ccor_b - ccor_c - ccor_d
    #porosity corrections
    pcor_a = 3*lam**2 + 4*lam*mu + (mu**2)*(36+20*v)/(7-5*v)
    pcor_b = (1 + 0.75*(kappa/mu))*(3*kappa*d1 + lam*d2)
    pcor = (3/(4*mu))*((1 - v)/(1 + v))*pcor_a - pcor_b
    # fracture correction
    fcor = (lam**2/sigma_c) - 3*f1*((lam*kappa)/sigma_c + kappa) - f2*(lam**2 / sigma_c + lam)
    # Combine for c11
    c11 = (lam + 2*mu) - cpor*ccor - ppor*pcor - fpor*fcor
    return c11

def calc_c33(lam, mu, aspect, sigma_c, cpor, ppor, fpor, d1, d2, f1, f2, g1, g2, g3):
    '''
    Calculates the elastic constant c33 (or c33 in ijkl notation) using
    equation 52 of Chapman (2003)
    
    Parameters
    ----------
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    aspect : float
        aspect ratio of the crack [-]
    sigma_c : float
        normal stress acting on crack faces [Pa]
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    f1 : float
        Chapman's F1 parameter
    f2 : float
        Chapman's F2 parameter
    g1 : float
        Chapman's G1 parameter
    g2 : float 
        Chapman's G2 parameter
    g3 : float
        Chapman's G3 parameter

    Returns 
    ------- 
    c33 : float
        complex elastic constant c11 (or c3333) [Pa]
    '''
    l2 = lam**2 + (4/3)*(lam*mu) + (4/5)*mu**2
    v = calc_poisson_ratio(lam, mu)
    kappa = lam + (2/3)*mu    
    #crack correction
    ccor_a = l2/sigma_c + mu*(32/15)*(1 - v)/((2 -v)*np.pi*aspect)
    ccor_b = (l2/sigma_c + kappa)*g1
    ccor_c = g2*3*kappa*(1 + (kappa/sigma_c))
    ccor_d = g3*((lam + 2*mu)*kappa/sigma_c + lam + 2*mu)
    ccor = ccor_a - ccor_b - ccor_c - ccor_d
    # Porosity correction
    pcor_a = 3*lam**2 + 4*lam*mu + (mu**2)*(36+20*v)/(7-5*v)
    pcor_b = (1 + (3*kappa)/(4*mu))*(3*kappa*d1 + (lam + 2*mu)*d2)
    pcor = (3/(4*mu))*((1 - v)/(1 + v))*pcor_a - pcor_b
    # Fracture correction
    fcor_a = 3*f1*kappa*(((lam + 2*mu)/sigma_c) + 1)
    fcor_b = f2*(lam + 2*mu + ((lam + 2*mu)**2)/sigma_c)
    fcor = ((lam + 2*mu)**2)/sigma_c - fcor_a - fcor_b
    # combine for c33
    c33 = (lam + 2*mu) - cpor*ccor - ppor*pcor - fpor*fcor
    return c33

def calc_c44(lam, mu, aspect, sigma_c, cpor, ppor, fpor, g1):
    '''
    Calculates the elastic constant c44 (or c2323 in ijkl notation) using
    equation 54 of Chapman (2003)

    Parameters
    ----------
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    aspect : float
        aspect ratio of the crack [-]
    sigma_c : float
        normal stress acting on crack faces [Pa]
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    g1 : float
        Chapman's g1 parameter

    Returns 
    ------- 
    c44 : float
        complex elastic constant c44 (or c2323) [Pa]
    '''
    v = calc_poisson_ratio(lam, mu)
    # crack correction
    ccor_a = (4/15)*((mu**2)/sigma_c)*(1 - g1)
    ccor_b = (8/5)*mu*(1 - v)/((2 - v)*np.pi*aspect)
    ccor = ccor_a + ccor_b
    # porosity correction
    pcor = 15*mu*(1 - v)/(7 - 5*v)
    # fracture correction
    fcor = 4*(1 - v)*mu/((2 - v)*np.pi*aspect)
    # combine for c44
    c44 = mu - cpor*ccor - ppor*pcor - fpor*fcor

    return c44

def calc_c12(c11, lam, mu, aspect, sigma_c, cpor, ppor, fpor, d1, d2, f1, f2, g1, g2, g3):
    '''
    Calculates the elastic constant c12 (or c1122 in ijkl notation) using
    equation 56 of Chapman (2003)

    Parameters
    ----------
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    aspect : float
        aspect ratio of the crack [-]
    sigma_c : float
        normal stress acting on crack faces [Pa]
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    f1 : float
        Chapman's F1 parameter
    f2 : float
        Chapman's F2 parameter
    g1 : float
        Chapman's G1 parameter
    g2 : float 
        Chapman's G2 parameter
    g3 : float
        Chapman's G3 parameter

    Returns 
    ------- 
    c12 : float
        complex elastic constant c12 (or c1122) [Pa]
    '''
    v = calc_poisson_ratio(lam, mu)
    l3 = 4*(lam**2 + (4/3)*(lam*mu) + (8/15)*mu**2)
    kappa = lam + (2/3)*mu
    # crack correction
    ccor_a = l3/sigma_c + (32/15)*(mu)*((1 - v)/((2 - v)*np.pi*aspect))
    ccor_b = (l3/sigma_c + 4*kappa)*g1
    ccor_c = 12*kappa*(1 + kappa/sigma_c)*g2
    ccor_d = 4*lam*(1 + kappa/sigma_c)*g3
    ccor = ccor_a - ccor_b - ccor_c - ccor_d
    # porosity correction
    pcor_a = 12*lam**2 + 16*lam*mu + (mu**2)*(64/(7 - 5*v))
    pcor_b = (2 + (3*kappa)/(2*mu))*(6*kappa*d1 + 2*lam*d2)
    pcor = (3/(4*mu))*((1 - v)/(1 + v))*pcor_a - pcor_b
    # fracture correction
    fcor_a = (4*lam**2)/sigma_c 
    fcor_b = 12*kappa*(1 + lam/sigma_c)*f1
    fcor_c = 4*lam*(1 + lam/sigma_c)*f2
    fcor = fcor_a - fcor_b - fcor_c
    # Solve a re-arranged version of equation 56
    c12 = 0.5*(4*(lam + mu) - 2*c11 - cpor*ccor - ppor*pcor - fpor*fcor)
    return c12

def calc_c13(c11, c33, lam, mu, aspect, sigma_c, cpor, ppor, fpor, d1, d2, f1, f2, g1, g2, g3):
    '''
    Calculates the elastic constant c13 (or c1133 in ijkl notation) using
    equation 60 of Chapman (2003)

    Parameters
    ----------
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    aspect : float
        aspect ratio of the crack [-]
    sigma_c : float
        normal stress acting on crack faces [Pa]
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    f1 : float
        Chapman's F1 parameter
    f2 : float
        Chapman's F2 parameter
    g1 : float
        Chapman's G1 parameter
    g2 : float 
        Chapman's G2 parameter
    g3 : float
        Chapman's G3 parameter

    Returns 
    ------- 
    c13 : float
        complex elastic constant c13 (or c1133) [Pa]
    '''
    v = calc_poisson_ratio(lam, mu)
    l3 = 4*(lam**2 + (4/3)*(lam*mu) + (8/15)*mu**2)
    kappa = lam + (2/3)*mu
    # crack correction
    ccor_a = l3/sigma_c + (32*mu*(1 - v)/(15*(2 - v)*np.pi*aspect))
    ccor_b = ((l3/sigma_c) + 4*kappa)*g1
    ccor_c = 12*kappa*(1+(kappa/sigma_c))*g2
    ccor_d = 4*(lam + mu)*(1 + kappa/sigma_c)*g3
    ccor = ccor_a - ccor_b - ccor_c - ccor_d
    # porosity correction
    pcor_a = (3/(4*mu))*((1 - v)/(1 + v))*(12*lam**2 + 16*lam*mu + (64/(7 - 5*v))*(mu**2))
    pcor_b = (2 + 1.5*(kappa/mu))*(6*kappa*d1 + 2*(lam+mu)*d2)
    pcor = pcor_a - pcor_b
    # fracture correction
    fcor_a = 4*((lam+mu)**2)/sigma_c
    fcor_b = 12*kappa*(1 + (lam + mu)/sigma_c)*f1
    fcor_c = 4*(((lam+mu)**2)/sigma_c + lam + mu)*f2
    fcor = fcor_a - fcor_b - fcor_c
    # Solve a re-arranged version of equation 60 for c13
    c13 = 2*(lam + mu) - 0.5*(c11 + c33 + cpor*ccor + ppor*pcor + fpor*fcor)
    return c13

# "Under the hood" functions

def calc_taus(bulk_c, visc_f, tau, frac_length, grainsize):
    '''
    Calculates to mineral tau_m and fracture, tau_f, scale relaxation times. 
    
    N.B in a partially saturated case bulk_c should be sig_c/bulk_f and does not
    use the effictive bulkmodulus of the fluid+gas mixture.
    In our fully saturated case tau_m = tau, but I am leaving this in here for if/when
    we want to modify to account for partial saturation.
    
    Parameters
    ----------
    bulk_c : float
        parameter Kc from Chapman (2003)
    visc_f : float
        fluid crack fill viscosity [Pa.s]
    tau : float
        fully saturated mineral (grain-scale) relaxation time [s]
    frac_length : float
        meso-scale fracture length [m]
    grainsize : float
        size of mineral grains [m]

    Returns
    -------
    tau_m : float
        mineral relaxation time [s]
    tau_f : float
        fracture relaxation time [s]
    '''
    p0 = (1 + bulk_c)*visc_f/tau 

    tau_m = ((1 + bulk_c)*visc_f)/p0 
    tau_f = frac_length*tau_m/grainsize
    return tau_m, tau_f

def calc_gammas(lam, mu, bulk_f, bulk_c):
    '''
    Calculates the terms gamma and gamma prime using equations 21 annd 22
    of Chapman (2003)

    Parameters
    ----------
    lam : float 
        1st Lame parameter of the uncracked solid [Pa]
    mu : float
        shear modulus of the uncracked solid [Pa]
    bulk_f : float
        bulk modulus of the crack fill [Pa]
    bulk_c : float
        parameter Kc from Chapman (2003)

    Returns
    -------
    gamma : float
        Chapman (2003)'s gamma parameter
    gamma_prime : float
        Chapman (2003)'s gamma prime parameter
    '''
    v = calc_poisson_ratio(lam, mu)
    kp = 4*mu/(3*bulk_f)
    gamma = (3 * np.pi * (1 + kp))/(8*(1 - v)*(1 - bulk_c))
    gamma_prime = gamma*((1-v)/(1+v))*(1/(1+kp))
    return gamma, gamma_prime

def split_porosity(por, cden, fden, aspect):
    '''
    Calculates the crack and fracture porosity, which contribute to the overall porosity.

    Parameters
    ----------
    por : float
        total porosity of the medium
    cden : float
        micro-crack density
    fden : float
        meso-scale fracture density
    aspect : float
        aspect ratio of cracks

    Returns
    -------
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    '''
    cpor=4/3*np.pi*cden*aspect; # crack porosity
    fpor=4/3*np.pi*fden*aspect; # fracture porosity
    ppor=por-cpor-fpor; # pore porosity

    if (fpor > por):
        raise ValueError(f'Invalid fracture density {fden}, fpor {fpor} > total porosity {por}!!') 
    elif (ppor < 0):
        raise ValueError(f'Invalid ppor {ppor}, check cpor {cpor} and fpor {fpor}!')
        
    return cpor, ppor, fpor 

def calc_poisson_ratio(lam, mu):
    '''
    Calculate poissons ratio
    
    Parameters
    ----------
    lam : float
        1st lamee parameter of the uncracked solid
    mu : float
        shear modulus of the uncracked solid
    
    Returns
    -------
    v : float
        Poisson's ratio of the two elastic modulii
    '''
    v = lam / (2*lam + 2*mu)
    return v

def calc_crack_parameters(lam, mu, bulk_f, aspect):
    '''
    Calculates the terms sigma and K for cracks following the formulas in Chapman (2003) [p.371]

    Parameters
    ----------
    lam : float
        1st lamee parameter of the solid
    mu : float
        shear modulus of the uncracked solid
    bulk_f : float
        bulk modulus of the crack fill material (assumed to be a fluid)
    aspect : float
        aspect ratio of fractures

    Returns
    -------
    sigma_c : float
        normal stress acting on crack faces [Pa]
    k_c : float
        term Kc as described by Champan (2003)
    '''
    v = calc_poisson_ratio(lam, mu)
    sigma_c = (np.pi * mu * aspect)/(2 - 2*v)
    k_c = sigma_c / bulk_f
    return sigma_c, k_c

# Function to calculate D, F, G terms used to evalute elastic constants

def calc_d_terms(gamma, gammap, bulk_c, tau_m, tau_f, cpor, ppor, fpor, aspect, omega):
    '''
    Calculate the terms D1 and D2 following equations 30 and 31 of Chapman (2003)
    
    Parameters
    ----------
    gamma : float
        Chapman (2003)'s gamma parameter
    gamma_prime : float
        Chapman (2003)'s gamma prime parameter
    bulk_c : float
        Chapman (2003)'s Kc parameter
    tau_m : float
        mineral relaxation time [s]
    tau_f : float
        fracture relaxation time [s] 
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    fpor : float
        porosity (or volume fraction) associated with meso-scale fractures
    aspect : float
        aspect ratio of cracks
    omega : float
        angular frequency of sampling wave (omerga = 2*pi*f)
    Returns
    -------
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    '''
    io = (cpor/aspect)/((cpor/aspect) + ppor)
    beta = (fpor/aspect)/((cpor/aspect) + ppor)
    
    d_denom_a = (1-io)*gamma + ((1-io)*beta)/(1+1j*omega*tau_f)
    d_denom_b = io*(1+(beta/(1 + 1j*omega*tau_f)))*((1+1j*omega*gamma*tau_m)/(1+1j*omega*tau_m))
    d_denom = d_denom_a + d_denom_b
    
    d1_num_a = (io/(3*(1+bulk_c))) + (1-io)*gammap
    d1_num_b = ((1j*omega*tau_m)/(1+1j*tau_m))*((1/(3+3*bulk_c)) - gammap)
    d1_num_c = io*(1 + (beta/(1 + 1j*omega*tau_f)))
    d1_num = d1_num_a - (d1_num_b*d1_num_c)
    
    d2_num = beta/((1+bulk_c)*(1+1j*omega*tau_f)) 
    
    d1 = d1_num/d_denom
    d2 = d2_num/d_denom

    return d1, d2

def calc_f_terms(d1, d2, gamma, gammap, bulk_c, tau_m, tau_f, cpor, ppor, aspect, omega):
    '''
    Calculate the terms F1 and F2 following equations 37 and 38 of Chapman (2003)
    
    Parameters
    ----------
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    gamma : float
        Chapman (2003)'s gamma parameter
    gamma_prime : float
        Chapman (2003)'s gamma prime parameter
    bulk_c : float
        Chapman (2003)'s Kc parameter
    tau_m : float
        mineral relaxation time [s]
    tau_f : float
        fracture relaxation time [s] 
    cpor : float
        porosity (or volume fraction) associated with micro-scale cracks
    ppor : float
        porosity (or volume fraction) associated with pores
    aspect : float
        aspect ratio of cracks
    omega : float
        angular frequency of sampling wave (omerga = 2*pi*f)
        
    Returns
    -------
    f1 : float 
        Chapman's F1 parameter
    f2 : float
        Chapman's F2 parameter
    '''
    io = (cpor/aspect)/((cpor/aspect) + ppor)
    
    f1a = 1/(1+1j*omega*tau_f)
    f1b = ((1+1j*omega*gamma*tau_m)/(1 + 1j*omega*tau_m))*io*d1
    f1c = (1-io)*d1
    f1d = (1/(3+3*bulk_c) - gammap)*(1j*io*omega*tau_m)/(1 + 1j*omega*tau_m)
    f1 = f1a*(f1b + f1c + f1d)
    
    f2a = (1j*omega*tau_f)/(1+bulk_c)
    f2b = io*d2*(1 + 1j*omega*gamma*tau_m)/(1 + 1j*omega*tau_m)
    f2c = d2*(1-io)
    f2 = (f2a + f2b + f2c)/(1+1j*omega*tau_f)
    return f1, f2

def calc_g_terms(d1, d2, gamma, gammap, bulk_c, tau_m, omega):
    '''
    Calculates the terms G1, G2, G3 following equations 33, 34, 35 of Chapman (2003)
    
    Parameters
    ----------
    d1 : float 
        Chapman's D1 parameter
    d2 : float
        Chapman's D2 parameter
    gamma : float
        Chapman (2003)'s gamma parameter
    gamma_prime : float
        Chapman (2003)'s gamma prime parameter
    bulk_c : float
        Chapman (2003)'s Kc parameter
    tau_m : float
        mineral relaxation time [s]
    omega : float
        angular frequency of sampling wave (omerga = 2*pi*f)
        
    Returns
    -------
    g1 : float
        Chapman's G1 parameter
    g2 : float 
        Chapman's G2 parameter
    g3 : float
        Chapman's G3 parameter
    '''
    g1 = (1j*omega*tau_m)/((1 + bulk_c)*(1 + 1j*omega*tau_m))
    g2a = (1 + 1j*omega*gamma*tau_m)/(1 + 1j*omega*tau_m)*d1
    g2 = g2a - (1j*omega*tau_m*gammap)/(1 + 1j*omega*tau_m)
    g3 = d2*(1 + 1j*omega*gamma*tau_m)/(1 + 1j*omega*tau_m)
    return g1, g2, g3

if __name__ == '__main__':
# Use parameters from Chapman (2003) to calculate C for 40Hz
    lam = 1.75e10
    mu = 1.75e10
    rho = 2300
    cden = 0.1
    por = 0.1
    aspect = 1e-4
    tau = 2e-5
    bulk_f = 2.2e9
    visc_f = 6e-4
    frac_length = 0.1
    fden = 0.05
    freq = 40/((2*np.pi)**2)
    C = chapman_ani(freq, lam, mu, bulk_f, visc_f, aspect, cden, fden, frac_length, por, tau)
