#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:03:59 2022

@author: ja17375
"""
import numpy as np


def hudson_c0(lam, mu):
    '''
    Form isotropic matrix based on lamda and mu
    '''
    c11 = lam + 2*mu
    c = np.array([
                  [c11, lam, lam,  0,  0,  0],
                  [lam, c11, lam,  0,  0,  0],
                  [lam, lam, c11,  0,  0,  0],
                  [  0,   0,   0, mu,  0,  0],
                  [  0,   0,   0,  0, mu,  0],
                  [  0,   0,   0,  0,  0, mu]
                ])
    return c

def hudson_c1(cden, lam, mu, D):
    '''
    Calculate first order perturbations using Hudson (1981) equations (via Crampin (1984) eqn. 2)
    
    '''
    # Make a variable for lam + 2*mu for convenience

    b = lam + 2*mu
    M = np.array([ 
                  [ b**2, lam*b, lam*b,     0,     0,     0],
                  [lam*b,lam**2,lam**2,     0,     0,     0],
                  [lam*b,lam**2,lam**2,     0,     0,     0],
                  [    0,     0,     0,     0,     0,     0],
                  [    0,     0,     0,     0, mu**2,     0],
                  [    0,     0,     0,     0,     0, mu**2]
                ], dtype=float)
    c1 = -(cden/mu)*np.matmul(M, D)
    
    return c1

def hudson_c2(cden, lam, mu, D):
    '''
    Calculate second order perturbations using Hudson (1982) equations (via Crampin (1984) eqn. 3)
    
    '''
    b = lam + 2*mu
    q = 15*(lam/mu)**2 + 28*(lam/mu) + 28
    X = 2*mu*(3*lam + 8*mu)/b
    M = np.array([ 
                  [ b*q,        lam*q,       lam*q,     0,     0,     0],
                  [lam*q,(lam**2)*q/b,(lam**2)*q/b,     0,     0,     0],
                  [lam*q,(lam**2)*q/b,(lam**2)*q/b,     0,     0,     0],
                  [    0,           0,           0,     0,     0,     0],
                  [    0,           0,           0,     0,     X,     0],
                  [    0,           0,           0,     0,     0,     X]
                ])
    D2 = np.matmul(D, D)
    c2 = (cden**2/15)* np.matmul(M, D2)
    return c2

def hudson_cI(lam, mu, kappap, mup, rho, cr, aspr, cden, freq):
        
    c0 = hudson_c0(lam, mu)
    
    u11, u33 = calculate_u_coefficiants(lam, mu, kappap, mup, aspr)
    D = np.diag(np.array([u11, u11, u11, 0, u33, u33]))
    # Find real parts of complex elastic tensor (these give us velocity anisotropy)
    c1 = hudson_c1(cden, lam, mu, D)
    c2 = hudson_c2(cden, lam, mu, D)
    cR = c0 + c1 + c2
    # Use equation 6 of Crampin to estimate imaginary part of C
    vs = np.sqrt(mu/rho)
    vp = np.sqrt((lam + 2*mu)/rho)
    # Calculate specific values of Q as required by Crapin's method 
    # constants labelled A, B, C, D, E, F as per equation 6.
    qp0, qsr0, _ = approx_q_values(0, freq, cden, cr, vp, vs, u11, u33)
    qp45, _, _ = approx_q_values(45, freq, cden, cr, vp, vs, u11, u33)
    qp90, qsr90, _ = approx_q_values(90, freq, cden, cr, vp, vs, u11, u33)
    # terms A and B are defined by a combination of some of the other imaginary elastic constants
    # Crampin uses notation c_ijkl, I will use voight (C_mn) notation. remeber python indexing starts from 0
    C = 1/qp0
    D = 1/qp90
    E = 1/qsr90
    F = 1/qsr0
    ci_11 = cR[0,0]/C 
    ci_22 = cR[1,1]/D
    ci_44 = cR[3,3]/E # this is c2323 in Crampin (1984)
    ci_66 = cR[5,5]/F # this is c3131 in Crampin (1984)
    A = (0.5*(cR[0,0] + cR[1,1]) + cR[0,1] + 2*cR[5,5])*qp45 - 0.5*(ci_11 + ci_22) - 2*ci_66
    B = ci_22 - 2*ci_44
    # Now make cI
    cI = np.array([ 
                  [ ci_11,     A,     A,     0,     0,     0],
                  [     A, ci_22,     B,     0,     0,     0],
                  [     A,     B, ci_22,     0,     0,     0],
                  [     0,     0,     0, ci_44,     0,     0],
                  [     0,     0,     0,     0, ci_66,     0],
                  [     0,     0,     0,     0,     0, ci_66]
                ])
    c_out = cR + 1j*cI
    return c_out

def calculate_u_coefficiants(lam, mu, kappap, mup, aspr):
    '''
    Creates the diagonal trace matrix D
    '''
    t1 = lam + 2.0*mu ;
    t2 = 3.0*lam + 4.0*mu ;
    t3 = lam + mu ;
    t4 = np.pi*aspr*mu ;
    
    k = ((kappap+mup*4.0/3.0)/t4)*(t1/t3) ;
    m = (mup*4.0/t4)*(t1/t2) ;
    
    u11 = (4.0/3.0)*(t1/t3)/(1.0+k) ;
    u33 = (16.0/3.0)*(t1/t2)/(1.0+m) ;
    
    return u11, u33

def approx_q_values(theta, freq, cden, cr, vp, vs, u11, u33):
    '''
    Uses the expression derived by Hudson (1981) to estimate dissipation coefficiants 1/Qp, 1/Qsr, 1/Qsp. 
    
    Follows formulation of Crampin (9184) [eqn. 5] 
    
    Parameters
    ----------
    theta: 
        angle from the crack normal
    freq:
        frequency
    cden:
        crack density
    cr:
        crack radius
    vp:
        compressional velocity of uncracked solid
    vs: 
        shear-wave velocity of uncracked solid
    u11:
        quantity U11 for cracks normal to the x1-axis
    u33:
        quantity U33 for cracks normal to the x1-axis
    
    Returns
    -------
    qp_inv:
        1/Qp evaluated for an input theta
    '''
    
    vsvp = vs/vp
    x = (3/2 + (vsvp**5))*(u33**2)
    y = (2 + (15/4)/vsvp - 10*vsvp**3 + 8*vsvp**5)*(u11**2)
    omega = 2*np.pi*freq
    thetar = np.deg2rad(theta)
    # Calculate 1/Qp
    qp1 = ((vp*cden)/(15*np.pi*vs)) * ((omega*cr)/vp)**3
    qp2 = (x*np.sin(2*thetar)**2) + y*((vs/vp)**2 - 2*np.sin(thetar)**2)**2
    qp_inv = qp1*qp2
    
    # Calculate 1/Qsr
    qsr_inv = (cden/(15*np.pi))*((omega*cr/vs)**3)*(x*np.cos(thetar)**2)
    
    # Calculate 1/Qsp
    qsp1 = (cden/(15*np.pi))*((omega*cr/vs)**3)
    qsp2 = (x*np.cos(2*thetar)**2 + y*np.sin(2*thetar)**2)
    qsp_inv = qsp1*qsp2
    
    return qp_inv, qsr_inv, qsp_inv