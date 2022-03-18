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

    Parameters
    ----------

    cden: float
        crack density
    lam : float
        1st lamee parameter (lambda) of the uncracked solid
    mu : float
        shear modulus of the uncracked solid
    D : 2d-array
        diagonal trace matrix containing U11, U33

    Returns
    -------
    c1 : 2d-array
        matrix containing the 1st order purtubations of the elastic tensor
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

    Parameters
    ----------
    cden: float
        crack density
    lam : float
        1st lamee parameter (lambda) of the uncracked solid
    mu : float
        shear modulus of the uncracked solid
    D : 2d-array
        diagonal trace matrix containing U11, U33

    Returns
    -------
    c2 : 2d-array
        matrix containing the 2nd order purtubations of the elastic tensor
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

def hudson_c_real(lam, mu, u11, u33, cden):
    '''
    Creates the elastic tensor for a cracked solid , with cracks normal to the X1 axis.
    We use Crampin (1984)'s formulation of Hudson (1981, 1982)
    '''
    D = np.diag(np.array([u11, u11, u11, 0, u33, u33]))
    c0 = hudson_c0(lam, mu)
    c1 = hudson_c1(cden, lam, mu, D)
    c2 = hudson_c2(cden, lam, mu, D)
    cR = c0 + c1 + c2
    return cR


def hudson_complex_c(lam, mu, rho, kappap, mup, cden, crad, aspect, freq):
    '''
    Calculates the complex (anelastic) components of the elastic tensor for a cracked solid.

    This implementation follows equation 6 of Crampin (1984), using Hudson modelling (scattering).

    Parameters
    ----------
    lam : float
        1st lamee parameter of the uncracked solid
    mu : float
        shear modulus of the uncracked solid
    rho : 
        density of the uncracked solid
    kappap : float
        bulk modulus of the crack fill material 
    mup : float
        shear modulus of the crack fill material 
    cden : float
        crack density 
    aspect :
        aspect ratio of cracks
    freq : float
        frequency of sampling waves

    Returns
    -------

    c_cmplx : complex array
        the complex elastic tensor for a cracked solid expected from Hudson modelling
    '''
    vp = np.sqrt((lam + 2*mu)/rho)
    vs = np.sqrt(mu/rho)
    u11, u33 = calculate_u_coefficiants(lam, mu, kappap, mup, aspect)
    # Find real parts of complex elastic tensor (these give us velocity anisotropy)
    c_real = hudson_c_real(lam, mu, u11, u33, cden)
    # Use equation 6 of Crampin to estimate imaginary part of C
    # Calculate specific values of Q as required by Crapin's method
    qp0, qsr0, _ = approx_q_values(0, freq, cden, crad, vp, vs, u11, u33)
    qp45, _, _ = approx_q_values(45, freq, cden, crad, vp, vs, u11, u33)
    qp90, qsr90, _ = approx_q_values(90, freq, cden, crad, vp, vs, u11, u33)
    # terms A and B are defined by a combination of some of the other imaginary elastic constants
    # Crampin uses notation c_ijkl, I will use voight (C_mn) notation
    # Indexing starts from 0 so C_11 = c[0,0]
    ci_11 = c_real[0,0]*qp0
    ci_22 = c_real[1,1]*qp90
    ci_44 = c_real[3,3]*qsr90 # this is c2323 in Crampin (1984)
    ci_66 = c_real[5,5]*qsr0 # this is c3131 in Crampin (1984)
    A = (0.5*(c_real[0,0]+c_real[1,1])+c_real[0,1]+2*c_real[5,5])*qp45-0.5*(ci_11+ci_22)-2*ci_66
    B = ci_22 - 2*ci_44
    # Now make cI
    c_imag = np.array([
                  [ ci_11,     A,     A,     0,     0,     0],
                  [     A, ci_22,     B,     0,     0,     0],
                  [     A,     B, ci_22,     0,     0,     0],
                  [     0,     0,     0, ci_44,     0,     0],
                  [     0,     0,     0,     0, ci_66,     0],
                  [     0,     0,     0,     0,     0, ci_66]
                ])
    c_cmplx = c_real + 1j*c_imag
    return c_cmplx

def calculate_u_coefficiants(lam, mu, kappap, mup, aspr):
    '''
    Creates the diagonal trace matrix D

    Parameters
    ----------
    lam : float
        Ist Lamee parameters of the Uncracked Solid
    mu : float
        Shear modulus of the Uncracked Solid
    kappap : float
        Bulk modulus of Crack Fill
    mup : float
        Shear modulus of Crack Fill
    aspr : float    
    '''
    t1 = lam + 2.0*mu
    t2 = 3.0*lam + 4.0*mu
    t3 = lam + mu
    t4 = np.pi*aspr*mu
    k = ((kappap+mup*4.0/3.0)/t4)*(t1/t3)
    m = (mup*4.0/t4)*(t1/t2)
    u11 = (4.0/3.0)*(t1/t3)/(1.0+k)
    u33 = (16.0/3.0)*(t1/t2)/(1.0+m)

    return u11, u33

def approx_q_values(theta, freq, cden, cr, vp, vs, u11, u33):
    '''
    Uses the expression derived by Hudson (1981) to estimate dissipation coefficiants
    1/Qp, 1/Qsr, 1/Qsp.

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
