#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:55:58 2022



@author: ja17375
"""
import numpy as np

from hudson import hudson_complex_c, approx_q_values, calculate_u_coefficiants
from christoffel import calc_velocity_and_attenuation

class IsoMedium:
    '''
    Isotropic medium 
    
    Stores/populates/generates physical properties. Either velocities or elastic modulii can be provided 
    '''
    def __init__(self, **kwargs):
        '''
        Initialised Class. Sets name of medium as attribute.
        
        Can either provide vp, vs; lam, mu or kappa, mu to initialise.
        '''
        self.rho = kwargs['rho']
        
        if 'vp' in kwargs.keys():
            self.set_from_velocity(kwargs['vp'], kwargs['vs'])
        elif 'kappa' in kwargs.keys():
            self.set_from_kappa_mu(kwargs['kappa'], kwargs['mu'])
        elif 'lam' in kwargs.keys():
            self.set_from_lam_mu(kwargs['lam'], kwargs['mu'])
        else:
            raise ValueError('Must provide kwargs for one of (vp, vs), (lam, mu), (kappa, mu)')
        # Call set_from functions to add physical params 
        
        
    def set_from_velocity(self, vp, vs):
        '''
        Define elastic properties (lambda, mu, kappa) from observed velocties 
        
        Parameters
        ----------
        vp : float
            Compressional (P) wave velocity in m/s2
        vs : float
            Shear wave velocity in m/s2
        '''
        self.vp = vp
        self.vs = vs
        self.mu = self.rho*vs**2
        self.lam = self.rho*vp**2 - 2*self.mu
        self.kappa = self.lam + (2/3)*self.mu
        
    def set_from_lam_mu(self, lam, mu):
        '''
        Define velocities and kappa from a known lambda, mu and density

        Parameters
        ----------
        lam : float
            1st lamee parameter
        mu : float
            shear modulus
        Returns
        -------
        None.
        
        '''
        self.lam = lam
        self.mu = mu
        self.kappa = lam + (2/3)*mu
        self.vp = np.sqrt((lam - 2*mu)/self.rho)
        self.vs = np.sqrt(mu/self.rho)
        
    def set_from_kappa_mu(self, kappa, mu):
        '''
        Define velocities and kappa from a known kappa (bulk modulus), mu (shear modulus) and density

        Parameters
        ----------
        kappa : float
            bulk modulus
        mu : float
            shear modulus
        Returns
        -------
        None.
        
        '''
        self.lam = kappa - (2/3)*mu
        self.mu = mu
        self.kappa = kappa
        self.vp = np.sqrt((self.lam - 2*mu)/self.rho)
        self.vs = np.sqrt(mu/self.rho)
        
class CrackedSolid:
    '''
    Defines parameterisation of a cracked solid
    '''
    def __init__(self, Solid, Fill, model, **kwargs):
        self.Solid = Solid
        self.Fill = Fill
        if model == 'hudson':
            self.set_hudson_params(kwargs['cden'], kwargs['crad'], kwargs['aspect'])
            
            
    def set_hudson_params(self, density, radii, aspect):
        '''
        Sets up parameters for Hudson Modelling, this model only relies on crack density
        
        Parameters
        ----------
        density : float
            crack density [-]
        radii : float 
            crack radii [-]
        aspect : float
            crack aspect ratio [-]
        
        Returns
        -------
        None
        '''
        self.cden = density
        self.crad = radii
        self.aspect = aspect
        self.vol_frac = np.pi*aspect*density
        self.rho_eff = self.vol_frac*self.Fill.rho + (1- self.vol_frac)*self.Solid.rho


    def calc_hudson_tensor(self, freq):
        '''
        Calculates the frequency dependent complex elastic tensor using 
        Crampin (1984)'s implementation of Hudson modelling
        If more than one frequency is given returns a 3-D array of complex tensors.
    
        Parameters
        ----------
        freq : array
            frequencies of interest
    
        Returns
        -------
        cmplx_c : array
            complex elastic tensor
    
        '''
        self.cmplx_c = hudson_complex_c(self.Solid.lam, self.Solid.mu, self.Solid.rho, 
                                   self.Fill.kappa, self.Fill.mu, self.cden, self.crad,
                                   self.aspect, freq)
            
    def hudson_approx_attenuation(self, theta, freq):
        '''
        Approximates attenuation using Hudsons's forumulas as described in 
        Crampin (1984) eqn. 5'
        '''
        u11, u33 = calculate_u_coefficiants(self.Solid.lam, self.Solid.mu,
                                            self.Fill.kappa, self.Fill.mu,
                                            self.aspect)
        return approx_q_values(theta, freq, self.cden, self.crad, self.Solid.vp,
                               self.Solid.vs, u11, u33)
    

    
    # def calculate_tstar(self, theta, freq, path_length, vp, vs):
    #     '''
    #     Calculates differntial attenuation in terms of t* using hudson modelling
    #     '''
    #     azis = np.zeros(1)
    #     velo, attn = self.calc_velocity_and_attenuation(theta, azis)
        
    #     tstar = 
        
    #     return tp_star, tsr_star, tsp_star
    
 
    def calc_velocity_and_attenuation(self, theta, azis):
        '''
        Solves christoffel equation for rays propagating at theta degrees from the crack normal.
        '''
        incs = 90 - theta
        
        velocity, attenuation = calc_velocity_and_attenuation(self.cmplx_c, self.rho_eff, incs, azis)
        return velocity, attenuation

