#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:55:58 2022



@author: ja17375
"""
import numpy as np

from hudson import hudson_complex_c, approx_q_values, calculate_u_coefficiants
from chapman import chapman_ani
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
        self.rho = kwargs['rho'] # Density must be in kg/m3
        
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

class AnisotropicMedium:
    '''
    Anistotropic solid medium 

    To-Dos

    Re-organise Classes to allow for a generic anisotropic medium (which can or cannot have specified imaginary components)
    Want to be able to handle both LPO and SPO anisotopies
    Add a rough elasticDB style function to pre-populate if requetsed
    Calculate dt* for assumaed constant Q
    Absorb calculation of dt* into Class
    Make a ElasticTensor Class that handles the anisotorpy/rotation/decomposition aspects
    Seperate Medium/Material class that holds other properties (denisty, thickness etc.)
    edit
    '''


class CrackedSolid:
    '''
    Defines parameterisation of a cracked solid to be used for effective medium modelling of 
    aligned fluid-filled fracture sets.
    '''
    def __init__(self, Solid, Fill, model, **kwargs):
        self.Solid = Solid
        self.Fill = Fill
        self.aspect = kwargs['aspect']
        if model == 'hudson':
            self.set_hudson_params(kwargs['cden'], kwargs['crad'], kwargs['aspect'])
        elif model == 'chapman':
            self.set_chapman_param(kwargs)
        else:
            raise ValueError(f'Unknown model <{model}>')
      
    def set_hudson_params(self, density, radii, aspect):
        '''
        Sets up parameters for effective medium modelling using the theory of Hudson (1982)
        
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

    def set_chapman_param(self, params):
        
        self.cden = params['cden']
        self.fden = params['fden']
        self.flen = params['flen']
        self.por = params['por']
        self.tau_m = params['tau_m']
        self.rho_eff = self.por*self.Fill.rho + (1- self.por)*self.Solid.rho
        self.grainsize = params['grainsize']

        if 'visc_f' not in params:
            self.visc_f = 1
        else :
            self.visc_f = params['visc_f']
    
    def calc_chapman_tensor(self, freq):
        '''
        Calculate a frequency dependent elastic tensor using Chapman (2003)'s squirt flow model

        Parameters
        ----------
        freq : float
            frequency of interest
        Returns
        -------
        cmplx_c : array
            squirt flow elastic tensor
        '''
        self.cmplx_c = chapman_ani(freq, self.Solid.lam, self.Solid.mu, self.Fill.kappa,
                                   self.visc_f, self.aspect, self.cden, self.fden, self.flen,
                                   self.por, self.tau_m, self.grainsize)


    def hudson_approx_attenuation(self, theta, freq):
        '''
        Approximates attenuation using Hudsons's forumulas as described in 
        Crampin (1984) eqn. 5'

        Parameters:
        ----------
        theta : 
            propagtion angle relative to fracture normal (0 = fracture perpendicular)
        freq : float
            frequency of interest
        Returns:
        ----------
        qp_inv:
            1/Qp evaluated for an input theta
        qsr_inv:
            1/Qsr (radial shear-wave)
        qsp_inv: 
            1/Qsp (ray perpendicular shear-wave)
        '''
        u11, u33 = calculate_u_coefficiants(self.Solid.lam, self.Solid.mu,
                                            self.Fill.kappa, self.Fill.mu,
                                            self.aspect)
        qp_inv, qsr_inv, qsp_inv = approx_q_values(theta, freq, self.cden, self.crad, self.Solid.vp,
                                                    self.Solid.vs, u11, u33)
        return qp_inv, qsr_inv, qsp_inv
 
    def calc_velocity_and_attenuation(self, incs, azis):
        '''
        Solves christoffel equation for rays propagating at a given inclination and azimuth.

        Makes function from christoffel.py inbuilt

        Parameters
        -----------
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
        fast_pol : nd-array
            S1 polarisation vector for each inclination and azimuth
        '''
        
        velocity, attenuation, fast_pol = calc_velocity_and_attenuation(self.cmplx_c, self.rho_eff, incs, azis)
        return velocity, attenuation, fast_pol

    def rotate_tensor(self, alpha, beta, gamma, inplace=False):
        '''
        Takes elastic tensor and rotates it about X1, X2, X3 axes. 

        Rotations are applied in order X1, X2, X3

        Parameters:
        ----------
        alpha : float
            rotation angle about X1 axis
        beta : float 
            rotation angle about X2 axis
        gamma : float
            rotation angle about X3 
        inplace : bool
            switch for if rotate elastic tensor should be returned in place
        
        Returns:
        ----------
        cmplx_c_rot :
            rotated complex elastic tensor (taken from self.cmplx_c)
        '''
        rad_alpha = np.deg2rad(alpha)
        rad_beta = np.deg2rad(beta)
        rad_gamma = np.deg2rad(gamma)
        cmplx_c_rot = _rotate_tensor(self.cmplx_c, rad_alpha, rad_beta, rad_gamma)
        if inplace==True:
            self.cmplx_c = cmplx_c_rot
        
        return cmplx_c_rot

def _rotate_tensor(c, alpha ,beta, gamma):
    '''
    Adapted from MS_rot3

    Angles are given in degrees and correspond to yaw, -dip and aximuth,
    respectvly. The rotations are applied in order, ie: alpha, then beta
    then gamma (by default).
    
    Unlike MS_rot3 this function only handles a single rotation of a single matrix.
    Rotations are always applied in the order alpha, beta, gamma

     Parameters:
    ----------
    c : array, shape (6,6)
        elastic tensor to rotate
    alpha : float
        rotation angle about X1 axis
    beta : float 
        rotation angle about X2 axis
    gamma : float
        rotation angle about X3 
    inplace : bool
        switch for if rotate elastic tensor should be returned in place
    
    Returns:
    ----------
    cr :
        rotated elastic tensor
    '''

    rot_order = [0, 1, 2]
    rot_mats = np.zeros((3,3,3)) 
    # rot_mats is 3D array holding rotation matrices for alpha, beta, gamaa
    # this allows us to support different rotation orders in theory
    rot_mats[0,:,:] = np.array([[1, 0, 0 ], [0, np.cos(alpha), np.sin(alpha)], [0, -1*np.sin(alpha), np.cos(alpha)]])
    rot_mats[1,:,:] = np.array([[np.cos(beta), 0, -1*np.sin(beta)], [0, 1, 0 ], [np.sin(beta), 0, np.cos(beta)]])
    rot_mats[2,:,:] = np.array([[np.cos(gamma), np.sin(gamma), 0 ], [-1*np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    rot_mat = rot_mats[rot_order[2],:,:] @ rot_mats[rot_order[1],:,:] @ rot_mats[rot_order[0],:,:]

    cr = _rotR(c, rot_mat)
    return cr 

def _rotR(C, R):
    '''
    Rotate a elastic tensor in Voight notated (6x6) form by a (3x3) rotation matrix

    Ported from MS_rotR, which is intended to be an under-the-hood function

    Routines are from 'Applied Mechanics of Solids' by Allen F. Bower, Chapter 3, 2010

    Parameters:
    ----------
    C : array 
        elastic tensor to rotate
    R : array
        roation matrix
    Returns:
    ----------
    CR : array 
        rotated elastic tensor
    '''
    # Form the K matrix (based on Bower (2010) [http://solidmechanics.org/Text/Chapter3_2/Chapter3_2.php]

    K = np.zeros((6,6))

    k1 = R**2 # k1 is each element of rotation matrix squared

    k2 = np.array([
                [R[0,1]*R[0,2], R[0,2]*R[0,0], R[0,0]*R[0,1]],
                [R[1,1]*R[1,2], R[1,2]*R[1,0], R[1,0]*R[1,1]],
                [R[2,1]*R[2,2], R[2,2]*R[2,0], R[2,0]*R[2,1]],
                  ])

    k3 = np.array([
                [R[1,0]*R[2,0], R[1,1]*R[2,1], R[1,2]*R[2,2]],
                [R[2,0]*R[0,0], R[2,1]*R[0,1], R[2,2]*R[0,2]],
                [R[0,0]*R[1,0], R[0,1]*R[1,1], R[0,2]*R[1,2]]
                ])

    k4 = np.array([
                [R[1,1]*R[2,2] + R[1,2]*R[2,1], R[1,2]*R[2,0] + R[1,0]*R[2,2], R[1,0]*R[2,1] + R[1,1]*R[2,0]],
                [R[2,1]*R[0,2] + R[2,2]*R[0,1], R[2,2]*R[0,0] + R[2,0]*R[0,2], R[2,0]*R[0,1] + R[2,1]*R[0,0]],
                [R[0,1]*R[1,2] + R[0,2]*R[1,1], R[0,2]*R[1,0] + R[0,0]*R[1,2], R[0,0]*R[1,1] + R[0,1]*R[1,0]]
                ])

    K[0:3, 0:3] = k1
    K[0:3, 3:] = 2*k2
    K[3:, 0:3] = k3
    K[3:, 3:] = k4
    # print(R)

    CRa = K @ C 
    CR = CRa @ np.transpose(K)

    return CR 