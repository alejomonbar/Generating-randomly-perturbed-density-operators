#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:41:23 2022
Randomly perturbed method: Method for Generating Randomly Perturbed Density Operators
Subject to Different Sets of Constraints 

Paper:<https://arxiv.org/abs/2112.12247>
@author: alejomonbar
"""


import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
from scipy.constants import hbar
import matplotlib.pyplot as plt

I = np.eye(4, dtype = complex)

si = np.eye(2, dtype = complex)
sx = np.array([[0,1],[1,0]], dtype = complex)
sy = np.array([[0,-1j],[1j,0]], dtype = complex)
sz = np.array([[1,0],[0,-1]], dtype = complex)
sn = [si, sx, sy, sz]

def anticommutator(A, B):
    return A @ B + B @ A

def gammar_fun(lambdas, G, gammae):
    """
    Returns the density operator once the constraints are applied.
    Parameters
    ----------
    lambdas : List
        List of multipliers, every multiplier is associated with a constraint.
    G : List
        List of symmetrized gradients.
    gammae:np.array 
        squared root of the perturbed density state.

    Returns
    -------
    gammar : np.array
        Modified perturbed density state based on the constraints applied.

    """
    N = len(G)
    gammar = np.zeros(gammae.shape, dtype=complex)
    gammar += gammae
    for i in range(N):
        gammar -= lambdas[i] * anticommutator(G[i], gammae)
    return gammar

def constrains(lambdas, G, po, gammae, C, C_exp=[]):
    """
    Cost function used to determine the lambda multipliers that restrict the problem.
    Parameters
    ----------
    lambdas : List
        indetermined multipliers.
    G : list of np.arrays
        Symmetrized gradients of the constraints.
    po : np.array
        unperturbed state.
    gammae : np.array
        perturbed state.
    C : List of lambda functions
        restrictions of the new state e.g. Entropy = lambda p: Tr(-p @ ln(p)).

    Returns
    -------
    cost : List
        List of constraints to be satisfy.

    """
    N = len(G)
    cost = N*[0]
    gammar = gammar_fun(lambdas, G, gammae)
    for i in range(N):
        pr = gammar @ gammar.T.conjugate()
        if len(C_exp) > 0:
            cost[i] = (C[i](pr) - C_exp[i]).real ** 2
        else:
            cost[i] = (C[i](pr) - C[i](po)).real ** 2
    return cost

  
def gammae_fun(gamma0, etas):
    gammaes = []
    for eta in etas:
        gammae = np.zeros(gamma0.shape, dtype = complex)
        gammae += gamma0
        for i in range(len(sn)):
            for j in range(len(sn)):
                gammae += 0.5*eta[i,j] * np.kron(sn[i],sn[j])
        gammaes.append(gammae)
    return np.array(gammaes)

def bell_diagonal_state(c):
    """
    Function to create a Bell diagonal state
    Parameters
    ----------
    c : list
        List of c parameters of Eq.3 in the paper

    Returns
    -------
    po : np.array
        Bell diagonal state.

    """
    po = np.kron(si, si)
    for i in range(1, len(sn)):
        po += c[i-1] * np.kron(sn[i], sn[i])
    po *= 0.25
    return po

def etas_norm(mean, std, N):
    """
    In any case that we have a distribution of the perturbation parameters eta,
    it can be extracted from the experiments.
    
    Parameters
    ----------
    mean : List
        mean value of the different perturbation parameters eta.
    std : List
        standard deviation of the different perturbation paramters eta.
    N : int
        Required eta randomly selected values needed..

    Returns
    -------
    etas : np.array
        array with the eta values, for a 2-qubit system it has dimensions Nx4x4.

    """
    n1, n2 = mean.shape
    etas = np.zeros((N,n1,n2))
    for i in range(n1):
        for j in range(n2):
            etas[:,i,j] = np.random.normal(loc=mean[i,j],scale=std[i,j],size = N)
    return etas

def apply_constraints(gammae, po, C, C_exp, G, init):
    """
    

    Parameters
    ----------
    gammae : np.array
        perturbed density state operator prior the application of the set of constraints.
    po : np.array
        Density state that wants to be perturbed.
    C : list
        Set of different constraints, they should be lambda function depending on the density 
        operator
    C_exp : List
        List of constraints, float numbres that represent the constraints Exp. [1, E_mean, S_mean]
        where E_mean is the mean energy and S_mean the mean entropy.
    G : TYPE
        DESCRIPTION.
    init : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    lambdas = fsolve(constrains, init, args=(G, po, gammae, C, C_exp))
    return lambdas, gammar_fun(lambdas, G, gammae)

def find_eta(gammar, gamma0):
    l = len(sn)
    eta = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            eta[i,j] = 0.5 * ((gammar - gamma0) @ np.kron(sn[i], sn[j])).trace().real
    return eta
