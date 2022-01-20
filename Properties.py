#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:16:33 2022

@author: alejomonbar
"""

import numpy as np
from scipy import linalg

si = np.eye(2, dtype = complex)
sx = np.array([[0,1],[1,0]], dtype = complex)
sy = np.array([[0,-1j],[1j,0]], dtype = complex)
sz = np.array([[1,0],[0,-1]], dtype = complex)
sn = [si, sx, sy, sz]

def partial_trace_mul(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a linalg
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])

def fidelity(p_ideal, p_real):
    """
    

    Parameters
    ----------
    p_ideal : square matrix or array of square matrices
        The ideal density state
    p_real : square matrix or array of square matrices
        The experimental or simulated density state.

    Returns
    -------
    F : value or array
        Fidelity of the output signal.

    """
    
    if len(p_real.shape) == 3:
        F = []
        for i,p in enumerate(p_real):
            if len(p_ideal.shape) == 3:
                sqrt_p_ideal = linalg.sqrtm(p_ideal[i,:,:])
            else:
                sqrt_p_ideal = linalg.sqrtm(p_ideal)
            F.append(np.trace(linalg.sqrtm(sqrt_p_ideal.dot(p).dot(sqrt_p_ideal)))**2)
        F = np.array(F)
    else:
        sqrt_p_ideal = linalg.sqrtm(p_ideal)
        F = np.trace(linalg.sqrtm(sqrt_p_ideal.dot(p_real).dot(sqrt_p_ideal)))**2
    return F

def concurrence(p):
    """
    Return the concurrence based on the paper of Shulman 2012
    "Demonstration of entanglement of electrostatically coupled singlet-triplet
    qubits"
    Arguments:
        p (array nn x nn): the evolution in time the density operator based
        in the evolution equation used
    Return:
        con(array n x 1): array with the values of concurrence for the density state.
    """
    sy = np.array([[0,-1j],[1j,0]])
    pb = np.kron(sy,sy) @ np.conjugate(p) @ np.kron(sy,sy)
    psqrt = linalg.sqrtm(p)
    R = linalg.sqrtm(psqrt @ pb @ psqrt)
    eig = sorted(np.abs(np.linalg.eigh(R)[0]))
    con = eig[3] - eig[2] - eig[1] - eig[0]
    return con

def distanceBS(gamma1, gamma2):
    theta = (gamma1 @ gamma2).trace().real
    return np.arccos(theta)

def mutualInf(p):
    pa = partial_trace_mul(p, [2,2], axis = 0)
    pb = partial_trace_mul(p, [2,2], axis = 1)
    return (-pa @ linalg.logm(pa)).trace() + (-pb @ linalg.logm(pb)).trace() - (-p @ linalg.logm(p)).trace()

def CHSH(p):
    """Clauser-Horne-Shimony-Holt"""
    
    T = np.zeros((3,3), dtype = complex) 
    for i in range(3):
        for j in range(3):
            T[i,j] = (p @ np.kron(sn[i+1], sn[j+1])).trace()
    eig = sorted(linalg.eig(T)[0])
    t11 = eig[-1]
    t22 = eig[-2]
    return 2 * np.sqrt(t11**2 + t22**2)

def unit_trace(p):
    return p.trace()

def energy(p, H):
    return np.trace(p @ H)

def entropy(p):
    return np.trace(-p @ linalg.logm(p))

