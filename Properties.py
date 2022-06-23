#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:16:33 2022

@author: alejomonbar
"""

import numpy as np
from scipy import linalg
import qutip as qt

si = np.eye(2, dtype = complex)
sx = np.array([[0,1],[1,0]], dtype = complex)
sy = np.array([[0,-1j],[1j,0]], dtype = complex)
sz = np.array([[1,0],[0,-1]], dtype = complex)
sn = [si, sx, sy, sz]

sTqt = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()] 

ST = [qt.tensor(si, sTqt[0]) for si in sTqt[1:]]


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
    """
    Distance between two density state operators

    Parameters
    ----------
    gamma1 array: 
        The square root of the first density operator.
    gamma2 : array
        The square root of the second density operator..

    Returns
    -------
    float distance measured in radians
        The abstract angle between two states.

    """
    theta = (gamma1 @ gamma2).trace().real
    return np.arccos(theta)

def mutualInf(p):
    """
    Mutual information of a two-qubit system

    Parameters
    ----------
    p : Matrix
        Density state of a two-qubit system.

    Returns
    -------
    floar
        Mutual information of a two-qubit system.

    """
    pa = partial_trace_mul(p, [2,2], axis = 0)
    pb = partial_trace_mul(p, [2,2], axis = 1)
    return (-pa @ linalg.logm(pa)).trace() + (-pb @ linalg.logm(pb)).trace() - (-p @ linalg.logm(p)).trace()

def CHSH(p):
    """Clauser-Horne-Shimony-Holt inequality"""
    
    T = np.zeros((3,3), dtype = complex) 
    for i in range(3):
        for j in range(3):
            T[i,j] = (p @ np.kron(sn[i+1], sn[j+1])).trace()
    eig = sorted(linalg.eig(T)[0])
    t11 = eig[-1]
    t22 = eig[-2]
    return 2 * np.sqrt(t11**2 + t22**2)

def Compute_element(SigmasBaseRho, evals, label1, label2):
    g = []# List where the elements are stored to calculate the matrix element W [label1.label2]
    for i in range(0,4):
        for j in range(0,4):
            if i == j or (evals[i] + evals[j]) < 1e-9:
                g.append(0)
                g.append(0) # If the denominator of the formula to obtain the element W [label1.label2] is close to zero, the saved element will be zero  
            else: 
                g.append(((2 * (evals[i] * evals[j]))/(evals[i]+evals[j]))*(SigmasBaseRho[label1][i,j]*SigmasBaseRho[label2][j,i]))  # Compute the matrix element W [label1.label2]                                
    return g

def QFI(p):
    """
    Quantum Fisher information

    Parameters
    ----------
    p : numpy array
        density state operator.

    Returns
    -------
    float
        Quantum Fisher information.

    """
    evals11, ekets11 = p.eigenstates()
    SigmasBaseRho = [(Si.transform(ekets11)) for Si in ST]
    W_A1 = np.zeros((3,3), dtype=complex)
    for i in range(3):
        for j in range(3):
            W_A1[i,j] = np.sum(Compute_element(SigmasBaseRho, evals11, i, j))
    W_A1 = qt.Qobj(W_A1)  
    evals1, ekets1 = W_A1.eigenstates()
    return 1 - np.amax(evals1)

def heaviside(x):
    if x < 0:
        return 0
    elif x == 0:
        return 1/2
    elif x > 0:
        return 1

def TDD(rho):
    """
    Trace-distance discord from https://github.com/jonasmaziero/libPyQ/blob/master/discord.py
    
    Parameters
    ----------
    rho : array
        density state operator.

    Returns
    -------
    trace-distance discord.

    """
    # trace distance discord for x states [arXiv:1304.6879]
    xA3 = 2 * (rho[0,0]+rho[1,1]) - 1; xA32 = xA3**2
    g1 = 2 * (rho[2,1] + rho[3,0]);  g12 = g1**2
    g2 = 2 * (rho[2][1] - rho[3][0]);  g22 = g2**2
    g3 = 1 - 2 * (rho[1,1] + rho[2,2]);  g32 = g3**2

    if g12 - g32 + xA32 < 0:
        disc = abs(g1)
    else:
        if abs(g3) >= abs(g1):
            disc = abs(g1)
        else:
            disc = heaviside(g12 - g32 + xA32)
            disc *= np.sqrt((g12 * (g22 + xA32) - g22 * g32)/(g12 - g32 + xA32))
            disc += heaviside(-(g12 - g32 + xA32)) * (abs(g3))
    return disc


def unit_trace(p):
    return p.trace()

def energy(p, H):
    return np.trace(p @ H)

def entropy(p):
    return np.trace(-p @ linalg.logm(p))



Id = qt.qeye(4) # Identity operator
bell = qt.bell_state(state='00') # Bell state 00
bell_operator = bell*(bell.dag()) # Bell operator |\phi > < \phi|
BellOps = qt.Qobj(bell_operator.data.toarray(),dims=[[4],[4]]) # Convert Bell Operator to Proper Dimensions in Qutip
Plist = np.linspace(0,1,350) # List of elements P from 0 to 1 with 350 elements


def WernerState(p): # Compute the Werner state given a p-value
    werner = ( (1-p))*BellOps + (p/4)*Id
    return werner

rho = []
for p in Plist: #Compute Werner state and save the density matrix in ro
    rho.append(WernerState(p))
  
qfi = []
for rho_i in rho:
    qfi.append(QFI(rho_i))
    
import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.linspace(0,1,350), qfi)
plt.grid()

