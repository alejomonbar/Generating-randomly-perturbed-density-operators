"""
This work is part of the paper: Method for Generating Randomly Perturbed 
Density Operators Subject to Different Sets of Constraints.

"""
import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
from tqdm import tqdm
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
    N = len(G)
    gammar = np.zeros(gammae.shape, dtype=complex)
    gammar += gammae
    for i in range(N):
        gammar -= lambdas[i] * anticommutator(G[i], gammae)
    return gammar

def constrains(lambdas, G, po, gammae, C, C_exp=[]):
    """
    

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

def po_fun(c):
    po = np.kron(si, si)
    for i in range(1, len(sn)):
        po += c[i-1] * np.kron(sn[i], sn[i])
    po *= 0.25
    return po

def etas_norm(sol, N):
    mean = sol["mean"]
    std = sol["std"]
    n1, n2 = mean.shape
    etas = np.zeros((N,n1,n2))
    for i in range(n1):
        for j in range(n2):
            etas[:,i,j] = np.random.normal(loc=mean[i,j],scale=std[i,j],size = N)
    return etas

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

# =============================================================================
# Experiment
# =============================================================================
omega1 = 5.037e9; omega2 = 4.951e9
experiment = np.load("./Data/etas_exp.npy", allow_pickle=True).item()
N = 200
H = -0.5 * hbar * (2 * np.pi)*(omega1 * np.kron(sz,si) + omega2 * np.kron(si, sz)) / (hbar * omega1)
# etas = etas_norm(experiment, N)
sigma = 0.05
np.random.seed(1)
etas = np.random.normal(loc = 0.0, scale = sigma, size = (N,len(sn),len(sn)))
po = np.array([[0.5,0,0,0.5],
               [0,0,0,0],
               [0,0,0,0],
               [0.5,0,0,0.5]])
gamma0 = linalg.sqrtm(po)
gammaes = gammae_fun(gamma0, etas)#, epsilon)
results = {"3.1":{},"3.2":{},"3.3":{},"3.4":{},"eta":etas,
           "H":H, "po":po, "gammaes":gammaes}
prs = experiment["Exp"]
E_exps = np.array([(pi @ H).trace().real for pi in prs])
S_exps = np.array([(-pi @ linalg.logm(pi)).trace().real for pi in prs])
S_mean = S_exps.mean(); S_std = S_exps.std()
E_mean = E_exps.mean(); E_std = E_exps.std()

# =============================================================================
# Constraints
# =============================================================================
identity = lambda p: np.trace(p)
energy = lambda p: np.trace(p @ H)
entropy = lambda p: np.trace(-p @ linalg.logm(p))
# =============================================================================
# Case 3.1 Tr(pr@I) = 1
# =============================================================================
init = [0.01]
gammars = []
C = [identity]
G = [I]
convergence = []
for gammae in tqdm(gammaes):
    pe = gammae @ gammae.T.conjugate()
    x = fsolve(constrains, init, args=(G, po, gammae, C),full_output=True)
    convergence.append(x[-1])
    gammars.append(gammar_fun(x[0], G, gammae))
results["3.1"]["gammars"] = np.array(gammars)    

# =============================================================================
# Case 3.2 Tr(pr@I) = 1 & Tr(pr@H) = Tr(po@H) Constant Energy
# =============================================================================

init = [0.01, 1e-2]
gammars = []
C = [identity, energy]
G = [I, H]
convergence = []
for gammae in tqdm(gammaes):
    pe = gammae @ gammae.T.conjugate()
    E_random = np.random.normal(loc=E_mean, scale=E_std)
    C_exp = [1, E_random]  
    x = fsolve(constrains, init, args=(G, po, gammae, C, C_exp))
    gammars.append(gammar_fun(x, G, gammae))
results["3.2"]["gammars"] = np.array(gammars)

# =============================================================================
# Case 3.3 Tr(pr@I) = 1 & Tr(pr @ log(pr)) = Tr(po @ log(po)) Constant Entropy
# =============================================================================
max_num_cases = 100
gammars = []
C = [identity, entropy]
for gammae in tqdm(gammaes):
    ii = 0
    error_best = [10, 10]
    init = [0.01, 1e-2]
    pe = gammae @ gammae.T.conjugate()
    G = [I, - I - linalg.logm(pe)]
    S_random = np.random.normal(loc=S_mean, scale=S_std)
    C_exp = [1, S_random]
    while ii < max_num_cases:
        x = fsolve(constrains, init, args=(G, po, gammae, C, C_exp), full_output=True)
        if all(x[1]["fvec"] < 1e-3):
            x_best = x[0]
            break
        if x[1]["fvec"][1] < error_best[1]:
            x_best = x[0]
            error_best = x[1]["fvec"]

        init[1] = np.random.rand()
        ii += 1
    if ii == max_num_cases: 
        print(f"it didn't converge after {max_num_cases} iterations")
    gammars.append(gammar_fun(x_best, G, gammae))
results["3.3"]["gammars"] = np.array(gammars)
# =============================================================================
# Case 3.4 Unitary trace & Constant Energy & Constant Entropy
# =============================================================================
max_num_cases = 100
gammars = []
C = [identity, energy, entropy]
for gammae in tqdm(gammaes):
    ii = 0
    error_best = [10, 10, 10]
    init = [0.01, 1e-2, 1e-2]
    pe = gammae @ gammae.T.conjugate()
    G = [I, H, -I - linalg.logm(pe)]
    S_random = np.random.normal(loc=S_mean, scale=S_std)
    E_random = np.random.normal(loc=E_mean, scale=E_std)
    C_exp = [1, E_random, S_random]
    while ii < max_num_cases:
        x = fsolve(constrains, init, args=(G, po, gammae, C, C_exp), full_output=True)
        if all(x[1]["fvec"] < 1e-3):
            x_best = x[0]
            break
        if x[1]["fvec"][1] < error_best[1] and x[1]["fvec"][2] < error_best[2]:
            x_best = x[0]
            error_best = x[1]["fvec"]
        init[1] = np.random.rand()
        init[2] = np.random.rand()
        ii += 1
    if ii == max_num_cases: 
        print(f"it didn't converge after {max_num_cases} iterations")
        
    gammars.append(gammar_fun(x_best, G, gammae))
results["3.4"]["gammars"] = np.array(gammars)
# =============================================================================
# Adding the properties of the gamma_rs of sections 3.1 to 3.4
# =============================================================================
sections = [f"3.{i+1}" for i in range(4)]
fid = lambda p: fidelity(po, p)
funs = {"E":energy, "S":entropy, "C":concurrence, "F":fid, "MI":mutualInf, "DBS":distanceBS, "CHSH":CHSH}
for sec in sections:
    gammars = results[sec]["gammars"]
    variables = {"E":[], "S":[], "C":[], "F":[], "MI":[], "DBS":[], "CHSH":[]}
    for gammar in tqdm(gammars):
        pr = gammar @ gammar.T.conjugate()
        for key in variables.keys():
            if key == "DBS":
                variables[key].append(distanceBS(gamma0, gammar))
            else:
                variables[key].append(funs[key](pr))
    for var in variables:
        results[sec][var] = variables[var]
# =============================================================================
# Save results
# =============================================================================
results["e0"] = energy(po)
results["s0"] = entropy(po)
results["F0"] = 1
results["MI0"] = mutualInf(po)
results["DBS0"] = distanceBS(gamma0, gamma0)
results["CHSH0"] = CHSH(po)
results["C0"] = concurrence(po)
results["etas_exp"] = experiment["etas"]
results["etas_sim"] = etas
results["E_exps"] = E_exps
results["S_exps"] = S_exps
np.save("./Data/Results_IIIB.npy", results)
