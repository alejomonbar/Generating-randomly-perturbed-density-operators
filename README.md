# Method for Generating Randomly Perturbed Density Operators Subject to Different Sets of Constraints

This code presents a general method for producing randomly perturbed density operators subject to different sets of constraints. The perturbed density operators are a specified “distance” away from the state described by the original density operator. This approach is applied to a bipartite system of qubits and used to examine the sensitivity of various entanglement measures on the perturbation magnitude. The constraint sets used include constant energy, constant entropy, and both constant energy and entropy. The method is then applied to produce perturbed random quantum states that correspond with those obtained experimentally for Bell states on the IBM quantum device ibmq manila. The results show that the methodology can be used to simulate the outcome of real quantum devices where noise, which is important both in theory and simulation, is present.

<img src="./Figures/S_E.png" width="500">

Energy-entropy diagram of the simulation of the ibmq\_manila device based on the $\eta_{i,j}$ values using the normal distributions $\mathcal{N}(\tilde E_\mu, \tilde E_\sigma)$ and $\mathcal{N}(\tilde S_\mu, \tilde S_\sigma)$ for the energy and entropy, respectively. The yellow crosses are the 200 experimental values.
