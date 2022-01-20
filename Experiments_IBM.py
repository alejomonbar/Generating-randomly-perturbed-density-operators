#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 07:21:25 2021
Experimental setup to generate the 200 random cases.
@author: alejomonbar
"""

from qiskit import pulse, QuantumCircuit, transpile, schedule, IBMQ, assemble
from qiskit.ignis.verification.tomography import state_tomography_circuits
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.verification.tomography import StateTomographyFitter
import numpy as np

provider = IBMQ.load_account() # You need an account to generate the random states on a real device


backend = provider.get_backend('ibmq_manila') # Backend used for the experiments
backend_config = backend.configuration()
dt = backend_config.dt
backend_defaults = backend.defaults()
inst_sched_map = backend_defaults.instruction_schedule_map

meas = inst_sched_map.get("measure", qubits=range(backend_config.n_qubits))

qc = 0
qt = 1
N = 200 # Random cases
# =============================================================================
# Circuit to create the Bell state
# =============================================================================
bell = QuantumCircuit(4) 
bell.h(qc)
bell.cx(qc, qt)
# =============================================================================

qst_bell = state_tomography_circuits(bell, [qc, qt]) # Quantum tomography to get back
# the density state
p = []
circuit = transpile(qst_bell, backend)
for i in range(N):
    job = backend.run(circuit,
                    meas_level = 2,
                    meas_return = 'avg',
                    shots = 2000,
                    qubit_lo_freq=backend_defaults.qubit_freq_est,
                    job_name="Bell")
    job_monitor(job)

    result = job.result()
    p.append(StateTomographyFitter(result, circuit).fit())


p = np.array(p)
np.save("./Data/prs_exp.npy", p)
