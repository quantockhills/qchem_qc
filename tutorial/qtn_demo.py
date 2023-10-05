
# * First that for N qubits, the index of the final qubit is N - 1 
# * ncil: no class in loop: yet to implement
import time

t_init = time.time()
import scipy 
import qtn_help as qtn
import numpy as np
import math as m
import qutip as qt 
import openfermion as of
import openfermionpyscf as scfwrap
from qutip_qip.circuit import QubitCircuit, CircuitSimulator
import qutip_qip.circuit as qip_circ
import qutip_qip.operations as qip_op
import matplotlib.pyplot as plt 
import pickle 

# now, some imports for speeding up the code and benchmarking 
# should eventually also import tensorflow

print_count = 0

def qcmps_run(thetas, qubitham, circuit_params): 
    
    """Evaluates the measurement outcome on a qMPS circuit of a given quantum 
    chemistry Hamiltonian. 
    Currently simulates density matrices instead of wavefunctions - that is, 
    one obtains the 'ideal' result (corresponding to infinite measurements). 
    This can be changed soon with sampling from the density matrix. 
    Parameters
    ----------
    thetas : np.array or list
        The parameters for the circuit ansatz. 
    qubitham : openfermion's QubitOperator class
        The qubit Hamiltonian (obtained with OpenFermion) that we need to measure.
        This is already decomposed into Pauli strings (along with the respective coefficients)
        with the Jordan Wigner mapping. 
    circuit_params: dictionary
        This is used to provide the circuit architecture: the number of orbitals (which determines 
        the no. of circuit blocks), the number of bond qubits N (determines the bond dim. 2^N) and 
        the number of LU layers (see arXiv:2301.06376).
    
    Returns
    -------
    energy
        The measured energy of the Hamiltonian 
    
    Caveats:
    ________
    
    Runs currently only for the LU layer architecture (see arXiv:2301.06376). We want to generalize this of course.
    """
    global print_count
    orbitals = circuit_params["orbital no."]           # length of longest pauli string 
    n_bond_qbits = circuit_params["bond_qbit"]

    qbits = n_bond_qbits + 1 + 1    # log2(Bond dimension) +T physical qubit + ancilla for Hadamard measurement 
    noOfLayers = circuit_params["layers"]        # Number of LU layers + appending 1qrot layers 
    # * 0 : physical qubit 
    # * 1 : ancilla qubit
    params_in_a_pl = (qbits - 1) * 4 # * -1 to remove the ancilla 
    params_in_an_lu = 4 * (qbits - 1 - 1)

    vqeh2 = qubitham.terms # * The list of terms to be measured 
    thetas = thetas.reshape(orbitals, int(qtn.no_of_variational_params_lu(orbitals, noOfLayers, n_bond_qbits)/orbitals))
    
    coeffs = []
    energy = 0
    for i1 in vqeh2: #Iterating over different Pauli strings
        termmeas = vqeh2[i1]
        oneqmeas_arr = []
        if i1 != ():
            pstr = i1 
            length = len(pstr)

            init_state = qt.ket2dm(qt.tensor([qt.basis(2, 0)] * qbits))
            for p in np.arange(0, orbitals):
                qc = QubitCircuit(qbits, num_cbits = 0)
                qc.user_gates = {"U": qtn.U, "CU": qtn.CU}
                if p == 0:
                    qc.add_gate("SNOT", targets=[0])
                layer_count = 0
                for i in range(noOfLayers):
                    qc = qtn.PL(qc, thetas[p][layer_count:layer_count + params_in_a_pl])
                    qc = qtn.LU(qc, thetas[p][layer_count + params_in_a_pl:layer_count + params_in_a_pl + params_in_an_lu])
                    layer_count += params_in_a_pl + params_in_an_lu
                qc = qtn.PL(qc, thetas[p][layer_count: layer_count + params_in_a_pl])
                layer_count += params_in_a_pl
                pstr = dict(pstr)
                if p in pstr.keys():
                    if pstr[p] == "X":
                        qc.add_gate("CX", controls = [0] , targets = [1])
                    elif pstr[p] == "Y":
                        qc.add_gate("CY", controls = [0], targets = [1])
                    elif pstr[p] == "Z":
                        qc.add_gate("CZ", controls = [0], targets = [1])

                if p == orbitals - 1: 
                
                    qc.add_gate("SNOT", targets = [0]) 
                    
                circ_instance = CircuitSimulator(qc, mode = "density_matrix_simulator")
                result = circ_instance.run(init_state) # * now a density matrix
                init_state = circ_instance.state
                if p != (orbitals - 1): # * qubit reset (conditional)
                    init_state = qtn.kraus_reset_arbitrary(init_state, 1)    
                
            expec_val = qt.expect(circ_instance.state.ptrace(0), qt.sigmaz())
            termmeas = termmeas * expec_val
         
        energy = energy + termmeas
    if print_count%100 == 0: 
        print(energy)
    print_count += 1
    
    return energy

circuit_params = {"bond_qbit": 2, "layers": 5, "orbital no.": 4}

conv_mps = lambda qubitham, circuit_params: (lambda x: qcmps_run(x, qubitham, circuit_params))

[layers, orbitals, bond_qbit] = circuit_params["layers"], circuit_params["orbital no."], circuit_params["bond_qbit"]
length = 0.8
desc = str(round(length, 2))
geom = [('H', (0, 0, 0)), ('H', (0, 0, length))]
mc = of.chem.MolecularData(geom, 'sto-3g', multiplicity=1, charge=0, description = desc)
mol = scfwrap.run_pyscf(mc,
                         run_scf=1,
                         run_fci=1,
                         verbose=0)
    
k1 = mol.get_molecular_hamiltonian()
qubitham = of.transforms.jordan_wigner(of.transforms.get_fermion_operator(k1))
optimf = conv_mps(qubitham, circuit_params)
theta1 = 2*np.pi*np.random.rand(qtn.no_of_variational_params_lu(orbitals, layers, bond_qbit)) 
opt_dict = {'eps': 0.15e-2, 'ftol' : 1e-2, 'gtol' : 1e-2}
thetaopt = scipy.optimize.minimize(optimf, theta1, method='L-BFGS-B', options = opt_dict) #*l-bfgs-b
save = True
if save == True: 
    dic = {"Energy": thetaopt.fun, "Status": thetaopt.status, "Thetas": thetaopt.x, "Solver options": opt_dict,
            "circ_params": circuit_params, "molecule": 'H2', "bond length": length}
    qtn.save_mps_info(dic)

t_fin = time.time()
print(t_fin - t_init)