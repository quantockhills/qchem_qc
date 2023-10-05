import qutip as qt
import numpy as np 
import qutip_qip.operations as qip_op

def n_qubit_unitary(theta_vector, n = 2):
    """
    Creates a n-qubit unitary 'gate'. In general will have 4^n - 1 parameters, for now it takes 4^n parameters. 

    Parameters
    ----------
    theta_vector : np.array or list
        The parameters for the n-qubit unitary. 

    n : int
        The number of qubits the gate acts upon. 
    
    Returns
    -------
    unitary: qutip.Qobj 
        The Qobj class object corresponding to the unitary 

    """
    # is a 2^n x 2^n matrix 
    if len(theta_vector) != 4**n:
        print("Error: Too many arguments given.")
    dict_pauli = {'0': qt.qeye(2), '1': qt.sigmax(), '2': qt.sigmay(), '3': qt.sigmaz()}
    mat = np.zeros((2**n, 2**n), dtype = complex)
    str_tens = 0.0
    for i in np.arange(0, 4**n):
        aux_list = [] 
        pstr = np.base_repr(i, base=4)
        while len(pstr) < n:
            pstr = '0' + pstr
        for j in pstr: 
            aux_list.append(dict_pauli[j]) 
        str_tens += 1.0j*theta_vector[i]*qt.tensor(aux_list)
    
    return str_tens.expm()

# * Below function is a controlled unitary gate (see arXiv:2301.06376), that applies (conditional on the control qubit) a 
# * unitary U on the target qubit 
CU = lambda arg_value : qip_op.controlled_gate(
    n_qubit_unitary(arg_value, 1), controls = 0, targets = 1, N = 2)


# * Below function is an arbitrary single qubit unitary gate (parametrized) 
U = lambda arg_value : n_qubit_unitary(arg_value, 1)

def PL(qcirc, thetas):   
    """
    Creates a layer of single-qubit parametrized unitaries in the circuit.  
    Parameters
    ----------
    qcirc : qutip's QubitCircuit class
        The quantum circuit to which you wish to append a single qubit variational layer

    thetas : np.array or list
        The parameters for the single qubit layer 

    Returns
    -------
    qcirc : QubitCircuit class
        The circuit with the appended parametrized layer.
    """

    if len(thetas) != (qcirc.N - 1)*4:
        print("PLerr: excess no. of parameters added. ")
    for i in range(1, qcirc.N): 

        qcirc.add_gate("U", targets=[i], arg_value=thetas[4*(i-1) : 4*i])
        
    return qcirc

def LU(qcirc, thetas):   
    """
    Creates a entangled layer of parametrized unitaries in the circuit. See the arxiv paper cited elsewhere in the code for more 
    details on the LU layer. 

    Parameters
    ----------
    qcirc : qutip's QubitCircuit class
        The quantum circuit to which you wish to append a single qubit variational layer

    thetas : np.array or list
        The parameters for the multiqubit layer 

    Returns
    -------
    qcirc : QubitCircuit class
        The circuit with the appended parametrized layer.
    """
  
    lu_params = (qcirc.N - 2)*4    
    # Linearly Entangled Layer
    if len(thetas) != lu_params:
        print("LUerr: excess no. of parameters added. ")
    for i in range(1, qcirc.N - 1): 
        qcirc.add_gate("CU", targets = [i, i+1], arg_value = thetas[4*(i-1) : 4*i])
    return qcirc

def ORBG(qcirc, thetas):

    return qcirc
     
def kraus_reset_arbitrary(densop, target):
    """
    Function to carry out partial measurement and reset (without measuring) on a target qubit. 

    Parameters
    ----------
    densop : qutip's Qobj density matrix object 
        The density matrix corresponding to a quantum state 
        
    target : integer
        A single site at which we want to carry out partial measurement and reset 
    
    Returns
    -------
    Density operator: Qobj corresponding to the new quantum state 
    """

    sites = len(densop.dims[0])
    m0, m1 = [qt.qeye(2)]*sites, [qt.qeye(2)]*sites
    m0[target] = qt.basis(2, 0) * qt.basis(2, 0).dag()
    m1[target] = qt.basis(2, 0) * qt.basis(2, 1).dag()
    m0, m1 = qt.tensor(m0), qt.tensor(m1)

    return (m0 * densop * m0.dag() + m1 * densop * m1.dag())

def no_of_variational_params_lu(orbitals, nlayers, bond_qbits):
    """
    Helper function for the LU layer: not too relevant in the future but for 
    now it helps keep track of the number of variational parameters 
    """

    tot_qubits = bond_qbits + 1 # * assuming phys_dimension = 2
    single_qubit_params = 4 * tot_qubits * 2 + 4 * (nlayers - 1) * tot_qubits 
    two_qubit_params = 4 * (tot_qubits - 1) * nlayers
    num = single_qubit_params + two_qubit_params 
    num = num * orbitals
    return num

def save_mps_info(dictionary, create_entry = True):
    """
    Function to save the details of the run after the program terminates, 
    in a dictionary. 

    Parameters
    ----------
    dictionary : dictionary consisting of a few parameters 
        such as no. of bond qubits, circuit layers, converged energy, 
        circuit's parameters that minimize the energy etc. 
        
    create_entry : Bool
        True by default - adds the name of the dictionary to the 'database' stored at 
        entries.pickle. This makes for easier lookup, but can be set to False if one does 
        not want to spam the database/just wants a trial run. 

    Returns
    -------
    None
    """

    circ_params = dictionary["circ_params"]
    id_no = np.random.randint(1000)
    filename = 'dict_storage/LU_layers_'+str(circ_params["layers"])\
    +'_bonddim_'+str(circ_params["bond_qbit"]**2)+'_orbitals_'\
    +str(circ_params["orbital no."])+'_id_'+str(id_no)+'.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if create_entry == True: 
        with open('entries.pickle', 'rb') as handle:
            aux = pickle.load(handle)
        aux[str(id_no)] = filename 
        with open("entries.pickle", 'wb') as handle:
            pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_mps_info(id_no):
    """
    Function to load dictionary with qMPS run info from a file. 
    Parameters
    ----------
    id_no : int 
        The id number of the run you wish to study. 
    Returns
    -------
    loaded dictionary 
    """

    with open('entries.pickle', 'rb') as handle:
        aux = pickle.load(handle)
    filename = aux[str(id_no)]
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)    
    return dic     