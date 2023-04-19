from qutip import tensor, basis
import qutip as qt
import numpy as np

#HELPER FUNCTIONS:
#conv returns a single parameter function from a multiparam. one, to plug into scipy.optimize
def conv(ansatz, times, qubitham):
    def optimfriendly(x):
         energy = vqerun(x, ansatz, times, qubitham)
         return energy
    return optimfriendly

def vqerun(x, ansatz, times, qubitham):
   #INPUT: (circuit parameters theta, ansatz structure, number of measurements, hamiltonian/quantity to be measured in Pauli strings
   #OUTPUT: Measurement, in our case, energy
   vqeh2 = qubitham.terms
   coeffs = []
   for i1 in vqeh2: #Iterating over different Pauli strings
       termmeas = vqeh2[i1] * meas(i1, ansatz(x), times, ideal = 0)  # coefficient*outcome of measurement of pauli string
       coeffs.append(termmeas)
   energy = sum(coeffs)
   return energy.real

def meas(pstr, cirq, times, ideal = 0):
    #If ideal = 1, we do an infinite number of measurements, this is easier
    #HELPER FUNCTION!
    #measures one pauli string
    #Input: Pauli string in tuple form, ansatz, no. of measurements (times)
    pauli = {'X': ["SNOT"], 'Y': ["PHASEGATE","SNOT"], 'Z': [""]}
    # 0 is [1,0], eigenvalue is 1 for Z, 1 is [0,1], eigenvalue is -1
    # X measurement: Hadamard, maps + state to 0, - state to 1
    # Y measurement: Hadamard*sdag, maps -i state to 1, +i state to 0
    # Z measurement: I
    # We transform (for example) phi = c_+|+> + c_-|->
    # to phi' = c_+|0> + c_-|1> by applying a Hadamard gate, so a Z measurement now is effectively a X measurement
    numq = cirq.N
    qubitlocs = []
    ctr = 0
    for i in pstr: #one Pauli string
        qubitloc = pstr[ctr][0]
        qubitlocs.append(qubitloc)
        qubitstr = pstr[ctr][1]
        for k in pauli[qubitstr]: #look up pauli string
           if k=="PHASEGATE":
             cirq.add_gate(k, targets=[qubitloc], arg_value = -np.pi/2)
           elif k=="":
             pass
           elif k=="SNOT":
             cirq.add_gate(k, targets=[qubitloc])
        cirq.add_measurement("M"+str(qubitloc), targets = [qubitloc],classical_store = qubitloc)
        ctr += 1
    multiqubop = []
    for i1 in range(numq):
        if i1 in qubitlocs:
            onequbop = qt.sigmaz()
        else:
            onequbop = qt.identity(2)
        multiqubop.append(onequbop)
    measop = tensor(multiqubop)
    hf_init = tensor(basis(2, 1), basis(2, 1), basis(2,0), basis(2, 0))
    outcomes = []
    stats = cirq.run_statistics(state = hf_init)
    states = stats.get_final_states()
    probdist = stats.get_probabilities()
    expe = qt.expect(measop, states) #returns array of measurements
    #result = cirq.run(state=hf_init)
    if ideal == 0:
       outcomes = np.random.choice(expe, size = (times,1), p = probdist)
       expval = outcomes.sum()/(len(outcomes))
    #one notes that the resulting state is a (2^N)x1 object for N qubits, such that the binary representation corresponds to the qubit information
    if ideal == 1:
        expval = sum(np.multiply(probdist, expe))
    return expval