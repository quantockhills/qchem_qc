# qchem_qc

A gentle introduction to variational quantum algorithms for quantum chemistry.

This material was put together by Madhav Mohan at TU Eindhoven in 2023, for bachelor's and master's thesis students getting started with research in quantum algorithms. It covers the Variational Quantum Eigensolver (VQE) — one of the first quantum chemistry algorithms demonstrated on real quantum hardware — along with a quantum matrix product state (qMPS) ansatz for going beyond small molecules. It is shared here in case other students beginning similar research find it useful.

## What's inside

| File / directory | Description |
|---|---|
| `tutorial/vqe_introduction.ipynb` | A step-by-step Jupyter notebook introducing VQE, reproducing the H₂ potential energy curve from Kandala et al. (Nature, 2017) |
| `tutorial/helper.py` | Helper functions for the VQE notebook (circuit measurements, optimization wrappers) |
| `tutorial/qtn_demo.py` | Standalone script demonstrating the qMPS ansatz on H₂ |
| `tutorial/qtn_help.py` | Circuit-building utilities for the qMPS ansatz (parametrized layers, reset operations) |
| `tutorial/images/` | Figures referenced in the notebook |
| `tutorial/resources.md` | Further reading: computational chemistry primers, key papers, QuTiP documentation |

## Prerequisites

### Software

- Python 3.8+
- [QuTiP](https://qutip.org/) — quantum circuit construction and simulation
- [OpenFermion](https://quantumai.google/openfermion) — quantum chemistry on classical computers
- [OpenFermion-PySCF](https://github.com/quantumlib/OpenFermion-PySCF) — OpenFermion + PySCF integration
- [PySCF](https://pyscf.org/) — electronic structure calculations
- SciPy, NumPy, Matplotlib

```bash
pip install qutip openfermion openfermionpyscf pyscf scipy numpy matplotlib
```

### Background knowledge

Students are expected to be comfortable with Python and have some familiarity with quantum mechanics. The [resources document](tutorial/resources.md) lists lecture series and papers that cover the necessary computational chemistry background.

## Getting started

1. Clone the repository:
   ```bash
   git clone https://github.com/quantockhills/qchem_qc.git
   cd qchem_qc
   ```

2. Start with the notebook:
   ```bash
   cd tutorial
   jupyter notebook vqe_introduction.ipynb
   ```

3. For the qMPS ansatz demo:
   ```bash
   python qtn_demo.py
   ```

## Key references

- Kandala et al., *Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets*, Nature 549, 242–246 (2017)
- McArdle et al., *Quantum computational chemistry*, Rev. Mod. Phys. 92, 015003 (2020)
- Foss-Feig et al., *Variational quantum simulation of correlated quantum matter using matrix product states*, arXiv:2301.06376 (2023)
