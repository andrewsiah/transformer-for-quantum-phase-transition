from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, spinless_fermion_basis_1d
from quspin.tools.measurements import ent_entropy, diag_ensemble
import numpy as np

# import scipy.linalg as sla
# from numpy.random import ranf,seed

from abc import ABC, abstractmethod


class Hamiltonian(ABC):
    """represents a hamiltonian"""

    L: int
    params: dict
    H: hamiltonian

    def get_GS(self):
        """return GS energy,psi of H using sparse method"""
        E_GS, psi_GS = self.H.eigsh(
            self.H.aslinearoperator(), k=1, which="SA", maxiter=1e5
        )
        return E_GS, psi_GS[:, 0]

    def get_GS_full(self):
        """return GS energy,psi of H without sparse method"""
        E_GS, psi_GS = self.H.eigh(self.H.aslinearoperator())
        return E_GS[0], psi_GS[:, 0]

    def get_1stState(self, k=5):
        """return 1st excited-state energy,psi of H using sparse method"""
        E_GS, psi_GS = self.H.eigsh(
            self.H.aslinearoperator(), k=k, which="SA", maxiter=1e5
        )
        try:
            s = self.excitedIndex(E_GS)
            return E_GS[s], psi_GS[:, s]
        except ValueError:
            return get_1stState(self, k=k + 5)

    def get_kStates(self, k=5):
        """return 1st excited-state energy,psi of H using sparse method"""
        E, psi = self.H.eigsh(self.H.aslinearoperator(), k=k, which="SA", maxiter=1e5)
        return E, psi

    def get_1stState_full(self):
        """return 1st excited-state energy,psi of H without sparse method"""
        E_GS, psi_GS = self.H.eigh(self.H.aslinearoperator())
        s = self.excitedIndex(E_GS)
        return E_GS[s], psi_GS[:, s]

    """
    def get_ES(self,state=0):
        if state == 1 :
            psi = self.get_1stState()[1]
        else :
            psi = self.get_GS()[1]
        p_A = ent_entropy(psi,self.basis,return_rdm_EVs=True)["p_A"]
        return p_A
     """

    def get_ES(self, state=0, blkA=None, psi=None):
        if psi != np.ndarray:
            if state == 1:
                # psi = self.get_1stState()[1] old implemetnation uses sparse method
                psi = self.get_1stState_full()[1]
            else:
                # psi = self.get_GS()[1] old implemetnation uses sparse method
                psi = self.get_GS()[1]

        p_A = self.basis.ent_entropy(
            psi, sub_sys_A=blkA, return_rdm="A", return_rdm_EVs=True
        )["p_A"]
        return p_A

    def get_rdm_A(self, blkA=None, state=None, psi=None, full=False):
        if psi != np.ndarray:
            if state == None:
                if not full:
                    psi = self.get_GS()[1]
                else:
                    psi = self.get_GS_full()[1]
            elif state == 1:
                if not full:
                    psi = self.get_1stState()[1]
                else:
                    psi = self.get_1stState_full()[1]
        rdm_A = self.basis.ent_entropy(
            psi, sub_sys_A=blkA, return_rdm="A", return_rdm_EVs=True
        )["rdm_A"]
        return rdm_A

    def get_Sent_A(self, state=0):
        if state == 1:
            psi = self.get_1stState()[1]
        else:
            psi = self.get_GS()[1]
        Sent_A = ent_entropy(psi, self.basis, return_rdm="A")["Sent_A"]
        return Sent_A

    def excitedIndex(self, eigs):
        """return the index of the 1st change in eigenvalues"""
        for i, eig in enumerate(eigs):
            if eig > eigs[0]:
                return i
        raise ValueError("no excited state found")

    @classmethod  # register H_{model} as sublacces of Hamiltonian
    def __subclasshook__(cls, C):
        if cls is Hamiltonian:
            if any("__Hamiltonian__" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


class H_XXZ(Hamiltonian):
    def __init__(self, L=6, delta=0, second_delta = 1, sz=1, D=0, lmbda=0, kwargs={}):
        # compute spin-1 basis
        if sz == 0:
            basis = spin_basis_1d(
                L, S="1/2", pauli=0, Nup=L // 2
            )  # GS for AFM and XY phases in subspace of tot Sz=0
        else:
            basis = spin_basis_1d(L, S=str(sz), pauli=0, **kwargs)

        # TODO: Add a param called second_delta that we use to generate more data.
        # define operators with using site-coupling lists
        S_pm = [[0.5 * second_delta, i, (i + 1) % L] for i in range(L)]  # PBC (SxSx+SySy)
        S_zz = [[delta, i, (i + 1) % L] for i in range(L)]  # PBC
        S_z2 = [[D, i, i] for i in range(L)]  # Sz^2
        S_z = [[-1 * lmbda * (-1) ** i, i] for i in range(L)]

        # static lists
        static = [["+-", S_pm], ["-+", S_pm], ["zz", S_zz], ["zz", S_z2], ["z", S_z]]
        dynamic = []
        # compute exact ground state
        H_xxz = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        self.L = L
        self.params = {"delta": delta, "sz": sz}
        self.basis = basis
        self.H = H_xxz


class H_KH(Hamiltonian):
    def __init__(self, L=6, Jh=1, Jk=1):
        # compute spin-1 basis
        basis = spin_basis_1d(L, S="1", pauli="False")

        # define operators with using site-coupling lists
        S_zz = [[Jh, i, (i + 1) % L] for i in range(L)]  # PBC
        S_pm = [[0.5 * Jh + 0.25 * Jk, i, (i + 1) % L] for i in range(L)]
        S_pp = []
        for i in range(L):
            if np.mod(i, 2) == 0:
                Jk_ = Jk
            else:
                Jk_ = -Jk
            S_pp.append([0.25 * Jk_, i, (i + 1) % L])

        # static lists
        static = [["zz", S_zz], ["+-", S_pm], ["-+", S_pm], ["++", S_pp], ["--", S_pp]]
        dynamic = []
        # compute exact ground state
        H_kh = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        self.L = L
        self.params = {"Jh": Jh, "Jk": Jk}
        self.basis = basis
        self.H = H_kh


class H_SSH(Hamiltonian):
    def __init__(self, L=6, eta=-0.6, U=4, V=-4):
        basis = spinless_fermion_basis_1d(L, Nf=L // 2)  # half filling

        # define operators with using site-coupling lists
        hop_pm = [[-1 - eta * (-1) ** i, i, (i + 1) % L] for i in range(L)]
        hop_mp = [[1 + eta * (-1) ** i, i, (i + 1) % L] for i in range(L)]
        inter = []
        for i in range(L):
            if np.mod(i, 2) == 0:
                U_ = U
            else:
                U_ = V
            inter.append([U_, i, (i + 1) % L])

        # static lists
        static = [["+-", hop_pm], ["-+", hop_mp], ["nn", inter]]
        dynamic = []

        # compute exact ground state
        H_ssh = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        self.L = L
        self.params = {"eta": eta, "U": U, "V": V}
        self.basis = basis
        self.H = H_ssh


class H_SSH2(Hamiltonian):
    # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.115113
    # OBC
    def __init__(self, L=6, eta=-1, t=1, delta=0):
        basis = spinless_fermion_basis_1d(L, Nf=L // 2)  # half filling

        # define operators with using site-coupling lists
        """
        hop_pm1 = [[-t*(1+eta),i,(i+1)] for i in range(0,L-1,2)]
        hop_pm2 = [[-t*(1+eta),i,(i+1)] for i in range(1,L-1,2)]
        hop_pm3 = [[-t*(1-eta),i,(i-1)] for i in range(2,L-1,2)]
        hop_pm4 = [[-t*(1-eta),i,(i-1)] for i in range(3,L-1,2)]
        pair_pp1 = [[delta*(1+eta),i,(i+1)] for i in range(1,L-1,2)]
        pair_mm1 = [[delta*(1+eta),i,(i+1)] for i in range(0,L-1,2)]
        pair_pp2 = [[delta*(1-eta),i,(i-1)] for i in range(3,L-1,2)]
        pair_mm2 = [[delta*(1-eta),i,(i-1)] for i in range(2,L-1,2)]"""
        hop_pm = [[-1 - eta * (-1) ** i, i, (i + 1)] for i in range(L - 1)]
        hop_mp = [[1 + eta * (-1) ** i, i, (i + 1)] for i in range(L - 1)]
        hop_pp = [[delta * (1 + eta * (-1) ** i), i, (i + 1)] for i in range(L - 1)]
        hop_mm = [[delta * (1 - eta * (-1) ** i), i, (i + 1)] for i in range(L - 1)]
        # static lists
        static = [["+-", hop_pm], ["-+", hop_mp], ["nn", hop_pp], ["--", hop_mm]]
        dynamic = []

        # compute exact ground state
        H_ssh = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        self.L = L
        self.params = {"eta": eta, "delta": delta, "t": t}
        self.basis = basis
        self.H = H_ssh


class H_Heisenberg(Hamiltonian):
    def __init__(self, L=6, J=1, h=1):
        # compute spin-1 basis
        basis = spin_basis_1d(L, S="1", pauli="False")

        # define operators with using site-coupling lists
        S_zz = [[J, i, (i + 1) % L] for i in range(L)]  # PBC
        S_pm = [[0.5 * J, i, (i + 1) % L] for i in range(L)]
        S_z = [[-h, i] for i in range(L)]

        # static lists
        static = [["zz", S_zz], ["+-", S_pm], ["-+", S_pm], ["z", S_z]]
        dynamic = []
        # compute exact ground state
        H_heisenberg = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        self.L = L
        self.params = {"J": J, "h": h}
        self.basis = basis
        self.H = H_heisenberg


class H_KH_OBC(Hamiltonian):
    def __init__(self, L=6, Jh=1, Jk=1):
        # compute spin-1 basis
        basis = spin_basis_1d(L, S="1", pauli="False")

        # define operators with using site-coupling lists
        S_zz = [[Jh, i, (i + 1)] for i in range(L - 1)]  # PBC
        S_pm = [[0.5 * Jh + 0.25 * Jk, i, (i + 1)] for i in range(L - 1)]
        S_pp = []
        for i in range(L - 1):
            if np.mod(i, 2) == 0:
                Jk_ = Jk
            else:
                Jk_ = -Jk
            S_pp.append([0.25 * Jk_, i, (i + 1)])

        # static lists
        static = [["zz", S_zz], ["+-", S_pm], ["-+", S_pm], ["++", S_pp], ["--", S_pp]]
        dynamic = []
        # compute exact ground state
        H_kh = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        self.L = L
        self.params = {"Jh": Jh, "Jk": Jk}
        self.basis = basis
        self.H = H_kh


class H_XXZ2(Hamiltonian):
    def __init__(self, L=6, delta=0, sz=1, D=0, lmbda=0, h=0, kwargs={}):
        # compute spin-1 basis
        if sz == 0:
            basis = spin_basis_1d(
                L, S="1/2", pauli=0, Nup=L // 2
            )  # GS for AFM and XY phases in subspace of tot Sz=0
        else:
            basis = spin_basis_1d(L, S=str(sz), pauli=0, **kwargs)

        # define operators with using site-coupling lists
        S_pm = [[0.5, i, (i + 1) % L] for i in range(L)]  # PBC (SxSx+SySy)
        S_zz = [[delta, i, (i + 1) % L] for i in range(L)]  # PBC
        S_z2 = [[D, i, i] for i in range(L)]  # Sz^2
        S_z = [[-1 * lmbda * (-1) ** i, i] for i in range(L)]
        S_x = [[h, i] for i in range(L)]

        # static lists
        static = [
            ["+-", S_pm],
            ["-+", S_pm],
            ["zz", S_zz],
            ["zz", S_z2],
            ["z", S_z],
            ["x", S_x],
        ]
        dynamic = []
        # compute exact ground state
        H_xxz = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)

        self.L = L
        self.params = {"delta": delta, "sz": sz}
        self.basis = basis
        self.H = H_xxz


def mutual_info(basis, psi, d, blkSize, i=0):
    blk_i = [i for i in range(i, i + blkSize)]
    blk_j = [i for i in range(i + d + blkSize, i + d + 2 * blkSize)]
    blk_ij = blk_i + blk_j
    S_i = basis.ent_entropy(psi, sub_sys_A=blk_i, density=False)["Sent_A"] / np.log(2)
    S_j = basis.ent_entropy(psi, sub_sys_A=blk_j, density=False)["Sent_A"] / np.log(2)
    S_ij = basis.ent_entropy(psi, sub_sys_A=blk_ij, density=False)["Sent_A"] / np.log(2)
    return S_i + S_j - S_ij


if __name__ == "__main__":
    # testing
    print(H_KH().get_ES(1))
