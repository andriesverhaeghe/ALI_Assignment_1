"""angmom.py -- angular momentum toolkit, QLM Atom-Light Interactions.
Built in Session 1; imported in Sessions 2 and 3, and in Assignment A1.

Convention: dimensionless operators (physical J is hbar * this),
basis ordered with m DESCENDING.
"""
import numpy as np


def dim(j: int | float) -> int:
    """Hilbert space dimension 2j+1, as an int."""
    d = int(round(2 * j + 1))
    if d < 1:
        raise ValueError(f"invalid j = {j}")
    return d


def mvalues(j: int | float) -> np.ndarray:
    """m values in DESCENDING order: [j, j-1, ..., -j]."""
    return np.array([j - k for k in range(dim(j))], dtype=float)


def commutator(j1: np.ndarray, j2: np.ndarray) -> np.ndarray:
    """[J1, J2]"""
    for j in (j1, j2):
        if len(j.shape) != 2 or j.shape[0] != j.shape[1]:
            raise ValueError("Commutator only supported for square 2D matrices!")

    try:
        return j1 @ j2 - j2 @ j1
    except Exception as e:
        raise ValueError(f"Matrix multiplication failed: {e}")


def hermitian(j: np.ndarray) -> np.ndarray:
    """J^dagger"""
    if len(j.shape) != 2 or j.shape[0] != j.shape[1]:
        raise ValueError("Commutator only supported for square 2D matrices!")

    return j.conj().T


def jplus(j: int | float) -> np.ndarray:
    """Raising operator J+ for angular momentum j, in the m-descending basis."""
    m = mvalues(j)[1:]
    matrix_elements = np.sqrt(j*(j+1) - m*(m+1))
    return np.diag(matrix_elements, 1)


def jminus(j: int | float):
    """Lowering operator J+ for angular momentum j, in the m-descending basis."""
    m = mvalues(j)[:-1]
    matrix_elements = np.sqrt(j*(j+1) - m*(m-1))
    return np.diag(matrix_elements, -1)


def jops(j: int | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Jx, Jy, Jz) for angular momentum j."""
    jplus_arr = jplus(j)
    jminus_arr = jminus(j)

    jx = 0.5 * (jplus_arr + jminus_arr)
    jy = 0.5j * (jminus_arr - jplus_arr)
    jz = 0.5 * (jplus_arr @ jminus_arr - jminus_arr @ jplus_arr)

    return jx, jy, jz


def kron_ops(j1, j2):
    """Embed both angular momenta into the joint space.
    Returns (J1, J2), each a 3-tuple (x, y, z) acting on the full space."""
    d1, d2 = dim(j1), dim(j2)
    o1, o2 = jops(j1), jops(j2)
    I1, I2 = np.eye(d1), np.eye(d2)
    J1 = tuple(np.kron(a, I2) for a in o1)
    J2 = tuple(np.kron(I1, b) for b in o2)
    return J1, J2


def dot_product(A, B):
    """A . B for two 3-tuples of operators on the same space."""
    return A[0] @ B[0] + A[1] @ B[1] + A[2] @ B[2]


def condon_shortley(F, MF, vecs, Fx, Fy):
    """Sort the coupled basis by (F, M_F) and fix its phases the Condon-Shortley way.

    F, MF: quantum numbers of each column of vecs; Fx, Fy: total angular momentum
    in the uncoupled basis. Within each F multiplet the top state (M_F = F) gets a
    positive m1 = j1 component, and each lower state is chosen so that F- takes the
    state above it there with a positive coefficient.
    """
    order = np.lexsort((-MF, F))                  # each multiplet, top state first
    F, MF, vecs = F[order], MF[order], vecs[:, order]
    Fminus = Fx - 1j * Fy
    for k in range(len(F)):
        if k == 0 or F[k] != F[k - 1]:            # top state: m1 = j1 part > 0
            lead = vecs[np.argmax(np.abs(vecs[:, k]) > 1e-9), k]
            vecs[:, k] *= abs(lead) / lead
        else:                                     # <F,M|F-|F,M+1> > 0
            a = vecs[:, k].conj() @ Fminus @ vecs[:, k - 1]
            vecs[:, k] *= a / abs(a)
    order = np.lexsort((MF, F))
    return F[order], MF[order], vecs[:, order]


def r_diag(j: np.ndarray) -> np.ndarray:
    """Real helper for floating point error imaginary components"""
    return np.real(np.diag(j))


def coupled_basis(j1, j2, lmbd = np.sqrt(2)):
    """Coupled basis |F, M_F> for the coupling of j1 and j2.

    Returns (F, MF, vecs) where columns of `vecs` hold the Clebsch-Gordan
    coefficients <m1, m2 | F, MF>, sorted by (F, MF).
    Finish with: return condon_shortley(F, MF, vecs, Fx, Fy)
    """
    J1, J2 = kron_ops(j1, j2)
    Fx, Fy, Fz = (J1[i] + J2[i] for i in range(3))

    F2 = Fx @ Fx + Fy @ Fy + Fz @ Fz

    compound = F2 + lmbd * Fz
    _eigval, eigvec = np.linalg.eigh(compound)

    F2_diag = r_diag(hermitian(eigvec) @ F2 @ eigvec)
    MF = r_diag(hermitian(eigvec) @ Fz @ eigvec)

    F = 0.5 * (-1 + np.sqrt(1 + 4 * F2_diag))

    # Fix floating point errors
    F = np.round(F * 2) / 2.0
    MF = np.round(MF * 2) / 2.0

    return condon_shortley(F, MF, vecs=eigvec, Fx=Fx, Fy=Fy)


def group_levels(E, tol=1e-4):
    """Cluster near-degenerate eigenvalues. Returns [(energy, degeneracy), ...].

    Do NOT use round()-then-set for this: two eigenvalues differing by 1e-13
    can straddle a rounding boundary and be reported as distinct levels.
    """
    out = []
    for e in np.sort(E):
        if out and abs(e - out[-1][0]) < tol:
            n = out[-1][1]
            out[-1] = ((out[-1][0]*n + e)/(n + 1), n + 1)
        else:
            out.append((e, 1))
    return out


def check_algebra(j: int | float, tol: float = 1e-10):
    """Verify the defining properties of an angular momentum. Raises on failure."""    
    jx, jy, jz = jops(j)

    # [J_x, J_y] = iJ_z
    assert np.allclose(commutator(jx, jy), 1j * jz, tol)

    # J_x^2 + J_y^2 + J_z^2 = j(j+1) I
    assert np.allclose(
        jx @ jx + jy @ jy + jz @ jz,
        (j * (j + 1)) * np.eye(dim(j)),
        tol
    )

    # J_k = J_k^dag
    for j_matrix in (jx, jy, jz):
        assert np.allclose(j_matrix, hermitian(j_matrix), tol)

    # diag(J_z) = (j, j-1, ... , -j+1, -j)
    assert np.allclose(np.diag(jz), np.arange(j, -(j+1), -1), tol)
