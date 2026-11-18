import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter
from scipy.optimize import OptimizeResult, minimize_scalar
from angmom import (
    kron_ops,
    dot_product,
    dim,
    check_algebra,
    hermitian,
    r_diag,
)

# Constants - Set USE_TEX = True if you want the report's formatting
MU_B = 1.3996244936 # MHz / G
USE_TEX = False

# Preamble
np.set_printoptions(precision=4, suppress=True, linewidth=110)

if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    })

for j in [0.5, 1.5, 2.5]:
    check_algebra(j)
print("angmom.py imported and verified.")


def lande_gJ(L, S, J, gS=2.0023193):
    """Lande g-factor for a fine-structure level."""
    if J == 0:
        return 0.0
    gL = 1.0
    return (gL * (J*(J+1) - S*(S+1) + L*(L+1)) / (2*J*(J+1))
            + gS * (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1)))

def lande_gF(F, I, J, gJ, gI):
    """Lande g-factor for a hyperfine level (weak-field limit)."""
    if F == 0:
        return 0.0
    return (gJ * (F*(F+1) + J*(J+1) - I*(I+1)) / (2*F*(F+1))
            + gI * (F*(F+1) + I*(I+1) - J*(J+1)) / (2*F*(F+1)))

print(f"{'term':<10}{'L':>3}{'J':>5}   g_J")
for nm, L, J in [("S_1/2", 0, 0.5), ("P_1/2", 1, 0.5),
                 ("P_3/2", 1, 1.5), ("D_5/2", 2, 2.5)]:
    print(f"{nm:<10}{L:>3}{J:>5.1f}   {lande_gJ(L, 0.5, J):.4f}")


# Main functions
def H_breit_rabi(I, J, A, Bq, gJ, gI, Bfield):
    """Hyperfine + Zeeman Hamiltonian in the |m_J> (x) |m_I> basis.

    A, Bq in MHz; Bfield in Gauss; returns H in MHz.
    """
    # electronic space first
    Jf, If = kron_ops(J, I)
    IdotJ = dot_product(If, Jf)

    J_z, I_z = Jf[2], If[2]

    H_HFS = A * IdotJ

    if Bq != 0.0 and I >= 1 and J >= 1:
        Id = np.eye(dim(J) * dim(I))
        num = 3*(IdotJ @ IdotJ) + 1.5*IdotJ - I*(I+1)*J*(J+1)*Id
        den = 2*I*(2*I - 1)*J*(2*J - 1)
        H_QUAD = H_HFS + Bq * num / den

    # Zeeman Term
        H_Z = H_QUAD + MU_B * Bfield * (gJ * J_z + gI * I_z)
    else:
        H_Z = H_HFS + MU_B * Bfield * (gJ * J_z + gI * I_z)

    return H_Z

def sweep(I, J, A, Bq, gJ, gI, Bfield):
    """Eigenvalues at each field. Returns array (n_fields, n_states)."""
    return np.array([
        np.linalg.eigvalsh(H_breit_rabi(I, J, A, Bq, gJ, gI, B))
        for B in Bfield
    ])

def breit_rabi_formula(F, MF, I, A, gJ, gI, Bfield, tol = 0.001):
    """The analytic Breit-Rabi energy, exactly as written above."""
    delta_W = A * (I + 0.5)
    x = (gJ - gI) * MU_B * Bfield / delta_W

    first = - delta_W / (2 * (2 * I + 1))
    second = gI * MU_B * MF * Bfield
    third =  0.5 * delta_W * np.sqrt(1 + 4 * MF * x / (2 * I + 1) + x ** 2)

    if np.isclose(MF, - I - 0.5):
        return first + second + 0.5 * delta_W * (1 - x)
    elif np.isclose(MF, I + 0.5):
            return first + second + 0.5 * delta_W * (1 + x)

    e1 = first + second + third
    e2 = first + second - third

    return e1 if F > I else e2

def analytic_levels(fn, I, A, gJ, gI, Bfield):
    out = []
    for F in [I - 0.5, I + 0.5]:
        for MF in np.arange(-F, F + 1):
            out.append(fn(F, MF, I, A, gJ, gI, Bfield))
    return np.sort(np.array(out))

def analytic_lines(fn, I, A, gJ, gI, Bfields: np.ndarray):
    out = []
    for F in [I - 0.5, I + 0.5]:
        for MF in np.arange(-F, F + 1):
            line = []
            for b in Bfields:
                line.append(fn(F, MF, I, A, gJ, gI, b))
            out.append(line)
    return out

def sweep_with_MF(I, J, A, Bq, gJ, gI, fields):
    lambda_ = np.sqrt(2)

    Jf, If = kron_ops(J, I)
    F_z = Jf[2] + If[2]

    hamiltonians = [H_breit_rabi(I, J, A, Bq, gJ, gI, f) for f in fields]
    compounds = [h + lambda_ * F_z for h in hamiltonians]

    E = []
    MF = []

    for idx, compound in enumerate(compounds):
        _eigvals, eigvec = np.linalg.eigh(compound)

        E_vals = r_diag(hermitian(eigvec) @ hamiltonians[idx] @ eigvec)
        M_vals = r_diag(hermitian(eigvec) @ F_z @ eigvec)
        E.append(np.round(E_vals, 4))
        MF.append(np.round(M_vals, 4))

    return E, MF

def F_z_diagnostic(MF_Rb: list):
    MF_Rb_arr = np.asarray(MF_Rb)

    half_int_err = np.max(
        np.abs(MF_Rb_arr*2 - np.round(MF_Rb_arr*2))
    )

    print(f"max deviation of <Fz> from a half-integer: {half_int_err:.2e}")

    sets = {
        tuple(sorted(Counter(np.round(row, 6)).items())) # type: ignore
        for row in MF_Rb_arr
    }

    print(f"number of distinct M_F multisets across the sweep: {len(sets)} (must be 1)")
    print("M_F values present:", np.round(np.unique(np.round(MF_Rb_arr, 6)), 1))

    # Lastly plot MF occurrences
    a = sets.pop()
    vals, occur = zip(*a)
    fig, ax = plt.subplots(1,1,figsize=(8,2))
    ax.bar(vals, occur)
    ax.set_xticks(vals)
    ax.set_xlabel("$M_F$", fontsize=18)
    ax.set_ylabel("Occurrence", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14, top=True, right=True, direction='in')
    plt.show()

def curly_brace(fig, x, y0, y1, w=0.005, lw=0.6, color="k", n=60):
    ym = (y0 + y1) / 2
    t = np.linspace(0, np.pi / 2, n)

    r = w
    bx = x + r * np.sin(t)
    by = y0 + r * (1 - np.cos(t))
    vx = np.array([x + r, x + r])
    vy = np.array([y0 + r, ym - r])
    tx = x + r + r * (1 - np.cos(t))
    ty = ym - r + r * np.sin(t)

    lower_x = np.concatenate([bx, vx, tx])
    lower_y = np.concatenate([by, vy, ty])

    upper_x = lower_x[::-1]
    upper_y = 2 * ym - lower_y[::-1]

    xs = np.concatenate([lower_x, upper_x])
    ys = np.concatenate([lower_y, upper_y])
    fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color=color, lw=lw))

def tabulate(E_K0: np.ndarray, MF_K0: np.ndarray, reference: bool = True):
    energies, inverse = np.unique(E_K0, return_inverse=True)
    print(energies)

    # Reference values (Tiecke), keyed by F
    tiecke_vals = {4.5: -571.5, 3.5: 714.3}

    if reference:
        header = (f"{'F':>4}  {'Energy (MHz)':>13}  {'MF range':>12}  "
              f"{'Degeneracy':>10}  {'Reference (MHz)':>15}")
        print(header)
        print("-" * len(header))
        for i, e in enumerate(energies):
            m = MF_K0[inverse == i]
            F_val = float(np.max(m))
            degeneracy = len(m)
            mf_range = f"[-{F_val:g}, +{F_val:g}]"
            ref = tiecke_vals[F_val]
            print(f"{F_val:>4.1f}  {e:>13.1f}  {mf_range:>12}  "
                f"{degeneracy:>10d}  {ref:>15.1f}")
    else:
        header = (f"{'F':>4}  {'Energy (MHz)':>13}  {'MF range':>12}  "
              f"{'Degeneracy':>10}")
        print(header)
        print("-" * len(header))
        for i, e in enumerate(energies):
            m = MF_K0[inverse == i]
            F_val = float(np.max(m))
            degeneracy = len(m)
            mf_range = f"[-{F_val:g}, +{F_val:g}]"
            print(f"{F_val:>4.1f}  {e:>13.1f}  {mf_range:>12}  "
                f"{degeneracy:>10d}")

def summarise(K40_4S12: dict, fields: np.ndarray, *, reference_tab: bool = True):
    name = K40_4S12.pop("name") or "Atomic State"
    E_K, MF_K = sweep_with_MF(fields=fields, **K40_4S12)

    tabulate(E_K[0], MF_K[0], reference_tab)
    F_z_diagnostic(MF_K)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.13, right=0.87)

    MFvals = np.unique(np.round(MF_K[-1], 3))
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(MFvals)))

    for i, mf in enumerate(MFvals):
        E_mf = np.array([np.sort(e[np.isclose(m, mf, atol=1e-2)]) for e, m in zip(E_K, MF_K)])
        ax.plot(fields, E_mf, color=cmap[i], lw=1.5)
        ax.plot([], [], color=cmap[i], lw=2, label=f"$M_F$ = {round(mf*2)}/2")

    ax.legend(fontsize=10, ncol=2, loc="upper left", framealpha=0.9)
    
    ax.set_xlabel("Magnetic field / G", fontsize=14)
    ax.set_ylabel("Energy / MHz", fontsize=14)
    
    ax.set_title(name + ", levels coloured by $M_F$")

    ax.tick_params(axis="both", which="major", labelsize=14, top=True, right=True, direction='in')

    # Annotate
    ax.annotate("$F=7/2$", (45, 500), ha="center",va="center", fontsize=12)
    ax.annotate("$F=9/2$", (45, -800), ha="center",va="center", fontsize=12)

    fig.text(0.95,0.74,"$m_J = + 1/2$",ha="center",va="center", fontsize=12)
    fig.text(0.95,0.23,"$m_J = - 1/2$",ha="center",va="center", fontsize=12)

    # Curly bracket

    h1 = 0.11
    h2 = 0.11
    curly_brace(fig, 0.885, 0.74 - h1, 0.74 + h1)
    curly_brace(fig, 0.885, 0.23 - h2, 0.23 + h2)

    plt.show()

    return E_K, MF_K

def analytical_comparison(K40_4S12: dict, fields: np.ndarray, E_K: np.ndarray):
    ana2 = np.array(
        analytic_lines(
            breit_rabi_formula, K40_4S12["I"], K40_4S12["A"], K40_4S12["gJ"], K40_4S12["gI"], fields
        )
    )

    max_residuals = []
    for E_K_num, E_K_ana in zip(E_K, ana2.T):
        ordered_num = np.sort(E_K_num)
        ordered_ana = np.sort(E_K_ana)

        residual = ordered_num - ordered_ana
        max_residuals.append(residual[np.argmax(np.abs(residual))])

    max_residuals = np.asarray(max_residuals)

    fig, axs = plt.subplots(
        2, 1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    for mf in ana2:
        axs[0].plot(fields, mf, color='tab:red')

    axs[1].scatter(fields, max_residuals*1e6, s=8)

    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=14, top=True, right=True, direction='in')

    axs[1].set_xlabel("Magnetic field / G", fontsize=20)
    axs[1].set_ylabel("Residuals / Hz", fontsize=20)
    axs[0].set_ylabel("Energy / MHz", fontsize=20)

    fig.text(0.025, 0.92, "\\textbf{a)}", fontsize=20)
    fig.text(0.025, 0.38, "\\textbf{b)}", fontsize=20)

    plt.tight_layout()
    plt.show()

def stationary(F: int | float, MF: int | float, K40_4S12: dict, fields: np.ndarray):
    result = minimize_scalar(
        lambda B: breit_rabi_formula(
            F, MF,
            K40_4S12["I"],
            K40_4S12["A"],
            K40_4S12["gJ"],
            K40_4S12["gI"],
            B
        ),
        bounds=(0, fields.max()),
        method='bounded'
    )

    if isinstance(result, OptimizeResult):
        print(f"Analytical Stationary at B: {result.x:.3e} G")
    else:
        raise ValueError("Unknown return type of minimize_scalar")

def validate_stationary(K40_4S12, small_range):
    E_K, _MF_K = sweep_with_MF(fields=small_range, **K40_4S12)
    E_K = np.asarray(E_K)

    # fit ax^2 + bx + c -> stationary at x = -b / 2a
    params = np.polyfit(small_range, E_K[:,10], 2)

    x_stat = - params[1] / (2 * params[0])
    linear = lambda x, a, b, c: a * x ** 2 + b * x + c
    print(f"Numerical Stationary at B: {x_stat:.3e} G")

    fig, axs = plt.subplots(2,1,figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    axs[0].scatter(small_range, E_K[:,10], s=40, label="Numerical")

    x = np.linspace(320, 400, 1000)
    axs[0].plot(x, linear(x, *params), c = 'tab:blue', alpha=0.8, linestyle='dashed', label="$ax^2 + bx + c$")

    B_label = "$B_{\\text{stat}}: " + f"{x_stat:.2f}" + "\\ \\text{G}$"

    axs[0].axvline(x_stat, linestyle='dotted', lw=1.5, label=B_label, color='tab:red')

    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=14, top=True, right=True, direction='in')

    residuals_khz = (E_K[:,10] - linear(small_range, *params)) * 1000
    axs[1].scatter(small_range, residuals_khz,s=30,marker='D')
    axs[1].set_ylim(np.max(np.abs(residuals_khz))*-1.2, np.max(np.abs(residuals_khz))*1.2)

    axs[1].axhline(0, color='tab:grey', linestyle='dashed', lw=0.5, alpha=0.8)

    axs[0].legend(fontsize=14)

    axs[1].set_xlabel("Magnetic field / G", fontsize=20)
    axs[1].set_ylabel("Residuals / kHz", fontsize=20)
    axs[0].set_ylabel("Energy / MHz", fontsize=20)

    fig.text(0.025, 0.94, "\\textbf{c)}", fontsize=20)
    fig.text(0.025, 0.33, "\\textbf{d)}", fontsize=20)

    plt.tight_layout()
    plt.show()

# Entrypoint
if __name__ == "__main__":
    fields = np.linspace(0, 1000, 1000)
    K40_4S12 = dict(
        name="$^{40}$K $4\\,^2S_{1/2}$",
        I=4,
        J=0.5, 
        A=-285.731,
        Bq=0,
        gJ=lande_gJ(0, 0.5, 0.5),
        gI=0.00017649
    )

    E_K, MF_K = summarise(K40_4S12, fields)
    analytical_comparison(K40_4S12, fields, np.asarray(E_K))
    stationary(7/2, 7/2, K40_4S12, fields)

    # Choose small range to validate stationary point
    small_range = np.linspace(320, 400, 30)
    validate_stationary(K40_4S12, small_range)
