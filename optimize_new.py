#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

Ug2 = 900.0  # [V]

IA_CSV_PATH = "./ia_arr.csv"
IG2_CSV_PATH = "./ig2_arr.csv"

# These are the current values represented by each curve.
# Units should match your data/model convention.
ia_current_vals = np.array([0, 5, 10, 20, 30, 40, 60, 80], dtype=float)
ig2_current_vals = np.array([1, 2, 5, 10, 15, 20], dtype=float)

# Initial parameter guesses:
# Original:
# params_0 = [alpha, beta, K, mu_c, mu_s, exp_coeff]
params_0_physical = np.array([
    0.9596,
    0.7408,
    2.99e-10,
    521.793,
    144.766,
    1.5,
], dtype=float)

# Optimization settings
USE_ROBUST_LOSS = True
MAX_NFEV = 20000


# ============================================================
# Model definitions
# ============================================================

def model_A(alpha, beta, ug2, ua):
    """
    A model:
        A = alpha * (1 - exp(-ua / (beta * ug2)))
    """
    return alpha * (1.0 - np.exp(-ua / (beta * ug2)))


def model_Ii(K, mu_c, mu_s, exp_coeff, ug1, ug2, ua):
    """
    Ii model:
        Ii = K * (mu_c * ug1 + mu_s * ug2 + ua)^exp_coeff

    The drive term is clipped to avoid invalid fractional powers.
    """
    drive = mu_c * ug1 + mu_s * ug2 + ua

    # Avoid invalid values when exp_coeff is fractional.
    # You can replace this with a penalty approach later if needed.
    drive = np.maximum(drive, 0.0)

    return K * drive**exp_coeff


def unpack_params(x):
    """
    Optimizer variables:
        x = [alpha, beta, log_K, mu_c, mu_s, exp_coeff]

    Physical parameters:
        [alpha, beta, K, mu_c, mu_s, exp_coeff]
    """
    alpha = x[0]
    beta = x[1]
    K = np.exp(x[2])
    mu_c = x[3]
    mu_s = x[4]
    exp_coeff = x[5]

    return alpha, beta, K, mu_c, mu_s, exp_coeff


def pack_params(params_physical):
    """
    Convert physical params to optimizer params.
    """
    alpha, beta, K, mu_c, mu_s, exp_coeff = params_physical

    return np.array([
        alpha,
        beta,
        np.log(K),
        mu_c,
        mu_s,
        exp_coeff,
    ], dtype=float)


# ============================================================
# Data loading and cleaning
# ============================================================

def load_curve_arrays(ia_path, ig2_path):
    """
    Expected file format:
        row 0: UA domain
        rows 1 onward: UG1 curves

    Same format as your current script.
    """
    ia_raw = np.loadtxt(ia_path, delimiter=",")
    ig2_raw = np.loadtxt(ig2_path, delimiter=",")

    ua_domain = ia_raw[0].astype(float)

    ia_ug1_curves = ia_raw[1:].astype(float)
    ig2_ug1_curves = ig2_raw[1:].astype(float)

    return ua_domain, ia_ug1_curves, ig2_ug1_curves


def clean_limited_domain_curves(curves, mode="auto_argmax"):
    """
    Convert undefined sections of limited-domain curves to NaN.

    For your IG2 curves, your original code used np.argmax() as a domain cutoff.
    This function does the same thing, but converts everything after that point
    to NaN so the optimizer can use normal masks.

    Parameters
    ----------
    curves:
        2D array of shape [num_curves, num_points]

    mode:
        "auto_argmax":
            Keep samples up to and including the peak index, set the rest to NaN.
            This mimics your previous max_ig2_idx logic.

        "none":
            Do not alter curves.

    Returns
    -------
    cleaned:
        Same shape as curves, with invalid sections set to NaN.

    limits:
        List of last valid indices for each curve.
    """
    cleaned = curves.copy().astype(float)
    limits = []

    if mode == "none":
        for row in cleaned:
            valid = np.where(np.isfinite(row))[0]
            limits.append(valid[-1] if valid.size else -1)
        return cleaned, limits

    if mode == "auto_argmax":
        for i in range(cleaned.shape[0]):
            row = cleaned[i]
            peak_idx = int(np.nanargmax(row))

            # Keep through peak_idx inclusive.
            cleaned[i, peak_idx + 1:] = np.nan
            limits.append(peak_idx)

        return cleaned, limits

    raise ValueError(f"Unknown mode: {mode}")


def sort_domain_and_curves(ua, *curve_sets):
    """
    Ensure UA domain is increasing, and apply same sorting to all curve sets.
    """
    order = np.argsort(ua)
    ua_sorted = ua[order]

    sorted_sets = []
    for curves in curve_sets:
        sorted_sets.append(curves[:, order])

    return (ua_sorted, *sorted_sets)


# ============================================================
# Residual construction
# ============================================================

def residuals(
    x,
    ua_domain,
    ia_targets,
    ig2_targets,
    ia_ug1_curves,
    ig2_ug1_curves,
    verbose_invalid=False,
):
    """
    Return a flat residual vector for scipy.optimize.least_squares.

    IA model:
        IA = A * Ii

    IG2 model:
        IG2 = (1 - A) * Ii
    """
    alpha, beta, K, mu_c, mu_s, exp_coeff = unpack_params(x)

    res_chunks = []

    A_all = model_A(alpha, beta, Ug2, ua_domain)

    # ----------------------------
    # IA residuals
    # ----------------------------
    for idx, ia_target in enumerate(ia_targets):
        ug1_curve = ia_ug1_curves[idx]

        mask = (
            np.isfinite(ua_domain)
            & np.isfinite(ug1_curve)
            & np.isfinite(A_all)
        )

        if np.count_nonzero(mask) < 2:
            if verbose_invalid:
                print(f"Skipping IA curve {idx}: not enough valid samples.")
            continue

        ua = ua_domain[mask]
        ug1 = ug1_curve[mask]
        A = A_all[mask]

        pred = A * model_Ii(
            K=K,
            mu_c=mu_c,
            mu_s=mu_s,
            exp_coeff=exp_coeff,
            ug1=ug1,
            ug2=Ug2,
            ua=ua,
        )

        # Normalize by target scale so all current curves get comparable weight.
        scale = max(abs(ia_target), 1.0)
        res_chunks.append((pred - ia_target) / scale)

    # ----------------------------
    # IG2 residuals
    # ----------------------------
    for idx, ig2_target in enumerate(ig2_targets):
        ug1_curve = ig2_ug1_curves[idx]

        mask = (
            np.isfinite(ua_domain)
            & np.isfinite(ug1_curve)
            & np.isfinite(A_all)
        )

        if np.count_nonzero(mask) < 2:
            if verbose_invalid:
                print(f"Skipping IG2 curve {idx}: not enough valid samples.")
            continue

        ua = ua_domain[mask]
        ug1 = ug1_curve[mask]
        A = A_all[mask]

        pred = (1.0 - A) * model_Ii(
            K=K,
            mu_c=mu_c,
            mu_s=mu_s,
            exp_coeff=exp_coeff,
            ug1=ug1,
            ug2=Ug2,
            ua=ua,
        )

        scale = max(abs(ig2_target), 1.0)
        res_chunks.append((pred - ig2_target) / scale)

    if not res_chunks:
        raise RuntimeError("No valid residuals were generated. Check NaN masks and data arrays.")

    return np.concatenate(res_chunks)


# ============================================================
# Prediction helpers
# ============================================================

def predict_ia_curve(params_physical, ua_domain, ug1_curve):
    alpha, beta, K, mu_c, mu_s, exp_coeff = params_physical

    A = model_A(alpha, beta, Ug2, ua_domain)
    Ii = model_Ii(K, mu_c, mu_s, exp_coeff, ug1_curve, Ug2, ua_domain)

    return A * Ii


def predict_ig2_curve(params_physical, ua_domain, ug1_curve):
    alpha, beta, K, mu_c, mu_s, exp_coeff = params_physical

    A = model_A(alpha, beta, Ug2, ua_domain)
    Ii = model_Ii(K, mu_c, mu_s, exp_coeff, ug1_curve, Ug2, ua_domain)

    return (1.0 - A) * Ii


# ============================================================
# Plotting
# ============================================================

def plot_fit_results(
    ua_domain,
    ia_targets,
    ig2_targets,
    ia_ug1_curves,
    ig2_ug1_curves,
    params_physical,
):
    """
    Plot model-predicted current against target current for each curve.

    Since each extracted curve is an UG1-vs-UA contour for a fixed current,
    the target current is a horizontal line. The fitted model evaluated along
    the same UG1 curve should be flat and equal to that target if the fit is good.
    """

    # ----------------------------
    # IA family
    # ----------------------------
    plt.figure(figsize=(10, 6))

    for idx, ia_target in enumerate(ia_targets):
        ug1_curve = ia_ug1_curves[idx]
        mask = np.isfinite(ug1_curve)

        pred = predict_ia_curve(
            params_physical,
            ua_domain[mask],
            ug1_curve[mask],
        )

        plt.plot(
            ua_domain[mask],
            pred,
            label=f"fit IA={ia_target:g}",
        )

        plt.plot(
            ua_domain[mask],
            ia_target * np.ones(np.count_nonzero(mask)),
            "--",
            linewidth=1,
            label=f"data IA={ia_target:g}",
        )

    plt.xlabel("UA")
    plt.ylabel("IA")
    plt.title("IA curves: fitted model vs extracted datasheet values")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    # ----------------------------
    # IG2 family
    # ----------------------------
    plt.figure(figsize=(10, 6))

    for idx, ig2_target in enumerate(ig2_targets):
        ug1_curve = ig2_ug1_curves[idx]
        mask = np.isfinite(ug1_curve)

        pred = predict_ig2_curve(
            params_physical,
            ua_domain[mask],
            ug1_curve[mask],
        )

        plt.plot(
            ua_domain[mask],
            pred,
            label=f"fit IG2={ig2_target:g}",
        )

        plt.plot(
            ua_domain[mask],
            ig2_target * np.ones(np.count_nonzero(mask)),
            "--",
            linewidth=1,
            label=f"data IG2={ig2_target:g}",
        )

    plt.xlabel("UA")
    plt.ylabel("IG2")
    plt.title("IG2 curves: fitted model vs extracted datasheet values")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    # ----------------------------
    # Residual-style plots
    # ----------------------------
    plt.figure(figsize=(10, 6))

    for idx, ia_target in enumerate(ia_targets):
        ug1_curve = ia_ug1_curves[idx]
        mask = np.isfinite(ug1_curve)

        pred = predict_ia_curve(
            params_physical,
            ua_domain[mask],
            ug1_curve[mask],
        )

        err = pred - ia_target

        plt.plot(
            ua_domain[mask],
            err,
            label=f"IA={ia_target:g}",
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("UA")
    plt.ylabel("IA fit error")
    plt.title("IA residuals")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    plt.figure(figsize=(10, 6))

    for idx, ig2_target in enumerate(ig2_targets):
        ug1_curve = ig2_ug1_curves[idx]
        mask = np.isfinite(ug1_curve)

        pred = predict_ig2_curve(
            params_physical,
            ua_domain[mask],
            ug1_curve[mask],
        )

        err = pred - ig2_target

        plt.plot(
            ua_domain[mask],
            err,
            label=f"IG2={ig2_target:g}",
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("UA")
    plt.ylabel("IG2 fit error")
    plt.title("IG2 residuals")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    plt.show()


def plot_ug1_input_curves(ua_domain, ia_ug1_curves, ig2_ug1_curves):
    """
    Optional sanity-check plot showing the extracted UG1 curves.
    """

    plt.figure(figsize=(10, 6))
    for i, row in enumerate(ia_ug1_curves):
        mask = np.isfinite(row)
        plt.plot(ua_domain[mask], row[mask], label=f"IA curve {i}")
    plt.xlabel("UA")
    plt.ylabel("UG1")
    plt.title("Extracted IA UG1-vs-UA curves")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for i, row in enumerate(ig2_ug1_curves):
        mask = np.isfinite(row)
        plt.plot(ua_domain[mask], row[mask], label=f"IG2 curve {i}")
    plt.xlabel("UA")
    plt.ylabel("UG1")
    plt.title("Extracted IG2 UG1-vs-UA curves")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.show()


# ============================================================
# Main optimization
# ============================================================

def main():
    ua_domain, ia_ug1_curves, ig2_ug1_curves = load_curve_arrays(
        IA_CSV_PATH,
        IG2_CSV_PATH,
    )

    ua_domain, ia_ug1_curves, ig2_ug1_curves = sort_domain_and_curves(
        ua_domain,
        ia_ug1_curves,
        ig2_ug1_curves,
    )

    # Convert limited IG2 regions to NaN.
    # This replaces manual slicing using max_ig2_idx.
    ig2_ug1_curves, ig2_limits = clean_limited_domain_curves(
        ig2_ug1_curves,
        mode="auto_argmax",
    )

    print("Loaded arrays:")
    print(f"  UA domain shape: {ua_domain.shape}")
    print(f"  IA UG1 curves shape: {ia_ug1_curves.shape}")
    print(f"  IG2 UG1 curves shape: {ig2_ug1_curves.shape}")
    print(f"  IG2 valid limits: {ig2_limits}")

    # Optional sanity check before fitting
    plot_ug1_input_curves(ua_domain, ia_ug1_curves, ig2_ug1_curves)

    # Pack parameters for optimizer
    x0 = pack_params(params_0_physical)

    # Bounds in optimizer space:
    # x = [alpha, beta, log_K, mu_c, mu_s, exp_coeff]
    lower_bounds = np.array([
        0.0,            # alpha
        1e-6,           # beta
        np.log(1e-15),  # log_K
        0.0,            # mu_c
        0.0,            # mu_s
        0.5,            # exp_coeff
    ])

    upper_bounds = np.array([
        2.0,            # alpha
        10.0,           # beta
        np.log(1e-6),   # log_K
        5000.0,         # mu_c
        5000.0,         # mu_s
        3.0,            # exp_coeff
    ])

    kwargs = dict(
        fun=residuals,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        args=(
            ua_domain,
            ia_current_vals,
            ig2_current_vals,
            ia_ug1_curves,
            ig2_ug1_curves,
        ),
        x_scale="jac",
        max_nfev=MAX_NFEV,
        verbose=2,
    )

    if USE_ROBUST_LOSS:
        kwargs["loss"] = "soft_l1"
        kwargs["f_scale"] = 1.0

    result = sp.optimize.least_squares(**kwargs)

    opt_params_physical = np.array(unpack_params(result.x))

    print("\nOptimization result:")
    print(f"  success: {result.success}")
    print(f"  status:  {result.status}")
    print(f"  message: {result.message}")
    print(f"  cost:    {result.cost}")
    print(f"  nfev:    {result.nfev}")

    names = ["alpha", "beta", "K", "mu_c", "mu_s", "exp_coeff"]

    print("\nParameter comparison:")
    for name, p0, popt in zip(names, params_0_physical, opt_params_physical):
        print(f"  {name:10s}: initial = {p0: .8e}, optimized = {popt: .8e}")

    # Save optimized parameters
    np.savetxt(
        "optimized_params.csv",
        opt_params_physical[None, :],
        delimiter=",",
        header="alpha,beta,K,mu_c,mu_s,exp_coeff",
        comments="",
    )

    print("\nSaved optimized parameters to optimized_params.csv")

    plot_fit_results(
        ua_domain=ua_domain,
        ia_targets=ia_current_vals,
        ig2_targets=ig2_current_vals,
        ia_ug1_curves=ia_ug1_curves,
        ig2_ug1_curves=ig2_ug1_curves,
        params_physical=opt_params_physical,
    )


if __name__ == "__main__":
    main()
