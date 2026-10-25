"""Funciones auxiliares del TFM: evaluación/minimización con Cobaya y cálculo de C_l con CAMB.

Uso básico (en Colab o en local):

    import funciones as f
    f.set_likelihood_paths("/content/planck_2018")   # carpeta donde están los .clik
    chi2, partes, tabla = f.evaluar_modelo_early_late_curvatura(
        H0=73.5, Omega_E0=1e-4, w_E=0.2, Omega_k0=0.0, likelihood="lite")
"""

import glob
import logging
import os
from contextlib import redirect_stderr, redirect_stdout

import camb
import numpy as np
import pandas as pd
from camb import model
from camb.dark_energy import DarkEnergyFluid
from cobaya.run import run

# ---------------------------------------------------------------------------
# Parámetros de referencia (Planck 2018, baseline) y constantes
# ---------------------------------------------------------------------------
ombh2 = 0.02236
omch2 = 0.1202
tau = 0.0544
As = 2.101e-9
ns = 0.9649
mnu = 0.06
TCMB = 2.7255
Neff = 3.046

# Rutas a los ficheros .clik (se rellenan con set_likelihood_paths)
plik_TTTEEE_file = None
lowl_TT_file = None
lowl_EE_file = None
plik_lite_file = None


# ---------------------------------------------------------------------------
# Rutas de la likelihood de Planck
# ---------------------------------------------------------------------------
def find_one_file(pattern, base_dir):
    files = glob.glob(base_dir + "/**/" + pattern, recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"No he encontrado ningún archivo con patrón: {pattern}")
    if len(files) > 1:
        print(f"Hay varios archivos para {pattern}. Uso el primero:")
        for f in files:
            print(f)
    return files[0]


def set_likelihood_paths(planck_dir):
    """Busca los .clik dentro de planck_dir y los guarda como variables del módulo."""
    global plik_TTTEEE_file, lowl_TT_file, lowl_EE_file, plik_lite_file
    plik_TTTEEE_file = find_one_file("plik_rd12_HM_v22b_TTTEEE.clik", planck_dir)
    lowl_TT_file = find_one_file("commander_dx12_v3_2_29.clik", planck_dir)
    lowl_EE_file = find_one_file("simall_100x143_offlike5_EE_Aplanck_B.clik", planck_dir)
    plik_lite_file = find_one_file("plik_lite_v22_TTTEEE.clik", planck_dir)


def _likelihood_block(likelihood):
    if likelihood == "highl":
        return {"planck_2018_highl_plik.TTTEEE": {"clik_file": plik_TTTEEE_file}}
    if likelihood == "lowl":
        return {"planck_2018_lowl.TT_clik": {"clik_file": lowl_TT_file},
                "planck_2018_lowl.EE_clik": {"clik_file": lowl_EE_file}}
    if likelihood == "lite":
        return {"planck_2018_highl_plik.TTTEEE_lite": {"clik_file": plik_lite_file}}
    if likelihood == "baseline":
        return {"planck_2018_highl_plik.TTTEEE": {"clik_file": plik_TTTEEE_file},
                "planck_2018_lowl.TT_clik": {"clik_file": lowl_TT_file},
                "planck_2018_lowl.EE_clik": {"clik_file": lowl_EE_file}}
    raise ValueError("likelihood debe ser 'highl', 'lowl', 'lite' o 'baseline'.")


# ---------------------------------------------------------------------------
# Utilidades de Cobaya
# ---------------------------------------------------------------------------
def producto_cobaya_a_dataframe(obj):
    """Convierte objetos de Cobaya (SampleCollection, OnePoint...) a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame().T

    for attr in ("data", "_data"):
        if hasattr(obj, attr):
            data = getattr(obj, attr)
            if isinstance(data, pd.DataFrame):
                return data
            if isinstance(data, pd.Series):
                return data.to_frame().T
            if isinstance(data, np.ndarray):
                return pd.DataFrame(data)

    try:
        return pd.DataFrame(obj)
    except Exception:
        raise TypeError("No he podido convertir el producto de Cobaya a DataFrame.")


def extraer_chi2(sampler):
    """Devuelve (chi2_total, chi2_por_likelihood, tabla) del primer punto del sampler."""
    products = sampler.products()

    if "sample" in products:
        tabla = producto_cobaya_a_dataframe(products["sample"])
    elif "minimum" in products:
        tabla = producto_cobaya_a_dataframe(products["minimum"])
    else:
        raise KeyError(f"No encuentro ni 'sample' ni 'minimum'. Claves: {products.keys()}")

    fila = tabla.iloc[0]
    chi2_total = float(fila["chi2"])
    chi2_partes = {c: float(fila[c]) for c in tabla.columns if c.startswith("chi2__")}
    return chi2_total, chi2_partes, tabla


def _info_modelo(H0, Omega_E0, w_E, Omega_k0, likelihood, sampler, output, debug):
    params_modelo = {"H0": H0, "ombh2": ombh2, "omch2": omch2, "tau": tau,
                     "As": As, "ns": ns, "mnu": mnu, "nnu": Neff,
                     "Omega_E0": Omega_E0, "w_E": w_E, "Omega_k0": Omega_k0,
                     "Omega_m0_derived": {"derived": True},
                     "Omega_r0_derived": {"derived": True},
                     "Omega_L0_derived": {"derived": True}}
    if likelihood == "lite":
        params_modelo["A_planck"] = 1

    return {"theory": {"early_late_theory_curvatura.EarlyLateTheoryCurvatura": {
                "lmax": 2508, "TCMB": TCMB, "a_min": 1e-5, "n_a": 5000,
                "stop_at_error": True}},
            "likelihood": _likelihood_block(likelihood),
            "params": params_modelo,
            "sampler": sampler,
            "debug": debug,
            "output": output}


def evaluar_modelo_early_late_curvatura(H0, Omega_E0, w_E, Omega_k0,
                                        likelihood="highl", output=None, debug=False):
    """Evalúa chi2 del modelo en un punto fijo (sampler 'evaluate')."""
    info = _info_modelo(H0, Omega_E0, w_E, Omega_k0, likelihood,
                        {"evaluate": None}, output, debug)

    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        _, sampler = run(info, force=True)
    logging.disable(logging.NOTSET)
    return extraer_chi2(sampler)


def minimize_modelo_early_late_cobaya_curvatura(H0, Omega_E0, w_E, Omega_k0, likelihood,
                                                output=None, debug=False):
    """Minimiza chi2 respecto a los nuisance parameters, con el modelo fijo."""
    if likelihood == "lowl":
        raise ValueError("Para minimizar usa 'highl', 'lite' o 'baseline'.")
    info = _info_modelo(H0, Omega_E0, w_E, Omega_k0, likelihood,
                        {"minimize": {"method": "scipy", "ignore_prior": False,
                                      "max_evals": 10000}},
                        output, debug)

    logging.disable(logging.CRITICAL)
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        _, sampler = run(info, force=True)
    logging.disable(logging.NOTSET)
    return extraer_chi2(sampler)


# ---------------------------------------------------------------------------
# Modelo Early/Late: w efectiva y parámetros de CAMB
# ---------------------------------------------------------------------------
def w_lambda_eff(a, Omega_E0, w_E, Omega_L0):
    """Ecuación de estado efectiva de la componente oscura Early + Late."""
    a = np.asarray(a, dtype=float)
    X_E = Omega_E0 * a**(-3.0 * (1.0 + w_E))
    X_L = Omega_L0
    return -1.0 + ((1.0 + w_E) * X_E) / (X_E + X_L)


def _omegas_fondo(H0, ombh2_, omch2_, mnu_, TCMB_, Neff_):
    h = H0 / 100.0
    Omega_m0 = ombh2_ / h**2 + omch2_ / h**2 + (mnu_ / 93.14) / h**2
    omega_gamma_h2 = 2.4728e-5 * (TCMB_ / 2.7255)**4
    omega_nu_rel_h2 = omega_gamma_h2 * 0.22710731766 * Neff_
    Omega_r0 = (omega_gamma_h2 + omega_nu_rel_h2) / h**2
    return Omega_m0, Omega_r0


def camb_params_early_late_curvo(Omega_E0, w_E, Omega_k0, H0,
                                 ombh2=ombh2, omch2=omch2, tau=tau, As=As, ns=ns,
                                 mnu=mnu, TCMB=TCMB, Neff=Neff,
                                 lmax=2499, a_min=1e-5, n_a=50000):
    """Construye los CAMBparams del modelo early/late con curvatura.

    Omega_DE0 = 1 - Omega_m0 - Omega_r0 - Omega_k0
    Omega_L0  = Omega_DE0 - Omega_E0
    """
    Omega_m0, Omega_r0 = _omegas_fondo(H0, ombh2, omch2, mnu, TCMB, Neff)
    Omega_DE0 = 1.0 - Omega_m0 - Omega_r0 - Omega_k0
    Omega_L0 = Omega_DE0 - Omega_E0

    if Omega_L0 <= 0:
        raise ValueError(f"Omega_L0 = {Omega_L0:.5e} <= 0.")
    if w_E < -1:
        raise ValueError("Régimen phantom w_E < -1")

    a_table = np.logspace(np.log10(a_min), 0.0, n_a).astype(float)
    w_table = np.asarray(w_lambda_eff(a_table, Omega_E0, w_E, Omega_L0), dtype=float)
    if not np.all(np.isfinite(w_table)):
        raise ValueError("La tabla w(a) contiene valores no finitos.")

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=Omega_k0, tau=tau,
                       mnu=mnu, TCMB=TCMB, nnu=Neff)
    pars.InitPower.set_params(As=As, ns=ns, r=0)

    de = DarkEnergyFluid()
    de.set_w_a_table(a_table, w_table)
    pars.DarkEnergy = de

    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.NonLinear = model.NonLinear_none
    pars.WantTensors = False

    info = {"Omega_m0": Omega_m0, "Omega_r0": Omega_r0, "Omega_k0": Omega_k0,
            "Omega_DE0": Omega_DE0, "Omega_E0": Omega_E0, "Omega_L0": Omega_L0,
            "w_E": w_E, "a_table": a_table, "w_table": w_table}
    return pars, info


def get_Cl_early_late_curvo(Omega_E0, w_E, Omega_k0, H0=73.5, lmax=2499):
    """C_l (TT, EE, BB, TE) lensados, en muK^2 sin factor l(l+1)/2pi."""
    pars, info = camb_params_early_late_curvo(Omega_E0=Omega_E0, w_E=w_E,
                                              Omega_k0=Omega_k0, H0=H0, lmax=lmax)
    results = camb.get_results(pars)
    cls_lensed = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit="muK", raw_cl=True)
    ell = np.arange(cls_lensed.shape[0])
    return (ell, cls_lensed[:, 0], cls_lensed[:, 1], cls_lensed[:, 2],
            cls_lensed[:, 3], info)


def E2_model(Omega_E0, w_E, Omega_k0, H0, a,
             ombh2=ombh2, omch2=omch2, mnu=mnu, TCMB=TCMB, Neff=Neff):
    """E^2(a) = H^2(a)/H0^2 del modelo early/late con curvatura."""
    Omega_m0, Omega_r0 = _omegas_fondo(H0, ombh2, omch2, mnu, TCMB, Neff)
    Omega_L0 = 1.0 - Omega_m0 - Omega_r0 - Omega_k0 - Omega_E0

    return (Omega_r0 * a**(-4)
            + Omega_m0 * a**(-3)
            + Omega_k0 * a**(-2)
            + Omega_E0 * a**(-3 * (1 + w_E))
            + Omega_L0)
