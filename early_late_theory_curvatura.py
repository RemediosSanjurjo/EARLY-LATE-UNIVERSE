"""Theory de Cobaya para el modelo Early/Late Dark Energy con curvatura.

Omega_DE(a) = Omega_E0 * a^[-3(1+w_E)] + Omega_L0, con curvatura Omega_k0.
"""

import numpy as np
import camb
from camb import model
from camb.dark_energy import DarkEnergyFluid
from cobaya.theory import Theory


class EarlyLateTheoryCurvatura(Theory):
    """Omega_DE(a)=Omega_E0*a^[-3(1+w_E)] + Omega_L0, con curvatura Omega_k0."""

    params = {
        "H0": None, "ombh2": None, "omch2": None, "tau": None, "As": None, "ns": None,
        "mnu": None, "nnu": None, "Omega_E0": None, "w_E": None, "Omega_k0": None,
        "Omega_m0_derived": {"derived": True},
        "Omega_r0_derived": {"derived": True},
        "Omega_L0_derived": {"derived": True},
    }

    lmax, TCMB, a_min, n_a = 2501, 2.7255, 1e-5, 5000

    def initialize(self):
        self.requested_lmax = self.lmax

    def get_can_provide(self):
        return ["Cl"]

    def get_can_provide_params(self):
        return ["Omega_m0_derived", "Omega_r0_derived", "Omega_L0_derived"]

    def must_provide(self, **requirements):
        if "Cl" in requirements:
            lmax_req = 0
            for spec, lmax_spec in requirements["Cl"].items():
                if isinstance(lmax_spec, int):
                    lmax_req = max(lmax_req, lmax_spec)
            self.requested_lmax = max(self.requested_lmax, lmax_req)
        return {}

    def _background_numbers(self, H0, ombh2, omch2, mnu, nnu):
        h = H0 / 100.0
        Omega_b0, Omega_c0, Omega_nu0 = ombh2 / h**2, omch2 / h**2, (mnu / 93.14) / h**2
        Omega_m0 = Omega_b0 + Omega_c0 + Omega_nu0
        omega_gamma_h2 = 2.4728e-5 * (self.TCMB / 2.7255)**4
        omega_nu_rel_h2 = omega_gamma_h2 * 0.22710731766 * nnu
        Omega_r0 = (omega_gamma_h2 + omega_nu_rel_h2) / h**2
        return Omega_m0, Omega_r0

    def _w_eff_table(self, Omega_E0, w_E, Omega_L0):
        a_table = np.logspace(np.log10(self.a_min), 0.0, self.n_a)
        X_E, X_L = Omega_E0 * a_table**(-3.0 * (1.0 + w_E)), Omega_L0
        w_table = -1.0 + ((1.0 + w_E) * X_E) / (X_E + X_L)
        return a_table, w_table

    def calculate(self, state, want_derived=True, **params):
        H0, ombh2, omch2, tau = params["H0"], params["ombh2"], params["omch2"], params["tau"]
        As, ns, mnu, nnu = params["As"], params["ns"], params["mnu"], params["nnu"]
        Omega_E0, w_E, Omega_k0 = params["Omega_E0"], params["w_E"], params["Omega_k0"]

        Omega_m0, Omega_r0 = self._background_numbers(H0, ombh2, omch2, mnu, nnu)
        Omega_DE0 = 1.0 - Omega_m0 - Omega_r0 - Omega_k0
        Omega_L0 = Omega_DE0 - Omega_E0

        if Omega_L0 <= 0 or w_E < -1.0:
            return False

        a_table, w_table = self._w_eff_table(Omega_E0, w_E, Omega_L0)
        if not np.all(np.isfinite(w_table)):
            return False

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=Omega_k0,
                           tau=tau, mnu=mnu, TCMB=self.TCMB, nnu=nnu)
        pars.InitPower.set_params(As=As, ns=ns, r=0.0)

        de = DarkEnergyFluid()
        de.set_w_a_table(a_table, w_table)
        pars.DarkEnergy = de

        pars.set_for_lmax(self.requested_lmax, lens_potential_accuracy=1)
        pars.NonLinear, pars.WantTensors = model.NonLinear_none, False

        try:
            results = camb.get_results(pars)
            cls_lensed = results.get_lensed_scalar_cls(
                lmax=self.requested_lmax, CMB_unit="muK", raw_cl=True)
        except Exception:
            return False
        ell = np.arange(cls_lensed.shape[0])

        state["Cl"] = {"ell": ell, "tt": cls_lensed[:, 0], "ee": cls_lensed[:, 1],
                       "bb": cls_lensed[:, 2], "te": cls_lensed[:, 3]}

        if want_derived:
            state["derived"] = {"Omega_m0_derived": Omega_m0,
                                "Omega_r0_derived": Omega_r0,
                                "Omega_L0_derived": Omega_L0}

    def get_Cl(self, ell_factor=False, units="muK2"):
        Cl = self.current_state["Cl"]
        out = {key: Cl[key].copy() for key in ["ell", "tt", "ee", "bb", "te"]}

        if units in ["K2", "FIRASK2"]:
            for key in ["tt", "ee", "bb", "te"]:
                out[key] = out[key] / 1e12

        if ell_factor:
            factor = out["ell"] * (out["ell"] + 1.0) / (2.0 * np.pi)
            for key in ["tt", "ee", "bb", "te"]:
                out[key] = out[key] * factor

        return out
