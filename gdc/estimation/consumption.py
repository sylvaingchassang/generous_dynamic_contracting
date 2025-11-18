import numpy as np
import pandas as pd
from scipy import optimize, special
from abc import ABC, abstractmethod

from gdc.data_access import (df_temp_simulated_normalized,
                             df_load_simulated_normalized)
from gdc.utils import ExtendedNamespace


class ConsumptionModel(ABC):
    def __init__(self, y=df_load_simulated_normalized,
                 temp=df_temp_simulated_normalized):
        tau_h, tau_c = 15.0, 20.0
        heating_degree_days = (tau_h - temp).clip(lower=0)
        cooling_degree_days = (temp - tau_c).clip(lower=0)
        self.Yv = y.to_numpy(dtype=np.float32, copy=False)
        self.HDDv = heating_degree_days.to_numpy(dtype=np.float32, copy=False)
        self.CDDv = cooling_degree_days.to_numpy(dtype=np.float32, copy=False)

        self.m_idx, self.dow, self.hod = self._get_m_d_h(y.index)

    def _get_m_d_h(self, index: pd.DatetimeIndex):
        m_idx = index.month.values -1 # 0..11
        dow = index.dayofweek.values # 0..6
        hod = index.hour.values # 0..23
        return m_idx, dow, hod

    @abstractmethod
    def _single_var_demean_func(self, v):
        # returns v_within, v_means
        pass

    @abstractmethod
    def _broadcast_means(self, means, indices=None):
        pass

    def demeaned_variables(self):
        y_within, y_means = self._single_var_demean_func(self.Yv)
        hdd_within, hdd_means = self._single_var_demean_func(self.HDDv)
        cdd_within, cdd_means = self._single_var_demean_func(self.CDDv)
        return ((y_within, hdd_within, cdd_within),
                ExtendedNamespace(
                    y_means=y_means,
                    hdd_means=hdd_means,
                    cdd_means=cdd_means)
                )

    def _estimate_beta(self, y_within, hdd_within, cdd_within):
        # 2×2 normal equations
        s_hh = np.einsum('ij,ij->', hdd_within, hdd_within)
        s_cc = np.einsum('ij,ij->', cdd_within, cdd_within)
        s_hc = np.einsum('ij,ij->', hdd_within, cdd_within)
        s_hy = np.einsum('ij,ij->', hdd_within, y_within)
        s_cy = np.einsum('ij,ij->', cdd_within, y_within)
        det = s_hh * s_cc - s_hc * s_hc
        beta = np.array(
            [(s_hy * s_cc - s_cy * s_hc) / det,
             (-s_hy * s_hc + s_cy * s_hh) / det],
            dtype=np.float64)
        return beta

    def fit(self):
        (y_within, hdd_within, cdd_within), means = self.demeaned_variables()
        beta = self._estimate_beta(y_within, hdd_within, cdd_within)
        return beta, means

    def predict_static_means(self, beta, means, hddv=None, cddv=None):
        if hddv is not None:
            indices = self._get_m_d_h(hddv.index)
        else:
            indices = None
        if hddv is None:
            hddv = self.HDDv
        if cddv is None:
            cddv = self.CDDv

        mu = np.zeros_like(hddv, dtype=float)
        mu += sum(self._broadcast_means(means.y_means, indices))
        demeaned_hddv = hddv - sum(
            self._broadcast_means(means.hdd_means, indices))
        demeaned_cddv = cddv - sum(self._broadcast_means(means.cdd_means,
                                                         indices))
        mu += beta[0] * demeaned_hddv + beta[1] * demeaned_cddv
        return mu

    def static_resids(self, beta, means):
        resids = self.Yv - self.predict_static_means(beta, means)
        return resids

    def summary(self, beta, means, label="Model"):

        static_resids = self.static_resids(beta, means)
        static_sse = float(np.sum(static_resids ** 2))
        sst = float(np.sum((self.Yv - self.Yv.mean()) ** 2))
        r2 = 1.0 - static_sse / sst
        rmse = float(np.sqrt(static_sse / self.Yv.size))

        summary = {
            "label": label,
            "beta_HDD": float(beta[0]),
            "beta_CDD": float(beta[1]),
            "static_fit": {
                "r2": float(r2),
                "rmse": float(rmse)
            }
        }

        # --- dynamic one-step fit (if rho provided) ---
        # if rho is not None:
        #     yhat_dyn = mu.copy()
        #     yhat_dyn[1:, :] += rho * (Yv[:-1, :] - mu[:-1, :])
        #     resid_d = Yv - yhat_dyn
        #     sse_d = float(np.sum(resid_d ** 2))
        #     r2_d = 1.0 - sse_d / sst
        #     rmse_d = float(np.sqrt(sse_d / Yv.size))
        # else:
        #     r2_d = rmse_d = None

        # --- return summary as dict ---
        #
        #
        # if rho is not None:
        #     summary["dynamic_fit"] = {
        #         "rho": float(rho),
        #         "r2": float(r2_d),
        #         "rmse": float(rmse_d)
        #     }

        return summary


class PooledMDHUncorrelatedErrors(ConsumptionModel):

    def _broadcast_means(self, means, indices=None):
        if indices is None:
            indices = self.m_idx, self.dow, self.hod
        m_idx, dow, hod = indices
        v_m, v_dow, v_h = means
        v_m_b = v_m[m_idx, None]
        v_dow_b = v_dow[dow, None]
        v_h_b = v_h[hod, None]
        return v_m_b, v_dow_b, v_h_b

    def _single_var_demean_func(self, v):
        v_m = np.array(
            [v[self.m_idx == k, :].mean() for k in range(12)])  # (12, 1)
        v_dow = np.array(
            [v[self.dow == d, :].mean() for d in range(7)])  # (7, 1)
        v_h = np.array(
            [v[self.hod == h, :].mean() for h in range(24)])  # (24,1)
        v_h0 = v_h - v_h.mean(axis=0, keepdims=True)
        v_dow0 = v_dow - v_dow.mean(axis=0, keepdims=True)
        means = (v_m, v_dow0, v_h0)
        v_within = v - sum(self._broadcast_means(means))
        return v_within, means  # month is not demeaned -- replaces alphas


class IndividualMDHUncorrelatedErrors(ConsumptionModel):

    def _broadcast_means(self, means, indices=None):
        if indices is None:
            indices = self.m_idx, self.dow, self.hod
        m_idx, dow, hod = indices
        v_m, v_dow, v_h = means
        v_m_b = v_m[m_idx, :]
        v_dow_b = v_dow[dow, :]
        v_h_b = v_h[hod, :]
        return v_m_b, v_dow_b, v_h_b

    def _single_var_demean_func(self, v):
        v_im = np.array(
            [v[self.m_idx == k, :].mean(axis=0) for k in range(12)])  # (12,N)
        v_idow = np.array(
            [v[self.dow == d, :].mean(axis=0) for d in range(7)])  # (7,N)
        v_ih = np.array([v[self.hod == h, :].mean(axis=0) for h in
                          range(24)])  # (24,N)
        v_ih0 = v_ih - v_ih.mean(axis=0, keepdims=True)
        v_idow0 = v_idow - v_idow.mean(axis=0, keepdims=True)
        means = (v_im, v_idow0, v_ih0)
        v_within = v - sum(self._broadcast_means(means))
        return v_within, (v_im, v_idow0, v_ih0)


class ARErrorModel:
    def __init__(self, resids, lags=(1, 24)):
        """
        resids : (T, N)
        lags   : list of AR lags
        """
        self.resids = np.asarray(resids)
        self.lags = np.array(lags, dtype=int)

        self.phi_ = None
        self.sigma2_ = None
        self.nu_ = None
        self.dist_ = None

    # ------------------------------------------------------------
    # Build flattened y and X by stacking consumers vertically
    # ------------------------------------------------------------
    def _build_yx(self):
        r = self.resids               # shape (num_dates, num_cons)
        num_dates, num_cons = r.shape
        max_lags = self.lags.max()

        y = r[max_lags:, :]
        xlags = [r[max_lags - lag: num_dates - lag, :] for lag in self.lags]

        # Flatten y (column-major = per consumer stacking)
        y = y.reshape(-1)

        # Build X = [lag1 | lag2 | …], then flatten rows
        x = np.stack(xlags, axis=2)
        x = x.reshape(-1, len(self.lags))

        # Remove rows with NaNs (rare but necessary)
        ok = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
        return y[ok], x[ok, :]

    # ------------------------------------------------------------
    # Gaussian estimation (closed form)
    # ------------------------------------------------------------
    def _estimate_normal(self, y, x):
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        eps = y - x @ beta
        sigma = np.sqrt((eps @ eps) / len(eps))
        return beta, sigma

    # ------------------------------------------------------------
    # Student-t estimation (MLE)
    # ------------------------------------------------------------
    def _estimate_student(self, y, x, nu_init):
        n, k = x.shape

        # OLS initialization
        beta0, *_ = np.linalg.lstsq(x, y, rcond=None)
        eps0 = y - x @ beta0
        sigma20 = (eps0 @ eps0) / n
        nu0 = nu_init

        # parameter vector: [beta_k, log_sigma2, log(nu-2)]
        theta0 = np.concatenate([
            beta0,
            [np.log(sigma20)],
            [np.log(max(nu0 - 2, 1e-3))]
        ])

        def neg_loglik(theta):
            beta = theta[:k]
            sigma2 = np.exp(theta[k])
            nu = 2.0 + np.exp(theta[k + 1])

            eps = y - x @ beta
            z = eps**2 / (nu * sigma2)

            c = (special.gammaln((nu + 1) / 2)
                 - special.gammaln(nu / 2)
                 - 0.5 * np.log(nu * np.pi * sigma2))

            ll = n * c - 0.5 * (nu + 1) * np.sum(np.log1p(z))
            return -ll

        res = optimize.minimize(neg_loglik, theta0, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError("Student-t optimization did not converge")

        theta = res.x
        beta = theta[:k]
        sigma2 = np.exp(theta[k])
        nu = 2.0 + np.exp(theta[k + 1])

        return beta, sigma2, nu

    # ------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------
    def estimate_rhos(self, dist="normal", nu_init=8.0):
        y, X = self._build_yX()
        y = y - y.mean()

        if dist == "normal":
            beta, sigma2 = self._estimate_normal(y, X)
            self.phi_ = beta
            self.sigma2_ = sigma2
            self.nu_ = None
            self.dist_ = "normal"

        elif dist == "student":
            beta, sigma2, nu = self._estimate_student(y, X, nu_init)
            self.phi_ = beta
            self.sigma2_ = sigma2
            self.nu_ = nu
            self.dist_ = "student"

        else:
            raise ValueError("dist must be 'normal' or 'student'")

        return {
            "phi": self.phi_,
            "sigma2": self.sigma2_,
            "nu": self.nu_,
            "dist": self.dist_
        }

    # ------------------------------------------------------------
    # Stationarity check
    # ------------------------------------------------------------
    def check_stationarity(self):
        if self.phi_ is None:
            raise RuntimeError("Call estimate_rhos first")

        Lmax = self.lags.max()
        coeffs = np.zeros(Lmax)
        for lag, phi in zip(self.lags, self.phi_):
            coeffs[lag - 1] = phi

        # Polynomial: 1 - φ₁ z - ... - φ_p zᵖ
        poly = np.concatenate(([1.0], -coeffs))
        roots = np.roots(poly)
        return np.all(np.abs(roots) > 1), roots
