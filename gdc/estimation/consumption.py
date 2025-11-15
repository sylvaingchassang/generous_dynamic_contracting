import numpy as np
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

        self.month = y.index.month.values  # (nT,)
        self.m_idx = self.month - 1  # 0..11
        self.nT, self.nI = y.shape
        self.months = np.arange(12)

        self.n_dates, self.n_cons = self.Yv.shape
        idx = np.arange(self.n_dates)
        self.hod = idx % 24  # 0..23
        self.dow = (idx // 24) % 7

    @abstractmethod
    def _single_var_demean_func(self, v):
        # returns v_within, v_means
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

    def print_coeffs_and_forecast_metrics(self,
            beta,
            alpha_m=None,  # optional pure month FE (12,)
            alpha_im=None,  # (12,N) i×month
            alpha_ih=None,  # (24,N) i×hour (if using i×hour model)
            Hy0=None,  # (24,N) or (24,1) centered pooled hour
            Dy0=None,  # (7,N) or (24,1)
            alpha_id=None,  # (7,N)  i×DOW (if using i×DOW model)
            rho=None,
            label="Model"
    ):
        Yv, HDDv, CDDv, m_idx = self.Yv, self.HDDv, self.CDDv, self.m_idx
        mu = np.zeros_like(Yv, dtype=float)

        if alpha_m is not None:
            mu += alpha_m[m_idx, None]

        if alpha_im is not None:
            mu += alpha_im[m_idx, :]

        if alpha_ih is not None:
            mu += alpha_ih[self.hod, :]
        elif Hy0 is not None:
            mu += Hy0[self.hod, :]

        if alpha_id is not None:
            mu += alpha_id[self.dow, :]
        elif Dy0 is not None:
            mu += Dy0[self.dow, :]

        mu += beta[0] * HDDv + beta[1] * CDDv

        # --- static fit ---
        resid_s = Yv - mu
        sse_s = float(np.sum(resid_s ** 2))
        sst = float(np.sum((Yv - Yv.mean()) ** 2))
        r2_s = 1.0 - sse_s / sst
        rmse_s = float(np.sqrt(sse_s / Yv.size))

        # --- dynamic one-step fit (if rho provided) ---
        if rho is not None:
            yhat_dyn = mu.copy()
            yhat_dyn[1:, :] += rho * (Yv[:-1, :] - mu[:-1, :])
            resid_d = Yv - yhat_dyn
            sse_d = float(np.sum(resid_d ** 2))
            r2_d = 1.0 - sse_d / sst
            rmse_d = float(np.sqrt(sse_d / Yv.size))
        else:
            r2_d = rmse_d = None

        # --- return summary as dict ---
        summary = {
            "label": label,
            "beta_HDD": float(beta[0]),
            "beta_CDD": float(beta[1]),
            "static_fit": {
                "r2": float(r2_s),
                "rmse": float(rmse_s)
            }
        }

        if rho is not None:
            summary["dynamic_fit"] = {
                "rho": float(rho),
                "r2": float(r2_d),
                "rmse": float(rmse_d)
            }

        return summary


class PooledSeasonalUncorrelatedErrors(ConsumptionModel):

    def _single_var_demean_func(self, v):
        v_m = np.array(
            [v[self.m_idx == k, :].mean() for k in self.months])  # (12, 1)
        v_dow = np.vstack(
            [v[self.dow == d, :].mean() for d in range(7)])  # (7, 1)
        v_h = np.vstack(
            [v[self.hod == h, :].mean() for h in range(24)])  # (24,1)
        v_h0 = v_h - v_h.mean(axis=0, keepdims=True)
        v_dow0 = v_dow - v_dow.mean(axis=0, keepdims=True)
        v_within = (v - v_m[self.m_idx, None]
                    - v_h0[self.hod, None] - v_dow0[self.dow, None])
        return v_within, (v_m, v_dow0, v_h0)  # month is not demeaned -- replaces alphas


class IndividualSeasonalUncorrelatedErrors(ConsumptionModel):

    def _single_var_demean_func(self, v):
        v_im = np.array(
            [v[self.m_idx == k, :].mean(axis=0) for k in
             self.months])  # (12,N)
        v_idow = np.vstack(
            [v[self.dow == d, :].mean(axis=0) for d in range(7)])  # (7,N)
        v_ih = np.vstack([v[self.hod == h, :].mean(axis=0) for h in
                          range(24)])  # (24,N)
        v_ih0 = v_ih - v_ih.mean(axis=0, keepdims=True)
        v_idow0 = v_idow - v_idow.mean(axis=0, keepdims=True)
        v_within = (v - v_im[self.m_idx, :]
                    - v_ih0[self.hod, :] - v_idow0[self.dow, :])
        return v_within, (v_im, v_idow0, v_ih0)
