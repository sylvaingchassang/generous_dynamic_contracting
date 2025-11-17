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

    def  fit(self):
        (y_within, hdd_within, cdd_within), means = self.demeaned_variables()
        beta = self._estimate_beta(y_within, hdd_within, cdd_within)
        return beta, means

    def predict_static_means(self, beta, means, hddv=None, cddv=None):
        if hddv is None:
            hddv = self.HDDv
        if cddv is None:
            cddv = self.CDDv

        mu = np.zeros_like(hddv, dtype=float)
        mu += sum(means.y_means)
        demeaned_hddv = hddv - sum(means.hdd_means)
        demeaned_cddv = cddv - sum(means.cdd_means)
        mu += beta[0] * demeaned_hddv + beta[1] * demeaned_cddv

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

    def _single_var_demean_func(self, v):
        v_m = np.array(
            [v[self.m_idx == k, :].mean() for k in self.months])  # (12, 1)
        v_dow = np.vstack(
            [v[self.dow == d, :].mean() for d in range(7)])  # (7, 1)
        v_h = np.vstack(
            [v[self.hod == h, :].mean() for h in range(24)])  # (24,1)
        v_h0 = v_h - v_h.mean(axis=0, keepdims=True)
        v_dow0 = v_dow - v_dow.mean(axis=0, keepdims=True)
        v_m = v_m[self.m_idx, None]
        v_h0 = v_h0[self.hod, None]
        v_dow0 = v_dow0[self.dow, None]
        v_within = v - v_m - v_h0 - v_dow0
        return v_within, (v_m, v_dow0, v_h0)  # month is not demeaned -- replaces alphas


class IndividualMDHUncorrelatedErrors(ConsumptionModel):

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
        v_im = v_im[self.m_idx, :]
        v_ih0 = v_ih0[self.hod, :]
        v_idow0 = v_idow0[self.dow, :]
        v_within = v - v_im - v_ih0 - v_idow0
        return v_within, (v_im, v_idow0, v_ih0)
