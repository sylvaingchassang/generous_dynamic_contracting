import numpy as np
import pandas as pd
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
