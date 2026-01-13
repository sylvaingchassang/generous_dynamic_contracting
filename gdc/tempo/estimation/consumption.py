import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings

from gdc.tempo.data_access import (df_temp_simulated_normalized,
                                   df_load_simulated_normalized)
from gdc.utils import ExtendedNamespace
from gdc.tempo.estimation.random_variables import RandomVariable


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

    def predict_dynamic_means(self, beta, means, error_model_):
        mu = self.predict_static_means(beta, means)
        yhat_dyn = mu.copy()
        phi_ = error_model_.coeffs
        max_lag = error_model_.lags.max()
        for lag, phi_lag in zip(error_model_.lags, phi_):
            yhat_dyn[max_lag:, :] += phi_lag * (
                self.Yv[max_lag - lag:-lag, :] - mu[max_lag - lag:-lag, :])
        return yhat_dyn

    def dynamic_resids(self, beta, means, error_model_):
        yhat_dyn = self.predict_dynamic_means(beta, means, error_model_)
        resids_d = self.Yv - yhat_dyn
        return resids_d

    def summary(self, beta, means, error_model_=None, label="Model"):

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

        if error_model_ is not None:
            dynamic_resids = self.dynamic_resids(beta, means, error_model_)
            dynamic_sse = float(np.sum(dynamic_resids ** 2))
            r2_d = 1.0 - dynamic_sse / sst
            rmse_d = float(np.sqrt(dynamic_sse / self.Yv.size))
            summary["dynamic_fit"] = {
                "lags": error_model_.lags.tolist(),
                "coeffs": error_model_.coeffs.tolist(),
                "r2": float(r2_d),
                "rmse": float(rmse_d)
            }

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

    def _estimate_coeffs(self, y, x):
        phi_, *_ = np.linalg.lstsq(x, y, rcond=None)
        innovs_ = y - x @ phi_
        return phi_, innovs_

    def _estimate_gaussian_innovations(self, innovs_):
        sigma2 = (innovs_ @ innovs_) / len(innovs_)
        return sigma2

    def _estimate_sample_dist(self, innovs_, num_quantiles=100):

        sorted_innovs = np.sort(innovs_)
        quantiles = np.linspace(1./num_quantiles,
                                1 - 1./num_quantiles,
                                num_quantiles)
        values = np.quantile(sorted_innovs, quantiles)
        innov_rv = RandomVariable(quantiles, values)
        return innov_rv

    def _check_stationarity(self, phi_):
        Lmax = self.lags.max()
        coeffs = np.zeros(Lmax)
        for lag, phi in zip(self.lags, phi_):
            coeffs[lag - 1] = phi

        # Polynomial: 1 - φ₁ z - ... - φ_p zᵖ
        poly = np.flip(np.concatenate(([1.0], -coeffs)))
        roots = np.roots(poly)
        is_stationary = np.all(np.abs(roots) > 1)
        if not is_stationary:
            warnings.warn("Error Process is NOT Stationary")
        return is_stationary

    def fit(self, num_quantiles=100):
        y, x = self._build_yx()
        phi_, innovs_ = self._estimate_coeffs(y, x)
        sigma2_ = self._estimate_gaussian_innovations(innovs_)
        innov_rv_ = self._estimate_sample_dist(
            innovs_, num_quantiles=num_quantiles)
        self._check_stationarity(phi_)
        return ExtendedNamespace(
            lags=self.lags, coeffs=phi_, sigma2=sigma2_,
            sample_innov=innov_rv_)