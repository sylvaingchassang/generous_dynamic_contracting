import numpy as np
from gdc.data_access import (df_temp_simulated_normalized,
                             df_load_simulated_normalized)


class ConsumptionModel:
    def __init__(self, Y=df_load_simulated_normalized,
                 T=df_temp_simulated_normalized):
        tau_h, tau_c = 15.0, 20.0
        HDD = (tau_h - T).clip(lower=0)
        CDD = (T - tau_c).clip(lower=0)
        self.Yv = Y.to_numpy(dtype=np.float32, copy=False)
        self.HDDv = HDD.to_numpy(dtype=np.float32, copy=False)
        self.CDDv = CDD.to_numpy(dtype=np.float32, copy=False)

        self.month = Y.index.month.values  # (nT,)
        self.m_idx = self.month - 1  # 0..11
        self.nT, self.nI = Y.shape
        self.months = np.arange(12)

        self.n_dates, self.n_cons = self.Yv.shape
        idx = np.arange(self.n_dates)
        self.hod = idx % 24  # 0..23
        self.dow = (idx // 24) % 7

    def individual_centered_day_hour_means(self):
        """
        Returns centered individual Day Hour intercept tables:
          Hy0: (24, N) hour-of-day FE, column-centered (sum over 24 = 0 per consumer)
          Dy0: ( 7, N) day-of-week FE, column-centered (sum over 7  = 0 per consumer)
        """
        Hy = np.vstack([self.Yv[self.hod == h, :].mean(axis=0)
                        for h in range(24)])  # (24,N) mean consumption at each hour
        Dy = np.vstack([self.Yv[self.dow == d, :].mean(axis=0)
                        for d in range(7)])   # (7,N) mean consumption at each DOW
        Hy0 = Hy - Hy.mean(axis=0, keepdims=True) # center columns
        Dy0 = Dy - Dy.mean(axis=0, keepdims=True) # center columns
        return Hy0, Dy0

    def pooled_centered_day_hour_means(self):
        """
        Returns centered individual Day Hour intercept tables:
          Hy0: (24, 1) hour-of-day FE, column-centered (sum over 24 = 0 overall)
          Dy0: ( 7, 1) day-of-week FE, column-centered (sum over 7  = 0 overall)
        """
        Hy = np.vstack([self.Yv[self.hod == h, :].mean()
                        for h in range(24)]).reshape(-1,1)  # (24,1) mean consumption at each hour across all consumers
        Dy = np.vstack([self.Yv[self.dow == d, :].mean()
                        for d in range(7)]).reshape(-1,1)  # (7,1) mean consumption at each DOW across all consumers
        Hy0 = Hy - Hy.mean(axis=0, keepdims=True)  # center columns
        Dy0 = Dy - Dy.mean(axis=0, keepdims=True)  # center columns
        return Hy0, Dy0

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

    def fit(self):
        # centered pooled seasonal intercepts
        Hy0, Dy0 = self.get_seasonal_effects()

        # pooled month means (scalars per month)
        mY = np.array([self.Yv[self.m_idx == k, :].mean() for k in self.months])
        mH = np.array([self.HDDv[self.m_idx == k, :].mean() for k in self.months])
        mC = np.array([self.CDDv[self.m_idx == k, :].mean() for k in self.months])

        # within by month on Y and X; subtract pooled seasonal from Y only
        Yw = self.Yv - mY[self.m_idx, None] - Hy0[self.hod, :] - Dy0[self.dow, :]
        HDDw = self.HDDv - mH[self.m_idx, None]
        CDDw = self.CDDv - mC[self.m_idx, None]

        # 2×2 normal equations
        Shh = np.einsum('ij,ij->', HDDw, HDDw)
        Scc = np.einsum('ij,ij->', CDDw, CDDw)
        Shc = np.einsum('ij,ij->', HDDw, CDDw)
        Shy = np.einsum('ij,ij->', HDDw, Yw)
        Scy = np.einsum('ij,ij->', CDDw, Yw)
        det = Shh * Scc - Shc * Shc
        beta_A = np.array(
            [(Shy * Scc - Scy * Shc) / det, (-Shy * Shc + Scy * Shh) / det],
            dtype=np.float64)

        # month intercepts (on original scale)
        alpha_m = mY - (beta_A[0] * mH + beta_A[1] * mC)

        seasonal = {"Hy0": Hy0, "Dy0": Dy0}
        return alpha_m, beta_A, seasonal

    def get_seasonal_effects(self):
        return self.pooled_centered_day_hour_means()


class IndividualSeasonalUncorrelatedErrors(PooledSeasonalUncorrelatedErrors):

    def get_seasonal_effects(self):
        return self.individual_centered_day_hour_means()