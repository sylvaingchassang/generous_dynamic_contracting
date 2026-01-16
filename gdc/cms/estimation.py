import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

from gdc.cms.data_access import *
from gdc.utils import filter_data, is_greater_than

def drop_na_yX(y, X):
    idx = X.dropna().index.intersection(y.dropna().index)
    return y.loc[idx], X.loc[idx]


dic_prediction = {}
for year in [2008, 2009, 2010]:
    year = str(year)
    y = df_merged_payments[relevant_cols(year, df_merged_payments)]
    X = sm.add_constant(df_merged_covariates[relevant_cols(year)])
    y, X = drop_na_yX(y, X)

    l1linear = sm.QuantReg(y, X, missing='drop').fit(q=.5)
    l1RF = RandomForestRegressor(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=50,      # critical for cost data
            max_features="sqrt",
            n_jobs=-1,
            random_state=123,
            oob_score=True,
            criterion="absolute_error"
            ).fit(X, y)
    dic_prediction[year] = pd.DataFrame({
        'y': y, 'l1linear': l1linear.predict(X),
        'l1RF': l1RF.predict(X)
    })
