import os
import zipfile
import io
import csv
import pandas as pd
import statsmodels.api as sm


from gdc.utils import data_path, ExtendedNamespace

CMS_DATA_PATH = data_path('cms_data.txt')


def get_zip_files():
    zip_files = [os.path.join(CMS_DATA_PATH, f)
                 for f in os.listdir(CMS_DATA_PATH)
                 if f.lower().endswith(".zip")]
    return ExtendedNamespace(
        **{z.split('/')[-1].rstrip('.zip'): z for z in zip_files})


ZF = get_zip_files()


def get_zipfile_metrics(zip_path):
    """
    Return (column_names, num_columns, num_rows) for a CSV inside a ZIP.
    Counts rows via streamed iteration (safe for very large files).
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))

        with z.open(csv_name, "r") as f:
            text_stream = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.reader(text_stream)

            # Read header
            try:
                header = next(reader)
            except StopIteration:
                return [], 0, 0

            num_cols = len(header)

            # Count rows line-by-line
            num_rows = sum(1 for _ in reader)

    return ExtendedNamespace(
        **{'columns': header, 'num_cols': num_cols, 'num_rows': num_rows})


def zip_chunk_generator(zip_path, batch_size=None, max_batches=None, usecols=None):
    zip_path = os.path.join(CMS_DATA_PATH, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        # detect inner CSV file
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))

        with z.open(csv_name) as f:
            # wrap raw bytes → text
            text_stream = io.TextIOWrapper(f, encoding="utf-8", newline="")

            if batch_size is None:
                df = pd.read_csv(
                    text_stream,
                    low_memory=False,
                    usecols=usecols,
                )
                yield df
                return


            # streamed chunk reader
            reader = pd.read_csv(
                text_stream,
                chunksize=batch_size,
                low_memory=False,
                usecols=usecols,
            )

            for i, chunk in enumerate(reader):
                yield chunk

                if max_batches is not None and (i + 1) >= max_batches:
                    break


df_beneficiaries_2008 = next(zip_chunk_generator(ZF.DE1_0_2008_Beneficiary_Summary_File_Sample_1))
df_beneficiaries_2009 = next(zip_chunk_generator(ZF.DE1_0_2009_Beneficiary_Summary_File_Sample_1))
df_beneficiaries_2010 = next(zip_chunk_generator(ZF.DE1_0_2010_Beneficiary_Summary_File_Sample_1))

list_df_beneficiaries = [df_beneficiaries_2008,
                         df_beneficiaries_2009,
                         df_beneficiaries_2010]

for df, y in zip(list_df_beneficiaries, [2008, 2009, 2010]):
    df['Year'] = y


def load_codebook():
    df = pd.read_csv(os.path.join(CMS_DATA_PATH, 'code_book.csv'))

    mapping = dict(zip(df["variable"], df["name"]))
    return ExtendedNamespace(**mapping)


def age_in_years(bdate_yyyymmdd: int, ref_year: int) -> int:
    y = bdate_yyyymmdd // 10_000
    m = (bdate_yyyymmdd // 100) % 100

    return (ref_year - y) - (m - 1) / 12


CB = load_codebook()

for df in list_df_beneficiaries:
    df['age'] = df[['Year', CB.date_birth]].apply(
        lambda r: age_in_years(
            r[CB.date_birth], r['Year']), axis=1)


binary_characteristics = [
    c for c in CB.__dict__.keys() if c.startswith('cc')]

medicare_payment = [CB.medicare_reimb_car, CB.medicare_reimb_ip, CB.medicare_reimb_op]

numerical = ['months_partA', 'months_partB', 'months_hmo',
             'months_partD']

cat_cols = [CB[c] for c in binary_characteristics] + [
    CB.sex]

list_df_beneficiaries_w_dummies = []

for df in list_df_beneficiaries:
    df_w_dummies = df.copy()
    df_w_dummies = pd.get_dummies(
        df_w_dummies,
        columns=cat_cols,
        drop_first=True,
        dtype=float
    )
    df_w_dummies.set_index(CB.patient_id, inplace=True)
    df_w_dummies['payments'] = df_w_dummies[medicare_payment].sum(axis=1)
    list_df_beneficiaries_w_dummies.append(df_w_dummies)

df_beneficiaries_2008_w_dummies = list_df_beneficiaries_w_dummies[0]
df_beneficiaries_2009_w_dummies = list_df_beneficiaries_w_dummies[1]
df_beneficiaries_2010_w_dummies = list_df_beneficiaries_w_dummies[2]

common_patients = set(df_beneficiaries_2008_w_dummies.index) & \
                  set(df_beneficiaries_2009_w_dummies.index) & \
                  set(df_beneficiaries_2010_w_dummies.index)
common_patients = list(common_patients)

# get payments for common patients only, append year to payment col
df_merged_payments = pd.DataFrame({
    'payments_2008': df_beneficiaries_2008_w_dummies['payments'],
    'payments_2009': df_beneficiaries_2009_w_dummies['payments'],
    'payments_2010': df_beneficiaries_2010_w_dummies['payments'],
})

cols_to_drop = ['payments', 'Year', CB.beneficiary_resp_car, CB.beneficiary_resp_ip, CB.beneficiary_resp_op,
                CB.medicare_reimb_car, CB.medicare_reimb_ip, CB.medicare_reimb_op, CB.county_code,
                CB.date_birth, CB.date_death, CB.race, CB.state_code, CB.primary_payer_car, CB.primary_payer_ip, CB.primary_payer_op,
               CB.months_hmo, CB.months_partA, CB.months_partB]

# get other characteristics for common patients, append year to col names,
# merge into df_merged_covariates
for year, df in zip([2008, 2009, 2010], list_df_beneficiaries_w_dummies):
    df_common = df.copy()
    df_common = df_common.drop(columns=cols_to_drop)
    df_common.columns = [f"{c}_{year}" for c in df_common.columns]
    if year == 2008:
        df_merged_covariates = df_common
    else:
        df_merged_covariates = df_merged_covariates.join(df_common, how='outer')


def keep_cols_except_age(df):
    return [c for c in df.columns if c not in cols_to_drop and c != 'age']


def get_y_x():
    dic_out = {}
    y2008 = df_beneficiaries_2008_w_dummies['payments']
    x2008 = df_beneficiaries_2008_w_dummies.drop(columns=cols_to_drop)
    dic_out['year_2008'] = ExtendedNamespace(Y=y2008, X=x2008)

    y2009 = df_beneficiaries_2009_w_dummies['payments']
    x2009 = df_beneficiaries_2009_w_dummies.drop(columns=cols_to_drop)
    # concatenate x2008[ref_cols] -- add L to eac col label
    ref_cols = keep_cols_except_age(x2008)
    x2008_L = x2008[ref_cols].copy()
    x2008_L.columns = [f"{c}_L" for c in x2008_L.columns]
    #x2009 = x2009.join(x2008_L, how='left')
    dic_out['year_2009'] = ExtendedNamespace(Y=y2009, X=x2009)

    y2010 = df_beneficiaries_2010_w_dummies['payments']
    x2010 = df_beneficiaries_2010_w_dummies.drop(columns=cols_to_drop)
    # concatenate x2009[ref_cols] -- add L to eac col label
    ref_cols = keep_cols_except_age(x2009)
    x2009_L = x2009[ref_cols].copy()
    x2009_L.columns = [f"{c}_L" for c in x2009_L.columns]
    #x2010 = x2010.join(x2009_L, how='left')
    dic_out['year_2010'] = ExtendedNamespace(Y=y2010, X=x2010)

    return ExtendedNamespace(**dic_out)


yx_by = get_y_x()


def get_resids():
    dic_resids = {}
    dic_res = {}
    for year in ['year_2008', 'year_2009', 'year_2010']:
        y = yx_by[year].Y
        X = yx_by[year].X
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing='drop')
        res = model.fit()
        dic_resids[year] = res.resid
        dic_res[year] = res
    return dic_resids, dic_res

dic_resids, dic_res = get_resids()

df_resids = pd.DataFrame(dic_resids)
