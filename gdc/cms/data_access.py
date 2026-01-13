import os
import zipfile
import io
import csv
import pandas as pd

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

