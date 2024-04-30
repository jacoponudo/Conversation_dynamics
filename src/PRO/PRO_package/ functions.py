import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from scipy.stats import ttest_1samp

# Functions
def load_data_frame(input_filename):
    input_format = input_filename.split(".")[-1].lower()
    if input_format == "csv":
        df_data = pd.read_csv(input_filename)
    elif input_format == "parquet":
        df_data = pq.read_table(input_filename).to_pandas()
    else:
        print("Error: Unsupported input format")
        return None
    return df_data

def compute_CI(x):
    result = ttest_1samp(x, popmean=np.mean(x))
    return result.confidence_interval

