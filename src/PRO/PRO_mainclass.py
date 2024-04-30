# Notes for Jacopo:
''' 

'''

# Define the module path + import packages
import sys
module_path = '/Users/jacoponudo/Documents/thesis/src/PRO'
sys.path.append(module_path)
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PRO_package import *
from scipy import stats
import random 
from scipy.stats import chi2

root='/Users/jacoponudo/Documents/thesis/'



# Variables
n_bin = 21
bin_threshold = 50
social_media_name = "voat"
thread_identifier = "root_submission"
labeled_folder = os.path.join("data", social_media_name)

input_filename = os.path.join(labeled_folder, f"{social_media_name}_labeled_data_unified.parquet")
output_filename = f"{social_media_name}_toxicity_percentage_binned.parquet"
output_threshold_filename = f"{social_media_name}_comments_with_bin_ge_07.parquet"

output_folder = "/Users/jacoponudo/Documents/thesis/src/PRO/output"
comments_with_binned_threshold_folder = "data/Processed/BinFiltering"
output_format = "parquet"

# Load data
df_data = load_data_frame('/Users/jacoponudo/Documents/thesis/data/voat/voat_labeled_data_unified.parquet')

# Preprocessing

df_data["toxicity_score"].fillna(0, inplace=True)
df_data["is_toxic"] = df_data["toxicity_score"] > 0.6

df_data = df_data.drop(columns=["text"])

# Compute thread length and toxicity percentage
df = df_data.groupby([thread_identifier, "topic"]).agg(thread_length=("is_toxic", "count"),
                                                      toxicity_percentage=("is_toxic", "mean")).reset_index()
df = df.sort_values(by=["thread_length", "topic"])

# Binning for thread length
df["discretized_bin_label"] = df.groupby("topic")["thread_length"].apply(lambda x: pd.qcut(x, q=n_bin, labels=False))

# Resize last bin
df_binned = pd.DataFrame()

for topic in df["topic"].unique():
    print("Topic:", topic)
    aux = resize_last_bin(df[df["topic"] == topic])
    df_binned = pd.concat([df_binned, aux])

# Compute mean and confidence interval for toxicity percentage
df_toxicity_percentage_binned = df_binned.groupby(["topic", "resize_discretized_bin_label"])\
                                         .agg(mean_t=("toxicity_percentage", "mean"),
                                              CI_toxicity=("toxicity_percentage", compute_CI))

# Filter data with discretized bin >= 0.7
df_data_with_bin_after_threshold = df_binned[df_binned["resize_discretized_bin_label"] >= 0.7]

# Write results
output_filename = os.path.join(output_folder, f"{output_filename}.{output_format}")
df_toxicity_percentage_binned.to_parquet(output_filename)
print("Result can be found in", output_filename, "file")

output_threshold_filename = os.path.join(comments_with_binned_threshold_folder, f"{output_threshold_filename}.{output_format}")
df_data_with_bin_after_threshold.to_parquet(output_threshold_filename)
print("Comments with discretized bin >= 0.7 can be found in", output_threshold_filename, "file")
