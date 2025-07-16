# %%
import numpy as np 
import pandas as pd 
import polars as pl
import matplotlib.pyplot as plt 
import csv
import os 
import math 

from collections import defaultdict, Counter

# %%
os.chdir(r"C:\Research Task 4")

# %%
fb_ads = pd.read_csv(r"2024_fb_ads_president_scored_anon.csv")
fb_posts = pd.read_csv(r"2024_fb_posts_president_scored_anon.csv")
tw_posts = pd.read_csv(r"2024_tw_posts_president_scored_anon.csv")

# %%
def analyze_dataframe_with_pandas(df):
    # Numeric summary stats
    numeric_stats = df.describe().T.reset_index().rename(columns={"index": "Column"})
    numeric_stats["Type"] = "Numeric"

    # Non-numeric summary stats
    non_numeric_stats = []
    for col in df.select_dtypes(include=['object']).columns:
        value_counts = df[col].value_counts(dropna=True)
        most_freq_val = value_counts.index[0] if not value_counts.empty else None
        most_freq_count = value_counts.iloc[0] if not value_counts.empty else None
        non_numeric_stats.append({
            "Column": col,
            "Type": "Non-Numeric",
            "Count": df[col].count(),
            "Unique Values": df[col].nunique(),
            "Most Frequent": most_freq_val,
            "Frequency": most_freq_count
        })

    df_non_numeric_stats = pd.DataFrame(non_numeric_stats)

    return numeric_stats, df_non_numeric_stats

# %%
ads_num, ads_non_num = analyze_dataframe_with_pandas(fb_ads)
posts_num, posts_non_num = analyze_dataframe_with_pandas(fb_posts)
twitter_num, twitter_non_num = analyze_dataframe_with_pandas(tw_posts)

# %%
ads_num

# %%
ads_non_num

# %%
posts_num

# %%
posts_non_num 

# %%
twitter_num

# %%
twitter_non_num
