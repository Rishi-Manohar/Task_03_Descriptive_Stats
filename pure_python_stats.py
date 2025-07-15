# %%
import numpy as np 
import pandas as pd 
import polars as pl
import matplotlib.pyplot as plt 
import csv
import os 
import math 
import pandas as pd

from collections import defaultdict, Counter

# %%
os.chdir("D:\# %%
import numpy as np 
import pandas as pd 
import polars as pl
import matplotlib.pyplot as plt 
import csv
import os 
import math 
import pandas as pd

from collections import defaultdict, Counter

# %%
os.chdir("C:\Research Task 4")

# %% [markdown]
# ### Using Python

# %%
def infer_column_dtypes(filepath):
    type_counts = defaultdict(lambda: defaultdict(int))
    null_counts_base = defaultdict(int)
    row_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row in reader:
            row_count += 1
            for i, value in enumerate(row):
                if i >= len(headers):
                    continue  # skip malformed rows
                col = headers[i]
                value = value.strip()

                if value == "" or value.lower() == "nan":
                    null_counts_base[col] += 1
                else:
                    try:
                        int(value)
                        type_counts[col]["int"] += 1
                    except ValueError:
                        try:
                            float(value)
                            type_counts[col]["float"] += 1
                        except ValueError:
                            type_counts[col]["str"] += 1

    inferred_dtypes = {}
    for col, counts in type_counts.items():
        inferred_dtypes[col] = max(counts, key=counts.get)

    # Build a DataFrame
    df_summary = pd.DataFrame({
        "Column": list(inferred_dtypes.keys()),
        "Inferred Type": list(inferred_dtypes.values()),
        "Null Count": [null_counts_base.get(col, 0) for col in inferred_dtypes.keys()]
    })

    return df_summary

# %%
df_fb_ads_info = infer_column_dtypes("2024_fb_ads_president_scored_anon.csv")
df_fb_ads_info

# %%
df_fb_posts_info = infer_column_dtypes("2024_fb_posts_president_scored_anon.csv")
df_fb_posts_info

# %%
df_tw_posts_info = infer_column_dtypes("2024_tw_posts_president_scored_anon.csv")
df_tw_posts_info

# %%
def analyze_csv_basic(filepath):
    column_data = defaultdict(list)
    numeric_columns = set()
    non_numeric_columns = set()

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for row in reader:
            for col in headers:
                value = row[col]
                if value is None or value.strip() == "" or value.strip().lower() == "nan":
                    continue
                value = value.strip()
                try:
                    val = float(value)
                    column_data[col].append(val)
                    numeric_columns.add(col)
                except ValueError:
                    column_data[col].append(value)
                    non_numeric_columns.add(col)

    # Numeric stats
    numeric_stats = []
    for col in numeric_columns:
        data = [float(x) for x in column_data[col]
                if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).replace('-', '', 1).isdigit()]
        count = len(data)
        mean = sum(data) / count if count else None
        min_val = min(data) if data else None
        max_val = max(data) if data else None
        stddev = math.sqrt(sum((x - mean) ** 2 for x in data) / count) if count else None
        numeric_stats.append({
            "Column": col,
            "Type": "Numeric",
            "Count": count,
            "Mean": round(mean, 2) if mean is not None else None,
            "Min": min_val,
            "Max": max_val,
            "Std Dev": round(stddev, 2) if stddev is not None else None
        })

    # Non-numeric stats
    non_numeric_stats = []
    for col in non_numeric_columns:
        data = column_data[col]
        count = len(data)
        unique_vals = len(set(data))
        most_common = Counter(data).most_common(1)[0] if data else (None, None)
        non_numeric_stats.append({
            "Column": col,
            "Type": "Non-Numeric",
            "Count": count,
            "Unique Values": unique_vals,
            "Most Frequent": most_common[0],
            "Frequency": most_common[1]
        })

    return numeric_stats, non_numeric_stats


# %%
num_stats, non_num_stats = analyze_csv_basic("2024_fb_ads_president_scored_anon.csv")
fb_ads_df = pd.DataFrame(num_stats + non_num_stats)
fb_ads_df

# %%
num_stats, non_num_stats = analyze_csv_basic("2024_fb_posts_president_scored_anon.csv")
fb_posts_df = pd.DataFrame(num_stats + non_num_stats)
fb_posts_df

# %%
num_stats, non_num_stats = analyze_csv_basic("2024_tw_posts_president_scored_anon.csv")
tw_posts_df = pd.DataFrame(num_stats + non_num_stats)
tw_posts_df

# %%
def aggregate_fb_ads_data(file_path):
    # Aggregation structures
    agg_by_page = defaultdict(set)     # page_id → set of ad_ids
    agg_by_page_ad = defaultdict(int)  # (page_id, ad_id) → count

    # Read and process the file
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            page_id = row['page_id']
            ad_id = row['ad_id']

            agg_by_page[page_id].add(ad_id)
            agg_by_page_ad[(page_id, ad_id)] += 1

    # Convert to pandas DataFrame: by page_id
    data_by_page = pd.DataFrame([
        {'page_id': page_id, 'unique_ads': len(ad_ids)}
        for page_id, ad_ids in agg_by_page.items()
    ])

    # Convert to pandas DataFrame: by page_id and ad_id
    data_by_page_ad = pd.DataFrame([
        {'page_id': page_id, 'ad_id': ad_id, 'count': count}
        for (page_id, ad_id), count in agg_by_page_ad.items()
    ])

    return data_by_page, data_by_page_ad

# %%
by_page, by_page_ad = aggregate_fb_ads_data('2024_fb_ads_president_scored_anon.csv')

# %%
by_page

# %%
by_page_ad

# %%
# Export to CSV
by_page.to_csv('aggregated_by_page.csv', index=False)
by_page_ad.to_csv('aggregated_by_page_ad.csv', index=False)

# %%
num_stats, non_num_stats = analyze_csv_basic("aggregated_by_page.csv")
by_page_df = pd.DataFrame(num_stats + non_num_stats)
by_page_df

# %%
num_stats, non_num_stats = analyze_csv_basic("aggregated_by_page_ad.csv")
by_page_ad_df = pd.DataFrame(num_stats + non_num_stats)
by_page_ad_df")

# %% [markdown]
# ### Using Python

# %%
def infer_column_dtypes(filepath):
    type_counts = defaultdict(lambda: defaultdict(int))
    null_counts_base = defaultdict(int)
    row_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row in reader:
            row_count += 1
            for i, value in enumerate(row):
                if i >= len(headers):
                    continue  # skip malformed rows
                col = headers[i]
                value = value.strip()

                if value == "" or value.lower() == "nan":
                    null_counts_base[col] += 1
                else:
                    try:
                        int(value)
                        type_counts[col]["int"] += 1
                    except ValueError:
                        try:
                            float(value)
                            type_counts[col]["float"] += 1
                        except ValueError:
                            type_counts[col]["str"] += 1

    inferred_dtypes = {}
    for col, counts in type_counts.items():
        inferred_dtypes[col] = max(counts, key=counts.get)

    # Build a DataFrame
    df_summary = pd.DataFrame({
        "Column": list(inferred_dtypes.keys()),
        "Inferred Type": list(inferred_dtypes.values()),
        "Null Count": [null_counts_base.get(col, 0) for col in inferred_dtypes.keys()]
    })

    return df_summary

# %%
df_fb_ads_info = infer_column_dtypes("2024_fb_ads_president_scored_anon.csv")
df_fb_ads_info

# %%
df_fb_posts_info = infer_column_dtypes("2024_fb_posts_president_scored_anon.csv")
df_fb_posts_info

# %%
df_tw_posts_info = infer_column_dtypes("2024_tw_posts_president_scored_anon.csv")
df_tw_posts_info

# %%
def analyze_csv_basic(filepath):
    column_data = defaultdict(list)
    numeric_columns = set()
    non_numeric_columns = set()

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for row in reader:
            for col in headers:
                value = row[col]
                if value is None or value.strip() == "" or value.strip().lower() == "nan":
                    continue
                value = value.strip()
                try:
                    val = float(value)
                    column_data[col].append(val)
                    numeric_columns.add(col)
                except ValueError:
                    column_data[col].append(value)
                    non_numeric_columns.add(col)

    # Numeric stats
    numeric_stats = []
    for col in numeric_columns:
        data = [float(x) for x in column_data[col]
                if isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).replace('-', '', 1).isdigit()]
        count = len(data)
        mean = sum(data) / count if count else None
        min_val = min(data) if data else None
        max_val = max(data) if data else None
        stddev = math.sqrt(sum((x - mean) ** 2 for x in data) / count) if count else None
        numeric_stats.append({
            "Column": col,
            "Type": "Numeric",
            "Count": count,
            "Mean": round(mean, 2) if mean is not None else None,
            "Min": min_val,
            "Max": max_val,
            "Std Dev": round(stddev, 2) if stddev is not None else None
        })

    # Non-numeric stats
    non_numeric_stats = []
    for col in non_numeric_columns:
        data = column_data[col]
        count = len(data)
        unique_vals = len(set(data))
        most_common = Counter(data).most_common(1)[0] if data else (None, None)
        non_numeric_stats.append({
            "Column": col,
            "Type": "Non-Numeric",
            "Count": count,
            "Unique Values": unique_vals,
            "Most Frequent": most_common[0],
            "Frequency": most_common[1]
        })

    return numeric_stats, non_numeric_stats


# %%
num_stats, non_num_stats = analyze_csv_basic("2024_fb_ads_president_scored_anon.csv")
fb_ads_df = pd.DataFrame(num_stats + non_num_stats)
fb_ads_df

# %%
num_stats, non_num_stats = analyze_csv_basic("2024_fb_posts_president_scored_anon.csv")
fb_posts_df = pd.DataFrame(num_stats + non_num_stats)
fb_posts_df

# %%
num_stats, non_num_stats = analyze_csv_basic("2024_tw_posts_president_scored_anon.csv")
tw_posts_df = pd.DataFrame(num_stats + non_num_stats)
tw_posts_df

# %%
def aggregate_fb_ads_data(file_path):
    # Aggregation structures
    agg_by_page = defaultdict(set)     # page_id → set of ad_ids
    agg_by_page_ad = defaultdict(int)  # (page_id, ad_id) → count

    # Read and process the file
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            page_id = row['page_id']
            ad_id = row['ad_id']

            agg_by_page[page_id].add(ad_id)
            agg_by_page_ad[(page_id, ad_id)] += 1

    # Convert to pandas DataFrame: by page_id
    data_by_page = pd.DataFrame([
        {'page_id': page_id, 'unique_ads': len(ad_ids)}
        for page_id, ad_ids in agg_by_page.items()
    ])

    # Convert to pandas DataFrame: by page_id and ad_id
    data_by_page_ad = pd.DataFrame([
        {'page_id': page_id, 'ad_id': ad_id, 'count': count}
        for (page_id, ad_id), count in agg_by_page_ad.items()
    ])

    return data_by_page, data_by_page_ad

# %%
by_page, by_page_ad = aggregate_fb_ads_data('2024_fb_ads_president_scored_anon.csv')

# %%
by_page

# %%
by_page_ad

# %%
# Export to CSV
by_page.to_csv('aggregated_by_page.csv', index=False)
by_page_ad.to_csv('aggregated_by_page_ad.csv', index=False)

# %%
num_stats, non_num_stats = analyze_csv_basic("aggregated_by_page.csv")
by_page_df = pd.DataFrame(num_stats + non_num_stats)
by_page_df

# %%
num_stats, non_num_stats = analyze_csv_basic("aggregated_by_page_ad.csv")
by_page_ad_df = pd.DataFrame(num_stats + non_num_stats)
by_page_ad_df
