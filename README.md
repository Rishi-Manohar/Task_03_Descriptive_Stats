# Facebook Political Ad Data Aggregation

This repository contains three scripts for summarizing and analyzing Facebook political advertisement data. Each script demonstrates a different data processing approach using either core Python, Pandas, or Polars.

---

## 📒 Script Descriptions

### 1. `pure_python_stats.py`
- Utilizes basic Python (no external libraries) to:
  - Load ad data from a CSV file.
  - Perform aggregations:
    - Count of unique ads by `page_id`
    - Frequency of each `(page_id, ad_id)` pair
  - Outputs two CSV files:
    - `aggregated_by_page.csv`

### 2. `pandas_stats.py`
- Uses the Pandas library to:
  - Import and explore the dataset
  - Generate summary statistics and insights

### 3. `polars_stats.py`
- Leverages the Polars library for high-performance processing:
  - Performs similar reporting as `pandas_stats.py`
  - Offers improved speed and efficiency for larger datasets

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.7 or higher

### Installation
Install the necessary libraries by running:

```bash
pip install pandas polars jupyter


