# ============================================================
# MOBILE PRODUCT SEGMENTATION AND RECOMMENDATION SYSTEM
# Step 1: Data Loading & Initial Inspection
# ============================================================

import os
import pandas as pd

# ============================================================
# 1. DEFINE DATA PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Mobile Reviews Sentiment null.csv")

# ============================================================
# 2. CHECK WHETHER FILE EXISTS
# ============================================================
if not os.path.exists(DATA_PATH):
    print("Dataset file not found!")
    print("Expected location:", DATA_PATH)
    exit()

print("=" * 70)
print("DATASET FOUND")
print("=" * 70)

# ============================================================
# 3. LOAD CSV INTO DATAFRAME
# ============================================================
df = pd.read_csv(DATA_PATH)
print("\nDataset loaded successfully!")

# ============================================================
# 4. DISPLAY DATASET SHAPE
# ============================================================
print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ============================================================
# 5. DISPLAY COLUMN NAMES
# ============================================================
print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# ============================================================
# 6. DISPLAY FIRST 5 ROWS
# ============================================================
print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head())

# ============================================================
# 7. DISPLAY DATA TYPES
# ============================================================
print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)

# ============================================================
# 9. CHECK DUPLICATE ROWS
# ============================================================
print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)

# ============================================================
# 10. BASIC STATISTICAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe())

# ============================================================
# 11. UNIQUE VALUES IN IMPORTANT CATEGORICAL COLUMNS
# ============================================================
print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

categorical_columns = [
    "brand",
    "model",
    "currency",
    "sentiment",
    "country",
    "language",
    "verified_purchase",
    "source"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print("Number of unique values:", df[column].nunique())
    print("Sample values:", df[column].dropna().unique()[:10])

# ============================================================
# 12. END MESSAGE
# ============================================================
print("\n" + "=" * 70)
print("DATA LOADING AND INITIAL INSPECTION COMPLETED")
print("=" * 70)