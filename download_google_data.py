import os
import urllib.request
import tarfile
import pandas as pd

print("=" * 60)
print("DOWNLOADING GOOGLE CLUSTER TRACE DATASET")
print("=" * 60)

# Google hosts the data here
url = "https://storage.googleapis.com/cluster-data-v3/clusterdata-2011-2.tar.gz"
filename = "clusterdata-2011-2.tar.gz"

print(f"\nDownloading from: {url}")
print("(This is ~200MB, may take a few minutes...)\n")

if not os.path.exists(filename):
    urllib.request.urlretrieve(url, filename)
    print("\n✓ Download complete!")
else:
    print(f"✓ {filename} already exists, skipping download")

# Extract
print(f"\nExtracting {filename}...")
if not os.path.exists("clusterdata-2011-2"):
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()
    print("✓ Extraction complete!")
else:
    print("✓ Already extracted")

print("\n" + "=" * 60)
print("EXPLORING THE DATA")
print("=" * 60)

# Read resource_usage file (this is the one with CPU)
print("\nReading resource_usage.csv...")
df = pd.read_csv("clusterdata-2011-2/resource_usage.csv")

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("CPU USAGE STATISTICS")
print("=" * 60)

if 'cpu_rate' in df.columns:
    print("\nCPU Rate (cores):")
    print(df['cpu_rate'].describe())
    print(f"\nMin: {df['cpu_rate'].min()}")
    print(f"Max: {df['cpu_rate'].max()}")
    print(f"Mean: {df['cpu_rate'].mean()}")
    print(f"Median: {df['cpu_rate'].median()}")

if 'memory_rate' in df.columns:
    print("\nMemory Rate:")
    print(df['memory_rate'].describe())

print("\n" + "=" * 60)
print("DATA INFO")
print("=" * 60)
print(df.info())

print("\n✓ Data exploration complete!")
