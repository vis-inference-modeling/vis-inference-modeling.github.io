"""Export a slim UMAP CSV (umap_1, umap_2, class, path) for the explorer app."""
from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CSV_PATH = OUT_DIR / "umap_2d.csv"
SLIM_PATH = OUT_DIR / "umap_2d_slim.csv"

def main():
    df = pd.read_csv(CSV_PATH, usecols=["umap_1", "umap_2", "class", "path"])
    df.to_csv(SLIM_PATH, index=False)
    print(f"Wrote {SLIM_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()
