"""
Visualize data visualizations (PNG) using DreamSim embeddings, UMAP, and clustering.

- Loads all .png images from the visualizations folder
- Encodes them with DreamSim (perceptual similarity model)
- Reduces to 2D with UMAP
- Assigns clusters via KMeans (default 10+ clusters)
- Plots 2D scatter colored by cluster; hover shows the .png image

Requires: pip install dreamsim umap-learn scikit-learn altair pandas torch pillow
"""

from pathlib import Path
import argparse
import base64
import io
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import umap
import altair as alt

# Optional: use GPU if available
import torch

# Project root (parent of analysis/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = PROJECT_ROOT / "visualizations"
OUT_DIR = PROJECT_ROOT / "analysis" / "outputs"


def get_png_paths(vis_dir: Path) -> List[Path]:
    """Return sorted list of .png paths in vis_dir (numeric where possible, else by name)."""
    paths = list(vis_dir.glob("*.png"))

    def sort_key(p: Path):
        if p.stem.isdigit():
            return (0, int(p.stem))
        return (1, p.name)

    return sorted(paths, key=sort_key)


def load_dreamsim(device: Optional[str] = None):
    """Load DreamSim model and preprocess function."""
    from dreamsim import dreamsim

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True, device=device)
    return model, preprocess, device


def embed_images(model, preprocess, paths: List[Path], device: str, batch_size: int = 16):
    """Compute DreamSim embeddings; returns (embeddings, paths_used) so indices align."""
    embeddings_list = []
    paths_used = []
    for p in paths:
        try:
            img = Image.open(p)
            if img.mode != "RGB":
                img = img.convert("RGB")
            x = preprocess(img).to(device)  # preprocess already returns (1, 3, 224, 224)
            with torch.no_grad():
                e = model.embed(x)
            embeddings_list.append(e.cpu().numpy().squeeze(0))
            paths_used.append(p)
        except Exception as e:
            print(f"Skip {p}: {e}")
    if not embeddings_list:
        raise ValueError("No images could be loaded.")
    return np.vstack(embeddings_list), paths_used


def path_to_data_url(p: Path, max_size: int = 400) -> str:
    """Embed image as data URL so it displays in HTML when opened from disk."""
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_cached_embeddings(out_dir: Path) -> Optional[tuple]:
    """If embeddings.npy and paths_used.txt exist, return (embeddings, paths); else None."""
    emb_path = out_dir / "embeddings.npy"
    paths_path = out_dir / "paths_used.txt"
    if emb_path.exists() and paths_path.exists():
        embeddings = np.load(emb_path)
        paths = [Path(line.strip()) for line in paths_path.read_text().strip().splitlines()]
        if len(paths) == len(embeddings):
            return embeddings, paths
    return None


def main():
    parser = argparse.ArgumentParser(description="DreamSim embeddings + UMAP + KMeans clustering for PNG visualizations")
    parser.add_argument("--vis-dir", type=Path, default=VIS_DIR, help="Folder containing .png visualizations")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Where to save scatter plot and artifacts")
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of KMeans clusters (min 10)")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DreamSim")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for UMAP")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even if cache exists")
    args = parser.parse_args()

    args.n_clusters = max(10, args.n_clusters)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use cached embeddings if available (and not --force)
    cached = load_cached_embeddings(args.out_dir) if not args.force else None
    if cached is not None:
        embeddings, paths = cached
        print(f"Using cached embeddings: {embeddings.shape} for {len(paths)} images")
    else:
        paths = get_png_paths(args.vis_dir)
        if not paths:
            print(f"No .png files in {args.vis_dir}")
            return
        print(f"Found {len(paths)} PNG images in {args.vis_dir}")
        print("Loading DreamSim...")
        model, preprocess, device = load_dreamsim(args.device)
        print("Computing DreamSim embeddings...")
        embeddings, paths = embed_images(model, preprocess, paths, device, batch_size=args.batch_size)
        print(f"Embeddings shape: {embeddings.shape} for {len(paths)} images")
        np.save(args.out_dir / "embeddings.npy", embeddings)
        with open(args.out_dir / "paths_used.txt", "w") as f:
            f.write("\n".join(str(p) for p in paths))

    # Normalize for cosine similarity / K-means in angle space (optional; DreamSim may already normalize)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_n = embeddings / norms

    # Clusters via KMeans (at least 10 clusters)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    labels = kmeans.fit_predict(embeddings_n)
    print(f"KMeans: {args.n_clusters} clusters.")

    # UMAP to 2D (on normalized embeddings)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    coords_2d = reducer.fit_transform(embeddings_n)
    print("UMAP 2D reduction done.")

    # Save coords and cluster labels for reuse
    np.save(args.out_dir / "umap_2d.npy", coords_2d)
    np.save(args.out_dir / "cluster_labels.npy", labels)

    # DataFrame with data URL for image-on-hover (embedded so it works when opening HTML from disk)
    df = pd.DataFrame({
        "umap_1": coords_2d[:, 0],
        "umap_2": coords_2d[:, 1],
        "class": labels.astype(int).astype(str),
        "path": [str(p.name) for p in paths],
        "url": [path_to_data_url(p) for p in paths],
    })
    # save dataframe to csv
    df.to_csv(args.out_dir / "umap_2d.csv", index=False)

    # Hover selection: show image only when cursor is directly on a dot (not nearest point elsewhere)
    hover = alt.selection_single(on="mouseover", nearest=False, empty="none", clear="mouseout")
    scatter = (
        alt.Chart(df, title=f"Visualizations: DreamSim → UMAP (2D), colored by KMeans (n={args.n_clusters})")
        .mark_circle(size=100, opacity=0.8, stroke="white", strokeWidth=0.5)  # size ~ pi*10^2 for radius 10
        .encode(
            x=alt.X("umap_1:Q", title="UMAP 1"),
            y=alt.Y("umap_2:Q", title="UMAP 2"),
            color=alt.Color("class:N", title="Class", scale=alt.Scale(scheme="category10")),
            tooltip=[alt.Tooltip("path:N", title="Image"), "umap_1:Q", "umap_2:Q", "class:N"],
        )
        .properties(width=600, height=500)
        .add_selection(hover)
        .interactive()
    )
    # Image panel: show hovered point's .png
    image_panel = (
        alt.Chart(df)
        .mark_image(width=200, height=200)
        .encode(url="url:N")
        .transform_filter(hover)
        .properties(title="Hover to show image", width=220, height=220)
    )
    chart = alt.hconcat(scatter, image_panel, spacing=20)

    # Save as HTML (embeds Vega-Lite spec, view in browser)
    out_html = args.out_dir / "visualizations_umap_clusters.html"
    chart.save(str(out_html))
    print(f"Saved Vega-Lite scatter plot (HTML) to {out_html}")

    # Save raw Vega-Lite spec as JSON
    out_spec = args.out_dir / "visualizations_umap_clusters.vg.json"
    spec = chart.to_json()
    with open(out_spec, "w") as f:
        f.write(spec)
    print(f"Saved Vega-Lite spec to {out_spec}")

    # Print cluster sizes
    for k in sorted(set(labels)):
        count = (labels == k).sum()
        print(f"  cluster {k}: {count} images")


if __name__ == "__main__":
    main()
