"""
UMAP visualization of SAR ship embeddings for re-identification analysis.

Loads embeddings.npz, filters to ships with enough appearances,
runs UMAP, and produces several plots:
  - All ships colored by MMSI (do same ships cluster?)
  - Large ships only (>200m) — cleaner signal
  - Colored by vessel type
  - Colored by SOG (speed) — motion vs anchored
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import umap

EMB_PATH = "/Volumes/Machine Learning/sar_ship_chips/embeddings.npz"
OUT_DIR = "/Users/mousashaya/Projects/defense"

MIN_APPEARANCES = 4   # minimum passes to include a ship
MIN_LENGTH = 100      # meters — skip small boats for cleaner plot
CLEAN_ONLY = True     # use only chips with no other large ship (≥150m) in the 512px window

VTYPE_LABELS = {
    range(70, 80): ('Cargo/Container', '#4FC3F7'),
    range(80, 90): ('Tanker/LNG', '#FFB74D'),
    range(30, 33): ('Towing/Tug', '#81C784'),
    range(60, 70): ('Passenger', '#CE93D8'),
}

def vtype_color(vtype):
    for r, (label, color) in VTYPE_LABELS.items():
        if vtype in r:
            return color
    return '#AAAAAA'

def vtype_label(vtype):
    for r, (label, color) in VTYPE_LABELS.items():
        if vtype in r:
            return label
    return 'Other'


def main():
    print(f"Loading {EMB_PATH}...")
    d = np.load(EMB_PATH, allow_pickle=True)
    emb = d['embeddings']          # (N, 384)
    mmsi = d['mmsi'].astype(int)
    vessel_name = d['vessel_name']
    vessel_type = d['vessel_type'].astype(int)
    length_m = d['length_m'].astype(float)
    sog = d['sog'].astype(float)
    acq_time = d['acq_time']

    print(f"Total chips: {len(emb)}")
    print(f"Unique ships: {len(np.unique(mmsi))}")

    # Optionally restrict to clean chips (no other large ship in the 512px window)
    if CLEAN_ONLY and 'clean' in d:
        clean = d['clean'].astype(bool)
        print(f"Clean chips (no large neighbour): {clean.sum()} / {len(clean)}")
    else:
        clean = np.ones(len(emb), dtype=bool)

    # Filter: clean + minimum appearances + minimum length
    unique_mmsi, counts = np.unique(mmsi[clean], return_counts=True)
    keep_mmsi = set(unique_mmsi[counts >= MIN_APPEARANCES])
    mask = np.array([c and m in keep_mmsi and l >= MIN_LENGTH
                     for c, m, l in zip(clean, mmsi, length_m)])

    emb_f = emb[mask]
    mmsi_f = mmsi[mask]
    name_f = vessel_name[mask]
    vtype_f = vessel_type[mask]
    length_f = length_m[mask]
    sog_f = sog[mask]

    print(f"\nAfter filter (>={MIN_APPEARANCES} passes, >={MIN_LENGTH}m):")
    print(f"  Chips: {len(emb_f)}")
    print(f"  Ships: {len(np.unique(mmsi_f))}")

    # Per-ship stats
    unique_f, cnt_f = np.unique(mmsi_f, return_counts=True)
    idx_map = {m: i for i, m in enumerate(unique_f)}
    ship_idx = np.array([idx_map[m] for m in mmsi_f])  # 0..N_ships-1

    # Print top ships
    print("\nTop ships by appearance count:")
    order = np.argsort(-cnt_f)[:20]
    for i in order:
        m = unique_f[i]
        mask_m = mmsi_f == m
        nm = name_f[mask_m][0]
        lt = length_f[mask_m][0]
        vt = vtype_f[mask_m][0]
        print(f"  {nm:30s}  MMSI={m}  len={lt:.0f}m  type={vt}  appearances={cnt_f[i]}")

    # L2-normalize embeddings before UMAP
    norms = np.linalg.norm(emb_f, axis=1, keepdims=True)
    emb_norm = emb_f / np.maximum(norms, 1e-8)

    print("\nRunning UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(emb_norm)
    print("  UMAP done.")

    # ── Plot 1: colored by MMSI (ship identity) ──────────────────────────────
    n_ships = len(unique_f)
    cmap = matplotlib.colormaps['tab20'].resampled(min(n_ships, 20))
    colors_by_ship = [cmap(ship_idx[i] % 20) for i in range(len(coords))]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors_by_ship,
               s=18, alpha=0.7, linewidths=0)
    ax.set_title(f'SAR Ship Re-ID — UMAP by Ship Identity (MMSI)\n'
                 f'{len(emb_f)} chips, {n_ships} ships (≥{MIN_APPEARANCES} passes, ≥{MIN_LENGTH}m)',
                 fontsize=12)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

    # Label the top-10 most frequent ships
    for i in order[:10]:
        m = unique_f[i]
        mask_m = mmsi_f == m
        cx = coords[mask_m, 0].mean()
        cy = coords[mask_m, 1].mean()
        nm = str(name_f[mask_m][0]).strip()
        ax.annotate(nm, (cx, cy), fontsize=6.5, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'umap_ship_reid_identity.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # ── Plot 2: colored by vessel type ───────────────────────────────────────
    type_colors = [vtype_color(v) for v in vtype_f]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c=type_colors,
               s=18, alpha=0.7, linewidths=0)
    ax.set_title('SAR Ship Re-ID — UMAP by Vessel Type', fontsize=12)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    legend_handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color, markersize=9, label=label)
        for _, (label, color) in VTYPE_LABELS.items()
    ] + [Line2D([0],[0], marker='o', color='w', markerfacecolor='#AAAAAA', markersize=9, label='Other')]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'umap_ship_reid_type.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # ── Plot 3: colored by speed (SOG) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=sog_f,
                    cmap='plasma', s=18, alpha=0.8, linewidths=0,
                    vmin=0, vmax=15)
    plt.colorbar(sc, ax=ax, label='Speed over ground (knots)')
    ax.set_title('SAR Ship Re-ID — UMAP colored by Speed\n(Do moving vs anchored ships separate?)', fontsize=12)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'umap_ship_reid_speed.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # ── Plot 4: large ships only (>200m), big markers, labeled ───────────────
    big_mask = length_f >= 200
    if big_mask.sum() >= 10:
        coords_big = coords[big_mask]
        mmsi_big = mmsi_f[big_mask]
        name_big = name_f[big_mask]
        vtype_big = vtype_f[big_mask]
        sidx_big = ship_idx[big_mask]
        sog_big = sog_f[big_mask]
        length_big = length_f[big_mask]

        unique_big = np.unique(mmsi_big)
        cmap_big = matplotlib.colormaps['tab20'].resampled(min(len(unique_big), 20))
        idx_big = {m: i for i, m in enumerate(unique_big)}
        colors_big = [cmap_big(idx_big[m] % 20) for m in mmsi_big]

        fig, ax = plt.subplots(figsize=(14, 11))
        sc = ax.scatter(coords_big[:, 0], coords_big[:, 1],
                        c=colors_big, s=np.clip(length_big / 5, 20, 120),
                        alpha=0.85, linewidths=0.4, edgecolors='white')
        ax.set_title(f'Large Ships Only (≥200m) — {len(unique_big)} ships, {big_mask.sum()} chips\n'
                     'Marker size ∝ ship length', fontsize=12)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')

        # Label every ship
        for m in unique_big:
            mask_m = mmsi_big == m
            cx = coords_big[mask_m, 0].mean()
            cy = coords_big[mask_m, 1].mean()
            nm = str(name_big[mask_m][0]).strip()
            ax.annotate(nm, (cx, cy), fontsize=7, color='white',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.65))

        plt.tight_layout()
        out = os.path.join(OUT_DIR, 'umap_ship_reid_large.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    # ── Cluster quality metric: intra vs inter-ship distance ─────────────────
    print("\nCluster quality (lower = tighter clusters):")
    intra_dists, inter_dists = [], []
    sample_ships = unique_f[order[:15]]  # top 15 most frequent
    all_centers = {}
    for m in unique_f:
        mask_m = mmsi_f == m
        all_centers[m] = coords[mask_m].mean(axis=0)

    for m in sample_ships:
        mask_m = mmsi_f == m
        pts = coords[mask_m]
        center = pts.mean(axis=0)
        intra = np.mean(np.linalg.norm(pts - center, axis=1))
        inter = np.mean([np.linalg.norm(center - all_centers[other])
                         for other in unique_f if other != m])
        intra_dists.append(intra)
        inter_dists.append(inter)
        nm = str(name_f[mask_m][0]).strip()
        print(f"  {nm:30s}  intra={intra:.2f}  inter={inter:.2f}  ratio={intra/inter:.3f}")

    print(f"\nMean intra-ship spread: {np.mean(intra_dists):.3f}")
    print(f"Mean inter-ship distance: {np.mean(inter_dists):.3f}")
    print(f"Mean intra/inter ratio: {np.mean(intra_dists)/np.mean(inter_dists):.3f}  (lower = better clustering)")


if __name__ == '__main__':
    main()
