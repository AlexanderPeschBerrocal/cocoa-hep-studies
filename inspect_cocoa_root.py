#!/usr/bin/env python3
"""
Inspect a COCOA ROOT output file and make material-study oriented plots.

Focus:
  - photon conversion vertices
  - conversion electron kinematics
  - track acceptance / layer hit maps
  - truth-node displaced production points as a first proxy for secondary interactions
  - calorimeter sanity plots

Usage:
  python inspect_cocoa_root_revised.py output.root --outdir python_plots

Notes:
  - COCOA branches are often jagged/event-wise arrays.
  - This script uses awkward arrays and then flattens where appropriate.
  - "node" displaced production plots are NOT a clean nuclear-interaction reconstruction.
    They are diagnostic truth-level plots to help you see displaced secondary production.
"""

import argparse
from pathlib import Path
from collections import Counter

import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# -----------------------------
# General helpers
# -----------------------------

def savefig(path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"saved {path}")


def branch_exists(tree, name: str) -> bool:
    return name in tree.keys()


def get_ak(tree, name: str):
    if not branch_exists(tree, name):
        print(f"[skip] missing branch: {name}")
        return None
    return tree[name].array(library="ak")


def flatten_to_numpy(tree, name: str, dtype=float):
    """Read a possibly-jagged branch and return a flat NumPy array."""
    a = get_ak(tree, name)
    if a is None:
        return None
    try:
        flat = ak.flatten(a, axis=None)
        return ak.to_numpy(flat).astype(dtype, copy=False)
    except Exception as exc:
        print(f"[skip] could not flatten {name}: {exc}")
        return None


def counts_per_event(tree, name: str):
    """Return number of entries per event for a jagged branch."""
    a = get_ak(tree, name)
    if a is None:
        return None
    try:
        return ak.to_numpy(ak.num(a, axis=1))
    except Exception as exc:
        print(f"[skip] could not count entries for {name}: {exc}")
        return None


def finite_mask(*arrays):
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def describe(name: str, a: np.ndarray) -> None:
    if a is None:
        return
    if len(a) == 0:
        print(f"{name}: empty")
        return
    print(
        f"{name}: n={len(a)}, "
        f"min={np.nanmin(a):.4g}, median={np.nanmedian(a):.4g}, "
        f"max={np.nanmax(a):.4g}"
    )


def maybe_downsample(*arrays, max_points: int = 200_000, seed: int = 12345):
    """Downsample consistently for scatter plots only."""
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return tuple(a[idx] for a in arrays)


def eta_from_rz(r, z):
    # eta = asinh(z/r), safe for r > 0
    return np.arcsinh(z / r)


# -----------------------------
# Plot groups
# -----------------------------

def plot_conversion_vertices(tree, outdir: Path, max_scatter: int) -> None:
    print("\n== Photon conversion vertices ==")

    x = flatten_to_numpy(tree, "conv_el_prod_x")
    y = flatten_to_numpy(tree, "conv_el_prod_y")
    z = flatten_to_numpy(tree, "conv_el_prod_z")
    if x is None or y is None or z is None:
        print("[skip] conversion vertex plots")
        return

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    mask = finite_mask(x, y, z, r) & (r > 0)
    x, y, z, r, phi = x[mask], y[mask], z[mask], r[mask], phi[mask]
    eta = eta_from_rz(r, z)

    nconv_evt = counts_per_event(tree, "conv_el_prod_x")

    describe("conversion x [mm]", x)
    describe("conversion y [mm]", y)
    describe("conversion z [mm]", z)
    describe("conversion r [mm]", r)
    if nconv_evt is not None:
        describe("conversions per event", nconv_evt.astype(float))
        print(f"events with >=1 conversion: {np.count_nonzero(nconv_evt > 0)} / {len(nconv_evt)}")

    if len(x) == 0:
        return

    xs, ys, zs, rs, phis, etas = maybe_downsample(
        x, y, z, r, phi, eta, max_points=max_scatter
    )

    # x-y conversion map
    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys, s=3, alpha=0.45)
    plt.xlabel("conversion x [mm]")
    plt.ylabel("conversion y [mm]")
    plt.title("Photon conversion vertices: transverse view")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)
    savefig(outdir / "01_conversions_xy_scatter.png")

    # r-z scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(zs, rs, s=3, alpha=0.45)
    plt.xlabel("conversion z [mm]")
    plt.ylabel("conversion radius r [mm]")
    plt.title("Photon conversion vertices: r-z view")
    plt.grid(alpha=0.3)
    savefig(outdir / "02_conversions_rz_scatter.png")

    # r-z density, linear
    plt.figure(figsize=(8, 6))
    plt.hist2d(z, r, bins=(140, 140))
    plt.xlabel("conversion z [mm]")
    plt.ylabel("conversion radius r [mm]")
    plt.title("Conversion vertex density in r-z")
    plt.colorbar(label="entries")
    savefig(outdir / "03_conversions_rz_density.png")

    # r-z density, log
    if len(z) > 0:
        plt.figure(figsize=(8, 6))
        plt.hist2d(z, r, bins=(140, 140), norm=LogNorm())
        plt.xlabel("conversion z [mm]")
        plt.ylabel("conversion radius r [mm]")
        plt.title("Conversion vertex density in r-z, log scale")
        plt.colorbar(label="entries")
        savefig(outdir / "04_conversions_rz_density_log.png")

    # radial distribution
    plt.figure(figsize=(8, 6))
    plt.hist(r, bins=120)
    plt.xlabel("conversion radius r [mm]")
    plt.ylabel("entries")
    plt.title("Radial distribution of conversion vertices")
    plt.grid(alpha=0.3)
    savefig(outdir / "05_conversions_radius_hist.png")

    # z distribution
    plt.figure(figsize=(8, 6))
    plt.hist(z, bins=120)
    plt.xlabel("conversion z [mm]")
    plt.ylabel("entries")
    plt.title("Longitudinal distribution of conversion vertices")
    plt.grid(alpha=0.3)
    savefig(outdir / "06_conversions_z_hist.png")

    # phi distribution
    plt.figure(figsize=(8, 6))
    plt.hist(phi, bins=72)
    plt.xlabel("conversion phi [rad]")
    plt.ylabel("entries")
    plt.title("Azimuthal distribution of conversion vertices")
    plt.grid(alpha=0.3)
    savefig(outdir / "07_conversions_phi_hist.png")

    # eta distribution
    plt.figure(figsize=(8, 6))
    plt.hist(eta, bins=100)
    plt.xlabel("conversion eta")
    plt.ylabel("entries")
    plt.title("Pseudorapidity distribution of conversion vertices")
    plt.grid(alpha=0.3)
    savefig(outdir / "08_conversions_eta_hist.png")

    # r-phi map
    plt.figure(figsize=(8, 6))
    plt.hist2d(phi, r, bins=(72, 120))
    plt.xlabel("conversion phi [rad]")
    plt.ylabel("conversion radius r [mm]")
    plt.title("Conversion density in r-phi")
    plt.colorbar(label="entries")
    savefig(outdir / "09_conversions_r_phi_density.png")

    # z-phi map
    plt.figure(figsize=(8, 6))
    plt.hist2d(phi, z, bins=(72, 140))
    plt.xlabel("conversion phi [rad]")
    plt.ylabel("conversion z [mm]")
    plt.title("Conversion density in z-phi")
    plt.colorbar(label="entries")
    savefig(outdir / "10_conversions_z_phi_density.png")

    # conversions per event
    if nconv_evt is not None:
        plt.figure(figsize=(8, 6))
        bins = np.arange(0, int(np.max(nconv_evt)) + 2) - 0.5 if len(nconv_evt) else 10
        plt.hist(nconv_evt, bins=bins)
        plt.xlabel("number of conversion electrons per event")
        plt.ylabel("events")
        plt.title("Conversion-electron multiplicity per event")
        plt.grid(alpha=0.3)
        savefig(outdir / "11_conversions_per_event.png")


def plot_conversion_kinematics(tree, outdir: Path) -> None:
    print("\n== Conversion electron kinematics ==")

    px = flatten_to_numpy(tree, "conv_el_px")
    py = flatten_to_numpy(tree, "conv_el_py")
    pz = flatten_to_numpy(tree, "conv_el_pz")
    q = flatten_to_numpy(tree, "conv_el_q")
    vx = flatten_to_numpy(tree, "conv_el_prod_x")
    vy = flatten_to_numpy(tree, "conv_el_prod_y")

    if px is None or py is None or pz is None:
        print("[skip] conversion kinematics")
        return

    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    mask = finite_mask(pt, p) & (p > 0)
    pt, p = pt[mask], p[mask]

    describe("conversion electron pT", pt)
    describe("conversion electron p", p)

    if len(pt) == 0:
        return

    plt.figure(figsize=(8, 6))
    plt.hist(pt, bins=100)
    plt.xlabel("conversion electron pT")
    plt.ylabel("entries")
    plt.title("Conversion-electron transverse momentum")
    plt.grid(alpha=0.3)
    savefig(outdir / "12_conversion_electron_pt_hist.png")

    plt.figure(figsize=(8, 6))
    plt.hist(p, bins=100)
    plt.xlabel("conversion electron momentum")
    plt.ylabel("entries")
    plt.title("Conversion-electron momentum")
    plt.grid(alpha=0.3)
    savefig(outdir / "13_conversion_electron_p_hist.png")

    if q is not None and len(q) > 0:
        q = q[np.isfinite(q)]
        plt.figure(figsize=(7, 5))
        vals, counts = np.unique(q.astype(int), return_counts=True)
        plt.bar([str(v) for v in vals], counts)
        plt.xlabel("conversion electron charge")
        plt.ylabel("entries")
        plt.title("Conversion-electron charge distribution")
        savefig(outdir / "14_conversion_electron_charge.png")

    if vx is not None and vy is not None and len(vx) == len(px):
        r = np.sqrt(vx**2 + vy**2)
        mask2 = finite_mask(r, pt) & (r > 0)
        if np.count_nonzero(mask2) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist2d(r[mask2], pt[mask2], bins=(100, 100), norm=LogNorm())
            plt.xlabel("conversion radius r [mm]")
            plt.ylabel("conversion electron pT")
            plt.title("Conversion-electron pT vs conversion radius")
            plt.colorbar(label="entries")
            savefig(outdir / "15_conversion_pt_vs_radius.png")


def plot_tracks(tree, outdir: Path, max_scatter: int) -> None:
    print("\n== Tracks and tracker-layer hit maps ==")

    d0 = flatten_to_numpy(tree, "track_d0")
    z0 = flatten_to_numpy(tree, "track_z0")
    theta = flatten_to_numpy(tree, "track_theta")
    phi = flatten_to_numpy(tree, "track_phi")
    qoverp = flatten_to_numpy(tree, "track_qoverp")
    reco = flatten_to_numpy(tree, "track_reconstructed")
    acc = flatten_to_numpy(tree, "track_in_acceptance")

    if d0 is not None:
        d0 = d0[np.isfinite(d0)]
        describe("track d0", d0)
        plt.figure(figsize=(8, 6))
        plt.hist(d0, bins=100)
        plt.xlabel("track d0 [mm]")
        plt.ylabel("tracks")
        plt.title("Track transverse impact parameter d0")
        plt.grid(alpha=0.3)
        savefig(outdir / "20_tracks_d0_hist.png")

    if z0 is not None:
        z0 = z0[np.isfinite(z0)]
        describe("track z0", z0)
        plt.figure(figsize=(8, 6))
        plt.hist(z0, bins=100)
        plt.xlabel("track z0 [mm]")
        plt.ylabel("tracks")
        plt.title("Track longitudinal impact parameter z0")
        plt.grid(alpha=0.3)
        savefig(outdir / "21_tracks_z0_hist.png")

    if qoverp is not None:
        qoverp = qoverp[np.isfinite(qoverp)]
        describe("track q/p", qoverp)
        plt.figure(figsize=(8, 6))
        plt.hist(qoverp, bins=100)
        plt.xlabel("track q/p")
        plt.ylabel("tracks")
        plt.title("Track q/p distribution")
        plt.grid(alpha=0.3)
        savefig(outdir / "22_tracks_qoverp_hist.png")

    if theta is not None and phi is not None:
        theta = theta[np.isfinite(theta)]
        phi = phi[np.isfinite(phi)]
        if len(theta) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(theta, bins=80)
            plt.xlabel("track theta [rad]")
            plt.ylabel("tracks")
            plt.title("Track theta distribution")
            plt.grid(alpha=0.3)
            savefig(outdir / "23_tracks_theta_hist.png")
        if len(phi) > 0:
            plt.figure(figsize=(8, 6))
            plt.hist(phi, bins=72)
            plt.xlabel("track phi [rad]")
            plt.ylabel("tracks")
            plt.title("Track phi distribution")
            plt.grid(alpha=0.3)
            savefig(outdir / "24_tracks_phi_hist.png")

    if reco is not None or acc is not None:
        labels = []
        vals = []
        if reco is not None and len(reco) > 0:
            labels.append("reconstructed")
            vals.append(float(np.mean(reco.astype(bool))))
        if acc is not None and len(acc) > 0:
            labels.append("in acceptance")
            vals.append(float(np.mean(acc.astype(bool))))
        if labels:
            for label, val in zip(labels, vals):
                print(f"{label} fraction: {val:.4f}")
            plt.figure(figsize=(7, 5))
            plt.bar(labels, vals)
            plt.ylim(0, 1)
            plt.ylabel("fraction")
            plt.title("Track reconstruction / acceptance fractions")
            savefig(outdir / "25_tracks_reco_acceptance_fractions.png")

    # Layer hit maps
    all_z = []
    all_r = []
    for i in range(6):
        lx = flatten_to_numpy(tree, f"track_x_layer_{i}")
        ly = flatten_to_numpy(tree, f"track_y_layer_{i}")
        lz = flatten_to_numpy(tree, f"track_z_layer_{i}")
        if lx is None or ly is None or lz is None:
            continue

        lr = np.sqrt(lx**2 + ly**2)
        mask = finite_mask(lx, ly, lz, lr) & (lr > 0)
        lx, ly, lz, lr = lx[mask], ly[mask], lz[mask], lr[mask]
        if len(lx) == 0:
            continue

        all_z.append(lz)
        all_r.append(lr)

        xs, ys = maybe_downsample(lx, ly, max_points=max_scatter)
        plt.figure(figsize=(7, 7))
        plt.scatter(xs, ys, s=2, alpha=0.35)
        plt.xlabel(f"track x at layer {i} [mm]")
        plt.ylabel(f"track y at layer {i} [mm]")
        plt.title(f"Track hit positions in layer {i}: x-y")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(alpha=0.3)
        savefig(outdir / f"26_tracks_layer_{i}_xy.png")

    if all_z:
        zcat = np.concatenate(all_z)
        rcat = np.concatenate(all_r)
        plt.figure(figsize=(8, 6))
        plt.hist2d(zcat, rcat, bins=(140, 80), norm=LogNorm())
        plt.xlabel("track hit z [mm]")
        plt.ylabel("track hit radius r [mm]")
        plt.title("All tracker-layer hit positions in r-z")
        plt.colorbar(label="entries")
        savefig(outdir / "32_tracks_all_layers_rz_density.png")


def plot_truth_nodes(tree, outdir: Path) -> None:
    print("\n== Truth-node displaced production diagnostics ==")

    px = flatten_to_numpy(tree, "node_prodx")
    py = flatten_to_numpy(tree, "node_prody")
    pz = flatten_to_numpy(tree, "node_prodz")
    pdg = flatten_to_numpy(tree, "node_pdg_id", dtype=int)

    if px is None or py is None or pz is None:
        print("[skip] truth-node plots")
        return

    r = np.sqrt(px**2 + py**2)
    mask = finite_mask(px, py, pz, r) & (r >= 0)
    px, py, pz, r = px[mask], py[mask], pz[mask], r[mask]
    if pdg is not None and len(pdg) == len(mask):
        pdg = pdg[mask]

    describe("node production r [mm]", r)
    describe("node production z [mm]", pz)

    if len(r) == 0:
        return

    displaced = r > 10.0
    print(f"nodes with production radius > 10 mm: {np.count_nonzero(displaced)} / {len(r)}")

    # Full production r-z density
    plt.figure(figsize=(8, 6))
    plt.hist2d(pz, r, bins=(160, 140), norm=LogNorm())
    plt.xlabel("node production z [mm]")
    plt.ylabel("node production radius r [mm]")
    plt.title("Truth-node production density in r-z")
    plt.colorbar(label="entries")
    savefig(outdir / "40_nodes_production_rz_density.png")

    # Displaced production points: possible secondary interactions / decays / material effects
    if np.count_nonzero(displaced) > 0:
        plt.figure(figsize=(8, 6))
        plt.hist2d(pz[displaced], r[displaced], bins=(160, 140), norm=LogNorm())
        plt.xlabel("node production z [mm]")
        plt.ylabel("node production radius r [mm]")
        plt.title("Displaced truth-node production density, r > 10 mm")
        plt.colorbar(label="entries")
        savefig(outdir / "41_nodes_displaced_rz_density.png")

        plt.figure(figsize=(8, 6))
        plt.hist(r[displaced], bins=120)
        plt.xlabel("displaced node production radius r [mm]")
        plt.ylabel("entries")
        plt.title("Displaced truth-node production radius, r > 10 mm")
        plt.grid(alpha=0.3)
        savefig(outdir / "42_nodes_displaced_radius_hist.png")

    if pdg is not None and len(pdg) == len(r):
        # Avoid making a huge unreadable category plot: show top 20 by absolute PDG ID.
        pdg_disp = pdg[displaced] if np.count_nonzero(displaced) > 0 else pdg
        counts = Counter(pdg_disp.tolist())
        most_common = counts.most_common(20)
        if most_common:
            labels = [str(k) for k, _ in most_common]
            vals = [v for _, v in most_common]
            plt.figure(figsize=(10, 6))
            plt.bar(labels, vals)
            plt.xlabel("PDG ID")
            plt.ylabel("entries")
            plt.title("Most common PDG IDs for displaced truth nodes")
            plt.xticks(rotation=45, ha="right")
            savefig(outdir / "43_nodes_displaced_pdg_top20.png")


def plot_cells(tree, outdir: Path) -> None:
    print("\n== Calorimeter sanity plots ==")

    e = flatten_to_numpy(tree, "cell_e")
    layer = flatten_to_numpy(tree, "cell_layer", dtype=int)
    eta = flatten_to_numpy(tree, "cell_eta")
    phi = flatten_to_numpy(tree, "cell_phi")

    if e is None:
        print("[skip] calorimeter plots")
        return

    e = e[np.isfinite(e)]
    describe("cell energy", e)

    if len(e) == 0:
        return

    plt.figure(figsize=(8, 6))
    plt.hist(e, bins=120)
    plt.xlabel("cell energy")
    plt.ylabel("cells")
    plt.title("Calorimeter cell energy distribution")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    savefig(outdir / "50_cells_energy_hist_logy.png")

    if layer is not None and len(layer) == len(e):
        # This branch matching can fail if masks were applied differently; use raw arrays more carefully.
        pass

    # Re-read with common mask if layer exists
    if branch_exists(tree, "cell_layer"):
        eraw = flatten_to_numpy(tree, "cell_e")
        lraw = flatten_to_numpy(tree, "cell_layer", dtype=int)
        if eraw is not None and lraw is not None and len(eraw) == len(lraw):
            mask = np.isfinite(eraw) & np.isfinite(lraw)
            eraw, lraw = eraw[mask], lraw[mask]
            unique_layers = np.unique(lraw)
            sums = [np.sum(eraw[lraw == lay]) for lay in unique_layers]
            plt.figure(figsize=(8, 6))
            plt.bar([str(int(l)) for l in unique_layers], sums)
            plt.xlabel("cell layer")
            plt.ylabel("sum cell energy")
            plt.title("Total calorimeter cell energy by layer")
            savefig(outdir / "51_cells_energy_by_layer.png")

    if eta is not None and phi is not None:
        eraw = flatten_to_numpy(tree, "cell_e")
        etaraw = flatten_to_numpy(tree, "cell_eta")
        phiraw = flatten_to_numpy(tree, "cell_phi")
        if eraw is not None and etaraw is not None and phiraw is not None:
            n = min(len(eraw), len(etaraw), len(phiraw))
            eraw, etaraw, phiraw = eraw[:n], etaraw[:n], phiraw[:n]
            mask = finite_mask(eraw, etaraw, phiraw) & (eraw > 0)
            if np.count_nonzero(mask) > 0:
                plt.figure(figsize=(8, 6))
                plt.hist2d(
                    phiraw[mask],
                    etaraw[mask],
                    bins=(72, 100),
                    weights=eraw[mask],
                    norm=LogNorm(),
                )
                plt.xlabel("cell phi")
                plt.ylabel("cell eta")
                plt.title("Calorimeter energy map in eta-phi")
                plt.colorbar(label="sum cell energy")
                savefig(outdir / "52_cells_eta_phi_energy_map.png")


def write_summary(tree, outdir: Path, rootfile: Path) -> None:
    summary_path = outdir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"ROOT file: {rootfile}\n")
        f.write(f"Tree: Out_Tree\n")
        f.write(f"Number of events: {tree.num_entries}\n")
        f.write("\nBranches:\n")
        for b in tree.keys():
            f.write(f"  {b}\n")
    print(f"saved {summary_path}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make COCOA ROOT diagnostic plots for tracker-material studies."
    )
    parser.add_argument("rootfile", help="COCOA ROOT output file")
    parser.add_argument("--outdir", default="plots", help="output directory for plots")
    parser.add_argument("--tree", default="Out_Tree", help="TTree name")
    parser.add_argument(
        "--max-scatter",
        type=int,
        default=200_000,
        help="maximum number of points shown in scatter plots",
    )
    parser.add_argument(
        "--no-cells",
        action="store_true",
        help="skip calorimeter sanity plots",
    )
    args = parser.parse_args()

    rootfile = Path(args.rootfile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    f = uproot.open(rootfile)
    print("Objects in file:", f.keys())

    if args.tree not in f:
        raise KeyError(f"Tree '{args.tree}' not found. Available objects: {f.keys()}")

    tree = f[args.tree]
    print(f"\nTree: {args.tree}")
    print(f"Number of events: {tree.num_entries}")

    print("\nBranches:")
    for b in tree.keys():
        print(" ", b)

    write_summary(tree, outdir, rootfile)
    plot_conversion_vertices(tree, outdir, args.max_scatter)
    plot_conversion_kinematics(tree, outdir)
    plot_tracks(tree, outdir, args.max_scatter)
    plot_truth_nodes(tree, outdir)
    if not args.no_cells:
        plot_cells(tree, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
