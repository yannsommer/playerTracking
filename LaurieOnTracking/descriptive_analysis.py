"""
Descriptive Analysis Pipeline for Football Event Data (Single-Match)

Generates:
- CSV tables:
    - outputs/top_passers.csv
    - outputs/top_receivers.csv
    - outputs/pass_pairs.csv (From->To counts)
    - outputs/player_centrality.csv (degree, betweenness)
    - outputs/zone_counts_start.csv / zone_counts_end.csv (zonal distributions)
    - outputs/possession_stats.json (chain lengths, tempo)
- Figures:
    - outputs/pass_network.png
    - outputs/pass_start_heatmap.png
    - outputs/pass_end_heatmap.png
    - outputs/pass_flow_top_edges.png (top passing links drawn on a pitch)

Usage:
    python descriptive_analysis_pipeline.py --csv /path/to/RawEventsData.csv --team All \
        --grid_x 6 --grid_y 4 --top_edges 25
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_passes(csv_path: str, team_filter: str = "All") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    p = df[(df["Type"] == "PASS") & df["To"].notna()].copy()
    if team_filter in ("Home", "Away"):
        p = p[p["Team"] == team_filter].copy()

    for col in ["Start X", "Start Y", "End X", "End Y"]:
        p[col] = p[col].astype(float).fillna(p[col].mean()).clip(0, 1)

    p["Start Time [s]"] = p["Start Time [s]"].astype(float)
    p["event_dt"] = p["Start Time [s]"].diff().fillna(0.0)
    dx = p["End X"] - p["Start X"]
    dy = p["End Y"] - p["Start Y"]
    p["pass_len"] = np.sqrt(dx**2 + dy**2)
    p["pass_ang"] = np.arctan2(dy, dx)

    p = p.sort_values(["Period", "Start Time [s]", "Start Frame"]).reset_index(drop=True)
    return p


def draw_pitch(ax, line_w=1.0):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], linewidth=line_w)
    ax.plot([0, 1], [0.5, 0.5], linewidth=line_w)
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy, r = 0.5, 0.5, 0.0915
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), linewidth=line_w)
    box_w, box_h = 0.16, 0.4
    ax.plot([0, box_w, box_w, 0], [0.3, 0.3, 0.7, 0.7], linewidth=line_w)
    ax.plot([1 - box_w, 1, 1, 1 - box_w], [0.3, 0.3, 0.7, 0.7], linewidth=line_w)


def build_pass_network(passes: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    players = sorted(set(passes["From"].astype(str)) | set(passes["To"].astype(str)))
    G.add_nodes_from(players)
    pair_counts = passes.groupby(["From", "To"]).size().reset_index(name="count")
    for _, r in pair_counts.iterrows():
        f, t, c = str(r["From"]), str(r["To"]), int(r["count"])
        if G.has_edge(f, t):
            G[f][t]["weight"] += c
        else:
            G.add_edge(f, t, weight=c)
    return G


def compute_network_metrics(G: nx.DiGraph) -> pd.DataFrame:
    in_strength = dict(G.in_degree(weight="weight"))
    out_strength = dict(G.out_degree(weight="weight"))
    bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    df = pd.DataFrame({
        "player": list(G.nodes()),
        "in_strength": [in_strength.get(n, 0) for n in G.nodes()],
        "out_strength": [out_strength.get(n, 0) for n in G.nodes()],
        "betweenness": [bet.get(n, 0.0) for n in G.nodes()],
    })
    df["total_strength"] = df["in_strength"] + df["out_strength"]
    return df.sort_values("total_strength", ascending=False)


def plot_pass_network(G: nx.DiGraph, out_path: str, top_edges: int = 25):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.axis('off')
    deg = dict(G.degree(weight="weight"))
    sizes = np.array([max(100, 80 * np.sqrt(deg.get(n, 1))) for n in G.nodes()])
    edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)[:top_edges]
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, d in edges:
        H.add_edge(u, v, weight=d["weight"])
    pos = nx.spring_layout(H, k=0.8, iterations=200, seed=42, weight="weight")
    nx.draw_networkx_nodes(H, pos, node_size=sizes, alpha=0.9)
    widths = [1 + 2.5 * (H[u][v]["weight"] / max(1, edges[0][2]["weight"])) for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, width=widths, arrowsize=15, alpha=0.6)
    nx.draw_networkx_labels(H, pos, font_size=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def heatmap_xy(points_xy: np.ndarray, bins=(50, 34), title="", out_path="heatmap.png"):
    H, _, _ = np.histogram2d(points_xy[:, 0], points_xy[:, 1], bins=bins, range=[[0, 1], [0, 1]])
    H = H.T
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    draw_pitch(ax)
    ax.imshow(H, origin="lower", extent=[0, 1, 0, 1], interpolation="bilinear", alpha=0.6)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pass_flow_top_edges(passes: pd.DataFrame, out_path: str, top_edges: int = 25):
    agg = passes.groupby(["From", "To"]).agg(
        count=("From", "size"),
        sx=("Start X", "mean"),
        sy=("Start Y", "mean"),
        ex=("End X", "mean"),
        ey=("End Y", "mean")
    ).reset_index().sort_values("count", ascending=False)
    top = agg.head(top_edges)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    draw_pitch(ax, line_w=1.0)
    for _, r in top.iterrows():
        x0, y0, x1, y1, c = float(r["sx"]), float(r["sy"]), float(r["ex"]), float(r["ey"]), int(r["count"])
        lw = 0.5 + 2.5 * (c / max(1, top["count"].max()))
        ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.02, length_includes_head=True, alpha=0.7, linewidth=lw)
    ax.set_title(f"Top {len(top)} passing links (average locations)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def zone_counts(passes: pd.DataFrame, grid_x=6, grid_y=4, which="start") -> pd.DataFrame:
    xcol = "Start X" if which == "start" else "End X"
    ycol = "Start Y" if which == "start" else "End Y"
    xs = passes[xcol].values
    ys = passes[ycol].values
    ix = np.clip((xs * grid_x).astype(int), 0, grid_x - 1)
    iy = np.clip((ys * grid_y).astype(int), 0, grid_y - 1)
    df = pd.DataFrame({"zone_x": ix, "zone_y": iy})
    out = df.groupby(["zone_x", "zone_y"]).size().reset_index(name="count")
    total = out["count"].sum()
    out["ratio"] = out["count"] / max(1, total)
    return out.sort_values(["zone_y", "zone_x"]).reset_index(drop=True)


def possessions(passes: pd.DataFrame) -> dict:
    chains = []
    cur_team = None
    cur_len = 0
    cur_times = []
    for _, r in passes.iterrows():
        team = r["Team"]
        t = r["Start Time [s]"]
        if team != cur_team:
            if cur_len > 0:
                chains.append((cur_team, cur_len, cur_times))
            cur_team = team
            cur_len = 1
            cur_times = [t]
        else:
            cur_len += 1
            cur_times.append(t)
    if cur_len > 0:
        chains.append((cur_team, cur_len, cur_times))
    lengths = [c[1] for c in chains]
    tempo = []
    for _, _, times in chains:
        if len(times) >= 2:
            tempo.extend(np.diff(times))
    return {
        "num_chains": len(chains),
        "avg_chain_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_chain_length": float(np.median(lengths)) if lengths else 0.0,
        "avg_time_between_passes": float(np.mean(tempo)) if tempo else 0.0,
        "median_time_between_passes": float(np.median(tempo)) if tempo else 0.0,
    }


def main(args):
    outdir = os.path.abspath(args.outdir)
    ensure_outdir(outdir)
    passes = load_passes(args.csv, team_filter=args.team)
    top_passers = passes.groupby("From").size().reset_index(name="passes").sort_values("passes", ascending=False)
    top_receivers = passes.groupby("To").size().reset_index(name="received").sort_values("received", ascending=False)
    pair_counts = passes.groupby(["From", "To"]).size().reset_index(name="count").sort_values("count", ascending=False)
    G = build_pass_network(passes)
    centrality_df = compute_network_metrics(G)
    z_start = zone_counts(passes, grid_x=args.grid_x, grid_y=args.grid_y, which="start")
    z_end = zone_counts(passes, grid_x=args.grid_x, grid_y=args.grid_y, which="end")
    poss_stats = possessions(passes)
    top_passers.to_csv(os.path.join(outdir, "top_passers.csv"), index=False)
    top_receivers.to_csv(os.path.join(outdir, "top_receivers.csv"), index=False)
    pair_counts.to_csv(os.path.join(outdir, "pass_pairs.csv"), index=False)
    centrality_df.to_csv(os.path.join(outdir, "player_centrality.csv"), index=False)
    z_start.to_csv(os.path.join(outdir, "zone_counts_start.csv"), index=False)
    z_end.to_csv(os.path.join(outdir, "zone_counts_end.csv"), index=False)
    with open(os.path.join(outdir, "possession_stats.json"), "w") as f:
        json.dump(poss_stats, f, indent=2)
    plot_pass_network(G, os.path.join(outdir, "pass_network.png"), top_edges=args.top_edges)
    start_xy = passes[["Start X", "Start Y"]].values.astype(float)
    end_xy = passes[["End X", "End Y"]].values.astype(float)
    heatmap_xy(start_xy, bins=(60, 40), title="Pass start heatmap", out_path=os.path.join(outdir, "pass_start_heatmap.png"))
    heatmap_xy(end_xy, bins=(60, 40), title="Pass end heatmap", out_path=os.path.join(outdir, "pass_end_heatmap.png"))
    plot_pass_flow_top_edges(passes, os.path.join(outdir, "pass_flow_top_edges.png"), top_edges=args.top_edges)
    print("Saved outputs to:", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--team", type=str, default="All", choices=["All", "Home", "Away"])
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--grid_x", type=int, default=6)
    ap.add_argument("--grid_y", type=int, default=4)
    ap.add_argument("--top_edges", type=int, default=25)
    args = ap.parse_args()
    main(args)

