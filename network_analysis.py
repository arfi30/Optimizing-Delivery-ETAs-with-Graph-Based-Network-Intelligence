import pandas as pd
import numpy as np
import networkx as nx

seg = pd.read_pickle("outputs/segments.pkl")

corridors = seg.groupby(["source_center", "destination_center"], as_index=False).agg(
    source_name=("source_name", "first"),
    destination_name=("destination_name", "first"),
    n_trips=("trip_uuid", "nunique"),
    median_ratio=("delay_ratio", "median"),
    mean_actual=("actual_time", "mean"),
    mean_osrm=("osrm_time", "mean"),
    late_share=("is_late", "mean"),
    total_late=("is_late", "sum"),
)
corridors["chronic"] = (corridors.median_ratio > 1.2) & (corridors.n_trips >= 10)

G = nx.DiGraph()
for r in corridors.itertuples():
    G.add_edge(r.source_center, r.destination_center,
               weight=r.median_ratio, n_trips=r.n_trips, late_share=r.late_share)

print("graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

k = min(600, G.number_of_nodes())
betweenness = nx.betweenness_centrality(G, weight="weight", k=k, seed=42)
pagerank = nx.pagerank(G, weight="n_trips")
clustering = nx.clustering(G.to_undirected())
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

nodes = pd.DataFrame({
    "center": list(G.nodes()),
    "betweenness": [betweenness[n] for n in G.nodes()],
    "in_degree": [in_deg[n] for n in G.nodes()],
    "out_degree": [out_deg[n] for n in G.nodes()],
    "clustering": [clustering[n] for n in G.nodes()],
    "pagerank": [pagerank[n] for n in G.nodes()],
})

names = seg.groupby("source_center").source_name.first()
fac = seg.groupby("source_center").agg(
    out_trips=("trip_uuid", "nunique"),
    out_median_ratio=("delay_ratio", "median"),
    out_late=("is_late", "sum"),
)
nodes = nodes.merge(names.rename("name"), left_on="center", right_index=True, how="left")
nodes = nodes.merge(fac, left_on="center", right_index=True, how="left")

nodes["sla_breach_contrib"] = nodes.out_late / seg.is_late.sum()

nodes["bottleneck_score"] = (
    nodes.betweenness.rank(pct=True) * 0.4
    + nodes.out_median_ratio.fillna(1).rank(pct=True) * 0.3
    + nodes.out_trips.fillna(0).rank(pct=True) * 0.3
)

nodes = nodes.sort_values("bottleneck_score", ascending=False)
nodes.to_csv("outputs/node_metrics.csv", index=False)
corridors.sort_values("total_late", ascending=False).to_csv("outputs/corridor_audit.csv", index=False)

show = ["name", "betweenness", "out_trips", "out_median_ratio", "sla_breach_contrib"]
print("\ntop 10 bottleneck hubs")
print(nodes[show].head(10).to_string(index=False))

bad = corridors[corridors.chronic].sort_values("total_late", ascending=False)
print("\nworst chronic corridors")
print(bad[["source_name", "destination_name", "n_trips", "median_ratio", "late_share"]]
      .head(10).to_string(index=False))
print("\nchronic corridors:", corridors.chronic.sum(), "out of", len(corridors))
