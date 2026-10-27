import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seg = pd.read_pickle("outputs/segments.pkl")
nodes = pd.read_csv("outputs/node_metrics.csv")
corridors = pd.read_csv("outputs/corridor_audit.csv")
bench = pd.read_csv("outputs/model_benchmark.csv")

seg["dist_band"] = pd.cut(seg.osrm_distance, [0, 25, 100, 300, 5000],
                          labels=["<25km", "25-100km", "100-300km", ">300km"])

tradeoff = seg.groupby(["dist_band", "route_type"], observed=True).agg(
    median_ratio=("delay_ratio", "median"),
    median_actual=("actual_time", "median"),
    n=("trip_uuid", "nunique"),
).round(2)
print("delay ratio by distance band and route type")
print(tradeoff.to_string())
tradeoff.to_csv("outputs/ftl_carting_tradeoff.csv")

bet_map = nodes.set_index("center").betweenness
seg["src_bet"] = seg.source_center.map(bet_map).fillna(0)
seg["bet_tier"] = pd.qcut(seg.src_bet.rank(method="first"), 3,
                          labels=["low", "mid", "high"])

cells = (seg.groupby(["dist_band", "time_of_day", "bet_tier", "route_type"], observed=True)
            .delay_ratio.median().unstack().dropna())
cells["recommend"] = np.where(cells.FTL < cells.Carting, "FTL", "Carting")
cells["delay_saved"] = cells[["FTL", "Carting"]].max(axis=1) - cells[["FTL", "Carting"]].min(axis=1)
cells.round(2).to_csv("outputs/route_type_framework.csv")

print("\nbiggest wins from switching route type")
print(cells.sort_values("delay_saved", ascending=False).head(8).round(2).to_string())

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
colors3 = ["#b23b3b", "#8a8a8a", "#2e7d32"]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].bar(bench.model, bench.MAE_min, color=colors3)
ax[0].set_title("ETA error (MAE, minutes)")
for i, v in enumerate(bench.MAE_min):
    ax[0].text(i, v + 1, f"{v:.0f}", ha="center")
ax[1].bar(bench.model, bench.within_15pct * 100, color=colors3)
ax[1].set_title("% trips within 15% of actual")
for i, v in enumerate(bench.within_15pct * 100):
    ax[1].text(i, v + 1, f"{v:.0f}%", ha="center")
plt.tight_layout()
plt.savefig("figures/model_benchmark.png")
plt.close()

G = nx.DiGraph()
for r in corridors.itertuples():
    G.add_edge(r.source_center, r.destination_center, late=r.late_share, n=r.n_trips)

main_comp = max(nx.weakly_connected_components(G), key=len)
H = G.subgraph(main_comp)
top10 = set(nodes.head(10).center)
pos = nx.spring_layout(H, k=0.12, seed=42)

fig, ax = plt.subplots(figsize=(10, 8))
edge_colors = ["#d9534f" if H[u][v]["late"] > 0.9 and H[u][v]["n"] >= 10 else "#cccccc"
               for u, v in H.edges()]
nx.draw_networkx_edges(H, pos, edge_color=edge_colors, alpha=0.4,
                       arrows=False, width=0.5, ax=ax)
sizes = [120 if n in top10 else 6 for n in H.nodes()]
node_colors = ["#d9534f" if n in top10 else "#4a6fa5" for n in H.nodes()]
nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=node_colors,
                       alpha=0.85, ax=ax)
name_map = nodes.set_index("center")["name"]
labels = {n: name_map[n].split(" (")[0] for n in top10 if n in H}
nx.draw_networkx_labels(H, pos, labels=labels, font_size=7, ax=ax)
ax.set_title("Delhivery network — red: top 10 bottleneck hubs and chronic corridors")
ax.axis("off")
plt.tight_layout()
plt.savefig("figures/network_bottlenecks.png")
plt.close()

t = tradeoff.reset_index().pivot(index="dist_band", columns="route_type",
                                 values="median_ratio")
t.plot(kind="bar", figsize=(6, 3.2), color={"FTL": "#2e7d32", "Carting": "#e08a00"})
plt.ylabel("median actual/OSRM ratio")
plt.title("Delay ratio by distance band and route type")
plt.axhline(1.2, ls="--", c="grey", lw=0.8)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/ftl_vs_carting.png")
plt.close()

top = nodes.head(10).iloc[::-1]
plt.figure(figsize=(7, 3.6))
plt.barh(top.name.str.split(" \\(").str[0], top.sla_breach_contrib * 100, color="#d9534f")
plt.xlabel("% of all late segments starting at hub")
plt.title("Top 10 bottleneck hubs")
plt.tight_layout()
plt.savefig("figures/bottleneck_hubs.png")
plt.close()

print("\ndone, figures saved")
