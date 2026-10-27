import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

seg = pd.read_pickle("outputs/segments.pkl")

train = seg[seg.data == "training"].copy()
test = seg[seg.data == "test"].copy()
print("train:", len(train), "test:", len(test))

train["is_ftl"] = (train.route_type == "FTL").astype(int)
test["is_ftl"] = (test.route_type == "FTL").astype(int)

base_features = ["osrm_time", "osrm_distance", "hour", "dow", "is_ftl"]

corridor_hist = train.groupby(["source_center", "destination_center"]).agg(
    hist_ratio=("delay_ratio", "median"),
    hist_n=("trip_uuid", "nunique"),
)

G = nx.DiGraph()
for (s, d), row in corridor_hist.iterrows():
    G.add_edge(s, d, weight=row.hist_ratio, n=row.hist_n)

betweenness = nx.betweenness_centrality(G, weight="weight",
                                        k=min(500, G.number_of_nodes()), seed=42)
pagerank = nx.pagerank(G, weight="n")
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

src_hist = train.groupby("source_center").delay_ratio.median()
dst_hist = train.groupby("destination_center").delay_ratio.median()
global_median = train.delay_ratio.median()

n2v = Node2Vec(G, dimensions=16, walk_length=20, num_walks=40,
               workers=4, seed=42, quiet=True)
emb_model = n2v.fit(window=5, min_count=1, seed=42)
embeddings = {n: emb_model.wv[n] for n in G.nodes()}
zero_vec = np.zeros(16)

hist_lookup = corridor_hist.hist_ratio.to_dict()

def add_graph_features(d):
    d = d.copy()
    pairs = list(zip(d.source_center, d.destination_center))
    d["corridor_hist"] = [hist_lookup.get(p, global_median) for p in pairs]
    d["src_bet"] = d.source_center.map(betweenness).fillna(0)
    d["dst_bet"] = d.destination_center.map(betweenness).fillna(0)
    d["src_pagerank"] = d.source_center.map(pagerank).fillna(0)
    d["src_outdeg"] = d.source_center.map(out_deg).fillna(0)
    d["dst_indeg"] = d.destination_center.map(in_deg).fillna(0)
    d["src_hist"] = d.source_center.map(src_hist).fillna(global_median)
    d["dst_hist"] = d.destination_center.map(dst_hist).fillna(global_median)
    src_emb = np.vstack(d.source_center.map(lambda x: embeddings.get(x, zero_vec)))
    dst_emb = np.vstack(d.destination_center.map(lambda x: embeddings.get(x, zero_vec)))
    for i in range(16):
        d[f"src_emb{i}"] = src_emb[:, i]
        d[f"dst_emb{i}"] = dst_emb[:, i]
    return d

train = add_graph_features(train)
test = add_graph_features(test)

graph_features = base_features + [
    "corridor_hist", "src_bet", "dst_bet", "src_pagerank",
    "src_outdeg", "dst_indeg", "src_hist", "dst_hist",
] + [f"src_emb{i}" for i in range(16)] + [f"dst_emb{i}" for i in range(16)]


def train_and_score(features, label):
    model = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                      learning_rate=0.08, subsample=0.8,
                                      random_state=42)
    model.fit(train[features], np.log1p(train.actual_time))
    pred = np.expm1(model.predict(test[features]))
    mae = mean_absolute_error(test.actual_time, pred)
    within15 = (np.abs(pred - test.actual_time) / test.actual_time <= 0.15).mean()
    print(f"{label}: MAE {mae:.1f} min, within 15% {within15:.1%}")
    return model, pred, mae, within15


print("\nbenchmark on test set")
mae_osrm = mean_absolute_error(test.actual_time, test.osrm_time)
w15_osrm = (np.abs(test.osrm_time - test.actual_time) / test.actual_time <= 0.15).mean()
print(f"OSRM as-is: MAE {mae_osrm:.1f} min, within 15% {w15_osrm:.1%}")

base_model, base_pred, mae_b, w15_b = train_and_score(base_features, "baseline GBM")
graph_model, graph_pred, mae_g, w15_g = train_and_score(graph_features, "graph GBM")

print(f"\ngraph model vs baseline: MAE down {(mae_b - mae_g) / mae_b:.1%}, "
      f"within-15% up {(w15_g - w15_b) * 100:.1f} points")

importance = pd.Series(graph_model.feature_importances_, index=graph_features)
importance = importance.sort_values(ascending=False)
print("\ntop features:")
print(importance.head(10).round(3).to_string())

pd.DataFrame({
    "model": ["OSRM", "Baseline", "Graph-enhanced"],
    "MAE_min": [mae_osrm, mae_b, mae_g],
    "within_15pct": [w15_osrm, w15_b, w15_g],
}).to_csv("outputs/model_benchmark.csv", index=False)

test.assign(pred_graph=graph_pred, pred_base=base_pred).to_pickle("outputs/test_predictions.pkl")
importance.to_csv("outputs/feature_importance.csv")
