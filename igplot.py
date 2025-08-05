import pandas as pd
import matplotlib.pyplot as plt
from igraph import Graph

# 1) Load your CSV (with columns "id" and "parent_id")
df = pd.read_csv("data.csv", sep=";", dtype=str)
df = df.where(df.notnull(), None)

# 1) Build person → index mapping
ids = df["id"].tolist()
id_to_idx = {pid: i for i, pid in enumerate(ids)}

# 2) Determine which IDs get a pair node: 
#    only those that appear as a row ID, not those that appear only as a spouse
spouse_ids = set(df["spouse_id"]) - {""}
pair_ids = [pid for pid in ids if pid not in spouse_ids]
pair_to_idx = {f"pair_{pid}": idx + len(ids) for idx, pid in enumerate(pair_ids)}

# Unified lookup
node_index = {**id_to_idx, **pair_to_idx}

edges = []
roots = []

for _, row in df.iterrows():
    pid = row["father_id"]
    cid = row["id"]
    cid_idx = id_to_idx[cid]
    pair_key = f"pair_{cid}"
    
    # 3a) Only create edges from parent → pair if this cid has a pair node
    if pair_key in pair_to_idx:
        pair_idx = pair_to_idx[pair_key]
        # parent → pair
        if pid and pid in id_to_idx:
            edges.append((id_to_idx[pid], pair_idx))
        else:
            # no parent ⇒ this pair is a root
            roots.append(pair_idx)
        # pair → person
        edges.append((pair_idx, cid_idx))

        # 3b) spouse → same pair (if exists)
        sp = row.get("spouse_id")
        if sp and sp in id_to_idx:
            edges.append((pair_idx, id_to_idx[sp]))
    else:
        # This CID has no pair (it is only ever a spouse), so attach it directly to its parent
        if pid and pid in id_to_idx:
            edges.append((id_to_idx[pid], cid_idx))
        else:
            roots.append(cid_idx)

# 4) Build the igraph and layout
g = Graph(edges=edges, directed=True)
layout = g.layout_reingold_tilford(root=roots[:1], mode="out")

coords = layout.coords

# normalize to positive values
offset = -min([x for x, y in coords])
coords = [(x + offset, y) for x, y in coords]


plt.figure(figsize=(10, 8))
for src, dst in edges:
    x1, y1 = coords[src]
    x2, y2 = coords[dst]
    plt.plot([x1, x2], [y1, y2], color="gray", linewidth=0.8)
xs, ys = zip(*coords)
plt.scatter(xs, ys, s=30, color="crimson")

# Optionally label nodes by their original IDs
for node_id, (x, y) in zip(ids, coords):
    plt.text(x, y, node_id, fontsize=8, ha="center", va="bottom")

plt.gca().invert_yaxis()
plt.title("Reingold–Tilford Layout via python-igraph")
plt.xlabel("X position")
plt.ylabel("Depth (Y)")
plt.tight_layout()
plt.show()
