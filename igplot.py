import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig

# Construct a genealogy graph with a nice layout of the person hierarchy
# 1. at each level, people from different sub trees are well distributed and non overlapping
# 2. spouses are well integrated in the graph
# 3. children are centered below spouses

# 1) Load your CSV (with columns "id" and "parent_id")
df = pd.read_csv("data.csv", sep=";", dtype=str)
df = df.where(df.notnull(), None)

edges = []
spouse_ids = set(df["spouse_id"]) - {None}

for _, entry in df.iterrows():
    person_id = entry["id"]
    parent_id = entry["father_id"]
    spouse_id = entry["spouse_id"]
    child_cnt = sum(df["father_id"] == person_id)

    # ignore spouse nodes - handled separately
    if person_id in spouse_ids:
        continue

    # create anchor node for marriage
    edges.append((f"anchor-{person_id}", person_id))

    # connect spouse if exists
    if spouse_id:
        edges.append((f"anchor-{person_id}", spouse_id))

    # connect parent's descendants hub with anchor if exists
    if parent_id:
        edges.append((f"hub1-{parent_id}", f"anchor-{person_id}"))

    # create descendants hub if children exist
    if child_cnt:
        edges.append((person_id, f"hub0-{person_id}"))
        edges.append((person_id, f"hub1-{person_id}"))

g = ig.Graph.TupleList(edges, directed=True, vertex_name_attr="name")
layout = g.layout_reingold_tilford(root=["anchor-0"], mode="out")

names = g.vs["name"]
coords = layout.coords  # list of (x, y)
# normalize to positive x values
offset = -min([x for x, y in coords])
coords = [(x + offset, y) for x, y in coords]

plt.figure(figsize=(12, 8))

# Draw edges, but skip any edge touching a hub0-* node
for src, dst in g.get_edgelist():
    n1, n2 = names[src], names[dst]
    if n1.startswith("hub0-") or n2.startswith("hub0-"):
        continue
    x1, y1 = coords[src]
    x2, y2 = coords[dst]
    plt.plot([x1, x2], [y1, y2], color="gray", linewidth=0.8)

# Draw nodes (skip hub0-* nodes)
for idx, (x, y) in enumerate(coords):
    name = names[idx]
    if "-" in name:
        continue
    plt.scatter(x, y, s=50, color="blue")
    plt.text(x, y, name, fontsize=8, ha="center", va="bottom")


plt.title("Reingold–Tilford Layout via python-igraph")
plt.gca().invert_yaxis()
# plt.axis("off")
plt.tight_layout()
plt.show()
