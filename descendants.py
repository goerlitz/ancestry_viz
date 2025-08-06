import pandas as pd
import svgwrite
import igraph as ig
from babel.dates import format_date
from datetime import datetime

# Load your CSV
df = pd.read_csv("data.csv", sep=";", dtype=str).set_index("id")
df = df.where(df.notnull(), None)
print(f"Loaded {len(df)} records from data.csv")


def create_graph(df: pd.DataFrame, exclude: list = []) -> ig.Graph:

    edges = []

    for index, entry in df.iterrows():
        person_id = index
        person_sx = entry["sex"]
        parent_id = entry["father_id"]
        spouse_id = entry["spouse_id"]
        child_cnt = sum(df["father_id"] == person_id)

        # ignore spouse nodes - handled separately
        if person_id in exclude:
            continue

        # connect male spouse before female person (if exist)
        if person_sx == "f" and spouse_id:
            edges.append((f"anchor-{person_id}", spouse_id))

        # create anchor node for marriage
        edges.append((f"anchor-{person_id}", person_id))

        # connect female spouse after male person (if exists)
        if person_sx == "m" and spouse_id:
            edges.append((f"anchor-{person_id}", spouse_id))

        # connect parent's descendants hub with anchor (if exists)
        if parent_id:
            edges.append((f"hub1-{parent_id}", f"anchor-{person_id}"))

        # create descendants hub for children (if exist)
        if child_cnt:
            edges.append((person_id, f"hub0-{person_id}"))
            edges.append((person_id, f"hub1-{person_id}"))

    return ig.Graph.TupleList(edges, directed=True, vertex_name_attr="name")


def create_layout(g: ig.Graph, root: list):
    layout = g.layout_reingold_tilford(root=root, mode="out")

    # swap x and y axis and normalize to positive values
    coords = layout.coords
    offset = -min([x for x, y in coords])
    return [(y, x + offset) for x, y in coords]


def to_canvas(coords):
    return [(x * x_unit + x_margin, y * y_unit + y_margin) for x, y in coords]


spouse_ids = set(df["spouse_id"]) - {None}
root_nodes = [p for p in df[df["father_id"].isnull()].index if not p in spouse_ids]

g = create_graph(df, exclude=spouse_ids)


# Create SVG with x/y coordinate swap for left-to-right layout
svg_width = 1000
svg_height = 1600
box_width = 160
box_height = 54
box_gap = 10
x_unit = box_width * 0.55
y_unit = box_height + box_gap
x_margin = -box_width * 0.5
y_margin = box_height * 0.8

coords = create_layout(g, [f"anchor-{id}" for id in root_nodes])
coords = to_canvas(coords)
names = g.vs["name"]


def date2str(value: str) -> str:
    if pd.isnull(value) or value == "":
        return ""

    try:
        date_obj = datetime.strptime(value, "%Y-%m-%d")
        # 'd. MMM y' = e.g., 15. Jan 1880 in German format
        return format_date(date_obj, format="d. MMM y", locale="de")
    except ValueError:
        return str(value)


# Create SVG drawing
dwg = svgwrite.Drawing("./descendants_tree.svg", size=(svg_width, svg_height))

# Add background
dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill="white"))


# Draw nodes (skip helper nodes)
for idx, (x, y) in enumerate(coords):
    name = names[idx]
    if "-" in name:
        continue

    person = df.loc[name]

    is_male = person["sex"] == "m"

    # Color based on gender - light backgrounds with colored borders
    fill_color = (
        "#E3F2FD" if is_male else "#FCE4EC"
    )  # Light blue for male, Light pink for female
    stroke_color = (
        "#4A90E2" if is_male else "#FF6EC7"
    )  # Blue border for male, Pink border for female

    # Draw box
    box = dwg.rect(
        insert=(x - box_width / 2, y - box_height / 2),
        size=(box_width, box_height),
        fill=fill_color,
        stroke=stroke_color,
        stroke_width=1.5,
        rx=4,  # rounded corners
    )
    dwg.add(box)

    # Add name text
    name_text = dwg.text(
        person["name"],
        insert=(x, y - 18),
        text_anchor="middle",
        dominant_baseline="middle",
        font_size="10px",
        font_family="Arial, sans-serif",
        fill="black",
        font_weight="bold",
    )
    dwg.add(name_text)

    name_text = dwg.text(
        person["occupation"] or "",
        insert=(x, y - 6),
        text_anchor="middle",
        dominant_baseline="middle",
        font_size="10px",
        font_family="Arial, sans-serif",
        fill="black",
    )
    dwg.add(name_text)

    # Add birth/death info if available
    info_lines = []
    birthdate = date2str(person["birth_date"])
    deathdate = date2str(person["death_date"])
    info_lines.append(f"* {birthdate}")
    info_lines.append(f"† {deathdate}")

    if info_lines:
        for k, info in enumerate(info_lines):
            info_text = dwg.text(
                info,
                insert=(x - 4, y - 16 + box_height / 2 + k * 12),
                text_anchor="end",
                font_size="10px",
                font_family="Arial, sans-serif",
                fill="#666666",
            )
            dwg.add(info_text)

    # Add birth/death place if available
    info_lines = []
    birthplace = person["place_of_birth"]
    deathplace = person["place_of_death"]
    info_lines.append(birthplace or "")
    info_lines.append(deathplace or "")

    if info_lines:
        for k, info in enumerate(info_lines):
            info_text = dwg.text(
                info,
                insert=(x, y - 16 + box_height / 2 + k * 12),
                text_anchor="start",
                font_size="10px",
                font_family="Arial, sans-serif",
                fill="#666666",
            )
            dwg.add(info_text)

# for coord lookup
name_to_idx = {name: idx for idx, name in enumerate(names)}

# draw spouse connections
parent_coords = {}
for idx, person in df[df.spouse_id.notna()].iterrows():
    (x1, y1), (x2, y2) = coords[name_to_idx[idx]], coords[name_to_idx[person.spouse_id]]

    x = x1 + box_width / 2 + 4
    parent_coords[idx] = (x, (y1 + y2) / 2)
    d = f"M {x},{y1} L {x+24},{y1} L {x+24},{y2} L {x},{y2}"
    dwg.add(dwg.path(d=d, stroke="lightgray", fill="none", stroke_width=1.5))

# draw parent connections
mask = df["father_id"].notna() & ~df.index.isin(spouse_ids)
for idx, person in df[mask].iterrows():

    (cx, cy), (px, py) = coords[name_to_idx[idx]], parent_coords[person.father_id]
    cx = cx - box_width / 2 - 4
    x_mid = cx - 24
    d = f"M {px+24},{py} L {x_mid},{py} L {x_mid},{cy} L {cx},{cy}"
    dwg.add(dwg.path(d=d, stroke="lightgray", fill="none", stroke_width=1.5))

dwg.save()
print("SVG file created: descendants_tree.svg")
