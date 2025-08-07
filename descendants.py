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
svg_width = 1040
svg_height = 1600
box_width = 168
box_height = 58
box_gap = 12
x_unit = box_width * 0.55
y_unit = box_height + box_gap
x_margin = -box_width * 0.5
y_margin = box_height * 0.8
# text_font="Apple Chancery, cursive"
# text_font = "Arial, sans-serif"
text_font = "Georgia, 'Times New Roman', Times, serif"

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
# dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill="white"))


def get_colors(is_male: bool, is_spouse: bool):
    """
    Returns (fill_color, stroke_color) based on gender and spouse-status.
    Spouses get a muted variant of the base palette.
    """
    # Base (non-spouse) palette
    if is_male:
        base_fill, base_stroke = "#E3F2FD", "#4A90E2"  # light blue / blue
    else:
        base_fill, base_stroke = "#FCE4EC", "#FF6EC7"  # light pink / pink

    if not is_spouse:
        return base_fill, base_stroke

    # Muted variants for spouses
    if is_male:
        mut_fill, mut_stroke = "#E3F2FD", "white"  # softer blues
    else:
        mut_fill, mut_stroke = "#FCE4EC", "white"  # softer pinks

    return mut_fill, mut_stroke


# Draw nodes (skip helper nodes)
for idx, (x, y) in enumerate(coords):
    name = names[idx]
    if "-" in name:
        continue

    person = df.loc[name]

    is_male = person["sex"] == "m"
    is_spouse = name in spouse_ids
    fill, stroke = get_colors(is_male, is_spouse)

    # Draw box
    box = dwg.rect(
        insert=(x - box_width / 2, y - box_height / 2),
        size=(box_width, box_height),
        fill=fill,
        stroke=stroke,
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
        font_family=text_font,
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
        font_family=text_font,
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
                insert=(x - 4, y - 18 + box_height / 2 + k * 12),
                text_anchor="end",
                font_size="10px",
                font_family=text_font,
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
                insert=(x, y - 18 + box_height / 2 + k * 12),
                text_anchor="start",
                font_size="10px",
                font_family=text_font,
                fill="#666666",
            )
            dwg.add(info_text)

# for coord lookup
name_to_idx = {name: idx for idx, name in enumerate(names)}
marriage_coords = {}

# draw spouse connections
for idx, person in df[df.spouse_id.notna()].iterrows():
    (x1, y1), (x2, y2) = coords[name_to_idx[idx]], coords[name_to_idx[person.spouse_id]]

    x = x1 + box_width / 2 + 4
    marriage_coords[idx] = (x, (y1 + y2) / 2)
    d = f"M {x},{y1} L {x+24},{y1} L {x+24},{y2} L {x},{y2}"
    dwg.add(dwg.path(d=d, stroke="lightgray", fill="none", stroke_width=1.2))

# draw parent connections
mask = df["father_id"].notna() & ~df.index.isin(spouse_ids)
for idx, person in df[mask].iterrows():

    (cx, cy), (px, py) = coords[name_to_idx[idx]], marriage_coords[person.father_id]
    cx = cx - box_width / 2 - 4
    x_mid = cx - 24
    d = f"M {px+24},{py} L {x_mid},{py} L {x_mid},{cy} L {cx},{cy}"
    dwg.add(dwg.path(d=d, stroke="lightgray", fill="none", stroke_width=1.2))

# place marriage info
for idx, person in df[df["marriage_date"].notna()].iterrows():
    marr_date = date2str(person.marriage_date)
    (x, y) = marriage_coords[idx]

    text = dwg.text(
        f"⚭ {marr_date}",
        insert=(x + 6, y),
        text_anchor="middle",
        dominant_baseline="middle",
        font_size="8px",
        font_family=text_font,
        fill="#666666",
    )
    text.rotate(-90, center=(x + 6, y))
    dwg.add(text)

    text = dwg.text(
        person.place_of_marriage,
        insert=(x + 16, y),
        text_anchor="middle",
        dominant_baseline="middle",
        font_size="8px",
        font_family=text_font,
        fill="#666666",
    )
    text.rotate(-90, center=(x + 16, y))
    dwg.add(text)

dwg.save()
print("SVG file created: descendants_tree.svg")
