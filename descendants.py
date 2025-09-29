import pandas as pd
import svgwrite
import igraph as ig
from babel.dates import format_date
from datetime import datetime
from collections import defaultdict
from PIL import ImageFont
import re

# Load your CSV
df = pd.read_csv("data.csv", sep=";", dtype=str).set_index("id")
df = df.where(df.notnull(), None)
print(f"Loaded {len(df)} records from data.csv")

# SVG path for protestant icon
book_d = (
    "M 8.2404 6.2054"
    "L -7.9606 6.2054"
    "C -9.1892 6.2054, -10.0000 7.8160, -9.9408 8.0984"
    "L -9.1704 -8.5156"
    "C -9.1260 -9.4446, -8.6814 -9.9534, -7.8534 -9.9562"
    "L 6.3718 -9.9996"
    "C 6.4914 -10.0000, 6.6478 -9.9024, 6.6596 -9.7812"
    "Z"
    "M 7.6022 6.2326"
    "L 7.9072 9.3592"
    "C 7.9370 9.6624, 7.8196 9.9218, 7.3876 9.9236"
    "L -7.9606 9.9942"
    "C -9.1892 10.0000, -10.0000 8.6244, -9.9408 8.0984"
    "M -0.8798 3.6152"
    "L -0.8798 -7.2304"
    "M -4.4950 -3.6152"
    "L 2.7354 -3.6152"
)

mitra_d = (
    "M -4.5394 10.0000"
    "L -7.5000 1.5790"
    "C -9.4736 -1.0526, -3.5526 -7.6316, 0.0000 -10.0000"
    "C 3.5526 -7.6316, 9.4736 -1.0526, 7.5000 1.5790"
    "L 4.5394 10.0000"
    "Q 0.0000 8.1578, -4.5394 10.0000"
    "Z"
    "M 0 -5 L 0 5"
    "M -3 -1 L 3 -1"
)


def draw_confession(dwg, x, y, is_kath=False):
    d = mitra_d if is_kath else book_d
    # Translate to desired position, e.g. (300, 300)
    path = dwg.path(
        d=d,
        fill="none",
        stroke="gray",
        stroke_width=2,
        # stroke_linecap="round",
        stroke_linejoin="round",
        transform=f"translate({x},{y}) scale(0.45)",
    )
    dwg.add(path)


def children_grouped_by_union(df_kids, focal_id):
    groups = (
        df_kids.reset_index()
        .groupby("parent2_id", dropna=False)
        .id.agg(list)
        .rename("children")
        .reset_index()
    )
    # print(df.loc[focal_id]["name"])
    # for row in groups.itertuples(index=False):
    #     print("  ", row.parent2_id, "→", [df.loc[id]["name"] for id in row.children])


def get_spouses(person_id, person_sx, spouse_id) -> list:
    """Create ordered list of spouse nodes and children hubs."""

    spouses = [] if not spouse_id else spouse_id.split(":")

    spid = None
    if len(spouses) != 0 and spouses[0] != "-":
        # check spouse of first spouse
        spid = df.loc[spouses[0]].spouse_id

    # contraint: spouse of spouse must be before current spouse
    if spid:
        if len(spouses) == 1:
            node_group = [spid, spouses[0], person_id]
            hub_group = [(spouses[0], spid), (person_id, spouses[0]), None, None]
        else:
            node_group = [spid, spouses[0], person_id, spouses[1]]
            hub_group = [
                (spouses[0], spid),
                (person_id, spouses[0]),
                (person_id, spouses[1]),
                None,
            ]
    else:
        if len(spouses) == 0:
            node_group = [person_id]
            hub_group = [(person_id, None)]
        elif len(spouses) == 1:
            if person_sx == "f":
                node_group = [*spouses, person_id]
                hub_group = [(person_id, None), (person_id, spouses[0])]
            else:
                node_group = [person_id, *spouses]
                hub_group = [(person_id, spouses[0]), (person_id, None)]
        else:
            node_group = [spouses[0], person_id, spouses[1]]
            hub_group = [(person_id, sp) for sp in spouses]

    hubs = [
        (person_id, f"hub-{x[0]}:{x[1]}" if x else f"hub-0{i}")
        for i, x in enumerate(hub_group)
    ]

    return (node_group, hubs)


def create_graph(df: pd.DataFrame, exclude: list = []) -> ig.Graph:
    """ """

    hub_nodes = defaultdict(list)
    edges = []

    # first pass: create nodes with hubs
    for index, entry in df.iterrows():

        person_id = index
        if not "sex" in entry:
            print(f"person {index}/{person_id} has no sex.")
            continue

        person_sx = entry["sex"]
        parent_id = entry["parent1_id"]
        parent2_id = entry["parent2_id"]
        if parent_id and "*" in parent_id:
            parent_id = parent_id.replace("*", "")
        spouse_id = entry["spouse_id"]
        child_match = df.parent1_id.str.replace("*", "") == person_id
        child_cnt = sum(child_match)

        # group children unions
        if not person_id in exclude and child_cnt != 0:
            children_grouped_by_union(df[child_match], person_id)

        # ignore spouse nodes - handled separately
        if person_id in exclude:
            continue

        (spouse_nodes, hubs) = get_spouses(person_id, person_sx, spouse_id)
        for sp in spouse_nodes:
            if sp != "-":
                edges.append((f"anchor-{person_id}", sp))

        # add hub nodes for children to connect
        if child_cnt:
            for node, hub in hubs:
                prefix = hub.split(":")[0] + ":"
                if sp != "-":
                    hub_nodes[prefix].append(hub)
                    edges.append((node, hub))

    # second pass: connect nodes to parent hubs
    for index, entry in df.iterrows():

        person_id = index
        parent_id = entry["parent1_id"]
        parent2_id = entry["parent2_id"]
        if parent_id and "*" in parent_id:
            parent_id = parent_id.replace("*", "")

        # ignore spouse nodes - handled separately
        if person_id in exclude:
            continue

        # find right parent hub node
        if parent_id:
            hubs = hub_nodes.get(f"hub-{parent_id}:", [])
            match len(hubs):
                case 0:
                    print(
                        f"no hub >>> {person_id}--{parent_id} -> {hubs} ({parent2_id})"
                    )
                case 1:
                    edges.append((hubs[0], f"anchor-{person_id}"))
                case 2:
                    hubs2 = [h for h in hubs if h == f"hub-{parent_id}:{parent2_id}"]
                    if len(hubs2) == 1:
                        edges.append((hubs2[0], f"anchor-{person_id}"))
                    else:
                        print(
                            f"no hub >>> {person_id}--{parent_id} -> {hubs} ({parent2_id})",
                        )
                case _:
                    print(f"does not support {len(hubs)} hubs")

    return ig.Graph.TupleList(edges, directed=True, vertex_name_attr="name")


def create_layout(g: ig.Graph, root: list):
    layout = g.layout_reingold_tilford(root=root, mode="out")

    # swap x and y axis and normalize to positive values
    coords = layout.coords
    offset = -min([x for x, y in coords])
    return [(y, x + offset) for x, y in coords]


def to_canvas(coords):
    return [(x * x_unit + x_margin, y * y_unit + y_margin) for x, y in coords]


# spouse_ids = set(df["spouse_id"]) - {None}
spouse_ids = df["spouse_id"].dropna().str.split(":").explode().pipe(set) - {None}

root_nodes = [p for p in df[df["parent1_id"].isnull()].index if not p in spouse_ids]
# root_nodes = ['531035']

g = create_graph(df, exclude=spouse_ids)


# Create SVG with x/y coordinate swap for left-to-right layout
svg_width = 1500
svg_height = 10000
box_width = 168
box_height = 58
box_gap = 8
sib_gap = 8
x_unit = box_width * 0.5
y_unit = box_height + box_gap
x_margin = -box_width * 0.4
y_margin = box_height * 0.8
# text_font="Apple Chancery, cursive"
# text_font = "Arial, sans-serif"
text_font = "Georgia, 'Times New Roman', Times, serif"
font_name = text_font.split(",")[0]
font_path = f"/System/Library/Fonts/Supplemental/{font_name} Bold.ttf"

roots = [f"anchor-{id}" for id in root_nodes]
coords = create_layout(g, roots)
coords = to_canvas(coords)
names = g.vs["name"]

# ---
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_debug_tree_svg(
    g: ig.Graph,
    roots,
    outfile: str = "tree.svg",
    box_w: int = 100,
    box_h: int = 20,
    hgap: int = 40,
    vgap: int = 10,
):
    """
    Draw a tree/forest laid out with Reingold–Tilford, with EXACT 100x30 boxes
    (configurable via box_w/box_h) and spacing that guarantees no overlap.
    Saves an SVG.

    Parameters
    ----------
    g : ig.Graph
        Directed graph.
    roots : int | list[int] | list[str]
        Root or roots (indices or names) for layout.
    outfile : str
        Output SVG filename.
    box_w, box_h : int
        box width and height in pixels.
    hgap, vgap : int
        Horizontal/vertical gaps between neighboring boxes (in pixels).
    """
    # 1) Get a tidy tree layout from igraph (relative coords)
    layout = g.layout_reingold_tilford(root=roots, mode="out")

    # 2) Normalize & scale layout to pixel centers with non-overlap spacing
    xs = [pt[0] for pt in layout]
    ys = [pt[1] for pt in layout]
    min_x, min_y = min(xs), min(ys)

    # Scale so each unit step in layout → box size + gap in pixels
    scale_w = box_w + hgap
    scale_h = box_h + vgap

    cy = [(x - min_x) * scale_h for x in xs]  # centers (top→bottom orientation for now)
    cx = [(y - min_y) * scale_w for y in ys]

    # 3) Prepare labels (prefer vertex 'name', fallback to index)
    labels = [
        (
            str(v["name"])
            if "name" in g.vs.attributes() and v["name"] is not None
            else str(v.index)
        )
        for v in g.vs
    ]

    # 4) Compute figure bounds with margins
    m = 10  # margin
    min_cx, max_cx = min(cx), max(cx)
    min_cy, max_cy = min(cy), max(cy)
    width_px = int((max_cx - min_cx) + box_w + 2 * m)
    height_px = int((max_cy - min_cy) + box_h + 2 * m)

    # Matplotlib figures are sized in inches; assume 100 dpi for easy px math
    dpi = 100
    fig_w_in = max(4, width_px / dpi)
    fig_h_in = max(3, height_px / dpi)

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)

    # 5) Draw edges (straight lines connecting box edges)
    for e in g.es:
        u, v = e.tuple
        x1, y1 = cx[u], cy[u]
        x2, y2 = cx[v], cy[v]

        # left_to_right:
        start = (x1 + box_w / 2, y1)
        end = (x2 - box_w / 2, y2)

        ax.plot([start[0], end[0]], [start[1], end[1]], linewidth=1.2, c="black")

    # 6) Draw boxes and labels
    for i in range(g.vcount()):
        x, y = cx[i], cy[i]
        # Rectangle expects lower-left corner; we have centers
        llx = x - box_w / 2
        lly = y - box_h / 2
        rect = Rectangle((llx, lly), box_w, box_h, fill=False, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, labels[i], ha="center", va="center")

    # 7) Finalize canvas
    ax.set_xlim(min_cx - box_w / 2 - m, max_cx + box_w / 2 + m)
    ax.set_ylim(min_cy - box_h / 2 - m, max_cy + box_h / 2 + m)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)

    # 8) Save SVG
    fig.savefig(outfile, format="svg", bbox_inches="tight")
    plt.close(fig)
    return outfile


draw_debug_tree_svg(g, roots=roots, outfile="debug.svg")
# ---

# post-process coordinates
levels = defaultdict(list)
for idx, (x, y) in enumerate(coords):
    levels[x].append(idx)
selected_levels = sorted(levels.keys())[::3]


def get_level_nodes(level):
    # get index of nodes in this level
    return [
        idx
        for idx, _ in sorted(
            ((i, y) for i, (x, y) in enumerate(coords) if abs(x - level) < 1e-6),
            key=lambda t: t[1],
        )
    ]


def apply_gaps(nodes, gaps):
    for idx, gap in zip(nodes, gaps):
        children = g.subcomponent(idx, mode="OUT")
        for node in children:
            x, y = coords[node]
            coords[node] = (x, y + gap)
        # parents = g.subcomponent(idx, mode="IN")
        # for node in parents:
        #     x, y = coords[node]
        #     coords[node] = (x, y + gap / 2)


# add gap between sibling trees on all levels (except leaf nodes)
for level in selected_levels[:-1]:
    level_idxs = get_level_nodes(level)
    gaps = [i * sib_gap for i in range(len(level_idxs))]
    # apply_gaps(level_idxs, gaps)


def date2str(value: str) -> str:
    if pd.isnull(value) or value == "":
        return ""
    if value.startswith("#"):
        return value
    if value.startswith("x"):
        value = value[1:]

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


def underline_quoted_text(dwg, text, font_size, x_pos, y_pos):
    # get length of text
    font = ImageFont.truetype(font_path, int(font_size.rstrip("'pxptem%")))
    text_width = font.getlength(text.replace("'", ""))

    # get locations of quotes and translate to angles from center
    positions = [m.start() for m in re.finditer("'", text)]
    widths = [font.getlength(text[0:pos].replace("'", "")) for pos in positions]
    pos = [x_pos - text_width / 2 + w for w in widths]

    if len(pos) != 2:
        print(f"invalid underline '{text}' ({pos})")
        return

    path_d = f"M {pos[0]},{y_pos} L {pos[1]},{y_pos}"

    path = dwg.path(d=path_d, fill="none", stroke="black", stroke_width=1.1)
    dwg.add(path)


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

    if (df.index == name).sum() > 1:
        print("WARN duplicate key:", name)

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
    # add link
    if len(name) > 4:
        link = dwg.a(
            f"https://meine-ahnen.eu/-/MA.dll?T=gedsql:db:indichart&home_id=0&indi={name}",
            target="_blank",
        )
        link.add(box)
        dwg.add(link)

        # add person id left side of box
        id_text = dwg.text(
            person.name,
            insert=(x - box_width / 2 + 6, y),
            text_anchor="middle",
            dominant_baseline="middle",
            font_size="8px",
            font_family=text_font,
            fill="white",
            # font_weight="bold",
        )
        id_text.rotate(-90, center=(x - box_width / 2 + 6, y))
        dwg.add(id_text)
    else:
        dwg.add(box)

    extra_space = 4 if person.occupation else 0

    name = person["name"] or ""

    # Add name text
    name_text = dwg.text(
        name.replace("'", ""),
        insert=(x, y - 14 - extra_space),
        text_anchor="middle",
        dominant_baseline="middle",
        font_size="10px",
        font_family=text_font,
        fill="black",
        font_weight="bold",
    )
    dwg.add(name_text)

    if "'" in name:
        underline_quoted_text(dwg, person["name"], "10px", x, y - 9 - extra_space)

    name_text = dwg.text(
        person["occupation"] or "",
        insert=(x, y - 4),
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
    dd = person["death_date"]
    kia = dd and dd.startswith("x")
    deathdate = date2str(dd)
    info_lines.append(f"* {birthdate}")
    info_lines.append(f"{'⚔' if kia else '†'} {deathdate}")

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

    # add religion
    if person.rel:
        draw_confession(
            dwg, x - box_width / 2 - 8, y - box_height / 2 + 8, person.rel == "kath"
        )

# for coord lookup
name_to_idx = {name: idx for idx, name in enumerate(names)}
marriage_coords = {}

# draw spouse connections
for idx, person in df[df.spouse_id.notna()].iterrows():
    spouses = person.spouse_id.split(":")
    for i, spouse_id in enumerate(spouses):
        idx_coords = coords[name_to_idx[idx]]

        if spouse_id != "-" and not spouse_id in name_to_idx:
            print("WARN spouse not found:", spouse_id)
            continue

        sp_coords = coords[name_to_idx[spouse_id]] if spouse_id != "-" else idx_coords

        # always apply a little offset in y direction
        if spouse_id == "-":
            # assuming "unknown father" to be first
            idx_coords = (idx_coords[0], idx_coords[1] - 2)
            sp_coords = (sp_coords[0], sp_coords[1] - 2)
        else:
            offset = 1.5 if idx_coords[1] > sp_coords[1] else -1.5
            idx_coords = (idx_coords[0], idx_coords[1] - offset)
            sp_coords = (sp_coords[0], sp_coords[1] + offset)

        (x1, y1), (x2, y2) = idx_coords, sp_coords
        x = x1 + box_width / 2 + 4

        marr_id = idx if len(spouses) == 1 else f"{idx}+{spouse_id}"
        marriage_coords[marr_id] = (x, (y1 + y2) / 2)
        d = f"M {x},{y1} L {x+24},{y1} L {x+24},{y2} L {x},{y2}"
        dwg.add(dwg.path(d=d, stroke="lightgray", fill="none", stroke_width=1.2))

# draw parent connections
mask = df["parent1_id"].notna() & ~df.index.isin(spouse_ids)
for idx, person in df[mask].iterrows():

    dashed = "none"
    parent_id = person.parent1_id
    if parent_id and "*" in parent_id:
        parent_id = parent_id.replace("*", "")
        dashed = "5 5"

    # get info about parent (and spouse)
    p_spouses = df.loc[parent_id].spouse_id.split(":")
    if p_spouses[0] != "-":
        spit = df.loc[p_spouses[0]].spouse_id
    indent = spit != None or len(p_spouses) > 1 and p_spouses[1] == person.parent2_id

    # add second parent
    if person.parent2_id:
        parent_id += f"+{person.parent2_id}"

    # must have both parents
    if parent_id not in marriage_coords:
        print("parent not found:", idx, person["name"], "->", parent_id)
        continue

    (cx, cy), (px, py) = coords[name_to_idx[idx]], marriage_coords[parent_id]
    cx = cx - box_width / 2 - 4
    x_mid = cx - 24
    x_mid += indent * 8
    d = f"M {px+24},{py} L {x_mid},{py} L {x_mid},{cy} L {cx},{cy}"
    dwg.add(
        dwg.path(
            d=d,
            stroke="lightgray",
            fill="none",
            stroke_width=1.2,
            stroke_dasharray=dashed,
        )
    )

# place marriage info
for idx, person in df[df["marriage_date"].notna()].iterrows():

    marr_dates = person.marriage_date.split(":")
    spouses = person.spouse_id.split(":")
    places = person.place_of_marriage.split(":")

    for dt, sp_id, place in zip(marr_dates, spouses, places):
        if sp_id == "-":
            continue

        marr_id = idx if len(spouses) == 1 else f"{idx}+{sp_id}"

        marr_date = date2str(dt)
        if marr_id not in marriage_coords:
            print("WARN marriage coords not found for", idx, person["name"])
            continue
        (x, y) = marriage_coords[marr_id]

        text = dwg.text(
            f"⚭ {marr_date}",
            insert=(x + 6, y),
            text_anchor="middle",
            dominant_baseline="middle",
            font_size="9px",
            font_family=text_font,
            fill="#666666",
        )
        text.rotate(-90, center=(x + 6, y))
        dwg.add(text)

        text = dwg.text(
            place or "",
            insert=(x + 16, y),
            text_anchor="middle",
            dominant_baseline="middle",
            font_size="9px",
            font_family=text_font,
            fill="#666666",
        )
        text.rotate(-90, center=(x + 16, y))
        dwg.add(text)


dwg.save()
print("SVG file created: descendants_tree.svg")
