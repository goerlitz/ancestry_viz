import pandas as pd
import svgwrite
from babel.dates import format_date
from datetime import datetime
from bigtree import Node, reingold_tilford, plot_tree

# Load your CSV
df = pd.read_csv("data.csv", sep=";", dtype=str)
df = df.where(df.notnull(), None)
print(f"Loaded {len(df)} records from data.csv")

# Detect the root (person with no FatherID or MotherID)
potential_roots = df[df["father_id"].isnull() & df["mother_id"].isnull()]
if potential_roots.empty:
    raise ValueError("No root person found")
root_id = potential_roots.iloc[0]["id"]
print(f"Root person ID: {root_id}")

# Build a tree of Nodes
nodes = {}


def build_tree(person_id):
    if person_id in nodes:
        return nodes[person_id]

    row = df[df["id"] == person_id].iloc[0]
    name = row.get("name", f"Person_{person_id}")

    # Create person node
    node = Node(
        name=name,
        person_id=str(person_id),
        birth_date=row.get("birth_date"),
        death_date=row.get("death_date"),
        occupation=row.get("occupation"),
        spouse_id=row.get("spouse_id"),
        sex=row.get("sex"),
    )
    nodes[person_id] = node

    # Always insert marriage node above person for layout positioning
    # (even if person has no spouse
    marriage_node = Node(name=f"Marriage_{person_id}")
    nodes[f"marriage_{person_id}"] = marriage_node

    spouse_id = row.get("spouse_id")
    if pd.notnull(spouse_id):
        # Ensure spouse node exists
        if spouse_id not in nodes:
            spouse_row = df[df["id"] == spouse_id].iloc[0]
            spouse_name = spouse_row.get("name", f"Person_{spouse_id}")
            spouse_node = Node(
                name=spouse_name,
                person_id=str(spouse_id),
                birth_date=spouse_row.get("birth_date"),
                death_date=spouse_row.get("death_date"),
                occupation=spouse_row.get("occupation"),
                sex=spouse_row.get("sex"),
            )
            nodes[spouse_id] = spouse_node
        else:
            spouse_node = nodes[spouse_id]
        # Attach both person and spouse as children of marriage node (female first)
        if node.sex == "f":
            node.parent = marriage_node
            spouse_node.parent = marriage_node
        else:
            spouse_node.parent = marriage_node
            node.parent = marriage_node
    else:
        # Attach only the person as child if no spouse
        node.parent = marriage_node

    # Attach children to main person ---
    children = df[(df["father_id"] == person_id) | (df["mother_id"] == person_id)]
    for _, child in children.iterrows():
        child_node = build_tree(child["id"])
        if child_node:
            child_node.parent = node

    return marriage_node


root = build_tree(root_id)
if not root:
    raise ValueError("Failed to build tree")

# Compute layout positions
reingold_tilford(
    root,
    sibling_separation=1.0,
    subtree_separation=1.1,
    level_separation=0.5,
    x_offset=0.0,
    y_offset=0.0,
    reverse=False,
)


def remove_marriage_nodes(nodes):
    # First collect all marriage nodes
    marriage_nodes = [node for node in nodes if str(node.name).startswith("Marriage_")]

    for marriage_node in marriage_nodes:
        # Move marriage_node's children to its parent
        parent = marriage_node.parent
        for child in list(marriage_node.children):
            child.parent = parent
        marriage_node.parent = None  # Detach from tree (will be garbage collected)

    # rewire spouses parents
    spouses = [node for node in nodes if node.get_attr("spouse_id")]
    for spouse in spouses:
        spouse_id = spouse.get_attr("spouse_id")
        others = [node for node in nodes if node.get_attr("person_id") == spouse_id]
        for other in others:
            other.parent = spouse

    # return list without marriage nodes
    return [node for node in all_nodes if not str(node.name).startswith("Marriage_")]


# Create SVG with x/y coordinate swap for left-to-right layout
svg_width = 1200
svg_height = 1600
box_width = 160
box_height = 60
box_gap = 10
x_unit = box_width * 1.5
y_unit = box_height + box_gap
x_margin = box_width * 0.6
y_margin = box_height * 0.7

# Get min and max coordinates of all nodes including root
all_nodes = [root] + list(root.descendants)

all_nodes = remove_marriage_nodes(all_nodes)

x_coords = [node.x for node in all_nodes if hasattr(node, "x")]
y_coords = [node.y for node in all_nodes if hasattr(node, "y")]

min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)


def date2str(value: str) -> str:
    if pd.isnull(value) or value == "":
        return ""

    try:
        date_obj = datetime.strptime(value, "%Y-%m-%d")
        # 'd. MMM y' = e.g., 15. Jan 1880 in German format
        return format_date(date_obj, format="d. MMM y", locale="de")
    except ValueError:
        return str(value)


# swap coordinates and prepare dates
for node in all_nodes:
    x, y = node.x, node.y
    node.x = (max_y - y) * x_unit + x_margin
    node.y = x * y_unit + y_margin
    bd = getattr(node, "birth_date", None)
    dd = getattr(node, "death_date", None)
    node.birthdate = date2str(bd) if bd else ""
    node.deathdate = date2str(dd) if dd else ""

fig = plot_tree(root)
fig.savefig("quick_tree.png")


# Create SVG drawing
dwg = svgwrite.Drawing("./descendants_tree.svg", size=(svg_width, svg_height))

# Add background
dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill="white"))

# Draw connections between nodes (parent to child)
for node in all_nodes:

    if node.parent:

        # spouse_id = getattr(node, "spouse_id", None)
        # if spouse_id:
        # spouses = [
        #     node
        #     for node in all_nodes
        #     if getattr(node, "person_id", None) == spouse_id
        # ]
        # line = dwg.line(
        #     start=(spouses[0].x, spouses[0].y),
        #     end=(node.x, node.y),
        #     stroke="lightgray",
        #     stroke_width=2,
        # )
        # else:
        # Draw connection line
        line = dwg.line(
            start=(node.parent.x, node.parent.y),
            end=(node.x, node.y),
            stroke="lightgray",
            stroke_width=2,
        )
        dwg.add(line)

for node in all_nodes:

    # Determine gender from the sex field
    is_male = hasattr(node, "sex") and node.sex == "m"

    # Color based on gender - light backgrounds with colored borders
    fill_color = (
        "#E3F2FD" if is_male else "#FCE4EC"
    )  # Light blue for male, Light pink for female
    stroke_color = (
        "#4A90E2" if is_male else "#FF6EC7"
    )  # Blue border for male, Pink border for female

    # Draw box
    box = dwg.rect(
        insert=(node.x - box_width / 2, node.y - box_height / 2),
        size=(box_width, box_height),
        fill=fill_color,
        stroke=stroke_color,
        stroke_width=2,
        rx=4,  # rounded corners
    )
    dwg.add(box)

    # Add name text
    name_text = dwg.text(
        node.name,
        insert=(node.x, node.y - 14),
        text_anchor="middle",
        dominant_baseline="middle",
        font_size="10px",
        font_family="Arial, sans-serif",
        fill="black",
        font_weight="bold",
    )
    dwg.add(name_text)

    # Add birth/death info if available
    info_lines = []
    info_lines.append(f"* {node.birthdate}")
    info_lines.append(f"† {node.deathdate}")

    if info_lines:
        for k, info in enumerate(info_lines):
            info_text = dwg.text(
                info,
                insert=(node.x, node.y - 24 + box_height / 2 + k * 12),
                text_anchor="end",
                font_size="10px",
                font_family="Arial, sans-serif",
                fill="#666666",
            )
            dwg.add(info_text)

dwg.save()
print("SVG file created: descendants_tree.svg")
