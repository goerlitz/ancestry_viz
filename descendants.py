import pandas as pd
import svgwrite
from bigtree import Node, reingold_tilford, plot_tree

# Step 1: Load your CSV
df = pd.read_csv("data.csv", sep=";")
print(f"Loaded {len(df)} records from data.csv")

# Step 2: Detect the root (person with no FatherID or MotherID)
potential_roots = df[df["father_id"].isnull() & df["mother_id"].isnull()]
if potential_roots.empty:
    raise ValueError("No root person found")
root_id = potential_roots.iloc[0]["id"]
print(f"Root person ID: {root_id}")

# Step 3: Build a tree of Nodes
nodes = {}


def build_tree(person_id):
    if person_id in nodes:
        return nodes[person_id]

    # Find the person in the dataframe
    person_data = df[df["id"] == person_id]
    if person_data.empty:
        print(f"Warning: Person ID {person_id} not found in data")
        return None

    row = person_data.iloc[0]

    # Create node with name as the first argument
    node = Node(
        name=row.get("name", f"Person_{person_id}"),
        person_id=str(person_id),
        birthdate=row.get("birth_date"),
        deathdate=row.get("death_date"),
        occupation=row.get("occupation"),
    )
    nodes[person_id] = node

    # Find children
    children = df[(df["father_id"] == person_id) | (df["mother_id"] == person_id)]
    for _, child in children.iterrows():
        child_node = build_tree(child["id"])
        if child_node:
            child_node.parent = node

    return node


root = build_tree(root_id)
if not root:
    raise ValueError("Failed to build tree")

print(f"Tree built successfully with root: {root.name}")

# Step 4: Compute layout positions
reingold_tilford(root)
reingold_tilford(
    root,
    sibling_separation=1.0,
    subtree_separation=1.25,
    level_separation=1,
    x_offset=0.0,
    y_offset=0.0,
    reverse=False,
)


# Create SVG with x/y coordinate swap for left-to-right layout
svg_width = 900
svg_height = 1200
box_width = 160
box_height = 60
box_gap = 10
x_unit = box_width * 1.5
y_unit = box_height + box_gap
x_margin = box_width * 0.6
y_margin = box_height * 0.7

# Get min and max coordinates of all nodes including root
all_nodes = [root] + list(root.descendants)

x_coords = [node.x for node in all_nodes if hasattr(node, "x")]
y_coords = [node.y for node in all_nodes if hasattr(node, "y")]

min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

# swap coordinates
for node in all_nodes:
    x, y = node.x, node.y
    node.x = (max_y - y) * x_unit + x_margin
    node.y = x * y_unit + y_margin

fig = plot_tree(root)
fig.savefig("quick_tree.png")

# Debug: print coordinate ranges
print("Coordinate ranges:")
for node in all_nodes:
    if hasattr(node, "x") and hasattr(node, "y"):
        print(f"  {node.name}: x={node.x}, y={node.y}")


# Create SVG drawing
dwg = svgwrite.Drawing("./descendants_tree.svg", size=(svg_width, svg_height))

# Add background
dwg.add(dwg.rect(insert=(0, 0), size=(svg_width, svg_height), fill="white"))

# Draw connections between nodes (parent to child)
for node in all_nodes:

    if node.parent:
        # Draw connection line
        line = dwg.line(
            start=(node.parent.x, node.parent.y),
            end=(node.x, node.y),
            stroke="lightgray",
            stroke_width=2,
        )
        dwg.add(line)

for node in all_nodes:

    # Draw box
    box = dwg.rect(
        insert=(node.x - box_width / 2, node.y - box_height / 2),
        size=(box_width, box_height),
        fill="#e3f2fd",
        stroke="#1976d2",
        stroke_width=2,
        rx=5,  # rounded corners
    )
    dwg.add(box)

    # Add name text
    name_text = dwg.text(
        node.name,
        insert=(node.x, node.y - 12),
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
    if hasattr(node, "birthdate") and node.birthdate:
        info_lines.append(f"* {node.birthdate}")
    if hasattr(node, "deathdate") and node.deathdate:
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
