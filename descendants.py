import pandas as pd
from bigtree import Node, reingold_tilford

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
        occupation=row.get("occupation")
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
try:
    reingold_tilford(root)
    print("Layout computed successfully")
except Exception as e:
    print(f"Error computing layout: {e}")
    exit(1)

# Step 5: Extract positions
positions = {}
try:
    # Get all nodes including root
    all_nodes = [root] + list(root.descendants)
    
    for node in all_nodes:
        if hasattr(node, 'x') and hasattr(node, 'y'):
            positions[node.person_id] = (node.x, node.y)
            print(f"{node.name} (ID {node.person_id}) → x={node.x:.3f}, y={node.y:.3f}")
        else:
            print(f"Warning: Node {node.name} missing x,y coordinates")
            
except Exception as e:
    print(f"Error extracting positions: {e}")

print(f"Extracted positions for {len(positions)} nodes")

# Optional: plot for quick visual check (requires matplotlib)
from bigtree import plot_tree
fig = plot_tree(root)
fig.savefig("quick_tree.png")
