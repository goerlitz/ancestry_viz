import svgwrite
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple
from svgwrite import gradients


@dataclass
class Person:
    ring: int
    name: str
    birthdate: str
    deathdate: str


# Parameters
center = (500, 500)
num_rings = 4
segments_per_ring = [2, 4, 8, 16]
total_angle = 200.0
segment_angles = [total_angle / i for i in segments_per_ring]  # 200/2, 200/4, ...
start_radius = 40
ring_thickness = 64
ring_gap = 44
ring_radii = [start_radius + i * (ring_thickness + ring_gap) for i in range(num_rings)]
start_angle_offset = 170
line_spacing = 16  # spacing between text lines
box_width = 120
box_height = 50
gap = 8

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import to_rgb, to_hex
# from scipy.interpolate import interp1d

# Define 4 base color pairs (dark to light) for each family line
color_families = {
    "Blue": ["#a8bedb", "#dbe9f6"],
    "Green": ["#a5d0b9", "#d9f2e6"],
    "Orange": ["#f8bd8d", "#ffe7cc"],
    "Magenta": ["#e3a7c6", "#f7dbef"],
}

# Function to interpolate colors in Lab space for perceptual uniformity
from skimage import color

def interpolate_colors_lab(start_hex, end_hex, n):
    start_rgb = np.array(to_rgb(start_hex)).reshape(1, 1, 3)
    end_rgb = np.array(to_rgb(end_hex)).reshape(1, 1, 3)

    start_lab = color.rgb2lab(start_rgb)[0, 0]
    end_lab = color.rgb2lab(end_rgb)[0, 0]

    labs = np.linspace(start_lab, end_lab, n)
    rgbs = color.lab2rgb(labs.reshape(n, 1, 3)).reshape(n, 3)
    hex_colors = [to_hex(np.clip(rgb, 0, 1)) for rgb in rgbs]
    return hex_colors

# Create the full palette: 4 families × 4 shades = 16 colors
palette = []
for family, (start, end) in color_families.items():
    shades = interpolate_colors_lab(start, end, 4)
    palette.extend(shades)

print(palette)


def polar_to_cartesian(
    radius: float, angle_degrees: float, center_point: Tuple[int, int] = center
) -> Tuple[float, float]:
    """Convert polar coordinates (radius, angle) to cartesian coordinates relative to center."""
    x = center_point[0] + radius * math.cos(math.radians(angle_degrees))
    y = center_point[1] + radius * math.sin(math.radians(angle_degrees))
    return (x, y)


def create_arc_path(
    radius: float,
    start_angle: float,
    end_angle: float,
    large_arc_flag: int = 0,
    gap_size: float = 5,
) -> str:
    """Create an SVG arc path string."""
    gap_angle = calculate_gap_angle(gap_size, radius)
    start_point = polar_to_cartesian(radius, start_angle + gap_angle)
    end_point = polar_to_cartesian(radius, end_angle - gap_angle)
    return f"M {start_point[0]},{start_point[1]} A {radius},{radius} 0 {large_arc_flag},1 {end_point[0]},{end_point[1]}"


def create_line_path(start_radius: float, end_radius: float, angle: float) -> str:
    """Create an SVG line path string."""
    start_point = polar_to_cartesian(start_radius, angle)
    end_point = polar_to_cartesian(end_radius, angle)
    return f"M {start_point[0]},{start_point[1]} L {end_point[0]},{end_point[1]}"


def calculate_gap_angle(gap_size: float, radius: float) -> float:
    """Calculate the angle offset needed to create a gap of specified size at a given radius."""
    # Convert gap size (in pixels) to angle (in degrees)
    # arc_length = radius * angle_in_radians
    # gap_size = radius * angle_in_radians
    # angle_in_radians = gap_size / radius
    # angle_in_degrees = (gap_size / radius) * (180 / pi)
    return (gap_size / radius) * (180 / math.pi)


def create_ring_segment(
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    gap_size: float = 5,
) -> str:
    """Create an SVG path that outlines a ring segment with rays and arcs and gaps between segments."""
    # Calculate angle offsets for even gaps at inner and outer radii
    inner_gap_angle = calculate_gap_angle(gap_size, inner_radius)
    outer_gap_angle = calculate_gap_angle(gap_size, outer_radius)

    # Calculate the four corners of the segment
    inner_start = polar_to_cartesian(inner_radius, start_angle + inner_gap_angle)
    inner_end = polar_to_cartesian(inner_radius, end_angle - inner_gap_angle)
    outer_start = polar_to_cartesian(outer_radius, start_angle + outer_gap_angle)
    outer_end = polar_to_cartesian(outer_radius, end_angle - outer_gap_angle)

    # Create path: inner_start -> outer_start -> outer_arc -> outer_end -> inner_end -> inner_arc -> inner_start
    large_arc_flag = 1 if (end_angle - start_angle) > 180 else 0

    path = f"M {inner_start[0]},{inner_start[1]} "  # Start at inner start
    path += f"L {outer_start[0]},{outer_start[1]} "  # Line to outer start
    path += f"A {outer_radius},{outer_radius} 0 {large_arc_flag},1 {outer_end[0]},{outer_end[1]} "  # Outer arc
    path += f"L {inner_end[0]},{inner_end[1]} "  # Line to inner end
    path += f"A {inner_radius},{inner_radius} 0 {large_arc_flag},0 {inner_start[0]},{inner_start[1]} Z"  # Inner arc back to start

    return path


# Load people data from CSV file
def load_people_from_csv(filename: str) -> List[List[Person]]:
    people_by_ring = [[] for _ in range(num_rings)]
    weddings_by_ring = [[] for _ in range(num_rings)]
    children = [[] for _ in range(2)]

    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ring_no = int(row["ring"])
            if ring_no >= 100:
                weddings_by_ring[ring_no - 100].append(row["name"])
            elif ring_no >= 10:
                person = Person(
                    ring=ring_no,
                    name=row["name"],
                    birthdate=row["birthdate"],
                    deathdate=row["birthplace"],
                )
                children[ring_no - 10].append(person)
            else:
                person = Person(
                    ring=ring_no,
                    name=row["name"],
                    birthdate=row["birthdate"],
                    deathdate=row["birthplace"],
                )
                people_by_ring[person.ring].append(person)

    # print("children", children[1])

    return (people_by_ring, weddings_by_ring, children)


# Load the data
(ring_data, wedd_data, children) = load_people_from_csv("people.csv")

# Create SVG drawing
dwg = svgwrite.Drawing("./radial_family.svg", size=("1000px", "1000px"))
# dwg.add(dwg.rect(insert=(0, 0), size=("1000px", "1000px"), fill="white"))

# Define shiny gold gradient
linear_gradient = dwg.linearGradient(
    id="gold_gradient", x1="10%", y1="0%", x2="15%", y2="100%"
)
linear_gradient.add_stop_color(offset="0%", color="#BF953F", opacity=1)
linear_gradient.add_stop_color(offset="25%", color="#FCF6BA", opacity=1)
linear_gradient.add_stop_color(offset="50%", color="#C19A2E", opacity=1)
linear_gradient.add_stop_color(offset="75%", color="#FBF5B7", opacity=1)
linear_gradient.add_stop_color(offset="100%", color="#B8860B", opacity=1)
dwg.defs.add(linear_gradient)

# Draw visible arcs and attach each text line to its own path
for ring_no, (base_radius, segments, angle_span) in enumerate(
    zip(ring_radii, segments_per_ring, segment_angles)
):
    for seg_no in range(segments):
        segment_angle = seg_no * angle_span + start_angle_offset
        start_angle = segment_angle
        end_angle = segment_angle + angle_span

        # Get person data for this segment
        person = ring_data[ring_no][seg_no]
        lines = (
            [person.name, person.birthdate]
            if ring_no == 0
            else [person.name, person.birthdate, person.deathdate]
        )

        inner_radius = base_radius
        outer_radius = base_radius + (130 if ring_no == 3 else ring_thickness)

        outline_path = create_ring_segment(
            inner_radius, outer_radius, start_angle, end_angle, gap_size=4
        )

        # Add the shaded box
        if ring_no == 0:
            fill = "url(#gold_gradient)"
        else:
            step = 16 // segments_per_ring[ring_no]
            fill = palette[seg_no * step]
        stroke = "#4A90E2" if seg_no % 2 == 0 else "#FF6EC7"
        box = dwg.path(
            d=outline_path,
            fill=fill,
            stroke=stroke,
            stroke_width=1.5,
        )
        dwg.add(box)

        for k, line in enumerate(lines):  # 3 lines per segment
            # Calculate center point of the segment
            center_angle = (start_angle + end_angle) / 2

            if ring_no == 3:  # Only outermost ring - use straight lines (rays)

                # Flip text for left half (upright)
                flip = 90 < (center_angle % 360) < 270

                # Calculate angles for the 3 lines: -3 degree, center, +3 degree
                line_angles = [
                    center_angle - 2.6,  # First line: -3 degree
                    center_angle,  # Second line: center angle
                    center_angle + 2.6,  # Third line: +3 degree
                ]
                # Center radius for all lines
                start_radius = base_radius + 8  # Start slightly inward
                end_radius = base_radius + 122  # End slightly outward
                line_angle = line_angles[2 - k if flip else k]

                # reverse the line path so text is upright
                if flip:
                    path_d = create_line_path(end_radius, start_radius, line_angle)
                else:
                    path_d = create_line_path(start_radius, end_radius, line_angle)

            else:  # Inner rings - use curved arcs
                radius = base_radius + (2.8 - k) * line_spacing  # offset each line
                if ring_no == 0:  # Innermost ring - center 2 lines
                    radius = base_radius + (2.4 - k) * line_spacing
                large_arc_flag = 1 if angle_span > 180 else 0
                path_d = create_arc_path(
                    radius, start_angle, end_angle, large_arc_flag, gap_size=12
                )

            text_anchor = "middle"
            start_offset = "50%"

            path_id = f"path_r{ring_no}_s{seg_no}_l{3-k}"
            path = dwg.path(
                d=path_d,
                fill="none",
                stroke="none",
                id=path_id,
                stroke_dasharray="2,2",
            )
            dwg.add(path)

            # Add text to each individual arc path
            text = dwg.text(
                "",
                font_size="13px" if ring_no != 0 else "15px",
                text_anchor=text_anchor,
                font_family="Georgia, 'Times New Roman', Times, serif",
            )
            text_path = dwg.textPath(f"#{path_id}", line, startOffset=start_offset)
            text.add(text_path)
            dwg.add(text)


def draw_parent_child_arcs(child_idx: int, parent_idx: int, dwg):
    parent_count = segments_per_ring[parent_idx]
    child_count = segments_per_ring[child_idx]
    parent_angle_span = segment_angles[parent_idx]
    child_angle_span = segment_angles[child_idx]
    arc_radius = ring_radii[parent_idx] - ring_gap / 2
    num_parents = parent_count
    num_children = child_count

    for i in range(num_children):
        child_center_angle = start_angle_offset + (i + 0.5) * child_angle_span
        start = polar_to_cartesian(
            ring_radii[parent_idx] - 4, child_center_angle - parent_angle_span / 2
        )
        end = polar_to_cartesian(
            ring_radii[parent_idx] - 4, child_center_angle + parent_angle_span / 2
        )
        arc_start = polar_to_cartesian(
            arc_radius, child_center_angle - parent_angle_span / 2
        )
        arc_end = polar_to_cartesian(
            arc_radius, child_center_angle + parent_angle_span / 2
        )
        arc_center = polar_to_cartesian(arc_radius, child_center_angle)
        child_center = polar_to_cartesian(
            ring_radii[child_idx] + 4 + ring_thickness, child_center_angle
        )

        wedding = wedd_data[child_idx + 1][i]

        path_d = f"M {start[0]},{start[1]}"
        path_d += f"L {arc_start[0]},{arc_start[1]}"
        path_d += f"A {arc_radius},{arc_radius} 0 0,1 {arc_end[0]},{arc_end[1]}"
        path_d += f"L {end[0]},{end[1]}"
        path_d += f"M {arc_center[0]},{arc_center[1]}"
        path_d += f"L {child_center[0]},{child_center[1]}"

        arc_path = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
        dwg.add(arc_path)

        # invisible arc for text path
        text_arc_start = polar_to_cartesian(
            arc_radius + 6, child_center_angle - parent_angle_span / 2
        )
        text_arc_end = polar_to_cartesian(
            arc_radius + 6, child_center_angle + parent_angle_span / 2
        )

        path_d = f"M {text_arc_start[0]},{text_arc_start[1]}"
        path_d += (
            f"A {arc_radius},{arc_radius} 0 0,1 {text_arc_end[0]},{text_arc_end[1]}"
        )
        path_id = f"path_w{child_idx}_s{i}"
        arc_path = dwg.path(
            d=path_d, fill="none", stroke="none", stroke_width=1, id=path_id
        )
        dwg.add(arc_path)

        text = dwg.text(
            "",
            font_size="12px",
            text_anchor="middle",
            font_family="Georgia, 'Times New Roman', Times, serif",
        )
        text_path = dwg.textPath(f"#{path_id}", wedding, startOffset="50%")
        text.add(text_path)
        dwg.add(text)


def draw_marriage_line(dwg, start, end, date_text="", font_size="12px"):
    path_d = f"M {start[0]},{start[1]}"
    path_d += f"L {start[0]},{start[1]+20}"
    path_d += f"L {end[0]},{end[1]+20}"
    path_d += f"L {end[0]},{end[1]}"

    line = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
    dwg.add(line)

    # Save SVG
    # Add centered text 'hello' under the center point
    wedd_text = dwg.text(
        date_text,
        insert=((start[0] + end[0]) / 2, end[1] + 14),
        text_anchor="middle",
        font_size=font_size,
        font_family="Georgia, 'Times New Roman', Times, serif",
    )
    dwg.add(wedd_text)


draw_parent_child_arcs(0, 1, dwg)
draw_parent_child_arcs(1, 2, dwg)
draw_parent_child_arcs(2, 3, dwg)

start = polar_to_cartesian(ring_radii[0] + ring_thickness / 2, 170)
end = polar_to_cartesian(ring_radii[0] + ring_thickness / 2, 370)
start_child = (center[0] - (box_width + gap) / 2, center[1] + 55)
end_child = (center[0] + (box_width + gap) / 2, center[1] + 55)

draw_marriage_line(dwg, start, end, wedd_data[0][0], font_size="14px")

path_d += f"M {(start[0]+end[0])/2},{end[1]+20}"
path_d += f"L {(start[0]+end[0])/2},{center[1] + 55}"
path_d += f"M {start_child[0]},{start_child[1]+10}"
path_d += f"L {start_child[0]},{start_child[1]}"
path_d += f"L {end_child[0]},{end_child[1]}"
path_d += f"L {end_child[0]},{end_child[1]+10}"
connect = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
dwg.add(connect)

x_offset = 0.5 * (box_width + gap)
y_offset = center[1] + 124
draw_marriage_line(
    dwg,
    (center[0] - 3 * x_offset, y_offset),
    (center[0] - x_offset, y_offset),
)
draw_marriage_line(
    dwg,
    (center[0] + x_offset, y_offset),
    (center[0] + 3 * x_offset, y_offset),
    "⚭ 14.7.2011",
)


def draw_children_boxes(dwg, y_offset):
    for line_no in range(len(children)):
        for i, child in enumerate(children[line_no]):

            # skip empty children
            if not child.name:
                continue

            # Calculate position
            x = (
                center[0]
                - (len(children[line_no]) - 1) * (box_width + gap) / 2
                + i * (box_width + gap)
            )
            # if the next child is empty, move the box to the right
            if i < len(children[line_no]) - 1 and not children[line_no][i + 1].name:
                x += (box_width + gap) / 2

            y = center[1] + y_offset + line_no * (box_height + ring_gap)

            # Draw the box
            child_box = dwg.rect(
                insert=(x - box_width / 2, y),
                size=(box_width, box_height),
                fill="#f0f0f0",
                stroke="lightgray",
                stroke_width=1,
            )
            dwg.add(child_box)

            # Add child information (3 lines like segments)
            lines = [child.name, child.birthdate, child.deathdate]
            for k, line in enumerate(lines):
                line_y = (
                    y + (k + 1.3) * line_spacing
                )  # Use same line_spacing as segments
                child_text = dwg.text(
                    line,
                    insert=(x, line_y),
                    text_anchor="middle",
                    font_size="13px",
                    font_family="Georgia, 'Times New Roman', Times, serif",
                )
                dwg.add(child_text)


draw_children_boxes(dwg, 70)

dwg.save()
