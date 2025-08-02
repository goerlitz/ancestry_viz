import base64
import csv
import math
import os
import re
import svgwrite
from svgwrite import gradients
from dataclasses import dataclass
from typing import List, Tuple
from PIL import ImageFont


@dataclass
class Person:
    ring: int
    name: str
    birthdate: str
    deathdate: str
    age: str
    occupation: str


title_font = "Apple Chancery, cursive"
title_font = "SnellRoundhand"
subtitle_font = "SnellRoundhand"
# text_font="Apple Chancery, cursive"
text_font = "Georgia, 'Times New Roman', Times, serif"

male_color = "#4A90E2"
female_color = "#FF6EC7"

# Parameters
center = (800, 920)
num_rings = 5
segments_per_ring = [2, 4, 8, 16, 32]
total_angle = 200.0
segment_angles = [total_angle / i for i in segments_per_ring]  # 200/2, 200/4, ...
start_radius = 40
ring_gap = 56
ring_thickness = 64  # total = 120
ring_thickness_outer = 144
ring_radii = [
    start_radius
    + min(i, 3) * ring_thickness
    + max(i - 3, 0) * ring_thickness_outer
    + i * ring_gap
    for i in range(num_rings)
]
start_angle_offset = 170
line_spacing = 16  # spacing between text lines
box_width = 120
box_height = 60
gap = 8

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import to_rgb, to_hex
from skimage import color  # interpolate colors in Lab space for perceptual uniformity

# Define 4 base color pairs (dark to light) for each family line
color_families = {
    "Blue": ["#a8bedb", "#dbe9f6"],
    "Green": ["#a5d0b9", "#d9f2e6"],
    "Orange": ["#f8bd8d", "#ffe7cc"],
    "Magenta": ["#e3a7c6", "#f7dbef"],
}

font_name = text_font.split(",")[0]
font_path = f"/System/Library/Fonts/Supplemental/{font_name}.ttf"


# def get_font_base64():
#     if not os.path.isfile(font_path):
#         raise FileNotFoundError(
#             "Georgia.ttf not found at the default macOS location. "
#             "Please point font_path to the .ttf file."
#         )

#     with open(font_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("ascii")


# def embed_font(dwg, font_b64):
#     css = f"""
#     @font-face {{
#       font-family: 'GeorgiaEmbed';
#       src: url("data:font/truetype;charset=utf-8;base64,{font_b64}") format('truetype');
#       font-weight: normal;
#       font-style: normal;
#     }}
#     """
#     dwg.defs.add(dwg.style(css))


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
    shades = interpolate_colors_lab(start, end, 8)
    palette.extend(shades)


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
    gap_left: bool = True,
    gap_right: bool = True,
) -> str:
    """Create an SVG arc path string."""
    gap_angle = calculate_gap_angle(gap_size, radius)
    start_point = polar_to_cartesian(
        radius, start_angle + (gap_angle if gap_left else 0)
    )
    end_point = polar_to_cartesian(radius, end_angle - (gap_angle if gap_right else 0))
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
    weddings_by_ring = [[] for _ in range(num_rings + 1)]
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
                    deathdate=row["deathdate"],
                    age=row["age"],
                    occupation=row["occupation"],
                )
                children[ring_no - 10].append(person)
            else:
                person = Person(
                    ring=ring_no,
                    name=row["name"],
                    birthdate=row["birthdate"],
                    deathdate=row["deathdate"],
                    age=row["age"],
                    occupation=row["occupation"],
                )
                people_by_ring[person.ring].append(person)

    return (people_by_ring, weddings_by_ring, children)


# Load the data
(ring_data, wedd_data, children) = load_people_from_csv("people.csv")

# Create SVG drawing
dwg = svgwrite.Drawing("./radial_family.svg", size=("1600px", "1200px"))
# dwg.add(dwg.rect(insert=(0, 0), size=("1000px", "1000px"), fill="white"))
# embed_font(dwg, get_font_base64())


title_font = "Apple Chancery, cursive"

title = dwg.text(
    "Stammbaum",
    insert=(center[0], 70),
    text_anchor="middle",
    font_size="72px",
    font_family=title_font,
    fill="#2c3e50",
    # font_weight="bold"
)
dwg.add(title)

title = dwg.text(
    "der Familie Görlitz",
    insert=(center[0], 120),
    text_anchor="middle",
    font_size="36px",
    font_family=subtitle_font,
    fill="#2c3e50",
    # font_weight="bold"
)
dwg.add(title)

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


def create_text_line_paths(
    start_angle: float,
    end_angle: float,
    start_radius: float,
    end_radius: float,
    gap_size: float = 2,
):
    """Create an SVG line paths for text."""
    # Calculate center point of the segment
    center_angle = (start_angle + end_angle) / 2
    line_spread = (end_angle - start_angle) / 5

    angle_adjust = (4 / start_radius) * (
        180 / math.pi
    )  # because text baseline sits on path

    # Flip text for left half (upright)
    flip = 90 < (center_angle % 360) < 270

    center_angle = center_angle - (angle_adjust if flip else -angle_adjust)

    # Calculate angles for 3 lines
    line_angles = [
        center_angle - line_spread + (angle_adjust if flip else -angle_adjust),
        center_angle,
        center_angle + line_spread,
    ]

    # reverse the line path so text is upright
    if flip:
        start = end_radius - gap_size
        end = start_radius + gap_size
        angles = line_angles[::-1]
    else:
        start = start_radius + gap_size
        end = end_radius - gap_size
        angles = line_angles

    return [
        (
            create_line_path(start, end, angle)
            if i == 0
            else [
                create_line_path(start, (start + end) / 2, angle),
                create_line_path((start + end) / 2, end, angle),
            ]
        )
        for i, angle in enumerate(angles)
    ]


def create_text_arc_paths(
    start_angle: float,
    end_angle: float,
    start_radius: float,
    end_radius: float,
    gap_size: float = 4,
    lines: int = 3,
):
    """Create an SVG arc paths for text."""
    center_angle = (start_angle + end_angle) / 2
    line_spread = (end_radius - start_radius) / (lines + 1)
    radii = [end_radius - (i + 1.3) * line_spread for i in range(lines)]

    # first line is full arc, rest are split into two arcs
    return [
        (
            create_arc_path(radius, start_angle, end_angle, gap_size=gap_size)
            if i == 0
            else [
                create_arc_path(
                    radius,
                    start_angle,
                    center_angle,
                    gap_size=gap_size,
                    gap_right=False,
                ),
                create_arc_path(
                    radius, center_angle, end_angle, gap_size=gap_size, gap_left=False
                ),
            ]
        )
        for i, radius in enumerate(radii)
    ]


def create_text_path(dwg, path_d, path_id):
    path = dwg.path(
        d=path_d,
        fill="none",
        stroke="none",
        id=path_id,
        # stroke_dasharray="2,2",
    )
    dwg.add(path)


def underline_quoted_text_arc(dwg, line, font_size, radius, center_angle):
    # get length of text
    font = ImageFont.truetype(font_path, int(font_size.rstrip("'pxptem%")))
    text_width = font.getlength(line.replace("'", ""))

    # TODO: fix bug with wrong line end because of first quote
    # get locations of quotes and translate to angles from center
    positions = [m.start() for m in re.finditer("'", line)]
    widths = [font.getlength(line[0:pos].replace("'", "")) for pos in positions]
    angles = [calculate_gap_angle(w - text_width / 2, radius) for w in widths]

    path_d = create_arc_path(
        radius,
        center_angle + angles[0],
        center_angle + angles[1],
        gap_size=0,
    )

    arc_path = dwg.path(d=path_d, fill="none", stroke="black", stroke_width=1.2)
    dwg.add(arc_path)


def underline_quoted_text_line(dwg, line, font_size, center_radius, angle):
    # get length of text
    font = ImageFont.truetype(font_path, int(font_size.rstrip("'pxptem%")))
    text_width = font.getlength(line.replace("'", ""))

    # get locations of quotes and translate to angles from center
    positions = [m.start() for m in re.finditer("'", line)]
    widths = [font.getlength(line[0:pos].replace("'", "")) for pos in positions]
    radii = [center_radius - (w - text_width / 2) for w in widths]

    path_d = create_line_path(radii[0], radii[1], angle)

    arc_path = dwg.path(d=path_d, fill="none", stroke="black", stroke_width=1.2)
    dwg.add(arc_path)


def create_text(dwg, font_size, text_anchor, path_id, line, start_offset, bold=False):

    text = dwg.text(
        "",
        font_size=font_size,
        text_anchor=text_anchor,
        font_family=text_font,
    )

    text_path = dwg.textPath(
        f"#{path_id}", line.replace("'", ""), startOffset=start_offset
    )

    text.add(text_path)
    dwg.add(text)


def draw_parent_child_arcs(child_idx: int, parent_idx: int, dwg):
    parent_count = segments_per_ring[parent_idx]
    child_count = segments_per_ring[child_idx]
    parent_angle_span = segment_angles[parent_idx]
    child_angle_span = segment_angles[child_idx]
    arc_radius = ring_radii[parent_idx] - ring_gap * 0.6

    for i in range(child_count):
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
            ring_radii[parent_idx] - ring_gap + 4, child_center_angle
        )

        # Split wedding date and place
        parts = wedd_data[child_idx + 1][i].split(":", 2)
        wedding = parts[0] if len(parts) > 0 else ""
        place = parts[1] if len(parts) > 1 else ""
        notes = parts[2] if len(parts) > 2 else ""

        # Split notes into parts
        parts = notes.split("/")
        note_male = parts[0] if len(parts) > 0 else ""
        note_female = parts[1] if len(parts) > 1 else ""

        path_d = f"M {start[0]},{start[1]}"
        path_d += f"L {arc_start[0]},{arc_start[1]}"
        path_d += f"A {arc_radius},{arc_radius} 0 0,1 {arc_end[0]},{arc_end[1]}"
        path_d += f"L {end[0]},{end[1]}"
        path_d += f"M {arc_center[0]},{arc_center[1]}"
        path_d += f"L {child_center[0]},{child_center[1]}"

        arc_path = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
        dwg.add(arc_path)

        # invisible arc for text path
        path_d = create_arc_path(
            arc_radius + 20,
            child_center_angle - parent_angle_span / 2,
            child_center_angle + parent_angle_span / 2,
            gap_size=0,
        )
        path_id = f"path_ring{child_idx}_mar_date{i}"
        create_text_path(dwg, path_d, path_id)
        create_text(dwg, "11px", "middle", path_id, wedding, "50%")

        path_d = create_arc_path(
            arc_radius + 6,
            child_center_angle - parent_angle_span / 2,
            child_center_angle + parent_angle_span / 2,
            gap_size=0,
        )
        path_id = f"path_ring{child_idx}_mar_place{i}"
        create_text_path(dwg, path_d, path_id)
        create_text(dwg, "11px", "middle", path_id, place, "50%")

        if note_male:

            # check for marriage info
            if "I." in note_male:
                mar_no = note_male.split(" ")[0]
                note_male = note_male.split(" ")[1]

                path_d = create_arc_path(
                    arc_radius + 6,
                    child_center_angle - parent_angle_span,
                    child_center_angle - parent_angle_span / 2,
                    gap_size=4,
                    gap_left=False,
                    gap_right=True,
                )
                path_id = f"path_ring{child_idx}_mar_note_male_no{i}"
                create_text_path(dwg, path_d, path_id)
                create_text(dwg, "11px", "end", path_id, mar_no, "95%")

            path_d = create_arc_path(
                arc_radius + 20,
                child_center_angle - parent_angle_span,
                child_center_angle - parent_angle_span / 2,
                gap_size=4,
                gap_left=False,
                gap_right=True,
            )
            path_id = f"path_ring{child_idx}_mar_note_male{i}"
            create_text_path(dwg, path_d, path_id)
            create_text(dwg, "11px", "end", path_id, note_male, "95%")

        if note_female:

            # check for marriage info
            if "I." in note_female:
                mar_no = note_female.split(" ")[0]
                note_female = note_female.split(" ")[1]

                path_d = create_arc_path(
                    arc_radius + 6,
                    child_center_angle + parent_angle_span / 2,
                    child_center_angle + parent_angle_span,
                    gap_size=4,
                    gap_left=True,
                    gap_right=False,
                )
                path_id = f"path_ring{child_idx}_mar_note_female_no{i}"
                create_text_path(dwg, path_d, path_id)
                create_text(dwg, "11px", "start", path_id, mar_no, "5%")

            path_d = create_arc_path(
                arc_radius + 20,
                child_center_angle + parent_angle_span / 2,
                child_center_angle + parent_angle_span,
                gap_size=4,
                gap_left=True,
                gap_right=False,
            )
            path_id = f"path_ring{child_idx}_mar_note_female{i}"
            create_text_path(dwg, path_d, path_id)
            create_text(dwg, "11px", "start", path_id, note_female, "5%")


def draw_marriage_line(dwg, start, end, date_text="", font_size="12px"):
    path_d = f"M {start[0]},{start[1]}"
    path_d += f"L {start[0]},{start[1]+30}"
    path_d += f"L {end[0]},{end[1]+30}"
    path_d += f"L {end[0]},{end[1]}"

    line = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
    dwg.add(line)

    if ":" in date_text:
        (mdate, place) = date_text.split(":")
    else:
        (mdate, place) = date_text, None

    wedd_text = dwg.text(
        mdate,
        insert=((start[0] + end[0]) / 2, end[1] + 8),
        text_anchor="middle",
        font_size=font_size,
        font_family=text_font,
    )
    dwg.add(wedd_text)

    if place:
        wedd_place = dwg.text(
            place,
            insert=((start[0] + end[0]) / 2, end[1] + 24),
            text_anchor="middle",
            font_size=font_size,
            font_family=text_font,
        )
        dwg.add(wedd_place)


def create_pill(radius, angle, direction=0, height: int = 16, width: int = 16):
    # the pill ends will be two half circles
    half = height / 2
    half_angle = calculate_gap_angle(half, radius)

    if direction == 0:
        pill_angle = calculate_gap_angle(width, radius) + half_angle

        # radius is center, angle is right aligned
        p1 = polar_to_cartesian(radius + half, angle - pill_angle)
        p2 = polar_to_cartesian(radius + half, angle - half_angle)
        p3 = polar_to_cartesian(radius - half, angle - half_angle)
        p4 = polar_to_cartesian(radius - half, angle - pill_angle)
    elif direction == -1:
        center_radius = 4 + radius - half - width / 2

        add_angle = -2 * half_angle * direction
        p1 = polar_to_cartesian(center_radius + half, angle)
        p2 = polar_to_cartesian(center_radius - half, angle)
        p3 = polar_to_cartesian(center_radius - half, angle - add_angle)
        p4 = polar_to_cartesian(center_radius + half, angle - add_angle)
    else:
        center_radius = 4 + radius - half - width / 2

        add_angle = -2 * half_angle * direction
        p1 = polar_to_cartesian(center_radius - half, angle)
        p2 = polar_to_cartesian(center_radius + half, angle)
        p3 = polar_to_cartesian(center_radius + half, angle - add_angle)
        p4 = polar_to_cartesian(center_radius - half, angle - add_angle)

    # Create pill path: p1 -> p2 -> arc to p3 -> p4 -> arc to p1
    pill_path = f"M {p1[0]},{p1[1]} L {p2[0]},{p2[1]} "
    pill_path += f"A {half},{half} 0 0,1 {p3[0]},{p3[1]} "
    pill_path += f"L {p4[0]},{p4[1]} "
    pill_path += f"A {half},{half} 0 0,1 {p1[0]},{p1[1]} Z"

    return pill_path


def create_pill_text(
    dwg, text, radius, angle, direction=0, height: int = 16, width: int = 16
):
    half = height / 2
    half_angle = calculate_gap_angle(half, radius)

    if direction == 0:
        pill_angle = calculate_gap_angle(width, radius) + half_angle
        center_angle = angle - (pill_angle + half_angle) / 2
        text_x, text_y = polar_to_cartesian(radius, center_angle)
    elif direction == -1:
        center_angle = angle - half_angle
        text_x, text_y = polar_to_cartesian(6 + radius - half - width / 2, center_angle)
    else:
        center_angle = angle + half_angle
        text_x, text_y = polar_to_cartesian(6 + radius - half - width / 2, center_angle)

    pill_text = dwg.text(
        text,
        insert=(text_x, text_y),
        text_anchor="middle",
        dominant_baseline="middle",
        font_size="10px",
        font_family=text_font,
        fill="black",
    )

    # Apply rotation transform
    add_angle = 90 if direction == 0 else (180 if direction == -1 else 0)
    pill_text.rotate(center_angle + add_angle, center=(text_x, text_y))
    dwg.add(pill_text)


# -------------


# Draw visible arcs and attach each text line to its own path
for ring_no, (base_radius, segments, angle_span) in enumerate(
    zip(ring_radii, segments_per_ring, segment_angles)
):
    for seg_no in range(segments):
        start_angle = seg_no * angle_span + start_angle_offset
        end_angle = start_angle + angle_span

        # Get person data for this segment
        person = ring_data[ring_no][seg_no]
        lines = [person.name, person.birthdate, person.deathdate]

        inner_radius = base_radius
        outer_radius = base_radius + (
            ring_thickness_outer if ring_no >= 3 else ring_thickness
        )

        outline_path = create_ring_segment(
            inner_radius, outer_radius, start_angle, end_angle, gap_size=4
        )

        # Add the shaded box
        if ring_no == 0:
            fill = "url(#gold_gradient)"
        else:
            step = 32 // segments_per_ring[ring_no]
            fill = palette[seg_no * step]
        stroke = male_color if seg_no % 2 == 0 else female_color
        box = dwg.path(
            d=outline_path,
            fill=fill,
            stroke=stroke,
            stroke_width=1.5,
        )
        dwg.add(box)

        if ring_no >= 3:  # Only outermost ring - use straight lines (rays)
            text_paths = create_text_line_paths(
                start_angle, end_angle, inner_radius, outer_radius, gap_size=4
            )
        else:
            text_paths = create_text_arc_paths(
                start_angle,
                end_angle,
                inner_radius,
                outer_radius,
                gap_size=6,
                lines=3,
            )

        for k, line in enumerate(lines):  # 3 lines per segment

            text_path = text_paths[k]
            if ring_no == 0:
                font_size = "15px" if k == 0 else "12px"
            elif ring_no == 1:
                font_size = "14px" if k == 0 else "12px"
            else:
                font_size = "13px" if k == 0 else "11px"

            if k == 0:
                path_id = f"path_r{ring_no}_s{seg_no}_l{k}"
                create_text_path(dwg, text_path, path_id)
                create_text(dwg, font_size, "middle", path_id, line, "50%")

                # underline quoted names
                if "'" in line:
                    if ring_no < 3:
                        underline_quoted_text_arc(
                            dwg,
                            line,
                            font_size,
                            (inner_radius + outer_radius) / 2 + 10,
                            (start_angle + end_angle) / 2,
                        )
                    else:
                        underline_quoted_text_line(
                            dwg,
                            line,
                            font_size,
                            (inner_radius + outer_radius) / 2,
                            (start_angle + end_angle) / 2 + 1.7,
                        )

            else:

                if ring_no > 0:
                    # Split value into date and place
                    if ":" in line:
                        date_part, place_part = line.split(":", 1)
                    else:
                        date_part, place_part = line, ""

                    # Date: right-aligned, left arc
                    path_id = f"path_r{ring_no}_s{seg_no}_l{k}_date"
                    create_text_path(dwg, text_path[0], path_id)
                    create_text(dwg, "11px", "end", path_id, date_part, "96%")

                    # Place: left-aligned, right arc
                    path_id = f"path_r{ring_no}_s{seg_no}_l{k}_place"
                    create_text_path(dwg, text_path[1], path_id)
                    create_text(dwg, "11px", "start", path_id, place_part, "4%")
                else:
                    path_id = f"path_r{ring_no}_s{seg_no}_l{k}"
                    create_text_path(dwg, text_path, path_id)
                    create_text(dwg, font_size, "middle", path_id, line, "50%")

        if person.age:
            if ring_no >= 3:
                angle = end_angle if end_angle <= 270 else start_angle
                direction = -1 if end_angle <= 270 else 1

                p = create_pill(outer_radius, angle, direction=direction)
                polygon = dwg.path(d=p, fill=fill, stroke=stroke)
                dwg.add(polygon)

                create_pill_text(
                    dwg, person.age, outer_radius - 2, angle, direction=direction
                )
            else:
                p = create_pill(outer_radius - 2, end_angle)
                polygon = dwg.path(d=p, fill=fill, stroke=stroke)
                dwg.add(polygon)

                create_pill_text(dwg, person.age, outer_radius - 2, end_angle)


draw_parent_child_arcs(0, 1, dwg)
draw_parent_child_arcs(1, 2, dwg)
draw_parent_child_arcs(2, 3, dwg)
draw_parent_child_arcs(3, 4, dwg)

start = polar_to_cartesian(ring_radii[0] + ring_thickness / 2, 170)
end = polar_to_cartesian(ring_radii[0] + ring_thickness / 2, 370)
start_child = (center[0] - (box_width + gap) / 2, center[1] + 55)
end_child = (center[0] + (box_width + gap) / 2, center[1] + 55)

draw_marriage_line(dwg, start, end, wedd_data[0][0], font_size="14px")

## draw more connections
path_d = f"M {(start[0]+end[0])/2},{end[1]+30}"
path_d += f"L {(start[0]+end[0])/2},{center[1] + 55}"
path_d += f"M {start_child[0]},{start_child[1]+10}"
path_d += f"L {start_child[0]},{start_child[1]}"
path_d += f"L {end_child[0]},{end_child[1]}"
path_d += f"L {end_child[0]},{end_child[1]+10}"
connect = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
dwg.add(connect)

x_offset = 0.5 * (box_width + gap)
y_offset = center[1] + 134
draw_marriage_line(
    dwg,
    (center[0] - 3 * x_offset, y_offset),
    (center[0] - x_offset, y_offset),
)
path_d = f"M {(center[0] - 2 * x_offset)},{y_offset+30}"
path_d += f"L {(center[0] - 2 * x_offset)},{y_offset+48}"
child1 = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
dwg.add(child1)

draw_marriage_line(
    dwg,
    (center[0] + x_offset, y_offset),
    (center[0] + 3 * x_offset, y_offset),
    wedd_data[5][1],
)
path_d = f"M {(center[0] + 2 * x_offset)},{y_offset+30}"
path_d += f"L {(center[0] + 2 * x_offset)},{y_offset+40}"
path_d += f"M {(center[0] + x_offset)},{y_offset+48}"
path_d += f"L {(center[0] + x_offset)},{y_offset+40}"
path_d += f"L {(center[0] + 3 * x_offset)},{y_offset+40}"
path_d += f"L {(center[0] + 3 * x_offset)},{y_offset+48}"
child2 = dwg.path(d=path_d, fill="none", stroke="lightgrey", stroke_width=1)
dwg.add(child2)


def draw_children_boxes(dwg, y_offset):

    # fixed male/female coding
    male = [[False, True, True, False], [True, True, True, False]]

    for line_no in range(len(children)):
        for i, child in enumerate(children[line_no]):

            is_male = male[line_no][i]

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
                fill="#ede7fa" if i < 2 else "#e7f0fa",
                stroke=male_color if is_male else female_color,
                stroke_width=1,
            )
            dwg.add(child_box)

            # Add child information (3 lines like segments)
            lines = [child.name, child.birthdate, child.deathdate]
            for k, line in enumerate(lines):
                line_y = (
                    y + (k + 1.2) * line_spacing
                )  # Use same line_spacing as segments
                child_text = dwg.text(
                    line,
                    insert=(x, line_y),
                    text_anchor="middle",
                    font_size="13px" if k == 0 else "11px",
                    font_family=text_font,
                )
                dwg.add(child_text)


draw_children_boxes(dwg, 70)

dwg.save()
