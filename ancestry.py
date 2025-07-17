import svgwrite
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Person:
    ring: int
    name: str
    birthdate: str
    birthplace: str


# Parameters
center = (500, 500)
num_rings = 4
segments_per_ring = [2, 4, 8, 16]
total_angle = 200.0
segment_angles = [
    total_angle / i for i in segments_per_ring
]  # 200/2, 200/4, 200/8, 200/16
start_radius = 32
ring_thickness = 56
ring_gap = 40
ring_radii = [start_radius + i * (ring_thickness + ring_gap) for i in range(num_rings)]
start_angle_offset = 170
line_spacing = 16  # spacing between text lines


def polar_to_cartesian(
    radius: float, angle_degrees: float, center_point: Tuple[int, int] = center
) -> Tuple[float, float]:
    """Convert polar coordinates (radius, angle) to cartesian coordinates relative to center."""
    x = center_point[0] + radius * math.cos(math.radians(angle_degrees))
    y = center_point[1] + radius * math.sin(math.radians(angle_degrees))
    return (x, y)


def create_arc_path(
    start_radius: float,
    end_radius: float,
    start_angle: float,
    end_angle: float,
    large_arc_flag: int = 0,
) -> str:
    """Create an SVG arc path string."""
    start_point = polar_to_cartesian(start_radius, start_angle)
    end_point = polar_to_cartesian(end_radius, end_angle)
    return f"M {start_point[0]},{start_point[1]} A {start_radius},{start_radius} 0 {large_arc_flag},1 {end_point[0]},{end_point[1]}"


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


def create_segment_outline_path_with_gaps(
    inner_radius: float,
    outer_radius: float,
    start_angle: float,
    end_angle: float,
    gap_size: float = 5,
) -> str:
    """Create an SVG path that outlines a segment with rays and arcs, with even gaps between segments."""
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

    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            person = Person(
                ring=int(row["ring"]),
                name=row["name"],
                birthdate=row["birthdate"],
                birthplace=row["birthplace"],
            )
            people_by_ring[person.ring].append(person)

    return people_by_ring


# Load the data
ring_data = load_people_from_csv("people.csv")

# Create SVG drawing
dwg = svgwrite.Drawing("./radial_family.svg", size=("1000px", "1000px"))
dwg.add(dwg.rect(insert=(0, 0), size=("1000px", "1000px"), fill="white"))

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
        lines = [person.name, person.birthdate, person.birthplace]

        # Draw shaded box for innermost and outermost ring segments

        inner_radius = base_radius
        outer_radius = base_radius + (120 if ring_no == 3 else ring_thickness)

        outline_path = create_segment_outline_path_with_gaps(
            inner_radius, outer_radius, start_angle, end_angle, gap_size=4
        )

        # Add the shaded box
        box = dwg.path(
            d=outline_path, fill="#f0f0f0", stroke="lightgray", stroke_width=1
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
                end_radius = base_radius + 112  # End slightly outward
                line_angle = line_angles[2 - k if flip else k]

                # reverse the line path so text is upright
                if flip:
                    path_d = create_line_path(end_radius, start_radius, line_angle)
                else:
                    path_d = create_line_path(start_radius, end_radius, line_angle)

            else:  # Inner rings - use curved arcs
                radius = base_radius + (2.5 - k) * line_spacing  # offset each line
                large_arc_flag = 1 if angle_span > 180 else 0
                path_d = create_arc_path(
                    radius, radius, start_angle, end_angle, large_arc_flag
                )

            text_anchor = "middle"
            start_offset = "50%"

            path_id = f"path_r{ring_no}_s{seg_no}_l{3-k}"
            path = dwg.path(d=path_d, fill="none", stroke="white", id=path_id)
            dwg.add(path)

            # Add text to each individual arc path
            text = dwg.text("", font_size="11px", text_anchor=text_anchor)
            text_path = dwg.textPath(f"#{path_id}", line, startOffset=start_offset)
            text.add(text_path)
            dwg.add(text)

# Save SVG
dwg.save()
