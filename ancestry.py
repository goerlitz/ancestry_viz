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
ring_thickness = 80
num_rings = 3
ring_radii = [100 + i * ring_thickness for i in range(num_rings)]
segments_per_ring = [4, 8, 16]
segment_angles = [50, 25, 12.5]
total_angle = 200
start_angle_offset = 170
line_spacing = 16  # spacing between text lines

def polar_to_cartesian(radius: float, angle_degrees: float, center_point: Tuple[int, int] = center) -> Tuple[float, float]:
    """Convert polar coordinates (radius, angle) to cartesian coordinates relative to center."""
    x = center_point[0] + radius * math.cos(math.radians(angle_degrees))
    y = center_point[1] + radius * math.sin(math.radians(angle_degrees))
    return (x, y)

def create_arc_path(start_radius: float, end_radius: float, start_angle: float, end_angle: float, large_arc_flag: int = 0) -> str:
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

def create_segment_outline_path_with_gaps(inner_radius: float, outer_radius: float, start_angle: float, end_angle: float, gap_size: float = 5) -> str:
    """Create an SVG path that outlines a segment with rays and arcs, with even gaps between segments."""
    # Calculate angle offsets for even gaps at inner and outer radii
    inner_gap_angle = calculate_gap_angle(gap_size, inner_radius)
    outer_gap_angle = calculate_gap_angle(gap_size, outer_radius)
    
    # Adjust angles to create even gaps
    start_angle_adjusted = start_angle + outer_gap_angle
    end_angle_adjusted = end_angle - outer_gap_angle
    
    # Calculate the four corners of the segment
    inner_start = polar_to_cartesian(inner_radius, start_angle_adjusted)
    inner_end = polar_to_cartesian(inner_radius, end_angle_adjusted)
    outer_start = polar_to_cartesian(outer_radius, start_angle_adjusted)
    outer_end = polar_to_cartesian(outer_radius, end_angle_adjusted)
    
    # Create path: inner_start -> outer_start -> outer_arc -> outer_end -> inner_end -> inner_arc -> inner_start
    large_arc_flag = 1 if (end_angle_adjusted - start_angle_adjusted) > 180 else 0
    
    path = f"M {inner_start[0]},{inner_start[1]} "  # Start at inner start
    path += f"L {outer_start[0]},{outer_start[1]} "  # Line to outer start
    path += f"A {outer_radius},{outer_radius} 0 {large_arc_flag},1 {outer_end[0]},{outer_end[1]} "  # Outer arc
    path += f"L {inner_end[0]},{inner_end[1]} "  # Line to inner end
    path += f"A {inner_radius},{inner_radius} 0 {large_arc_flag},0 {inner_start[0]},{inner_start[1]} Z"  # Inner arc back to start
    
    return path

# Load people data from CSV file
def load_people_from_csv(filename: str) -> List[List[Person]]:
    people_by_ring = [[] for _ in range(num_rings)]
    
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            person = Person(
                ring=int(row['ring']),
                name=row['name'],
                birthdate=row['birthdate'],
                birthplace=row['birthplace']
            )
            people_by_ring[person.ring].append(person)
    
    return people_by_ring

# Load the data
ring_data = load_people_from_csv('people.csv')

# Create SVG drawing
dwg = svgwrite.Drawing('./radial_family.svg', size=('1000px', '1000px'))
dwg.add(dwg.rect(insert=(0, 0), size=('1000px', '1000px'), fill='white'))

# Draw visible arcs and attach each text line to its own path
for i, (base_radius, segments, angle_span) in enumerate(zip(ring_radii, segments_per_ring, segment_angles)):
    for j in range(segments):
        segment_angle = j * angle_span + start_angle_offset
        start_angle = segment_angle
        end_angle = segment_angle + angle_span

        # Get person data for this segment
        person = ring_data[i][j]
        lines = [person.name, person.birthdate, person.birthplace]

        # Draw shaded box for innermost ring segments
        if i <= 1:  # Only for innermost ring
            inner_radius = base_radius + 6 + i*16 # Slightly inside the base radius
            outer_radius = base_radius + 3 * line_spacing + 18 + i*16  # Slightly outside the text area
            outline_path = create_segment_outline_path_with_gaps(inner_radius, outer_radius, start_angle, end_angle, gap_size=4)
            
            # Add the shaded box
            box = dwg.path(d=outline_path, fill="#f0f0f0", stroke="lightgray", stroke_width=1)
            dwg.add(box)

        for k, line in enumerate(lines):  # 3 lines per segment
            radius = base_radius + (3-k) * line_spacing + i*16  # offset each line outward
            large_arc_flag = 1 if angle_span > 180 else 0
            
            # Calculate center point of the segment
            center_angle = (start_angle + end_angle) / 2
            
            if i == 2:  # Only outermost ring - use straight lines (rays)
                # Calculate angles for the 3 lines: -3 degree, center, +3 degree
                line_angles = [
                    center_angle - 3,  # First line: -3 degree
                    center_angle,      # Second line: center angle
                    center_angle + 3   # Third line: +3 degree
                ]
                
                # Use the same radius for all 3 lines to center them on the same ray length
                ray_radius = base_radius + line_spacing  # Center radius for all lines
                ray_start_radius = ray_radius + 20  # Start slightly inward
                ray_end_radius = ray_radius + 100  # End slightly outward
                
                line_angle = line_angles[k]  # Use the appropriate angle for this line
                path_d = create_line_path(ray_start_radius, ray_end_radius, line_angle)
            else:  # Inner rings - use curved arcs
                path_d = create_arc_path(radius, radius, start_angle, end_angle, large_arc_flag)

            path_id = f"path_r{i}_s{j}_l{3-k}"
            path = dwg.path(d=path_d, fill="none", stroke="white", id=path_id)
            dwg.add(path)

            # Add text to each individual arc path
            text = dwg.text("", font_size="10px", text_anchor="middle")
            text_path = dwg.textPath(f"#{path_id}", line, startOffset="50%")
            text.add(text_path)
            dwg.add(text)

# Save SVG
dwg.save()
