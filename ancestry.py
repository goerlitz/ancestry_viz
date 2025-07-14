import svgwrite
import math
import csv
from dataclasses import dataclass
from typing import List

@dataclass
class Person:
    ring: int
    name: str
    birthdate: str
    birthplace: str

# Parameters
center = (500, 500)
ring_thickness = 40
num_rings = 3
ring_radii = [100 + i * ring_thickness for i in range(num_rings)]
segments_per_ring = [4, 8, 16]
segment_angles = [50, 25, 12.5]
total_angle = 200
start_angle_offset = 170
line_spacing = 12  # spacing between text lines

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

        for k, line in enumerate(lines):  # 3 lines per segment
            radius = base_radius + (3-k) * line_spacing + i*16  # offset each line outward
            large_arc_flag = 1 if angle_span > 180 else 0
            
            # Calculate center point of the segment
            center_angle = (start_angle + end_angle) / 2
            
            if i == 2:  # Only outermost ring - use straight lines (rays)
                # Calculate angles for the 3 lines: -4 degree, center, +4 degree
                line_angles = [
                    center_angle - 3,  # First line: -4 degree
                    center_angle,      # Second line: center angle
                    center_angle + 3   # Third line: +4 degree
                ]
                
                # Use the same radius for all 3 lines to center them on the same ray length
                ray_radius = base_radius + line_spacing  # Center radius for all lines
                ray_start_radius = ray_radius + 30  # Start slightly inward
                ray_end_radius = ray_radius + 120  # End slightly outward
                
                line_angle = line_angles[k]  # Use the appropriate angle for this line
                
                start = (
                    center[0] + ray_start_radius * math.cos(math.radians(line_angle)),
                    center[1] + ray_start_radius * math.sin(math.radians(line_angle))
                )
                end = (
                    center[0] + ray_end_radius * math.cos(math.radians(line_angle)),
                    center[1] + ray_end_radius * math.sin(math.radians(line_angle))
                )
                path_d = f"M {start[0]},{start[1]} L {end[0]},{end[1]}"
            else:  # Inner rings - use curved arcs
                start = (
                    center[0] + radius * math.cos(math.radians(start_angle)),
                    center[1] + radius * math.sin(math.radians(start_angle))
                )
                end = (
                    center[0] + radius * math.cos(math.radians(end_angle)),
                    center[1] + radius * math.sin(math.radians(end_angle))
                )
                path_d = f"M {start[0]},{start[1]} A {radius},{radius} 0 {large_arc_flag},1 {end[0]},{end[1]}"

            path_id = f"path_r{i}_s{j}_l{3-k}"
            path = dwg.path(d=path_d, fill="none", stroke="lightgray", id=path_id)
            dwg.add(path)

            # Add text to each individual arc path
            text = dwg.text("", font_size="10px", text_anchor="middle")
            text_path = dwg.textPath(f"#{path_id}", line, startOffset="50%")
            text.add(text_path)
            dwg.add(text)

# Save SVG
dwg.save()
