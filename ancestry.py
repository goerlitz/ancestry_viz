import svgwrite
import math

# Parameters
center = (500, 500)
ring_thickness = 40
num_rings = 4
ring_radii = [100 + i * ring_thickness for i in range(num_rings)]
segments_per_ring = [2, 4, 8, 16]
segment_angles = [100, 50, 25, 12.5]
total_angle = 200
start_angle_offset = 170
line_spacing = 12  # spacing between text lines

# Create SVG drawing
dwg = svgwrite.Drawing('./radial_family.svg', size=('1000px', '1000px'))
dwg.add(dwg.rect(insert=(0, 0), size=('1000px', '1000px'), fill='white'))

# Draw visible arcs and attach each text line to its own path
for i, (base_radius, segments, angle_span) in enumerate(zip(ring_radii, segments_per_ring, segment_angles)):
    for j in range(segments):
        segment_angle = j * angle_span + start_angle_offset
        start_angle = segment_angle
        end_angle = segment_angle + angle_span

        for k in range(3):  # 3 lines per segment
            radius = base_radius + k * line_spacing  # offset each line outward
            large_arc_flag = 1 if angle_span > 180 else 0
            start = (
                center[0] + radius * math.cos(math.radians(start_angle)),
                center[1] + radius * math.sin(math.radians(start_angle))
            )
            end = (
                center[0] + radius * math.cos(math.radians(end_angle)),
                center[1] + radius * math.sin(math.radians(end_angle))
            )

            path_id = f"path_r{i}_s{j}_l{k}"
            path_d = f"M {start[0]},{start[1]} A {radius},{radius} 0 {large_arc_flag},1 {end[0]},{end[1]}"
            path = dwg.path(d=path_d, fill="none", stroke="lightgray", id=path_id)
            dwg.add(path)

            # Add text to each individual arc path
            text = dwg.text("", font_size="10px", text_anchor="middle")
            text_path = dwg.textPath(f"#{path_id}", f"Line{k+1}", startOffset="50%")
            text.add(text_path)
            dwg.add(text)

# Save SVG
dwg.save()
