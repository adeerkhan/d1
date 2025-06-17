import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys
from rplan_data_coordtok import (
    TokenizationSchema,
    decode_tokens_to_polygons,
    pretty_token,
    IndexToName,
)

def load_any_data(json_file_path, schema=None):
    with open(json_file_path) as f:
        info = json.load(f)

    if "input_ids" in info:  # encoded sample produced by RPlanDataset
        schema = schema or TokenizationSchema()
        tokens = info["input_ids"]
        room_polygons, room_types = decode_tokens_to_polygons(tokens, schema)
        room_polygons = [np.array(p) for p in room_polygons]  # ensure ndarray for plotting
        return room_polygons, room_types, [], []
    else:  # fallback to original raw format
        room_polygons = [np.array(poly) for poly in info["room_polygons"]]
        room_types = info["rms_type"]
        return room_polygons, room_types, [], []

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate distance from a point to a line segment.
    A copy of this exists in wall_removal.py, but is included here
    to make the visualization script self-contained.
    """
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-6:
        return np.linalg.norm(point_vec)
    
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
    projection = line_start + t * line_vec
    
    return np.linalg.norm(point - projection)

def are_polygons_adjacent(poly1, poly2, tolerance=0.1):
    """
    Check if two polygons are adjacent by checking if any vertex of one 
    is very close to an edge of the other. This is more robust than
    checking for shared vertices, which can fail due to floating point inaccuracies.
    """
    # Check if any vertex of poly1 is close to an edge of poly2
    for p1_vertex in poly1:
        for j in range(len(poly2)):
            p2_start = poly2[j]
            p2_end = poly2[(j + 1) % len(poly2)]
            if point_to_line_distance(p1_vertex, p2_start, p2_end) < tolerance:
                return True

    # Check if any vertex of poly2 is close to an edge of poly1
    for p2_vertex in poly2:
        for i in range(len(poly1)):
            p1_start = poly1[i]
            p1_end = poly1[(i + 1) % len(poly1)]
            if point_to_line_distance(p2_vertex, p1_start, p1_end) < tolerance:
                return True
                
    return False

def visualize_floorplan(folder_path, show_doors=True, show_points=False):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return

    num_samples = min(9, len(json_files))
    selected_files = random.sample(json_files, num_samples)
    
    print(f"Visualizing {num_samples} randomly selected files from {folder_path}")
    
    room_names = {
        1: 'Living room', 
        2: 'Kitchen', 
        3: 'Bedroom', 
        4: 'Bathroom', 
        5: 'Balcony', 
        6: 'Dining room', 
        7: 'Study room',
        8: 'Storage'
    }
    
    room_type_colors = {
        1: '#FFB6C1',  # Light pink - Living room
        2: '#98FB98',  # Pale green - Kitchen
        3: '#87CEEB',  # Sky blue - Bedroom
        4: '#DDA0DD',  # Plum - Bathroom
        5: '#F0E68C',  # Khaki - Balcony
        6: '#E6E6FA',  # Lavender - Dining room
        7: '#FFE4B5',  # Moccasin - Study room
        8: '#D3D3D3',  # Light gray - Storage
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, selected_file_name in enumerate(selected_files):
        json_file_path = os.path.join(folder_path, selected_file_name)
        
        room_polygons, room_types, door_polygons, door_types = load_any_data(json_file_path)
        
        ax = axes[idx]
        ax.set_title(f'Sample {os.path.splitext(selected_file_name)[0]}', fontsize=10)
        all_polygons = room_polygons
        if all_polygons:
            all_coords = np.vstack(all_polygons)
            min_coord = np.min(all_coords)
            max_coord = np.max(all_coords)
        else:
            min_coord, max_coord = 0, 256
            
        if len(room_polygons) > 0:
            door_fill_colors = {}
            if not show_doors:
                rooms = [(p, t) for p, t in zip(room_polygons, room_types) if t not in [9, 10]]
                for i, (polygon, room_type) in enumerate(zip(room_polygons, room_types)):
                    if room_type in [9, 10]:  # It's a door
                        fill_color = '#F5F5F5'  # Default fallback color
                        for room_poly, r_type in rooms:
                            if are_polygons_adjacent(polygon, room_poly):
                                fill_color = room_type_colors.get(r_type, '#F5F5F5')
                                break
                        door_fill_colors[i] = fill_color

            for i, (polygon, room_type) in enumerate(zip(room_polygons, room_types)):
                plot_polygon = polygon
                
                room_color = room_type_colors.get(room_type, '#F5F5F5')

                # Determine visual style based on flags
                face_alpha = 0.0 if show_points else 0.9

                if room_type in [9, 10]:  # Door polygons
                    if show_doors:
                        face_alpha = 0.4 if not show_points else 0.0
                    else:
                        room_color = door_fill_colors.get(i, room_color)

                if show_points:
                    # Draw polygon edges explicitly so they remain visible
                    closed_poly = np.vstack([plot_polygon, plot_polygon[0]])
                    ax.plot(closed_poly[:, 0], closed_poly[:, 1], 
                            color=room_color, linewidth=1, zorder=4)
                else:
                    # Draw filled polygon as before
                    ax.fill(plot_polygon[:, 0], plot_polygon[:, 1], 
                            color=room_color, alpha=face_alpha, edgecolor=room_color, linewidth=1)

                # Overlay corner points if requested
                if show_points:
                    ax.scatter(plot_polygon[:, 0], plot_polygon[:, 1], 
                               color=room_color, edgecolors='k', s=20, zorder=5)
                
                # Only show labels for main room types (not doors/walls)
                if (not show_points) and (room_type not in [9, 10, 11]):  # Skip doors and walls when showing fills
                    centroid = np.mean(plot_polygon, axis=0)
                    room_name = room_names.get(room_type, f'Type {room_type}')
                    ax.text(centroid[0], centroid[1], room_name, 
                           ha='center', va='center', fontsize=6, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor='none'))
                           
        ax.set_xlim(min_coord, max_coord)
        ax.set_ylim(min_coord, max_coord)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
    
    for idx in range(num_samples, 9):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return selected_files