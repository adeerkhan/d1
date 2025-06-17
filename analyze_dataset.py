import json
import os
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import numpy as np

def check_file_coords(file_path: Path) -> tuple[str, float, float, list | None]:
    """
    Checks a single R-Plan JSON file for out-of-range coordinates.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A tuple containing: (
            file_name,
            min_coord_in_file,
            max_coord_in_file,
            list_of_out_of_range_coords | None
        )
    """
    min_coord, max_coord = float('inf'), float('-inf')
    bad_coords = []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        polygons = data.get('room_polygons', [])
        if not polygons:
            return str(file_path), 0, 0, None

        all_coords = [point for poly in polygons for point in poly]
        if not all_coords:
            return str(file_path), 0, 0, None
            
        coords_array = np.array(all_coords)
        min_coord = coords_array.min()
        max_coord = coords_array.max()

        if min_coord < 0 or max_coord > 256:
             # Find the specific coordinates that are out of bounds
            for x, y in all_coords:
                if not (0 <= x <= 256 and 0 <= y <= 256):
                    bad_coords.append((x,y))
            return str(file_path), min_coord, max_coord, bad_coords

        return str(file_path), min_coord, max_coord, None

    except (json.JSONDecodeError, KeyError) as e:
        # This file is problematic for other reasons
        return str(file_path), 0, 0, [f"Error: {e}"]
    except Exception as e:
        return str(file_path), 0, 0, [f"Unexpected Error: {e}"]

def analyze_dataset(root_dir: str = "dataset", workers: int = 8):
    """
    Scans the entire R-Plan dataset to find coordinate range issues.
    """
    data_dir = Path(root_dir)
    files = sorted([p for p in data_dir.glob("*.json") if p.stem.isdigit()])
    
    if not files:
        print(f"No suitable JSON files found in {root_dir}")
        return

    print(f"Analyzing {len(files)} files in '{root_dir}' using {workers} workers...")
    
    overall_min = float('inf')
    overall_max = float('-inf')
    bad_files = {}
    
    with mp.Pool(workers) as pool:
        results = tqdm(pool.imap_unordered(check_file_coords, files), total=len(files))
        
        for file_name, min_val, max_val, errors in results:
            if min_val < overall_min: overall_min = min_val
            if max_val > overall_max: overall_max = max_val
            if errors is not None:
                bad_files[file_name] = {
                    "min": min_val,
                    "max": max_val,
                    "errors": errors
                }

    print("\n--- Analysis Complete ---")
    print(f"Scanned {len(files)} files.")
    print(f"Overall coordinate range found: [{overall_min:.2f}, {overall_max:.2f}]")
    print("-" * 25)
    
    if not bad_files:
        print("✅ All files have coordinates within the expected [0, 256] range.")
    else:
        print(f"🚨 Found {len(bad_files)} files with out-of-range coordinates or errors:")
        # Sort bad files for consistent output
        sorted_bad_files = sorted(bad_files.items(), key=lambda item: int(Path(item[0]).stem))

        for file_name, details in sorted_bad_files:
            print(f"  - File: {Path(file_name).name}")
            print(f"    - Range: [{details['min']:.2f}, {details['max']:.2f}]")
            print(f"    - Details: {str(details['errors'][:5])[:100]}...") # Show first 5 errors, truncated

if __name__ == "__main__":
    # Use more workers if you have more CPU cores
    cpu_count = os.cpu_count() or 1
    num_workers = min(16, cpu_count) 
    
    analyze_dataset(workers=num_workers)