#!/usr/bin/env python3

import csv
import math
import numpy as np
from pathlib import Path


if __name__ == "__main__":
    # Dynamically get the directory this script lives in
    script_dir = Path(__file__).resolve().parent
    print(f'Currently inside {script_dir.parent}')

    # Construct paths relative to this script
    input_file = script_dir.parent / 'waypoints' / 'FinalRace3_optimal60.csv'
    output_file = script_dir.parent / 'waypoints' / 'FinalRace3_optimal60_yaw.csv'
    
    print(f'Pulling csv data from {input_file}')

    # Read input data
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    print(f'Extracted {len(data)} waypoints from the input file.')

    # Extract x and y only
    x_vals = [float(row[1]) for row in data]
    y_vals = [float(row[2]) for row in data]
    v_vals = [float(row[5]) for row in data]
    print(f'Extracted {len(x_vals)} waypoints from the input file.')
    print(f'x_vals: {x_vals[:5]}')
    print(f'y_vals: {y_vals[:5]}')
    print(f'v_vals: {v_vals[:5]}')

    # Reverse the data to go in the right direction
    # x_vals = x_vals[::-1]
    # y_vals = y_vals[::-1] 
    # v_vals = v_vals[::-1] 

    # Calculate yaw (heading) between each pair of points
    yaw_vals = []
    for i in range(len(data) - 1):
        dx = x_vals[i + 1] - x_vals[i]
        dy = y_vals[i + 1] - y_vals[i]
        yaw = math.atan2(dy, dx)
        yaw_vals.append(yaw)

    # For the last point, replicate the second-to-last yaw
    yaw_vals.append(yaw_vals[-1])

    # Write x, y, yaw to new CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'yaw','velocities'])
        for i in range(len(data)):
            writer.writerow([x_vals[i], y_vals[i], yaw_vals[i],v_vals[i]])

    print(f'Saving csv data to {output_file}')