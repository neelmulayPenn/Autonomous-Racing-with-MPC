import csv
import math
from pathlib import Path

"""
Reads a CSV file containing x and y coordinates of waypoints,
calculates the yaw (heading) between each pair of points, and writes a new CSV
file with x, y, and yaw values.     
Note: This assumes that the input CSV has a header and the first two columns 
x and y coordinates.

Inputs: 
input_filename: The file with x/y data in the first two rows that have to 
output_filename: The name the x/y/yaw file should be saved to 

Returns: None
"""

# Dynamically get the directory this script lives in
script_dir = Path(__file__).resolve().parent
print(f'Currently inside {script_dir.parent}')

# Construct paths relative to this script
input_file = script_dir.parent / 'waypoints' / 'zirui_race2_closed.csv'
output_file = script_dir.parent / 'waypoints' / 'zirui_race2_closed_yaw.csv'

print(f'Pulling csv data from {input_file}')
print(f'Saving csv data to {output_file}')

# Read input data
with open(input_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = [row for row in reader]

# Extract x and y only
x_vals = [float(row[0]) for row in data]
y_vals = [float(row[1]) for row in data]

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
    writer.writerow(['x', 'y', 'yaw', 'velocities'])
    for i in range(len(data)):
        writer.writerow([x_vals[i], y_vals[i], yaw_vals[i], 3.0])