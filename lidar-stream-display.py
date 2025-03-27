import redis
import time
import livox_pb2 as proto
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

MINZ = 0
MAXZ = 2
ZCLIP = 4
XCLIPMIN = 0
YCLIP = 15
PITCH_OFFSET = 2
VERTICAL_OFFSET = 2.2
ZCLIPGROUND = -1

# Function to generate colors based on the viridis colormap
def generate_colors(z_values):
    # Use the z-coordinate to determine color
    # z_values = points[:, 2]
    # min_z, max_z = np.min(z_values), np.max(z_values)
    min_z = MINZ
    max_z = MAXZ
    
    # Normalize z values to the range [0, 1]
    normalized_z = np.clip((z_values - min_z) / (max_z - min_z),0,1)
    
    # Apply the viridis colormap
    colormap = cm.get_cmap('viridis')
    colors = colormap(normalized_z)[:, :3]  # Drop the alpha channel
    return colors

# Function to create a grid
def create_grid(size=10, step=1):
    lines = []
    colors = []
    points = []
    for i in range(-size, size + 1, step):
        # Lines parallel to x-axis
        points.append([i, -size, 0])
        points.append([i, size, 0])
        lines.append([len(points) - 2, len(points) - 1])
        # Lines parallel to y-axis
        points.append([-size, i, 0])
        points.append([size, i, 0])
        lines.append([len(points) - 2, len(points) - 1])
        # Add color for each line
        colors.append([0.5, 0.5, 0.5])
        colors.append([0.5, 0.5, 0.5])
    points = np.array(points)
    lines = np.array(lines).reshape(-1, 2)
    colors = np.array(colors)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# Function to create range rings
def create_range_rings(radius_max=10, radius_step=1, num_segments=100):
    points = []
    lines = []
    colors = []

    for radius in np.arange(radius_step, radius_max + radius_step, radius_step):
        for i in range(num_segments):
            theta = 2.0 * np.pi * i / num_segments
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, 0])
            if i > 0:
                lines.append([len(points) - 2, len(points) - 1])
        # Connect the last segment to the first
        lines.append([len(points) - 1, len(points) - num_segments])
        for _ in range(num_segments):
            colors.append([0.5, 0.5, 0.5])

    points = np.array(points)
    lines = np.array(lines)
    colors = np.array(colors)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def rotate_coordinates(x, y, z, angle, axis='x'):
    """
    Rotate coordinates (x, y, z) around the specified axis.

    Parameters:
    - x, y, z: NumPy arrays representing the coordinates.
    - angle: The rotation angle in degrees.
    - axis: The axis of rotation ('x', 'y', or 'z').

    Returns:
    Rotated coordinates (x_rot, y_rot, z_rot).
    """
    angle_rad = np.radians(angle)

    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                    [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                                    [0, 1, 0],
                                    [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Use 'x', 'y', or 'z'.")

    rotated_coordinates = np.dot(rotation_matrix, np.vstack((x, y, z)))
    return rotated_coordinates[0], rotated_coordinates[1], rotated_coordinates[2]

r = redis.Redis(host='10.0.0.188', port=6379, db=0)

data_channel = "avia_points"

pubsub = r.pubsub()
pubsub.subscribe([data_channel])

time_start = time.time()
N = 0
fps_av = 0

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create a point cloud object
pcd = o3d.geometry.PointCloud()

# Initialize point cloud data
x_array = np.array([0,50])
y_array = np.array([0,0])
z_array = np.array([0,4])
r_array = np.array([1,1])
points = np.stack((x_array, y_array, z_array), axis=-1)
pcd.points = o3d.utility.Vector3dVector(np.stack((x_array, y_array, z_array), axis=-1))

# Generate colors based on the z-coordinate
colors = generate_colors(z_array)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Add point cloud to visualizer
vis.add_geometry(pcd)

# # Add coordinate frame to visualizer
# coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
# vis.add_geometry(coordinate_frame)

# Add grid to visualizer
grid = create_grid(size=50, step=5)
vis.add_geometry(grid)

# # Add range rings to visualizer
# range_rings = create_range_rings(radius_max=10, radius_step=1, num_segments=100)
# vis.add_geometry(range_rings)

# Set up the camera parameters
ctr = vis.get_view_control()

# Define camera parameters
camera_position = np.array([-5, 0, 2])  # Camera position in world coordinates
look_at = np.array([0, 0, 2])  # Look-at point
up_vector = np.array([0, 0, 1])  # Up vector

# Set the camera view
ctr.set_lookat(look_at)
ctr.set_front((camera_position - look_at) / np.linalg.norm(camera_position - look_at))
ctr.set_up(up_vector)
ctr.set_zoom(0.1)  # Adjust zoom level as needed

# Access render options to set point size
opt = vis.get_render_option()
opt.point_size = 2.0  # Adjust the point size as needed

for message in pubsub.listen():
    if message['type'] == 'message' and message['channel'] == data_channel.encode('ascii'):
        N += 1
        time_end = time.time()
        fps = 1/(time_end-time_start)
        fps_av = (N-1)*fps_av/N + fps/N
        print(f"Message received (fps {fps}, running average fps {fps_av})")
        time_start = time_end

        point_packet = proto.Point_Packet()
        point_packet.ParseFromString(message['data'])
        system_timestamp_us = point_packet.system_timestamp_us
        sensor_timestamp_us = point_packet.timestamp_us
        x_array = np.array([point.x_coord/1e3 for point in point_packet.list_points])
        y_array = np.array([point.y_coord/1e3 for point in point_packet.list_points])
        z_array = np.array([point.z_coord/1e3 for point in point_packet.list_points])
        r_array = np.array([point.reflectivity for point in point_packet.list_points])
        inds = (z_array < ZCLIP) & (x_array > XCLIPMIN) & (y_array > -YCLIP) & (y_array < YCLIP) & (z_array > ZCLIPGROUND)
        x_array = x_array[inds]
        y_array = y_array[inds]
        z_array = z_array[inds]
        r_array = r_array[inds]
        x_array, y_array, z_array = rotate_coordinates(x_array,y_array,z_array,PITCH_OFFSET,'y')
        z_array = z_array + VERTICAL_OFFSET

        # Update pt cloud data
        pcd.points = o3d.utility.Vector3dVector(np.stack((x_array, y_array, z_array), axis=-1))
        colors = generate_colors(z_array)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Update visualizer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

