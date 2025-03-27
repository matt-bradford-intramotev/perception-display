import redis
import time
import livox_pb2 as proto
import livox_detections_pb2 as proto_det
import numpy as np
import matlab.engine
import sys

# Settings
XMIN = 0
XMAX = 40
YMIN = -15
YMAX = 15
PITCH_OFFSET = 1.4
ROLL_OFFSET = -0.9
VERTICAL_OFFSET = 1.9
ZMIN = 0
ZMAX = 3
HMAX = 2.3

# Display settings
DEFAULT_AZ = 30
DEFAULT_EL = 30
DEFAULT_ZOOM = 1.5
DO_SET_DEFAULT_VIEW = False
DO_PRINT_CURRENT_VIEW = True

# Jetson IP 
if len(sys.argv) < 2:
    print("Using default IP 192.168.1.21 (TrinFlo A)")
    HOST_IP = '192.168.1.21'
else:
    HOST_IP = sys.argv[1]

# Flag to include detections from livox_detector
DO_DETS = False

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

r = redis.Redis(host=HOST_IP, port=6379, db=0)

lidar_channel = "avia_points"
detection_channel = "avia_detections"

# Parameters to estimate FPS
time_start = time.time()
N = 0
fps_av = 0

# Initialize point cloud data
x_array = np.array([0,50])
y_array = np.array([0,0])
z_array = np.array([0,4])
r_array = np.array([1,1])
points = np.vstack((x_array,y_array,z_array)).T

# Set up window
eng = matlab.engine.start_matlab()
eng.eval(f"lidarviewer = pcplayer([{XMIN} {XMAX}],[{YMIN} {YMAX}],[{ZMIN} {ZMAX}])", nargout=0)

eng.eval("figHandle = ancestor(lidarviewer.Axes, 'figure');", nargout=0)
eng.eval("figHandle.WindowState = 'maximized';", nargout=0)

if DO_SET_DEFAULT_VIEW:
    eng.eval("ax = lidarviewer.Axes;", nargout=0)
    eng.eval(f"view(ax, {DEFAULT_AZ}, {DEFAULT_EL});", nargout=0)
    eng.eval(f"camzoom(ax, {DEFAULT_ZOOM});", nargout=0)

pubsub = r.pubsub()
if DO_DETS:
    pubsub.subscribe([lidar_channel,detection_channel])
else:
    pubsub.subscribe([lidar_channel])

for message in pubsub.listen():
    if message['type'] == 'message' and message['channel'] == lidar_channel.encode('ascii'):
        # Estimate frame rate
        try:
            N += 1
            time_end = time.time()
            fps = 1/(time_end-time_start)
            fps_av = (N-1)*fps_av/N + fps/N
            print(f"Message received (fps {fps}, running average fps {fps_av})")
            time_start = time_end

            # Process timer
            timer_start = time.time()

            #Unpack message and align
            point_packet = proto.Point_Packet()
            point_packet.ParseFromString(message['data'])
            system_timestamp_us = point_packet.system_timestamp_us
            sensor_timestamp_us = point_packet.timestamp_us
            # x_array = np.array([point.x_coord/1e3 for point in point_packet.list_points])
            # y_array = np.array([point.y_coord/1e3 for point in point_packet.list_points])
            # z_array = np.array([point.z_coord/1e3 for point in point_packet.list_points])
            # r_array = np.array([point.reflectivity for point in point_packet.list_points])
            x_array = np.array(point_packet.x_coords)/1e3
            y_array = np.array(point_packet.y_coords)/1e3
            z_array = np.array(point_packet.z_coords)/1e3
            r_array = np.array(point_packet.reflectivity)
            z_array = z_array + VERTICAL_OFFSET
            # inds = (z_array < ZCLIP) & (x_array > XCLIPMIN) & (y_array > -YCLIP) & (y_array < YCLIP) & (z_array > ZCLIPGROUND)

            x_array, y_array, z_array = rotate_coordinates(x_array,y_array,z_array,PITCH_OFFSET,'y')
            x_array, y_array, z_array = rotate_coordinates(x_array,y_array,z_array,ROLL_OFFSET,'x')

            inds = (x_array > XMIN) & (x_array < XMAX) & (y_array > YMIN) & (y_array < YMAX) & (z_array > ZMIN) & (z_array < ZMAX)
            x_array = x_array[inds]
            y_array = y_array[inds]
            z_array = z_array[inds]
            r_array = r_array[inds]
            
            #Add in limits to stabilize color
            x_array = np.append(x_array,[XMIN, XMIN])
            y_array = np.append(y_array,[0, 0])
            z_array = np.append(z_array,[ZMIN, ZMAX])
            
            points = np.vstack((x_array,y_array,z_array)).T

            #Plot using matlab engine
            eng.workspace['points'] = matlab.double(points)

            eng.eval("ptCloud = pointCloud(points);", nargout = 0)
            eng.eval("view(lidarviewer,ptCloud)", nargout = 0)
            eng.drawnow(nargout=0)

            # Print current az, el, zoom
            if DO_PRINT_CURRENT_VIEW:
                eng.eval("[az, el] = view(lidarviewer.Axes);", nargout=0)
                eng.eval("zoomVal = camva(lidarviewer.Axes);", nargout=0)
                eng.eval("fprintf('Az = %.2f°, El = %.2f°, Zoom (FOV) = %.2f°\\n', az, el, zoomVal);", nargout=0)

            print(f"Time to process and display: {(time.time() - timer_start)*1e3} ms")
            
        except:
            print('Trouble with message.')

    elif DO_DETS and (N>0) and message['type'] == 'message' and message['channel'] == detection_channel.encode('ascii'):
        detections = proto_det.LidarDetectionList()
        detections.ParseFromString(message['data'])
        upstream_sensor_timestamp_us = detections.sensor_timestamp_us
        if upstream_sensor_timestamp_us == sensor_timestamp_us:
            det_x_coords = np.array([detection.x_coordinate_m for detection in detections.lidar_detections])
            det_y_coords = np.array([detection.y_coordinate_m for detection in detections.lidar_detections])
            det_z_coords = np.array([detection.z_coordinate_m for detection in detections.lidar_detections])

            det_heights = np.array([detection.height_m for detection in detections.lidar_detections])
            det_sizes = np.array([detection.size_m2 for detection in detections.lidar_detections])
            det_reflectances = np.array([detection.reflectance for detection in detections.lidar_detections])
            det_num_points = np.array([detection.num_points for detection in detections.lidar_detections])
            
            det_inds = (det_x_coords > XMIN) & (det_x_coords < XMAX) & (det_y_coords > YMIN) & (det_y_coords < YMAX) & (det_z_coords > ZMIN) & (det_z_coords < ZMAX) & (det_heights < HMAX)

            det_x_coords = det_x_coords[det_inds]
            det_y_coords = det_y_coords[det_inds]
            det_z_coords = det_z_coords[det_inds]
            det_heights = det_heights[det_inds]
            det_sizes = det_sizes[det_inds]
            det_reflectances = det_reflectances[det_inds]
            det_num_points = det_num_points[det_inds]

            # det_x_coords, det_y_coords, det_z_coords = rotate_coordinates(det_x_coords, det_y_coords, det_z_coords,PITCH_OFFSET,'y')
            # det_x_coords, det_y_coords, det_z_coords = rotate_coordinates(det_x_coords, det_y_coords, det_z_coords,ROLL_OFFSET,'x')   

            eng.workspace['dets'] = np.vstack((det_x_coords,det_y_coords,det_z_coords,det_heights,det_sizes,det_reflectances,det_num_points)).T

            eng.eval("showShape('cuboid',[dets(:,1) dets(:,2) dets(:,3) dets(:,5).^(1/4) dets(:,5).^(1/4) dets(:,4) 0*dets(:,1) 0*dets(:,1) 0*dets(:,1)],parent=lidarviewer.Axes,color='green',opacity=0.5);", nargout = 0)
        
        else:
            eng.eval("showShape('cuboid',[0 0 0 0 0 0 0 0 0],parent=lidarviewer.Axes,color='green',opacity=0.5);")
