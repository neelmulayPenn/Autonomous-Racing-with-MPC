#!/usr/bin/env python3
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
# from utils import nearest_point

# TODO CHECK: include needed ROS msg type headers and libraries


# Importing in utils function: nearest_point
from numba import njit

@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error 
            will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the 
        closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed] = [a, delta]
    TK: int = 8  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to TUNE the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]) # Keep yaw cost at 0 for now (13.0)
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0]) # Keep yaw rate as 0 for now  (OG: 13.0)
    )  # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03 #0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189*1.75  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189*1.75  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s] 6.0
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 2.0  # maximum acceleration [m/ss] 3.0


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        """
        Initialize the MPC node, create subscribers and publishers, load waypoints
        Note: 
        - The waypoints are loaded from a CSV file, which contains x, y, yaw, and speed. 
        - Should not have a header row.
        Inputs: None
        Returns: None
        """
        # LAB 8:
        # Levine2nd_Neel_1300 for Levine's Clean Sim Map.
        # waypoints_new_new for our remapped Levine SLAM Map.
        # csv_wrap_2pi = 0, since our csv is pi to -pi
        # 
        # FINAL RACE: 
        # final_race_pennovation_yaw: Naive closed loop just centerline s
        # final_race_pennovation_optimal_sim_mpc: Optimal and formatted for mpc
        waypoint_filepath = "/home/modlab/f1tenth_labs_ws/src/model-predictive-control-f1-5th/waypoints/zirui_race2_centerline_v_opt_yaw.csv"
        waypoint_data = self.load_waypoints(waypoint_filepath)
        self.waypoints = np.array(waypoint_data, dtype=np.float64)

        # self.waypoints = np.array(waypoint_data, dtype=np.float64)
        self.csv_wrap_2pi = 0 #0 for pi to -pi | 1 for 0-2pi 

        if self.waypoints.shape[0] == 0:
            self.get_logger().warn("No waypoints available")
            return
        print("Waypoints extracted successfully!")


        # Create subscribers and publishers
        self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10) # On sim
        # self.create_subscription(Odometry, '/pf/pose/odom', self.pose_callback, 10) # On Car
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10) # On Car
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, 'drive',10)
        self.ref_traj_publisher = self.create_publisher(Marker, 'reference_trajectory_marker', 10)
        self.mpc_solved_traj_publisher = self.create_publisher(Marker,'mpc_solved_trajectory_marker', 10)
        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # AEB Variable Initialization
        self.current_x_vel = 0.1
        self.iTTC = np.array([np.inf])
        self.safe_stop_threshold = 0.5 #s
        self.scan_angle_min = -np.pi/4 # -45 degrees
        self.scan_angle_max = np.pi/4 # 45 degrees

        # Gap Follow Variable Initialization  
        self.mpc_last_steering_angle = 0.0
        self.rb = 2.0
        self.max_range = 50
        self.prev_steering_angle=0
        # PID control parameters
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        # PID gains - you'll need to tune these values
        self.kp = 0.6  # Proportional gain
        self.ki = 0.0  # Integral gain
        self.kd = 0.03  # Derivative gain 0.05
        # Maximum integral windup
        self.max_integral = 1.0


        # initialize MPC problem
        self.mpc_prob_init()

    def scan_callback(self, scan_msg):
        #################################
        # AEB
        #################################
        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        range_min = scan_msg.angle_min
        range_max = scan_msg.range_max
        increment = scan_msg.angle_increment
        r = np.array(scan_msg.ranges) # Gather the range measurements
        # self.get_logger().info(f'len(r): {len(r)}')
        # self.get_logger().info(f'angle_min: {angle_min}, angle_max: {angle_max}, range_min: {range_min}, range_max: {range_max}, increment: {increment}')

        angles = angle_min + np.arange(len(r)) * increment

        # Mask: keep angles between scan_angle_min and scan_angle_max
        # (Make sure you fixed the min/max as discussed)
        angle_mask = (angles >= self.scan_angle_min) & (angles <= self.scan_angle_max)

        # Apply mask
        inside_range_data = r[angle_mask]
        inside_angles = angles[angle_mask]

        # Now clean inf/nan and range min/max
        good_idx = np.isfinite(inside_range_data) & (inside_range_data > range_min) & (inside_range_data < range_max)
        r = inside_range_data[good_idx]
        angle = inside_angles[good_idx]

        r_dot = self.current_x_vel * np.cos(angle) # Calculate r dot
        r_dot = np.maximum(1*r_dot, 0) # Take only shrinking values or 0
        r_dot[r_dot==-0.0] = 0.0

        self.iTTC = r/r_dot
        # self.get_logger().info(f'iTTC updated')

        #################################
        # Follow the Gap
        #################################
        proc_ranges = self.preprocess_lidar(r)
        
        # TODO:
        #Find closest point to LiDAR

        #Eliminate all points inside 'bubble' (set them to zero) 
        start,end = self.find_max_gap(proc_ranges)
        #Find max length gap 
        velocity = 0.
        steering_angle = 0.

        if not np.isnan(start):

            #Find the best point in the gap 
            max_index = self.find_best_point(start,end,proc_ranges)
            
            angles = np.linspace(angle_min, angle_max, len(r))
            desired_angle = angles[max_index]

            if abs(desired_angle) < np.deg2rad(10):
        
                # Compute error (difference between desired and current angle)
                error = desired_angle - self.prev_steering_angle
                
                # Apply PID control
                steering_angle = self.pid_control(error, self.get_clock().now())
            else:
                steering_angle = desired_angle
            
            # Clip the steering angle to valid range (if needed)
            steering_angle = np.clip(steering_angle, -np.pi/3, np.pi/3)
        
        # change speed by angle
        self.gap_follow_angle = steering_angle
        # self.get_logger().info(f'best gap follow angle updated')

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        
        ranges = np.clip(ranges, 0, self.max_range)  # Clip values between 0 and 3
        ranges = np.nan_to_num(ranges, nan=0)  # Replace NaN with 0
        
        # Assuming 'ranges' is already a NumPy array
        window = np.ones(self.max_range) / self.max_range  # Create a window of size 3 with equal weights
        ranges_new = np.convolve(ranges, window, mode='same')

        # Adjust the first and last elements
        ranges_new[0] = (ranges[0] + ranges[1]) / 2
        ranges_new[-1] = (ranges[-2] + ranges[-1]) / 2
        
        proc_ranges = ranges_new
        return proc_ranges
    
    #Method to return the longest consecutive non zero subarray of arr
    def longest_consecutive_non_zero(self,arr):
    # Convert to NumPy array if not already
        arr = np.asarray(arr)
        
        # Find where non-zero elements start and end
        zero_positions = np.where(arr == 0)[0]
        
        # Handle edge cases
        if len(zero_positions) == 0:
            return np.nan, np.nan
        if len(zero_positions) == len(arr):
            return 0, len(arr)-1
        
        # Add start and end markers
        zero_positions = np.concatenate(([0], zero_positions, [len(arr)]))
        
        # Calculate lengths of non-zero subarrays
        lengths = np.diff(zero_positions)
        
        # Find the maximum length and its position
        max_length = np.max(lengths)
        max_pos = np.argmax(lengths)
        
        # Return the slice of the original array
        start = zero_positions[max_pos]
        end = zero_positions[max_pos + 1]
        return start+1,end-1

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        # Finding the nearest index
        nearest_index = np.argmin(free_space_ranges)
        min_dist = free_space_ranges[nearest_index]
        
        safety_low = min_dist - self.rb
        safety_high = min_dist + self.rb
        
        condition = (free_space_ranges<=safety_high) & (free_space_ranges>=safety_low)
        indices = np.where(condition)
        free_space_ranges[indices]=0
        
        start,end=self.longest_consecutive_non_zero(free_space_ranges)
        return start,end
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        arr = np.array(ranges[start_i:end_i])
        max_index = len(arr)//2
        
        # Select a point closer to the inside of the turn (inner point)
        mid = len(ranges) // 2
        #if end_i < mid:  # Gap is on the left side
        #best_index = start_i + (end_i - start_i) // 5  # Choose a point 25% into the gap
        if start_i > mid:  # Gap is on the right side
            best_index = end_i - (end_i - start_i) // 3  # Choose a point 75% into the gap
        else:  # Gap is centered
            best_index = start_i + 3 * (end_i - start_i) // 5  # Default to midpoint
        return best_index
    

    def pid_control(self, error, current_time):
        """
        Compute steering adjustment using PID control
        """
        # Initialize time on first call
        if self.last_time is None:
            self.last_time = current_time
            return error * self.kp  # Only use P control on first iteration
        
        # Time delta
        dt = (current_time - self.last_time).nanoseconds / 1e9  # Convert to seconds
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        
        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        
        # PID output
        output = (self.kp * error +  # Proportional
                self.ki * self.integral +  # Integral
                self.kd * derivative)  # Derivative
        
        # Store values for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        return output

    def pose_callback(self, pose_msg):
        # TODO: extract pose from ROS msg
        vehicle_state = State()
        vehicle_quaternion = pose_msg.pose.pose.orientation
        vehicle_state.x = np.float64(pose_msg.pose.pose.position.x)
        vehicle_state.y = np.float64(pose_msg.pose.pose.position.y)

        vehicle_state.yaw = np.float64(self.get_yaw_from_quaternion(vehicle_quaternion))
        norm_speed = np.linalg.norm(np.array([pose_msg.twist.twist.linear.x, pose_msg.twist.twist.linear.y]))
        vehicle_state.v = norm_speed
        vehicle_state.v = max(norm_speed, 1.5)

        self.current_x_vel = pose_msg.twist.twist.linear.x

        # print(f"Current vehicle state: x={vehicle_state.x}, y={vehicle_state.y}, yaw={vehicle_state.yaw}, v={vehicle_state.v}")


        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        #       waypoints file need 4 columns, N*4
        # LAB 8: Our csv file is x,y,yaw, v.
        ref_x = self.waypoints[:,0]
        ref_y = self.waypoints[:,1]
        ref_v = self.waypoints[:,3]
        ref_yaw = self.waypoints[:,2]

        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        # print("x0k = ", x0)
        # print("x_ref = ", ref_path)
        # print(f'Reference trajectory: X {ref_path[0, 0]} Y {ref_path[1, 0]} Yaw {ref_path[3, 0]}, Vel {ref_path[2, 0]}')
        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)
        # print(f'MPC Solved for: Accel {self.oa} Vel {ov} Steer velocity {self.odelta_v}')
        # print(f'MPC Solved for: X {ox[0]} Y {oy[0]} Yaw {oyaw[0]}, Vel {ov[0]}')
        print(f'Current state: v={vehicle_state.v} | Ref Trajectory: Vel {ref_path[2, 0]} | MPC Solved for: Vel {ov[0]}')

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        for i in range (len(ox)):
            point = Point()
            point.x = ox[i]
            point.y = oy[i]
            point.z = 0.0
            marker.points.append(point)
        self.mpc_solved_traj_publisher.publish(marker)


        # TODO: publish drive message.
        steer_output = self.odelta_v[0] 
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

        drive_msg = AckermannDriveStamped()

        iTTC_stop_bool = self.iTTC < self.safe_stop_threshold
        if iTTC_stop_bool.any():
            print("########################################################")
            print("FOLLOW THE GAP")
            print("########################################################")
            drive_msg.drive.steering_angle = 1.0*(self.gap_follow_angle - self.mpc_last_steering_angle) + self.mpc_last_steering_angle
            drive_msg.drive.speed = self.current_x_vel*0.9
            self.drive_publisher.publish(drive_msg)
            self.get_logger().warning(f"Running follow the gap with steering angle {drive_msg.drive.steering_angle}")

        else:
            print("DRIVING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            drive_msg.drive.steering_angle = steer_output
            drive_msg.drive.speed = speed_output
            self.mpc_last_steering_angle = steer_output
            # drive_msg.drive.speed = 0.5
            self.drive_publisher.publish(drive_msg)

        # drive_msg.drive.steering_angle = steer_output
        # drive_msg.drive.speed = speed_output
        # # drive_msg.drive.speed = 0.5
        # self.drive_publisher.publish(drive_msg)
    

        
    # def get_yaw_from_quaternion(self, quaternion):
    #     """
    #     Convert quaternion to yaw angle
    #     :param quaternion: Quaternion from ROS message
    #     :return: yaw angle in radians
    #     """
    #     # Extract the components of the quaternion
    #     x = quaternion.x
    #     y = quaternion.y
    #     z = quaternion.z
    #     w = quaternion.w

    #     # Calculate the yaw angle (in radians)
    #     siny_cosp = 2.0 * (w * z + x * y)
    #     cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    #     yaw = math.atan2(siny_cosp, cosy_cosp)

    #     return yaw

    def get_yaw_from_quaternion(self, orientation):
        """ Convert quaternion to yaw angle """
        import tf_transformations
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        euler = tf_transformations.euler_from_quaternion(quaternion)
        return euler[2]

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somewhere

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective = cvxpy.quad_form(cvxpy.vec(self.uk),R_block)

        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        reference_trajectory_diff = cvxpy.vec(self.xk - self.ref_traj_k)
        objective += cvxpy.quad_form(reference_trajectory_diff, Q_block) 
        objective += cvxpy.quad_form(self.xk[:, -1] - self.ref_traj_k[:, -1], self.config.Qfk)

        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        control_input_diff = cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1])
        objective += cvxpy.quad_form(control_input_diff, Rd_block)

        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        # print(self.Ak_.shape, self.Bk_.shape, self.Ck_.shape)
        # print(self.xk.shape, self.uk.shape)
        constraints.append(cvxpy.vec(self.xk[:, 1:]) == self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_)

        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        constraints.append(self.uk[1,:] >= self.config.MIN_STEER) # Min steering angles
        constraints.append(self.uk[1,:] <= self.config.MAX_STEER) # Max steering angles
        constraints.append(self.uk[1, 1:] - self.uk[1, :-1] <= self.config.MAX_DSTEER * self.config.DTK) # Max steering rate

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        constraints.append(self.uk[0,:] <= self.config.MAX_ACCEL) # Max acceleration output

        constraints.append(self.xk[2, :] <= self.config.MAX_SPEED) # Max speed output
        constraints.append(self.xk[2, :] >= self.config.MIN_SPEED) # Min speed output
        
        constraints.append(self.xk[:, 0] == self.x0k) # Intial position constraint
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        ahead_ind = 0 #2
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)
        ind += ahead_ind  # Look ahead a few waypoints
        # [Erica] Update dlk to be the distance between waypoints
        self.config.dlk = np.linalg.norm(np.array([cx[1]-cx[0], cy[1]-cy[0]]))
        # print(self.config.dlk)

        # print(f'nearest point: {cx[ind]}, {cy[ind]} | Current position: {state.x}, {state.y}')

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK  
        
        dind = travel / self.config.dlk
        # print(f'Travel distance: {travel}, dind: {dind}, dlk: {self.config.dlk}, state velocity : {state.v}')
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]

        if self.csv_wrap_2pi:
            # Angle wraparound assumign 0-2pi
            cyaw[cyaw - state.yaw > 4.5] = np.abs(
                cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
            )
            cyaw[cyaw - state.yaw < -4.5] = np.abs(
                cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
            )
        else:
            # Angle wraparound assumign pi to -pi
           cyaw = state.yaw + ((cyaw - state.yaw + np.pi) % (2 * np.pi) - np.pi)
        
        ref_traj[3, :] = cyaw[ind_list]

        # Publishing reference trajectory rollout to trajectory marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        for i in range(len(ind_list)):
            point = Point()
            point.x = ref_traj[0, i]
            point.y = ref_traj[1, i]
            point.z = 0.0
            marker.points.append(point)

        self.ref_traj_publisher.publish(marker)

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C
    # def cal_point_ahead(self, x0):
    #     """
    #     Calculate the point ahead of the vehicle based on its current state
    #     :param x0: current position (x, y)
    #     :param v: current speed
    #     :param phi: current heading angle
    #     :param delta: current steering angle
    #     :return: point ahead (x, y)
    #     """
    #     v = 2.0  # current speed
    #     x_ahead = x0[0] + v * math.cos(x0[3]) * self.config.DTK
    #     y_ahead = x0[1] + v * math.sin(x0[3]) * self.config.DTK
    #     return [x_ahead, y_ahead, x0[2], x0[3]]
    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        # x0_ahead = self.cal_point_ahead(x0)
        # self.x0k.value = x0_ahead
        self.x0k.value = x0
        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        # print(self.MPC_prob.status)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()
            # print("the 1st xk = ", ox[0], oy[0], ov[0], oyaw[0])
        

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    def load_waypoints(self, file_path):
            """ Load waypoints from a CSV file into a NumPy array """
            try:
                waypoints = np.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows=1)
                self.get_logger().info(f"Loaded {waypoints.shape[0]} waypoints")
                return waypoints
            except Exception as e:
                self.get_logger().error(f"Failed to load waypoints: {e}")
                return np.empty((0, 2), dtype=np.float32) 

def main(args=None):
    print("Hello World!!!")
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()