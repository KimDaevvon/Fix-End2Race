import logging
import os
from PIL import Image
import yaml
from argparse import Namespace
from scipy.ndimage import distance_transform_edt as edt
from latticeplanner.utils import *
from latticeplanner.pure_pursuit import PurePursuitPlanner
from pyclothoids import Clothoid
import numpy as np
from numba import njit
logger = logging.getLogger(__name__)

class LatticePlanner:
    def __init__(self, conf, map_path, wpt_path, wb=0.33):

        self.wheelbase = wb

        ### load waypoints
        self.map_path = map_path
        self.map_ext = '.png'
        waypoints = np.loadtxt(wpt_path, delimiter=';', skiprows=2)
        waypoints = np.vstack((waypoints[:, 1], waypoints[:, 2], waypoints[:, 5], waypoints[:, 3], waypoints[:, 0])).T

        self.waypoints = waypoints
        self.traj_num = 55
        self.lh_grid_rows = 5
        self.lh_grid_lb = conf.lh_grid_lb
        self.lh_grid_ub = conf.lh_grid_ub

        self.traj_points = conf.traj_points
        self.traj_v_scale = conf.traj_v_scale
        self.s_max = self.waypoints[-1, 4]

        # sample and cost function
        self.sample_func = None
        self.add_sample_function(sample_lookahead_square)
        self.shape_cost_funcs = []
        self.constant_cost_funcs = []
        self.selection_func = None
        self.add_shape_cost_function(get_follow_optim_cost)
        self.add_constant_cost_function(get_map_collision)

        self.params_num = conf.params_num
        self.params_name = conf.params_name
        self.params_idx = {}
        count = 0
        for name, num in zip(self.params_name, self.params_num):
            self.params_idx[name] = count
            count += num
        try:
            self.cost_weights_num = conf.weights_num
        except:
            self.cost_weights_num = 4
        self.set_cost_weights(self.cost_weights_num)

        self.v_lattice_span = np.linspace(conf.traj_v_span_min, conf.traj_v_span_max, conf.traj_v_span_num)
        self.v_lattice_num = conf.traj_v_span_num
        self.best_traj = None
        self.best_traj_ref_v = 0.0
        self.best_traj_idx = 0
        self.prev_traj_local = np.zeros((self.traj_points, 2))
        self.prev_opp_pose = np.zeros((1, 2))
        self.goal_grid = None
        self.state_i = None
        self.state_t = None
        self.step_all_cost = {}
        self.all_costs = None
        self.time_interval = conf.tracker_steps * 0.01
        self.last_s = 0.0

        self.tracker = PurePursuitPlanner(conf, wpt_path, wb=wb)
        self.conf = conf
        self.step = 0

        # load map image
        map_img_path = os.path.splitext(self.map_path)[0] + self.map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.
        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        # load map yaml
        with open(self.map_path + '.yaml', 'r') as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            self.map_resolution = map_metadata['resolution']
            self.origin = map_metadata['origin']

        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        self.dt = self.map_resolution * edt(self.map_img)
        self.map_metainfo = (
            self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution)

        # scan
        self.scan_num = conf.scan_num
        self.angle_span = np.linspace(-0.75 * np.pi, 0.75 * np.pi, self.scan_num)
        self.ittc_thres = conf.ittc_thres
        self.collision_thres = 0.35

    def add_shape_cost_function(self, func):
        """
        Add cost function to list for eval.
        """
        if type(func) is list:
            self.shape_cost_funcs.extend(func)
        else:
            self.shape_cost_funcs.append(func)

    def add_constant_cost_function(self, func):
        self.constant_cost_funcs.append(func)

    def set_parameters(self, parameters, v_scale=6.0):
        if type(parameters) == np.ndarray:
            for name, num in zip(self.params_name, self.params_num):
                start = self.params_idx[name]
                if name != 'cost_weights':
                    self.__setattr__(name, parameters[start])
                else:
                    self.set_cost_weights(parameters[start:start + num])
        else:
            for key, value in parameters.items():
                if key == 'cost_weights':
                    self.set_cost_weights(value)
                else:
                    self.__setattr__(key, value)

    def set_cost_weights(self, cost_weights):
        if type(cost_weights) == int:
            n = cost_weights
            cost_weights = np.array([1 / n] * n)
        elif len(cost_weights) != self.cost_weights_num:
            raise ValueError('Length of cost weights must be the same as number of cost functions.')
        self.cost_weights = cost_weights

    def add_sample_function(self, func):
        self.sample_func = func

    def sample(self, pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid):

        if self.sample_func is None:
            raise NotImplementedError('Please set a sample function before sampling.')

        goal_grid = self.sample_func(pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid)

        return goal_grid

    def eval(self, all_traj, all_traj_clothoid, opp_poses, ego_pose):

        cost_weights = self.cost_weights
        n, k = self.traj_num, self.v_lattice_num

        # assume use the **first** weight to calculate curvature cost
        mean_k, _ = get_curvature(all_traj, all_traj_clothoid)  # (n, )
        cost = np.zeros(self.traj_num)

        ## other shape cost, current include length, similarity, map collision,
        for i, func in enumerate(self.shape_cost_funcs):
            cur_cost = func(all_traj, all_traj_clothoid, opp_poses, ego_pose, self.prev_traj_local, self.dt,
                            self.map_metainfo)
            cur_cost = cost_weights[i] * cur_cost
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        ## cost functions with constant scale
        for i, func in enumerate(self.constant_cost_funcs):
            cur_cost = func(all_traj, all_traj_clothoid, opp_poses, ego_pose, self.prev_traj_local, self.dt,
                            self.map_metainfo, self.collision_thres)
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        ## velocity cost, assume use the last two weight to calculate abs velocity cost
        all_traj_min_mean_k = np.min(mean_k)
        mean_k_lattice = np.repeat(mean_k, k).reshape(n, k)  # (n, k)
        all_traj_v = all_traj[:, -1, 2]  # (n, )
        traj_v_lattice = np.repeat(all_traj_v, k).reshape(n, k) * self.v_lattice_span * self.traj_v_scale  # (n, k)
        abs_v_cost = -cost_weights[-3] * np.log(1 + traj_v_lattice) + cost_weights[-2] * (mean_k_lattice - all_traj_min_mean_k) * traj_v_lattice
        self.step_all_cost['velocity_cost'] = -cost_weights[-3] * np.log(1 + traj_v_lattice) # Daewon
        self.step_all_cost['curvature_cost'] = cost_weights[-2] * (mean_k_lattice - all_traj_min_mean_k) * traj_v_lattice # Daewon
        self.step_all_cost['abs_v_cost'] = abs_v_cost

        ## collision cost
        collision_cost = cost_weights[-1] * get_obstacle_collision_with_v(all_traj, all_traj_clothoid, traj_v_lattice,
                                                                          opp_poses, self.prev_opp_pose,
                                                                          self.time_interval)
        self.step_all_cost['collision_cost'] = collision_cost

        cost = np.repeat(cost, k).reshape(n, k)
        cost = cost + abs_v_cost + collision_cost

        return cost

    def select(self, all_costs):
        """
        Select the best trajectory based on the selection function, defaults to argmin if no custom function is defined.
        """
        if self.selection_func is None:
            self.selection_func = np.argmin
        best_idx = self.selection_func(all_costs)
        return best_idx

    def plan(self, pose_x, pose_y, pose_theta, opp_poses, velocity, waypoints=None, debug=False):
        # Daewon
        # Add debug flag to print cost for ego planner

        self.step += 1
        if waypoints is None:
            waypoints = self.waypoints

        # state
        ego_pose = np.array([pose_x, pose_y, pose_theta])
        _, _, t, nearest_i = nearest_point(ego_pose[:2], waypoints[:, 0:2])
        self.state_i = nearest_i
        self.state_t = t

        # sample a grid based on current states
        min_L = self.tracker.get_L(velocity)
        lh_grid = np.linspace(min_L + self.lh_grid_lb, min_L + self.lh_grid_ub, self.lh_grid_rows)
        self.goal_grid = self.sample(pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid)

        # generate clothoids
        all_traj = []
        all_traj_clothoid = []
        for point in self.goal_grid:
            clothoid = Clothoid.G1Hermite(pose_x, pose_y, pose_theta, point[0], point[1], point[2])
            traj = sample_traj(clothoid, self.traj_points, point[3])
            all_traj.append(traj)
            # G1Hermite parameters are [xstart, ystart, thetastart, curvrate, kappastart, arclength]
            all_traj_clothoid.append(np.array(clothoid.Parameters))


        # evaluate all trajectory on all costs
        all_traj = np.array(all_traj)
        all_traj_clothoid = np.array(all_traj_clothoid)
        all_costs = self.eval(all_traj, all_traj_clothoid, opp_poses, ego_pose=ego_pose)
        self.all_costs = all_costs

        best_traj_idx = self.select(all_costs)
        self.best_traj_idx = best_traj_idx
        row_idx, col_idx = divmod(best_traj_idx, self.v_lattice_num)

        # Daewon
        # For Debugging
        if debug:
            for k in self.step_all_cost.keys():
                cost_array = self.step_all_cost[k]
                if k in ["get_map_collision", "abs_v_cost"]:
                    continue
                # 1. 1차원 배열인 경우 (경로에만 의존)
                if cost_array.ndim == 1:
                    print(f"{k} : {cost_array[row_idx]:.4f}", end=" | ", flush=False)
                    
                # 2. 2차원 배열인 경우 (경로와 속도 모두 의존)
                elif cost_array.ndim == 2:
                    print(f"{k} : {cost_array[row_idx, col_idx]:.4f}", end=" | ", flush=False)
            print("", flush=True)

        self.best_traj = all_traj[row_idx]
        self.best_traj_ref_v = self.best_traj[-1, 2]
        self.best_traj[:, 2] *= (self.v_lattice_span[col_idx] * self.traj_v_scale)
        self.prev_traj_local = traj_global2local(ego_pose, self.best_traj[:, :2])
        self.prev_opp_pose = opp_poses[:, :2]
        return self.best_traj

@njit(cache=True)
def sample_lookahead_square(pose_x,
                            pose_y,
                            pose_theta,
                            velocity,
                            waypoints,
                            lookahead_distances=np.array([1.6, 1.8, 2.0, 2.2]),
                            widths=np.linspace(-1.25, 1.25, num=11)):

    # get lookahead points to create grid along waypoints
    position = np.array([pose_x, pose_y])
    nearest_p, nearest_dist, t, nearest_i = nearest_point(position, waypoints[:, 0:2])
    local_span = np.vstack((np.zeros_like(widths), widths))
    xy_grid = np.zeros((2, 1))
    theta_grid = np.zeros((len(lookahead_distances), 1))
    v_grid = np.zeros((len(lookahead_distances), 1))
    for i, d in enumerate(lookahead_distances):
        lh_pt, i2, t2 = intersect_point(np.ascontiguousarray(nearest_p), d, waypoints[:, 0:2], t + nearest_i, wrap=True)
        i2 = int(i2)
        
        lh_pt_theta = waypoints[i2, 3]
        lh_pt_v = waypoints[i2, 2]
        lh_span_points = get_rotation_matrix(lh_pt_theta) @ local_span + lh_pt.reshape(2, -1)
        xy_grid = np.hstack((xy_grid, lh_span_points))
        theta_grid[i] = zero_2_2pi(lh_pt_theta)
        v_grid[i] = lh_pt_v
    xy_grid = xy_grid[:, 1:]
    theta_grid = np.repeat(theta_grid, len(widths)).reshape(1, -1)
    v_grid = np.repeat(v_grid, len(widths)).reshape(1, -1)
    grid = np.vstack((xy_grid, theta_grid, v_grid)).T
    return grid


@njit(cache=True)
def traj_global2local(ego_pose, traj):

    new_traj = np.zeros_like(traj)
    pose_x, pose_y, pose_theta = ego_pose
    c = np.cos(pose_theta)
    s = np.sin(pose_theta)
    new_traj[..., 0] = c * (traj[..., 0] - pose_x) + s * (traj[..., 1] - pose_y)  # (n, m, 1)
    new_traj[..., 1] = -s * (traj[..., 0] - pose_x) + c * (traj[..., 1] - pose_y)  # (n, m, 1)
    return new_traj


@njit(cache=True)
def get_follow_optim_cost(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None,
                          map_metainfo=None):
    n = traj.shape[0]
    center = np.array((5, 16, 27, 38, 49))
    center = np.repeat(center, 11)  # (33, 0)
    traj_idx = np.arange(0, n, 1)
    idx_diff = traj_idx - center
    cost = idx_diff * idx_diff
    return cost

def get_curvature(traj, traj_clothoid):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)  # (n, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)  # (n, 1)
    s = traj_clothoid[:, -1]  # (n, )
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T  # (n, m)
    traj_k = k0 + dk * s_pts  # (n, m)
    traj_k_abs = np.abs(traj_k)
    
    # Calculate steering angles from curvature (steering = arctan(L * kappa))
    wheelbase = 0.33
    traj_steer = np.arctan(wheelbase * traj_k)
    max_steer = np.max(np.abs(traj_steer), axis=1)
    
    # Apply graduated penalty with severe penalty for excessive steering
    mean_k = np.mean(traj_k_abs, axis=1)
    
    # Add steering penalty
    for i in range(len(mean_k)):
        if max_steer[i] > 0.32:  # Excessive steering threshold
            mean_k[i] *= 2.0  # Double penalty for excessive steering
    
    max_k = np.max(traj_k_abs, axis=1)
    return mean_k, max_k

@njit(cache=True)
def get_map_collision(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None, map_metainfo=None,
                      collision_thres=0.35):
    if dt is None:
        raise ValueError('Map Distance Transform dt has to be set when using this cost function.')
    # points: (n, 2)
    all_traj_pts = np.ascontiguousarray(traj).reshape(-1, 5)  # (nxm, 5)
    collisions = map_collision(all_traj_pts[:, 0:2], dt, map_metainfo, eps=collision_thres)  # (nxm)
    collisions = collisions.reshape(len(traj), -1)  # (n, m)
    cost = []
    for traj_collision in collisions:
        if np.any(traj_collision):
            cost.append(3000.0)
        else:
            cost.append(0.)
    return np.array(cost)


@njit(cache=True)
def get_obstacle_collision_with_v(traj, traj_clothoid, v_lattice, opp_poses, prev_oppo_pose, dt=None):
    max_cost = 20.0
    min_cost = 10.0
    width, length = 0.31, 0.58
    safety_width_distance = 0.15
    safety_length_distance = 0.2
    n, m, _ = traj.shape
    k = v_lattice.shape[1]

    traj_xyt = np.empty((traj.shape[0], traj.shape[1], 3), dtype=traj.dtype)
    traj_xyt[:, :, 0] = traj[:, :, 0]
    traj_xyt[:, :, 1] = traj[:, :, 1]
    traj_xyt[:, :, 2] = traj[:, :, 3]

    # Calculate opponent velocity vectors for future position prediction
    has_prev = (np.sum(np.abs(prev_oppo_pose)) > 1e-6 and
                prev_oppo_pose.shape[0] == opp_poses.shape[0])
    opp_vels = np.zeros((opp_poses.shape[0], 2))
    if has_prev and dt > 1e-6:
        for oi in range(opp_poses.shape[0]):
            opp_vels[oi, 0] = (opp_poses[oi, 0] - prev_oppo_pose[oi, 0]) / dt
            opp_vels[oi, 1] = (opp_poses[oi, 1] - prev_oppo_pose[oi, 1]) / dt

    # Calculate arc lengths for each trajectory (for time estimation)
    traj_arc_lengths = np.zeros((n, m))
    for i in range(n):
        arc_len = traj_clothoid[i, 5]  # total arc length of clothoid
        for j in range(m):
            traj_arc_lengths[i, j] = arc_len * j / max(m - 1, 1)

    # Collision cost is now (n, k) because time-to-reach depends on velocity
    cost = np.zeros((n, k))

    col_length = length + safety_length_distance
    col_width = width + safety_width_distance

    for i in range(n):
        tr = traj_xyt[i]  # (m, 3) - x, y, theta for each point

        for ki in range(k):
            ego_v = v_lattice[i, ki]
            if ego_v < 1e-3:
                ego_v = 1e-3  # avoid division by zero

            max_collision_cost = 0.0

            # Check collision at EVERY trajectory point with predicted opponent position
            for j in range(m):
                # Time for ego to reach trajectory point j
                t_reach = traj_arc_lengths[i, j] / ego_v
                if t_reach > 0.5:
                    t_reach = 0.5  # cap to prevent wild extrapolation at low speeds

                ego_point = tr[j]  # (3,) - x, y, theta
                ego_box = get_vertices(ego_point, col_length, col_width)

                for oi in range(opp_poses.shape[0]):
                    # Predict opponent position at time t_reach
                    predicted_opp = np.empty(3)
                    predicted_opp[0] = opp_poses[oi, 0] + opp_vels[oi, 0] * t_reach
                    predicted_opp[1] = opp_poses[oi, 1] + opp_vels[oi, 1] * t_reach
                    predicted_opp[2] = opp_poses[oi, 2]  # assume heading doesn't change much

                    opp_box = get_vertices(predicted_opp, col_length, col_width)

                    if collision(opp_box, ego_box):
                        # Earlier collision (smaller j) is more dangerous
                        point_cost = max_cost - j * (max_cost - min_cost) / m
                        if point_cost > max_collision_cost:
                            max_collision_cost = point_cost

            cost[i, ki] = max_collision_cost

    return cost
