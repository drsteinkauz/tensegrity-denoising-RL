import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os
from scipy.spatial.transform import Rotation
from collections import deque
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class tr_env_gym(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_file=os.path.join(os.getcwd(),"3prism_jonathan_steady_side.xml"),
        use_contact_forces=False,
        use_tendon_length=False,
        use_cap_velocity=True,
        use_stability_detection=False,
        use_obs_noise=False,
        use_intrinsic_params_dr=False,
        terminate_when_unhealthy=True,
        is_test = False,
        desired_action = "straight",
        desired_direction = 1,
        ctrl_cost_weight=0.01,
        contact_cost_weight=5e-4,
        healthy_reward=0.1, 
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.0, # reset noise is handled in the following 4 variables
        min_reset_heading = 0.0,
        max_reset_heading = 2*np.pi,
        use_reward_delay = False,
        reward_delay_seconds = 0.5,
        contact_with_self_penalty = 0.0,
        robot_type = "w",
        tendon_reset_mean_w = 0.05,
        tendon_reset_stdev_w = 0.0,
        tendon_max_length_w = 0.05,
        tendon_min_length_w = -0.05,
        tendon_max_vel_w = 0.05,
        tendon_reset_mean_j = 0.15,
        tendon_reset_stdev_j = 0.2,
        tendon_max_length_j = 0.15,
        tendon_min_length_j = -0.35,
        tendon_max_vel_j = 0.15,
        friction_noise_dist_w = (1.0, 4.0),
        damping_noise_dist_cross_w = (10.0, 4.0),
        stiffness_noise_dist_cross_w = (150.0, 2.0),
        mass_noise_dist_w = (0.1334, 4.0),
        friction_noise_dist_j = (1.0, 4.0),
        damping_noise_dist_cross_j = (100.0, 4.0),
        stiffness_noise_dist_cross_j = (700.0, 2.0),
        mass_noise_dist_j = (4.0, 4.0),
        obs_noise_tendon_stdev_w = 0.02,
        obs_noise_cap_pos_stdev_w = 0.01,
        obs_noise_tendon_stdev_j = 0.02,
        obs_noise_cap_pos_stdev_j = 0.05,
        iniyaw_bias_w = -np.pi/15,
        way_pts_range_w = (1.0, 1.0),
        way_pts_angle_range_w = (0.0, 0.0),
        curvature_w = -0.2,
        iniyaw_bias_j = 0.0, # np.pi/10,
        way_pts_range_j = (3.0, 3.0),
        way_pts_angle_range_j = (-np.pi/6, np.pi/6),
        curvature_j = 0.2,
        ditch_reward_max=150,
        ditch_reward_stdev_w=0.05,
        ditch_reward_stdev_j=0.15,
        waypt_reward_amplitude=50,
        waypt_reward_stdev_w=0.03,
        waypt_reward_stdev_j=0.1,
        threshold_waypt = 0.05,
        forrew_rate_w=3.0,
        forrew_rate_j=1.0,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            use_contact_forces,
            use_tendon_length,
            use_cap_velocity,
            use_stability_detection,
            use_obs_noise,
            use_intrinsic_params_dr,
            terminate_when_unhealthy,
            is_test,
            desired_action,
            desired_direction,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            contact_force_range,
            reset_noise_scale,
            min_reset_heading,
            max_reset_heading,
            use_reward_delay,
            reward_delay_seconds,
            contact_with_self_penalty,
            robot_type,
            tendon_reset_mean_w,
            tendon_reset_stdev_w,
            tendon_max_length_w,
            tendon_min_length_w,
            tendon_max_vel_w,
            tendon_reset_mean_j,
            tendon_reset_stdev_j,
            tendon_max_length_j,
            tendon_min_length_j,
            tendon_max_vel_j,
            friction_noise_dist_w,
            damping_noise_dist_cross_w,
            stiffness_noise_dist_cross_w,
            friction_noise_dist_j,
            damping_noise_dist_cross_j,
            stiffness_noise_dist_cross_j,
            obs_noise_tendon_stdev_w,
            obs_noise_cap_pos_stdev_w,
            obs_noise_tendon_stdev_j,
            obs_noise_cap_pos_stdev_j,
            iniyaw_bias_w,
            way_pts_range_w,
            way_pts_angle_range_w,
            iniyaw_bias_j,
            way_pts_range_j,
            way_pts_angle_range_j,
            ditch_reward_max,
            ditch_reward_stdev_w,
            ditch_reward_stdev_j,
            waypt_reward_amplitude,
            waypt_reward_stdev_w,
            waypt_reward_stdev_j,
            threshold_waypt,
            **kwargs
        )
        self._x_velocity = 1
        self._y_velocity = 1
        self._is_test = is_test
        self._desired_action = desired_action
        self._desired_direction = desired_direction
        self._reset_psi = 0
        self._psi_wrap_around_count = 0
        self._use_tendon_length = use_tendon_length
        self._use_cap_velocity = use_cap_velocity
        self._use_stability_detection = use_stability_detection

        self._use_obs_noise = use_obs_noise
        self._use_intrinsic_params_dr = use_intrinsic_params_dr

        self._min_reset_heading = min_reset_heading
        self._max_reset_heading = max_reset_heading
        self._robot_type = robot_type
        
        self._oripoint = None
        self._threshold_waypt = threshold_waypt
        self._ditch_reward_max = ditch_reward_max
        self._waypt_reward_amplitude = waypt_reward_amplitude
        self._waypt = None

        self._lin_vel_cmd = np.array([0.0, 0.0])
        self._ang_vel_cmd = 0.0


        if self._robot_type == "w":
            self._tendon_reset_mean = tendon_reset_mean_w
            self._tendon_reset_stdev = tendon_reset_stdev_w
            self._tendon_max_length = tendon_max_length_w
            self._tendon_min_length = tendon_min_length_w
            self._tendon_max_vel = tendon_max_vel_w

            self._obs_noise_tendon_stdev = obs_noise_tendon_stdev_w
            self._obs_noise_cap_pos_stdev = obs_noise_cap_pos_stdev_w
            self._friction_noise_dist = friction_noise_dist_w
            self._damping_noise_dist_cross = damping_noise_dist_cross_w
            self._stiffness_noise_dist_cross = stiffness_noise_dist_cross_w
            self._mass_noise_dist = mass_noise_dist_w
            self._inertia_mean = np.array([1.9170979e-03, 1.9170979e-03, 1.3436629e-05])

            self._iniyaw_bias = iniyaw_bias_w

            self._waypt_range = way_pts_range_w
            self._waypt_angle_range = way_pts_angle_range_w
            self._ditch_reward_stdev = ditch_reward_stdev_w
            self._waypt_reward_stdev = waypt_reward_stdev_w

            self._curvature = curvature_w

            self._forrew_rate = forrew_rate_w

        elif self._robot_type == "j":
            self._tendon_reset_mean = tendon_reset_mean_j
            self._tendon_reset_stdev = tendon_reset_stdev_j
            self._tendon_max_length = tendon_max_length_j
            self._tendon_min_length = tendon_min_length_j
            self._tendon_max_vel = tendon_max_vel_j

            self._obs_noise_tendon_stdev = obs_noise_tendon_stdev_j
            self._obs_noise_cap_pos_stdev = obs_noise_cap_pos_stdev_j
            self._friction_noise_dist = friction_noise_dist_j
            self._damping_noise_dist_cross = damping_noise_dist_cross_j
            self._stiffness_noise_dist_cross = stiffness_noise_dist_cross_j
            self._mass_noise_dist = mass_noise_dist_j
            self._inertia_mean = np.array([1.09641124, 1.09641124, 0.00377331])

            self._iniyaw_bias = iniyaw_bias_j

            self._waypt_range = way_pts_range_j
            self._waypt_angle_range = way_pts_angle_range_j
            self._ditch_reward_stdev = ditch_reward_stdev_j
            self._waypt_reward_stdev = waypt_reward_stdev_j

            self._curvature = curvature_j

            self._forrew_rate = forrew_rate_j

        else:
            raise ValueError("robot_type should be either w or j")

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_force_range = contact_force_range
        if self._desired_action == "turn":
            self._contact_force_range = (-1000.0, 1000.0)
        self._reset_noise_scale = reset_noise_scale
        self._use_contact_forces = use_contact_forces

        self._contact_with_self_penalty = contact_with_self_penalty

        obs_shape = 18
        if use_tendon_length:
            obs_shape += 9
        if use_cap_velocity:
            obs_shape += 18
        if use_stability_detection:
            obs_shape += 1
        if use_contact_forces:
            obs_shape += 84
        if desired_action == "vel_track":
            obs_shape += 3 # cmd lin_vel * 2 + ang_vel * 1
        elif desired_action == "tracking":
            obs_shape += 2 # tracking vector x, y
        
        self.state_shape = obs_shape

        self.intriparam_shape = 10 # 3 for friction coefficient, damping of cross, stiffness of cross
        self.intriparam_std = np.array([[self._friction_noise_dist[1]],
                                         [self._damping_noise_dist_cross[1]],
                                         [self._damping_noise_dist_cross[1]],
                                         [self._damping_noise_dist_cross[1]],
                                         [self._stiffness_noise_dist_cross[1]],
                                         [self._stiffness_noise_dist_cross[1]],
                                         [self._stiffness_noise_dist_cross[1]],
                                         [self._mass_noise_dist[1]],
                                         [self._mass_noise_dist[1]],
                                         [self._mass_noise_dist[1]]])

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
        frame_skip = 20
        MujocoEnv.__init__(
            self, xml_file, frame_skip, observation_space=observation_space, **kwargs
        )
        self._use_reward_delay = use_reward_delay
        self._reward_delay_steps = int(reward_delay_seconds/self.dt)
        self._heading_buffer = deque()

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action, tendon_length_6):
        if self._robot_type == "j":
            control_cost = self._ctrl_cost_weight * np.sum(np.square(action + 0.5 - tendon_length_6)) # 0.5 is the initial spring length for 6 tendons
        elif self._robot_type == "w":
            control_cost = self._ctrl_cost_weight * np.sum(np.square(action + 0.15 - tendon_length_6)) # 0.15 is the initial spring length for 6 tendons
        # control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        if self._desired_action == "turn":
            bar_speeds = np.abs(state[21:])
            min_velocity = 0.1
            is_healthy = np.isfinite(state).all() and (np.any(bar_speeds > min_velocity) )    

        else: #self._desired_action == "straight" or self._desired_action == "arc" or self._desired_action == "tracking" or self._desired_action == "vel_track":
            min_velocity = 0.0001
            is_healthy = np.isfinite(state).all() and ((self._x_velocity > min_velocity or self._x_velocity < -min_velocity) \
                                                        or (self._y_velocity > min_velocity or self._y_velocity < -min_velocity) )
            
        
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action_scaled):
        # action: [-1, 1] -> [tendon_min_length, tendon_max_length]
        action = action_scaled * (self._tendon_max_length - self._tendon_min_length) / 2 + (self._tendon_max_length + self._tendon_min_length) / 2
        
        xy_position_before = (self.get_body_com("r01_body")[:2].copy() + \
                            self.get_body_com("r23_body")[:2].copy() + \
                            self.get_body_com("r45_body")[:2].copy())/3
        
        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_before = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_before = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3

        orientation_vector_before = left_COM_before - right_COM_before
        psi_before = np.arctan2(-orientation_vector_before[0], orientation_vector_before[1])

        filtered_action = self._action_filter(action, self.data.ctrl[:].copy())
        # self.do_simulation(filtered_action, self.frame_skip)
        if self._robot_type == "w":
            self.do_simulation(action, self.frame_skip)
        elif self._robot_type == "j":
            self.data.ctrl[:] = action.copy()
            for _ in range(self.frame_skip):
                # crt_min_force = np.minimum(267 * (-self.data.actuator_velocity / 0.17 - 1), -4 * np.ones(6))
                # crt_min_force = np.maximum(crt_min_force, -267* np.ones(6))
                # self.model.actuator_forcerange[:, 0] = crt_min_force
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_rnePostConstraint(self.model, self.data)
        
        xy_position_after = (self.get_body_com("r01_body")[:2].copy() + \
                            self.get_body_com("r23_body")[:2].copy() + \
                            self.get_body_com("r45_body")[:2].copy())/3

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        self._x_velocity, self._y_velocity = xy_velocity

        x_position_before, y_position_before = xy_position_before
        x_position_after, y_position_after = xy_position_after

        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_after = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_after = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3

        orientation_vector_after = left_COM_after - right_COM_after
        psi_after = np.arctan2(-orientation_vector_after[0], orientation_vector_after[1])

        tendon_length = np.array(self.data.ten_length)
        tendon_length_6 = tendon_length[:6]

        state, observation, gt_log_intriparam = self._get_obs()


        if self._desired_action == "turn":
            if self._use_reward_delay:
                self._heading_buffer.append(psi_after)
                if len(self._heading_buffer) > self._reward_delay_steps:
                    old_psi = self._heading_buffer.popleft()

                    # unless the tensegrity is rotating faster than pi /(self.dt*self._reward_delay_steps) rad/s
                    # then this situation means that the tensegrity rolled from pi to -pi, and the delta should be positive
                    if psi_after < -np.pi/2 and old_psi > np.pi/2: 
                        psi_after = 2*np.pi + psi_after
                    # unless the tensegrity is rotating faster than pi /(self.dt*self._reward_delay_steps) rad/s
                    # then this situation means that the tensegrity rolled from -pi to pi, and the delta should be negative
                    elif psi_after > np.pi/2 and old_psi < -np.pi/2:
                        psi_after = -2*np.pi + psi_after
                    delta_psi = (psi_after - old_psi) / (self.dt*self._reward_delay_steps)
                    forward_reward = delta_psi * self._desired_direction 
                    costs = ctrl_cost = self.control_cost(action, tendon_length_6)

                    cone_potential_before = self._cone_potential_norm(xy_position_before)
                    cone_potential_after = self._cone_potential_norm(xy_position_after)
                    costs -= cone_potential_after - cone_potential_before

                else:
                    forward_reward = 0
                    costs = ctrl_cost =  0
                    delta_psi = 0
            
            else:
                if psi_after < -np.pi/2 and psi_before > np.pi/2: 
                    psi_after = 2*np.pi + psi_after
                elif psi_after > np.pi/2 and psi_before < -np.pi/2:
                    psi_after = -2*np.pi + psi_after
                delta_psi = (psi_after - psi_before) / self.dt
                forward_reward = delta_psi * self._desired_direction 
                costs = ctrl_cost = self.control_cost(action, tendon_length_6)

                cone_potential_before = self._cone_potential_norm(xy_position_before)
                cone_potential_after = self._cone_potential_norm(xy_position_after)
                costs -= cone_potential_after - cone_potential_before
            
            if self._terminate_when_unhealthy:
                healthy_reward = self.healthy_reward
            else:
                healthy_reward = 0
            
            terminated = self.terminated  


        elif self._desired_action == "straight":
            
            # psi_movement = np.arctan2(y_position_after-y_position_before, x_position_after-x_position_before)

            # psi_diff = np.abs(psi_movement-self._reset_psi)

            # forward_reward = self._desired_direction*\
            #                     (np.sqrt((x_position_after-x_position_before)**2 + \
            #                             (y_position_after - y_position_before)**2) *\
            #                     np.cos(psi_diff)/ self.dt) * self._forrew_rate

            before_potential = self._straight_potential(xy_position_before)
            after_potential = self._straight_potential(xy_position_after)
            forward_reward = after_potential - before_potential

            costs = ctrl_cost = self.control_cost(action, tendon_length_6)

            if self._terminate_when_unhealthy:
                # healthy_reward = self.healthy_reward
                healthy_reward = 0
            else:
                healthy_reward = 0
            
            terminated = self.terminated

        elif self._desired_action == "arc":
            forward_reward = self._arc_reward(xy_position_before, xy_position_after, self._curvature)

            costs = ctrl_cost = self.control_cost(action, tendon_length_6)
            healthy_reward = 0
            terminated = self.terminated
        
        elif self._desired_action == "tracking":
            # ditch tracking reward
            ditch_rew_after = self._tracking_potential(xy_position_after)
            ditch_rew_before = self._tracking_potential(xy_position_before)
            forward_reward = ditch_rew_after - ditch_rew_before

            costs = ctrl_cost = self.control_cost(action, tendon_length_6)

            healthy_reward = 0

            terminated = self.terminated  
            if self._step_num > 1000:
                terminated = True
            if np.linalg.norm(xy_position_after - self._waypt) < self._threshold_waypt:
                terminated = True
        
        elif self._desired_action == "vel_tracking":
            ang_vel_bwd = self._angle_normalize(psi_after - psi_before)/self.dt
            vel_bwd = np.array([self._x_velocity, self._y_velocity, ang_vel_bwd])
            vel_cmd = state[-3:]
            forward_reward = self._vel_track_rew(vel_cmd=vel_cmd, vel_bwd=vel_bwd)

            costs = ctrl_cost = self.control_cost(action, tendon_length_6)

            if self._terminate_when_unhealthy:
                healthy_reward = self.healthy_reward
            else:
                healthy_reward = 0
            
            terminated = self.terminated
        

        rewards = forward_reward + healthy_reward

        # if the contact between bars is too high, terminate the training run
        if np.any(self.data.cfrc_ext > 1500) or np.any(self.data.cfrc_ext < -1500):
            terminated = True


        
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            #"reward_contact_with_self": -contact_with_self_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "psi": psi_after,
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": self._x_velocity,
            "y_velocity": self._y_velocity,
            "tendon_length": tendon_length,
            "real_observation": observation,
            "forward_reward": forward_reward,
            "waypt": self._waypt,
            "oripoint": self._oripoint,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs #- contact_with_self_cost

        self._step_num += 1

        if self.render_mode == "human":
            self.render()
        
        return state, observation, gt_log_intriparam, reward, terminated, False, info

    def _get_obs(self):
        
        
        """ rotation_r01 = Rotation.from_matrix(self.data.geom("r01").xmat.reshape(3,3)).as_quat() # 4
        rotation_r23 = Rotation.from_matrix(self.data.geom("r23").xmat.reshape(3,3)).as_quat() # 4
        rotation_r45 = Rotation.from_matrix(self.data.geom("r45").xmat.reshape(3,3)).as_quat() # 4 """

        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()

        pos_center = (pos_r01_left_end + pos_r01_right_end + pos_r23_left_end + pos_r23_right_end + pos_r45_left_end + pos_r45_right_end) / 6

        pos_rel_s0 = pos_r01_left_end - pos_center # 3
        pos_rel_s1 = pos_r01_right_end - pos_center # 3
        pos_rel_s2 = pos_r23_left_end - pos_center # 3
        pos_rel_s3 = pos_r23_right_end - pos_center # 3
        pos_rel_s4 = pos_r45_left_end - pos_center # 3
        pos_rel_s5 = pos_r45_right_end - pos_center # 3

        rng = np.random.default_rng()
        random = rng.standard_normal(size=3)
        pos_rel_s0_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s0 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s1_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s1 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s2_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s2 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s3_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s3 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s4_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s4 # 3
        random = rng.standard_normal(size=3)
        pos_rel_s5_with_noise = random * self._obs_noise_cap_pos_stdev + pos_rel_s5 # 3

        # do not include positional data in the observation
        # position_r01 = self.data.geom("r01").xvelp
        # position_r23 = self.data.geom("r23").xvelp
        # position_r45 = self.data.geom("r45").xvelp


        tendon_lengths = self.data.ten_length[-9:] # 9
        
        random = rng.standard_normal(size=9)
        tendon_lengths_with_noise = random * self._obs_noise_tendon_stdev + tendon_lengths # 9

        if self._use_stability_detection:
            z_cap = [pos_r01_left_end[-1],pos_r01_right_end[-1],pos_r23_left_end[-1],pos_r23_right_end[-1],pos_r45_left_end[-1],pos_r45_right_end[-1]]
            stability = self._is_stable(z_cap,threshold=1e-3)
            state = np.concatenate(([stability],pos_rel_s0,pos_rel_s1,pos_rel_s2, pos_rel_s3, pos_rel_s4, pos_rel_s5))
            state_with_noise = np.concatenate(([stability],pos_rel_s0_with_noise, pos_rel_s1_with_noise, pos_rel_s2_with_noise, pos_rel_s3_with_noise, pos_rel_s4_with_noise, pos_rel_s5_with_noise))
        else:
            state = np.concatenate((pos_rel_s0, pos_rel_s1, pos_rel_s2, pos_rel_s3, pos_rel_s4, pos_rel_s5))
            state_with_noise = np.concatenate((pos_rel_s0_with_noise, pos_rel_s1_with_noise, pos_rel_s2_with_noise, pos_rel_s3_with_noise, pos_rel_s4_with_noise, pos_rel_s5_with_noise))
        
        if self._use_cap_velocity:
            velocity = self.data.qvel # 18

            vel_lin_r01 = np.array([velocity[0], velocity[1], velocity[2]])
            vel_ang_r01 = np.array([velocity[3], velocity[4], velocity[5]])
            vel_lin_r23 = np.array([velocity[6], velocity[7], velocity[8]])
            vel_ang_r23 = np.array([velocity[9], velocity[10], velocity[11]])
            vel_lin_r45 = np.array([velocity[12], velocity[13], velocity[14]])
            vel_ang_r45 = np.array([velocity[15], velocity[16], velocity[17]])

            s0_r01_pos = pos_r01_left_end - self.data.body("r01_body").xpos.copy()
            s1_r01_pos = pos_r01_right_end - self.data.body("r01_body").xpos.copy()
            s2_r23_pos = pos_r23_left_end - self.data.body("r23_body").xpos.copy()
            s3_r23_pos = pos_r23_right_end - self.data.body("r23_body").xpos.copy()
            s4_r45_pos = pos_r45_left_end - self.data.body("r45_body").xpos.copy()
            s5_r45_pos = pos_r45_right_end - self.data.body("r45_body").xpos.copy()

            vel_s0 = vel_lin_r01 + np.cross(vel_ang_r01, s0_r01_pos) # 3
            vel_s1 = vel_lin_r01 + np.cross(vel_ang_r01, s1_r01_pos) # 3
            vel_s2 = vel_lin_r23 + np.cross(vel_ang_r23, s2_r23_pos) # 3
            vel_s3 = vel_lin_r23 + np.cross(vel_ang_r23, s3_r23_pos) # 3
            vel_s4 = vel_lin_r45 + np.cross(vel_ang_r45, s4_r45_pos) # 3
            vel_s5 = vel_lin_r45 + np.cross(vel_ang_r45, s5_r45_pos) # 3

            random = rng.standard_normal(size=3)
            vel_s0_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s0 # 3
            random = rng.standard_normal(size=3)
            vel_s1_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s1 # 3
            random = rng.standard_normal(size=3)
            vel_s2_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s2 # 3
            random = rng.standard_normal(size=3)
            vel_s3_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s3 # 3
            random = rng.standard_normal(size=3)
            vel_s4_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s4 # 3
            random = rng.standard_normal(size=3)
            vel_s5_with_noise = random * self._obs_noise_cap_pos_stdev + vel_s5 # 3

            state = np.concatenate((state,\
                                        vel_s0, vel_s1, vel_s2, vel_s3, vel_s4, vel_s5))
            state_with_noise = np.concatenate((state,\
                                        vel_s0_with_noise, vel_s1_with_noise, vel_s2_with_noise, vel_s3_with_noise, vel_s4_with_noise, vel_s5_with_noise))
            
        if self._use_tendon_length:
            state = np.concatenate((state, tendon_lengths))
            state_with_noise = np.concatenate((state_with_noise, tendon_lengths_with_noise))

        if self._desired_action == "tracking":
            tracking_vec = self._waypt - pos_center[:2]
            tgt_drct = tracking_vec / np.linalg.norm(tracking_vec)
            pos_center_noise_del = (pos_rel_s0_with_noise + pos_rel_s1_with_noise + pos_rel_s2_with_noise + pos_rel_s3_with_noise + pos_rel_s4_with_noise + pos_rel_s5_with_noise)/6
            tracking_vec_with_noise = tracking_vec - pos_center_noise_del[:2]
            tgt_drct_with_noise = tracking_vec_with_noise / np.linalg.norm(tracking_vec_with_noise)

            tgt_yaw = np.array([np.arctan2(tgt_drct[1], tgt_drct[0])])
            tgt_yaw_with_noise = np.array([np.arctan2(tgt_drct_with_noise[1], tgt_drct_with_noise[0])])

            if np.linalg.norm(tracking_vec) < 1.0:
                state = np.concatenate((state,\
                                            tracking_vec))
                state_with_noise = np.concatenate((state_with_noise,\
                                                        tracking_vec_with_noise))
            else:
                state = np.concatenate((state,\
                                            tgt_drct))
                state_with_noise = np.concatenate((state_with_noise,\
                                                        tgt_drct_with_noise))
        
        elif self._desired_action == "vel_track":
            vel_cmd = np.array([self._lin_vel_cmd[0], self._lin_vel_cmd[1], self._ang_vel_cmd])
            state = np.concatenate((state, vel_cmd))
            state_with_noise = np.concatenate((state_with_noise, vel_cmd))
        
        if self._use_obs_noise == True:
            observation = state_with_noise
        else:
            observation = state
        
        if self._robot_type == "w":
            log_friction = np.log(self.model.geom_friction[0, 0] / self._friction_noise_dist[0])
            log_damping = np.array([np.log(self.model.tendon_damping[12] / self._damping_noise_dist_cross[0]),
                                    np.log(self.model.tendon_damping[13] / self._damping_noise_dist_cross[0]),
                                    np.log(self.model.tendon_damping[14] / self._damping_noise_dist_cross[0])])
            log_stiffness = np.array([np.log(self.model.tendon_stiffness[12] / self._stiffness_noise_dist_cross[0]),
                                      np.log(self.model.tendon_stiffness[13] / self._stiffness_noise_dist_cross[0]),
                                      np.log(self.model.tendon_stiffness[14] / self._stiffness_noise_dist_cross[0])])
            log_mass = np.array([np.log(self.model.body_mass[1] / self._mass_noise_dist[0]),
                                 np.log(self.model.body_mass[2] / self._mass_noise_dist[0]),
                                 np.log(self.model.body_mass[3] / self._mass_noise_dist[0])])
        elif self._robot_type == "j":
            log_friction = np.log(self.model.geom_friction[0, 0] / self._friction_noise_dist[0])
            log_damping = np.array([np.log(self.model.tendon_damping[6] / self._damping_noise_dist_cross[0]),
                                    np.log(self.model.tendon_damping[7] / self._damping_noise_dist_cross[0]),
                                    np.log(self.model.tendon_damping[8] / self._damping_noise_dist_cross[0])])
            log_stiffness = np.array([np.log(self.model.tendon_stiffness[6] / self._stiffness_noise_dist_cross[0]),
                                      np.log(self.model.tendon_stiffness[7] / self._stiffness_noise_dist_cross[0]),
                                      np.log(self.model.tendon_stiffness[8] / self._stiffness_noise_dist_cross[0])])
            log_mass = np.array([np.log(self.model.body_mass[1] / self._mass_noise_dist[0]),
                                 np.log(self.model.body_mass[2] / self._mass_noise_dist[0]),
                                 np.log(self.model.body_mass[3] / self._mass_noise_dist[0])])

        gt_log_intriparam = np.concatenate([np.array([log_friction]),  # friction coefficient
                                            log_damping,  # damping of cross tendon
                                            log_stiffness, # stiffness of cross tendon
                                            log_mass]) # mass of body 1, 2, 3

        return state, observation, gt_log_intriparam

    def _is_stable(self,arr, threshold=1e-3):
        if len(arr) < 3:
            return False  

        sorted_arr = sorted(arr)
        min1, min2, min3 = sorted_arr[0], sorted_arr[1], sorted_arr[2]

        if abs(min1 - min3) < threshold:
            return 1
        else:
            return 0

    def _angle_normalize(self, theta):
        if theta > np.pi:
            return self._angle_normalize(theta - 2 * np.pi)
        elif theta <= -np.pi:
            return self._angle_normalize(theta + 2 * np.pi)
        else:
            return theta
    
    def _cone_potential_norm(self, xy_position):
        odom_vec = xy_position - self._oripoint
        odom_dist = np.linalg.norm(odom_vec)
        threshold = 0.5
        affected_dist = np.maximum(odom_dist-threshold, 0)
        cone_potential = -self._forrew_rate * affected_dist**2 / self.dt
        return cone_potential
    
    def _tracking_potential(self, xy_position):
        pointing_vec = self._waypt - self._oripoint
        dist_pointing = np.linalg.norm(pointing_vec)
        pointing_vec_norm = pointing_vec / dist_pointing

        tracking_vec = self._waypt - xy_position
        dist_along = np.dot(tracking_vec, pointing_vec_norm)
        dist_bias = np.linalg.norm(tracking_vec - dist_along*pointing_vec_norm) * np.sign(np.linalg.det(np.array([tracking_vec, pointing_vec_norm])))

        ditch_potential = self._ditch_reward_max * (1.0 - np.abs(dist_along)/dist_pointing) * np.exp(-dist_bias**2 / (2*self._ditch_reward_stdev**2))
        waypt_potential = self._waypt_reward_amplitude * np.exp(-np.linalg.norm(xy_position - self._waypt)**2 / (2*self._waypt_reward_stdev**2))
        cone_potential = -self._forrew_rate * np.linalg.norm(tracking_vec) / self.dt
        ratio = 4.0
        ellipse_potential = -self._forrew_rate * (dist_along**2 + (ratio*dist_bias)**2)**0.5 / self.dt
        return (ditch_potential + cone_potential) / 2.0 * (200.0 / 150.0)
    
    def _straight_potential(self, xy_position):
        k_ALONG = self._forrew_rate / self.dt
        stdev_BIAS = self._ditch_reward_stdev

        odom_position = xy_position - self._oripoint
        distance = np.linalg.norm(odom_position)
        yaw_diff = np.arctan2(odom_position[1], odom_position[0]) - self._reset_psi
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))

        distance_along = distance * np.cos(yaw_diff)
        distance_bias = np.abs(distance * np.sin(yaw_diff))

        # potential = self._desired_direction * k_ALONG * distance_along * (1.0 + np.exp(-distance_bias**2 / (2*stdev_BIAS**2)))/2.0
        # potential = self._desired_direction * k_ALONG * distance_along * np.exp(-distance_bias**2 / (2*stdev_BIAS**2))
        ditch_potential = k_ALONG * np.exp(-distance_bias**2 / (2*self._ditch_reward_stdev**2))
        straight_potential = self._desired_direction * k_ALONG * distance_along
        return ditch_potential + straight_potential

    def _arc_reward(self, xy_position_before, xy_position_after, curvature):
        k_ALONG = self._forrew_rate / self.dt
        stdev_BIAS = self._ditch_reward_stdev

        radius = 1.0 / curvature if curvature != 0 else 1e6  # avoid division by zero
        rotation_center = self._oripoint + np.array([-radius * np.sin(self._reset_psi), radius * np.cos(self._reset_psi)])

        vec_center2before = xy_position_before - rotation_center
        vec_center2after = xy_position_after - rotation_center
        vec_movement = vec_center2after - vec_center2before

        vec_norm = (vec_center2after + vec_center2before) / 2.0
        vec_tangent = np.array([-vec_norm[1], vec_norm[0]])
        vec_tangent /= np.linalg.norm(vec_tangent)
        
        distance_along = np.dot(vec_movement, vec_tangent)
        along_reward = self._desired_direction * np.sign(curvature) * k_ALONG * distance_along

        distance_bias_before = np.linalg.norm(vec_center2before) - np.abs(radius)
        bias_potential_before = k_ALONG * np.exp(-distance_bias_before**2 / (2*stdev_BIAS**2))
        distance_bias_after = np.linalg.norm(vec_center2after) - np.abs(radius)
        bias_potential_after = k_ALONG * np.exp(-distance_bias_after**2 / (2*stdev_BIAS**2))
        bias_reward = bias_potential_after - bias_potential_before

        return along_reward + bias_reward
    
    def _vel_track_rew(self, vel_cmd, vel_bwd):
        track_stdev = np.array([5.0, 7.0])
        track_amplitude = np.array([1.0, 0.5])
        lin_vel_err = np.linalg.norm(vel_bwd[0:2] - vel_cmd[0:2])
        ang_vel_err = vel_bwd[2] - vel_cmd[2]

        lin_track_rew = track_amplitude[0] * np.exp(-track_stdev[0] * lin_vel_err**2)
        ang_track_rew = track_amplitude[1] * np.exp(-track_stdev[1] * ang_vel_err**2)

        return lin_track_rew + ang_track_rew
    
    def _action_filter(self, action, last_action):
        k_FILTER = 1.0
        vel_constraint = self._tendon_max_vel

        del_action = np.clip(action - last_action, -vel_constraint*self.dt, vel_constraint*self.dt)
        # del_action = np.clip(k_FILTER*(action - last_action)*self.dt, -vel_constraint*self.dt, vel_constraint*self.dt)
        # del_action = k_FILTER*(action - last_action)*self.dt
        # del_action = action / 0.05 * vel_constraint*self.dt

        filtered_action = last_action + del_action
        return filtered_action

    def _reset_intrinsic_params(self):
        rand_itpr = np.random.uniform(-1, 1, size=(self.intriparam_shape,))

        friction_coeff = np.exp(rand_itpr[0] * np.log(self._friction_noise_dist[1])) * self._friction_noise_dist[0]
        damping_coeff = np.exp(rand_itpr[1:4] * np.log(self._damping_noise_dist_cross[1])) * self._damping_noise_dist_cross[0]
        stiffness_coeff = np.exp(rand_itpr[4:7] * np.log(self._stiffness_noise_dist_cross[1])) * self._stiffness_noise_dist_cross[0]
        mass_coeff = np.exp(rand_itpr[7:10] * np.log(self._mass_noise_dist[1])) * self._mass_noise_dist[0]
        inertia_coeff = np.array([np.exp(rand_itpr[7] * np.log(self._mass_noise_dist[1])) * self._inertia_mean,
                                  np.exp(rand_itpr[8] * np.log(self._mass_noise_dist[1])) * self._inertia_mean,
                                  np.exp(rand_itpr[9] * np.log(self._mass_noise_dist[1])) * self._inertia_mean])

        if self._robot_type == "w":
            self.model.geom_friction[:, 0] = friction_coeff
            # self.model.tendon_damping[12:15] = damping_coeff
            self.model.tendon_damping[12] = damping_coeff[0]
            self.model.tendon_damping[13] = damping_coeff[1]
            self.model.tendon_damping[14] = damping_coeff[2]
            # self.model.tendon_stiffness[12:15] = stiffness_coeff
            self.model.tendon_stiffness[12] = stiffness_coeff[0]
            self.model.tendon_stiffness[13] = stiffness_coeff[1]
            self.model.tendon_stiffness[14] = stiffness_coeff[2]
            # self.model.body_mass[0:3] = mass_coeff
            self.model.body_mass[1] = mass_coeff[0]
            self.model.body_mass[2] = mass_coeff[1]
            self.model.body_mass[3] = mass_coeff[2]
            self.model.body_inertia[1] = inertia_coeff[0]
            self.model.body_inertia[2] = inertia_coeff[1]
            self.model.body_inertia[3] = inertia_coeff[2]
        elif self._robot_type == "j":
            self.model.geom_friction[:, 0] = friction_coeff
            # self.model.tendon_damping[6:9] = damping_coeff
            self.model.tendon_damping[6] = damping_coeff[0]
            self.model.tendon_damping[7] = damping_coeff[1]
            self.model.tendon_damping[8] = damping_coeff[2]
            # self.model.tendon_stiffness[6:9] = stiffness_coeff
            self.model.tendon_stiffness[6] = stiffness_coeff[0]
            self.model.tendon_stiffness[7] = stiffness_coeff[1]
            self.model.tendon_stiffness[8] = stiffness_coeff[2]
            # self.model.body_mass[0:3] = mass_coeff
            self.model.body_mass[1] = mass_coeff[0]
            self.model.body_mass[2] = mass_coeff[1]
            self.model.body_mass[3] = mass_coeff[2]
            self.model.body_inertia[1] = inertia_coeff[0]
            self.model.body_inertia[2] = inertia_coeff[1]
            self.model.body_inertia[3] = inertia_coeff[2]
        return


    def reset_model(self):
        self._psi_wrap_around_count = 0

        if self._use_intrinsic_params_dr:
            self._reset_intrinsic_params()

        # '''
        # with rolling noise start
        # rolling_qpos = [[0.25551711, -0.00069342, 0.22404039, -0.49720971, 0.24315431, 0.75327284, -0.35530059, 0.14409445, 0.0654207, 0.33662589, 0.42572066, 0.01379464, -0.53972521, 0.72613244, 0.28544944, -0.04883333, 0.38591159, 0.137357, 0.06898275, -0.85996553, 0.48665565],
        #                 [0.17072155, 0.12309229, 0.34540078, 0.84521031, 0.46789545, -0.25727243, -0.02245608, 0.28958816, 0.01081555, 0.39491017, 0.48941231, 0.78206488, -0.37311614, 0.09815532, 0.26757914, 0.06595669, 0.21556319, 0.55749435, 0.52139978, -0.59035693, -0.2623376],
        #                 [0.25175364, -0.07481714, 0.38328213, -0.34018568, -0.7272216, -0.46209704, 0.37668125, 0.24052312, -0.03980219, 0.21579878, -0.04044885, -0.77605179, -0.13528399, 0.61465906, 0.13432437, 0.04431492, 0.3472605, -0.39430158, -0.47235401, -0.24626466, 0.74884021]]
        # rolling_qpos = [[0.08369179, -0.28792231, 0.24830847, -0.49145555, 0.7539914, -0.27511722, -0.33805166, 0.14497616, -0.19291743, 0.35052097, -0.84766041, 0.27950622, 0.45085889, 0.00862359, 0.04557825, -0.29876206, 0.39531985, -0.35798606, -0.47531391, 0.72471075, 0.34744352],
        #                 [0.14497616, -0.19291743, 0.35052097, -0.84766041, 0.27950622, 0.45085889, 0.00862359, 0.04557825, -0.29876206, 0.39531985, -0.35798606, -0.47531391, 0.72471075, 0.34744352, 0.08369179, -0.28792231, 0.24830847, -0.49145555, 0.7539914, -0.27511722, -0.33805166],
        #                 [0.04557825, -0.29876206, 0.39531985, -0.35798606, -0.47531391, 0.72471075, 0.34744352, 0.08369179, -0.28792231, 0.24830847, -0.49145555, 0.7539914, -0.27511722, -0.33805166, 0.14497616, -0.19291743, 0.35052097, -0.84766041, 0.27950622, 0.45085889, 0.00862359]]
        if self._robot_type == "j":
            rolling_qpos = [[0.07900689, -0.32670045,  0.23079722,  0.49365198, -0.74001353,  0.26668361,  0.37090101,  0.13713385, -0.24342633,  0.32722167,  0.82936968, -0.31256817, -0.46189217, -0.03320677,  0.04903377, -0.3421725,   0.36675097,  0.33407281,  0.43794432, -0.72515863, -0.41321313],
                            [0.15521685, -0.20651043,  0.38922255,  0.85639289, -0.26723449, -0.44110818, -0.02450564,  0.02999107, -0.33576412,  0.43868814,  0.33839518,  0.48544838, -0.73094128, -0.33993149,  0.08083394, -0.31942006,  0.25783949,  0.51726058, -0.74281033,  0.29432583,  0.30667022],
                            [0.02985312, -0.33588999,  0.43866597,  0.33840617,  0.48522953, -0.73107566, -0.33994403,  0.08072907, -0.31942136,  0.25766037,  0.51740763, -0.74276722,  0.29421311,  0.30663471,  0.15537661, -0.20664637,  0.38923648,  0.85640002, -0.26722239, -0.44110397, -0.02446392],
                            [0.24191878,  0.30939576,  0.25838614,  0.04211683, -0.66689235, -0.44050762,  0.59952798,  0.1105878,   0.33967509,  0.38925944,  0.50825334,  0.20884794, -0.4715363,   0.68972067,  0.27475478,  0.2682452,   0.4387596,   0.47235593,  0.87732918, -0.01675131,  0.08302277],
                            [0.1105878,   0.33967509,  0.38925944,  0.50825334,  0.20884794, -0.4715363,   0.68972067,  0.27475478,  0.2682452,   0.4387596,   0.47235593,  0.87732918, -0.01675131,  0.08302277,  0.24191878,  0.30939576,  0.25838614,  0.04211683, -0.66689235, -0.44050762,  0.59952798],
                            [0.27475478,  0.2682452,   0.4387596,   0.47235593,  0.87732918, -0.01675131,  0.08302277,  0.24191878,  0.30939576,  0.25838614,  0.04211683, -0.66689235, -0.44050762,  0.59952798,  0.1105878,   0.33967509,  0.38925944,  0.50825334,  0.20884794, -0.4715363,   0.68972067]]
        elif self._robot_type == "w":
            rolling_qpos = [[0.2438013,  -0.23055046,  0.10995744,  0.46165276, -0.61078778, -0.64202933, -0.04016669,  0.23304155, -0.2781429,   0.0948906,   0.57252615,  0.17486495, -0.48006247, -0.64123013,  0.24824598, -0.2435365,   0.06010128,  0.12428316,  0.77737256,  0.16439319, -0.59432355],
                            [0.23304155, -0.2781429,   0.0948906,   0.57252615,  0.17486495, -0.48006247, -0.64123013,  0.24824598, -0.2435365,   0.06010128,  0.12428316,  0.77737256,  0.16439319, -0.59432355,  0.2438013,  -0.23055046,  0.10995744,  0.46165276, -0.61078778, -0.64202933, -0.04016669],
                            [0.24824598, -0.2435365,   0.06010128,  0.12428316,  0.77737256,  0.16439319, -0.59432355,  0.2438013,  -0.23055046,  0.10995744,  0.46165276, -0.61078778, -0.64202933, -0.04016669,  0.23304155, -0.2781429,   0.0948906,   0.57252615,  0.17486495, -0.48006247, -0.64123013],
                            [0.28037913, -0.18814138,  0.09807932,  0.45868198, -0.67775281, -0.53799085,  0.20205894,  0.26823767, -0.23681522,  0.1067152,   0.6793826,  -0.07348068, -0.46833013, -0.56009532,  0.26630231, -0.21850045,  0.0587917,   0.23642237,  0.60691942,  0.06582413, -0.75592358],
                            [0.26823767, -0.23681522,  0.1067152,   0.6793826,  -0.07348068, -0.46833013, -0.56009532,  0.26630231, -0.21850045,  0.0587917,   0.23642237,  0.60691942,  0.06582413, -0.75592358,  0.28037913, -0.18814138,  0.09807932,  0.45868198, -0.67775281, -0.53799085,  0.20205894],
                            [0.26630231, -0.21850045,  0.0587917,   0.23642237,  0.60691942,  0.06582413, -0.75592358,  0.28037913, -0.18814138,  0.09807932,  0.45868198, -0.67775281, -0.53799085,  0.20205894,  0.26823767, -0.23681522,  0.1067152,   0.6793826,  -0.07348068, -0.46833013, -0.56009532]]
        else:
            raise ValueError("Robot type not supported")

        idx_qpos = np.random.randint(0, 6)
        # idx_qpos = 0
        qpos = rolling_qpos[idx_qpos]
        
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        # with rolling noise end

        '''
        # without rolling noise start
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        if self._desired_action == "turn" or self._desired_action == "tracking":
            self.set_state(qpos, qvel)
        # without rolling noise end
        #'''
        
        position_r01 = qpos[0:3]
        rotation_r01 = Rotation.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_euler('xyz')
        position_r23 = qpos[7:10]
        rotation_r23 = Rotation.from_quat([qpos[11], qpos[12], qpos[13], qpos[10]]).as_euler('xyz')
        position_r45 = qpos[14:17]
        rotation_r45 = Rotation.from_quat([qpos[18], qpos[19], qpos[20], qpos[17]]).as_euler('xyz')

        ux = 0
        uy = 0
        uz = 1
        theta = np.random.uniform(low=self._min_reset_heading, high=self._max_reset_heading)
        # theta = -40 * np.pi / 180
        R = np.array([[np.cos(theta)+ux**2*(1-np.cos(theta)), 
                       ux*uy*(1-np.cos(theta))-uz*np.sin(theta),
                       ux*uz*(1-np.cos(theta))+uy*np.sin(theta)],
                       [uy*ux*(1-np.cos(theta))+uz*np.sin(theta),
                        np.cos(theta)+uy**2*(1-np.cos(theta)),
                        uy*uz*(1-np.cos(theta)-ux*np.sin(theta))],
                        [uz*ux*(1-np.cos(theta)) -uy*np.sin(theta),
                         uz*uy*(1-np.cos(theta)) + ux*np.sin(theta),
                         np.cos(theta)+uz**2*(1-np.cos(theta))]])

        
        position_r01_new = (R @ position_r01.reshape(-1,1)).squeeze()
        position_r23_new = (R @ position_r23.reshape(-1,1)).squeeze()
        position_r45_new = (R @ position_r45.reshape(-1,1)).squeeze()
        rot_quat_r01_new = Rotation.from_euler('xyz', rotation_r01 + [0, 0, theta]).as_quat()
        rot_quat_r01_new = [rot_quat_r01_new[3], rot_quat_r01_new[0], rot_quat_r01_new[1], rot_quat_r01_new[2]]
        rot_quat_r23_new = Rotation.from_euler('xyz', rotation_r23 + [0, 0, theta]).as_quat()
        rot_quat_r23_new = [rot_quat_r23_new[3], rot_quat_r23_new[0], rot_quat_r23_new[1], rot_quat_r23_new[2]]
        rot_quat_r45_new = Rotation.from_euler('xyz', rotation_r45 + [0, 0, theta]).as_quat()
        rot_quat_r45_new = [rot_quat_r45_new[3], rot_quat_r45_new[0], rot_quat_r45_new[1], rot_quat_r45_new[2]]

        qpos_new = np.concatenate((position_r01_new, rot_quat_r01_new, position_r23_new, rot_quat_r23_new,
                                   position_r45_new, rot_quat_r45_new))
        self.set_state(qpos_new, qvel)

        rng = np.random.default_rng()
        random = rng.standard_normal(size=6)
        tendons = random*self._tendon_reset_stdev + self._tendon_reset_mean
        for i in range(tendons.size):
            if tendons[i] > self._tendon_max_length:
                tendons[i] = self._tendon_max_length
            elif tendons[i] < self._tendon_min_length:
                tendons[i] = self._tendon_min_length
        
        for i in range(50):
            self.do_simulation(tendons, self.frame_skip)


        pos_r01_left_end = self.data.geom("s0").xpos.copy()
        pos_r23_left_end = self.data.geom("s2").xpos.copy()
        pos_r45_left_end = self.data.geom("s4").xpos.copy()
        left_COM_before = (pos_r01_left_end+pos_r23_left_end+pos_r45_left_end)/3
        pos_r01_right_end = self.data.geom("s1").xpos.copy()
        pos_r23_right_end = self.data.geom("s3").xpos.copy()
        pos_r45_right_end = self.data.geom("s5").xpos.copy()
        right_COM_before = (pos_r01_right_end+pos_r23_right_end+pos_r45_right_end)/3
        orientation_vector_before = left_COM_before - right_COM_before
        self._reset_psi = np.arctan2(-orientation_vector_before[0], orientation_vector_before[1])
        self._reset_psi += self._iniyaw_bias
        self._reset_psi = np.arctan2(np.sin(self._reset_psi), np.cos(self._reset_psi))
        self._oripoint = np.array([(left_COM_before[0]+right_COM_before[0])/2, (left_COM_before[1]+right_COM_before[1])/2])
        

        if self._desired_action == "tracking":
            min_waypt_range, max_waypt_range = self._waypt_range
            min_waypt_angle, max_waypt_angle = self._waypt_angle_range
            waypt_length = np.random.uniform(min_waypt_range, max_waypt_range)
            waypt_yaw = np.random.uniform(min_waypt_angle, max_waypt_angle) + self._reset_psi
            if self._is_test == True:
                kmm_length = 0.5
                kmm_yaw = 0.5
                waypt_length = kmm_length*max_waypt_range + (1-kmm_length)*min_waypt_range
                waypt_yaw = (kmm_yaw*max_waypt_angle + (1-kmm_yaw)*min_waypt_angle) + self._reset_psi
            self._waypt = np.array([self._oripoint[0] + waypt_length * np.cos(waypt_yaw), self._oripoint[1] + waypt_length * np.sin(waypt_yaw)])
            # if self._is_test == True: # for test3
            #     self._waypt = np.array([0, 0]) # for test3
        
        elif self._desired_action == "vel_track":
            lin_vel_scale = 0.5
            self._lin_vel_cmd = np.array([lin_vel_scale*np.cos(self._reset_psi), lin_vel_scale*np.sin(self._reset_psi)])
            self._ang_vel_cmd = 0.0
                
        obs_act_seq = []
        for i in range(64):
            self.do_simulation(tendons, self.frame_skip)
            _, observation, _ = self._get_obs()
            action = self.data.ctrl[:].copy()
            obs_act_seq.append(np.concatenate((observation, action)))
        obs_act_seq = np.array(obs_act_seq)

        self._step_num = 0
        if self._desired_action == "turn":
            for i in range(self._reward_delay_steps):
                self.step(tendons)
        
        state, observation, gt_log_intriparam = self._get_obs()

        return state, observation, gt_log_intriparam, obs_act_seq

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
