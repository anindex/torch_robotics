import os
import time
from math import ceil

import cv2

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch

from torch_robotics.environment.env_spheres_3d import EnvSpheres3D
from torch_robotics.environment.primitives import MultiSphereField
from torch_robotics.robot.robot_panda import RobotPanda
from torch_robotics.task.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_numpy


def create_assets_from_primitive_shapes(sim, gym, obj_list):
    object_assets_l = []
    object_poses_l = []

    for obj in obj_list or []:
        # get position and orientation
        obj_pos = to_numpy(obj.pos)
        obj_ori = to_numpy(obj.ori)

        for obj_field in obj.fields:
            if isinstance(obj_field, MultiSphereField):
                for center, radius in zip(obj_field.centers, obj_field.radii):
                    center_np = to_numpy(center)
                    radius_np = to_numpy(radius)

                    # create sphere asset
                    sphere_radius = radius_np
                    asset_options = gymapi.AssetOptions()
                    asset_options.fix_base_link = True
                    sphere_asset = gym.create_sphere(sim, sphere_radius, asset_options)
                    object_assets_l.append(sphere_asset)

                    # set position and orientation
                    obj_pose = gymapi.Transform()
                    obj_pose.p = gymapi.Vec3(*(center_np + obj_pos))
                    obj_pose.r = gymapi.Quat(obj_ori[1], obj_ori[2], obj_ori[3], obj_ori[0])
                    object_poses_l.append(obj_pose)
            else:
                raise NotImplementedError

    return object_assets_l, object_poses_l


class ViewerRecorder:

    def __init__(self, dt=1/50., fps=50):
        self.dt = dt
        self.fps = fps
        self.step_img = []

    def append(self, step, img):
        self.step_img.append((step, img))

    def make_video(self, video_path='./trajs_replay.mp4'):
        frame = self.step_img[0][1]
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))

        max_steps = len(self.step_img) - 1
        for step, frame in self.step_img:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            cv2.putText(frame,
                        f'Step: {step}/{max_steps}',
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_4)
            cv2.putText(frame,
                        f'Time: {self.dt * step:.2f} secs',
                        (50, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_4)
            video.write(frame)

        cv2.destroyAllWindows()
        video.release()


class PandaMotionPlanningIsaacGymEnv:

    def __init__(self, env, robot, task,
                 asset_root="/home/carvalho/Projects/MotionPlanningDiffusion/mpd/isaacgym/assets",
                 controller_type='position',
                 num_envs=8,
                 all_robots_in_one_env=False,
                 color_robots=False,
                 use_pipeline_gpu=False,
                 show_goal_configuration=True,
                 sync_with_real_time=False
                 ):

        self.env = env
        self.robot = robot
        self.task = task
        self.controller_type = controller_type
        self.num_envs = num_envs + 1 if show_goal_configuration else num_envs

        self.all_robots_in_one_env = all_robots_in_one_env
        self.color_robots = color_robots

        ###############################################################################################################
        # ISAAC
        ###############################################################################################################
        # Setup simulator
        # acquire gym interface
        self.gym = gymapi.acquire_gym()

        # gym arguments
        self.gym_args = gymutil.parse_arguments()
        self.gym_args.use_gpu_pipeline = use_pipeline_gpu
        self.tensor_args = {'device': get_torch_device(device='cuda' if use_pipeline_gpu else 'cpu')}

        self.sync_with_real_time = sync_with_real_time

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 50.0  # upper level policy frequency
        sim_params.substeps = 20  # 1/dt * substeps = lower level controller frequency
        # e.g. dt = 1/50 and substeps = 20, means a lower-level controller running at 1000 Hz
        sim_params.use_gpu_pipeline = self.gym_args.use_gpu_pipeline
        if self.gym_args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.gym_args.num_threads
            sim_params.physx.use_gpu = self.gym_args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")

        # create sim
        self.sim = self.gym.create_sim(
            self.gym_args.compute_device_id, self.gym_args.graphics_device_id, self.gym_args.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        ###############################################################################################################
        # Environment assets
        object_fixed_assets_l, object_fixed_poses_l = create_assets_from_primitive_shapes(self.sim, self.gym, self.env.obj_fixed_list)
        object_movable_assets_l, object_movable_poses_l = create_assets_from_primitive_shapes(self.sim, self.gym, self.env.obj_movable_list)

        ###############################################################################################################
        # Robot asset

        # load franka asset
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        self.franka_hand = 'panda_hand'

        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_lower_limits = franka_dof_props["lower"]
        self.franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = self.franka_upper_limits - self.franka_lower_limits
        franka_mids = 0.3 * (self.franka_upper_limits + self.franka_lower_limits)

        # use position or velocity drive for all joint dofs
        if self.controller_type == 'position':
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            franka_dof_props["stiffness"][:7].fill(400.0)
            franka_dof_props["damping"][:7].fill(40.0)
        elif self.controller_type == 'velocity':
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_VEL)
            franka_dof_props["stiffness"][:7].fill(0.0)
            franka_dof_props["damping"][:7].fill(600.0)
        else:
            raise NotImplementedError

        # use position control for grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        self.franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        # default_dof_pos[:7] = franka_mids[:7]

        # grippers open
        self.default_dof_pos[7:] = self.franka_upper_limits[7:]

        self.default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        ###############################################################################################################
        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print(f"Creating {self.num_envs} environments")

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 2
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # robot pose
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0)

        self.envs = []
        self.franka_handles = []
        self.obj_idxs = []
        self.hand_idxs = []

        color_obj_fixed = gymapi.Vec3(220. / 255., 220. / 255., 220. / 255.)
        color_obj_movable = gymapi.Vec3(1., 0., 0.)

        # create env
        if self.all_robots_in_one_env:
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

        self.show_goal_configuration = show_goal_configuration
        self.goal_joint_position = None

        for i in range(self.num_envs):
            # create env
            if not self.all_robots_in_one_env:
                env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
                self.envs.append(env)

            # add objects fixed
            for obj_asset, obj_pose in zip(object_fixed_assets_l, object_fixed_poses_l):
                object_handle = self.gym.create_actor(env, obj_asset, obj_pose, "obj_fixed", i, 0)
                self.gym.set_rigid_body_color(env, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_obj_fixed)
                # get global index of object in rigid body state tensor
                obj_idx = self.gym.get_actor_rigid_body_index(env, object_handle, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs.append(obj_idx)

            # add objects movable
            for obj_asset, obj_pose in zip(object_movable_assets_l, object_movable_poses_l):
                object_handle = self.gym.create_actor(env, obj_asset, obj_pose, "obj_movable", i, 0)
                self.gym.set_rigid_body_color(env, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_obj_movable)
                # get global index of object in rigid body state tensor
                obj_idx = self.gym.get_actor_rigid_body_index(env, object_handle, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs.append(obj_idx)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)
            self.franka_handles.append(franka_handle)

            # color franka
            n_rigid_bodies = self.gym.get_actor_rigid_body_count(env, franka_handle)
            if self.show_goal_configuration and i == self.num_envs - 1:
                color = gymapi.Vec3(128/255., 0., 128/255.)
                for j in range(n_rigid_bodies):
                    self.gym.set_rigid_body_color(env, franka_handle, j, gymapi.MESH_VISUAL_AND_COLLISION, color)

            if color_robots:
                if not(self.show_goal_configuration and i == self.num_envs - 1):
                    c = np.random.random(3)
                    color = gymapi.Vec3(c[0], c[1], c[2])
                    for j in range(n_rigid_bodies):
                        self.gym.set_rigid_body_color(env, franka_handle, j, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        ###############################################################################################################
        # CAMERA
        # point camera at middle env
        cam_pos = gymapi.Vec3(2., 0, 1.5)
        cam_target = gymapi.Vec3(-3, 0, -1)
        if len(self.envs) == 1:
            self.middle_env = self.envs[0]
        else:
            self.middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
        self.gym.viewer_camera_look_at(self.viewer, self.middle_env, cam_pos, cam_target)

        camera_props = gymapi.CameraProperties()
        camera_props.width = 1920
        camera_props.height = 1080
        self.viewer_camera_handle = self.gym.create_camera_sensor(self.middle_env, camera_props)
        self.gym.set_camera_location(self.viewer_camera_handle, env, cam_pos, cam_target)

        self.viewer_recorder = ViewerRecorder(dt=sim_params.dt, fps=ceil(1 / sim_params.dt))

        ###############################################################################################################
        # DEBUGGING visualization
        self.axes_geom = gymutil.AxesGeometry(0.15)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.005, 6, 6, gymapi.Transform(), color=(0, 0, 1))

        ###############################################################################################################
        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 9, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 9, 1)

        ###############################################################################################################
        self.step_idx = 0

    def reset(self, start_joint_positions=None, goal_joint_position=None):
        self.step_idx = 0

        if start_joint_positions is None:
            start_joint_positions = torch.zeros((self.num_envs, self.franka_num_dofs - 2), **self.tensor_args)  # w/o gripper
            start_joint_positions[..., :7] = to_torch(self.default_dof_pos[:7], **self.tensor_args)

        assert start_joint_positions.ndim == 2

        # make sure joint positions are in the same device as the pipeline
        start_joint_positions = start_joint_positions.to(**self.tensor_args)

        # Set dof states
        # reset to zero velocity
        # dof_state_tensor = torch.zeros((self.num_envs, self.franka_num_dofs * 2), **self.tensor_args)

        dof_pos_tensor = torch.zeros((self.num_envs, self.franka_num_dofs), **self.tensor_args)
        if self.show_goal_configuration:
            dof_pos_tensor[:-1, :7] = start_joint_positions
            assert goal_joint_position is not None
            self.goal_joint_position = goal_joint_position
            dof_pos_tensor[-1, :7] = goal_joint_position
        else:
            dof_pos_tensor[..., :7] = start_joint_positions

        # grippers open
        dof_pos_tensor[..., 7:] = to_torch(self.franka_upper_limits, **self.tensor_args)[None, 7:]

        # dof_state_tensor[..., :self.franka_num_dofs] = dof_pos_tensor

        if self.all_robots_in_one_env:
            envs = self.envs * self.num_envs
        else:
            envs = self.envs

        for env, handle, joints_pos in zip(envs, self.franka_handles, dof_pos_tensor):
            joint_state_des = self.gym.get_actor_dof_states(env, handle, gymapi.STATE_ALL)
            joint_state_des['pos'] = np.zeros_like(joint_state_des['pos'])
            joint_state_des['vel'] = np.zeros_like(joint_state_des['vel'])
            joint_state_des['pos'][:7] = to_numpy(joints_pos[:7])
            joint_state_des['pos'][-1] = joints_pos[-1]
            joint_state_des['pos'][-2] = joints_pos[-2]
            self.gym.set_actor_dof_states(env, handle, joint_state_des, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(env, handle, joint_state_des['pos'])

        # TODO - convert to tensor API
        # # set dof states
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_tensor))
        # # set position targets
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_pos_tensor))

    def step(self, actions, visualize=True, render_viewer_camera=False):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # Deploy control based on type
        action_dof = torch.zeros_like(self.dof_pos).squeeze(-1)
        if self.show_goal_configuration:
            action_dof[:-1, :7] = actions[..., :7]
            action_dof[-1, :7] = self.goal_joint_position
        else:
            action_dof[..., :7] = actions[..., :7]

        # gripper is open
        action_dof[..., 7:9] = torch.Tensor([[0.04, 0.04]] * self.num_envs)

        # Deploy actions
        if self.controller_type == 'position':
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_dof))
        elif self.controller_type == 'velocity':
            self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(action_dof))

        # Check collisions between robot and objects

        if visualize:
            # clean up
            self.gym.clear_lines(self.viewer)

            # draw EE reference frame
            # TODO - implement vectorized version
            if self.all_robots_in_one_env:
                envs = self.envs * self.num_envs
            else:
                envs = self.envs

            for env, franka_handle in zip(envs, self.franka_handles):
                body_dict = self.gym.get_actor_rigid_body_dict(env, franka_handle)
                props = self.gym.get_actor_rigid_body_states(env, franka_handle, gymapi.STATE_POS)
                ee_pose = props['pose'][:][body_dict[self.franka_hand]]
                ee_transform = gymapi.Transform(p=gymapi.Vec3(*ee_pose[0]), r=gymapi.Quat(*ee_pose[1]))
                # reference frame
                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, ee_transform)
                # sphere
                # gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, env, ee_transform)

            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            if self.sync_with_real_time:
                self.gym.sync_frame_time(self.sim)

            # render viewer camera
            if render_viewer_camera:
                self.gym.render_all_camera_sensors(self.sim)
                viewer_img = self.gym.get_camera_image(self.sim, self.middle_env, self.viewer_camera_handle, gymapi.IMAGE_COLOR)
                viewer_img = viewer_img.reshape(viewer_img.shape[0], -1, 4)[..., :3]  # get RGB part
                self.viewer_recorder.append(self.step_idx, viewer_img)

        self.step_idx += 1

    def check_viewer_has_closed(self):
        return self.gym.query_viewer_has_closed(self.viewer)

    def clean_up(self):
        # cleanup
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class MotionPlanningController:

    def __init__(self, motion_planning_isaac_env):
        self.mp_env = motion_planning_isaac_env

    def run_trajectories(
            self,
            trajectories,  # shape: (H, B, D)
            start_states_joint_pos=None, goal_state_joint_pos=None,
            visualize=True,
            render_viewer_camera=False,
            make_video=False,
            video_path='./trajs_replay.mp4'
    ):
        assert start_states_joint_pos is not None
        assert goal_state_joint_pos is not None

        # start at the initial position
        self.mp_env.reset(start_joint_positions=start_states_joint_pos, goal_joint_position=goal_state_joint_pos)

        # execute trajectory
        for actions in trajectories:
            if self.mp_env.check_viewer_has_closed():
                break

            self.mp_env.step(actions, visualize=visualize, render_viewer_camera=render_viewer_camera)

        # clean up isaac
        self.mp_env.clean_up()

        # create video
        if make_video:
            self.mp_env.viewer_recorder.make_video(video_path=video_path)


if __name__ == '__main__':
    seed = 0
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=tensor_args
    )

    robot = RobotPanda(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # workspace limits
        tensor_args=tensor_args
    )

    # -------------------------------- Physics ---------------------------------
    motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
        env, robot, task,
        controller_type='position',
        num_envs=8,
        all_robots_in_one_env=False,
        color_robots=False
    )

    motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
    trajectories_joint_pos = torch.zeros((100, 8, 7), **tensor_args) + 1.
    motion_planning_controller.run_trajectories(
        trajectories_joint_pos,
        start_states_joint_pos=trajectories_joint_pos[0], goal_state_joint_pos=trajectories_joint_pos[-1][0],
        visualize=True
    )
