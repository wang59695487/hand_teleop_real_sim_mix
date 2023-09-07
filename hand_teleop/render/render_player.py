import os
from copy import deepcopy
from typing import Optional, Dict, Sequence, Union, Any, Tuple

import numpy as np
import sapien.core as sapien
import transforms3d.quaternions
from sapien.core import Pose
from sapien.utils import Viewer

from hand_teleop.env.sim_env.constructor import add_default_scene_light
from hand_teleop.env.sim_env.constructor import get_engine_and_renderer
from hand_teleop.real_world import lab
from hand_teleop.utils.common_robot_utils import load_robot
from hand_teleop.utils.ycb_object_utils import load_ycb_object

MAX_DEPTH_RANGE = 2.5


class RenderPlayer:
    def __init__(self, meta_data, data, use_gui=False, robot_name="xarm6_allegro_modified_finger", **renderer_kwargs):
        engine, renderer = get_engine_and_renderer(use_gui=use_gui, need_offscreen_render=True, no_rgb=False,
                                                   **renderer_kwargs)
        self.use_gui = use_gui
        self.engine = engine
        self.renderer = renderer

        self.np_random = None
        self.viewer: Optional[Viewer] = None
        self.scene: Optional[sapien.Scene] = None
        self.robot: Optional[sapien.Articulation] = None
        self.cameras: Dict[str, sapien.CameraEntity] = {}
        self.camera_infos: Dict[str, Dict] = {}
        self.camera_pose_noise: Dict[str, Tuple[Optional[float], sapien.Pose]] = {}

        # Create scene
        scene_config = sapien.SceneConfig()
        if "address" in renderer_kwargs:
            scene_config.disable_collision_visual = True
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(0.004)

        # Viewer
        if use_gui:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)

        # Load table and object
        env_args = meta_data["env_kwargs"]
        self.env_class = meta_data["env_class"]
        if self.env_class == "PickPlaceEnv":
            self.tables = self.create_pick_place_table(table_height=0.91)
            self.plate = load_ycb_object(self.scene, "plate", render_only=True, static=True)
            self.manipulated_object = load_ycb_object(self.scene, env_args["object_name"], render_only=True)

        # Load robot
        self.robot = load_robot(self.scene, robot_name, disable_self_collision=True)
        self.meta_data = meta_data
        self.data = data

        # Load light and ground
        add_default_scene_light(self.scene, self.renderer)

        # Prepare player
        self.meta2scene_actor, self.meta2scene_articulation = self.prepare_player()
        self.scene.step()

    def prepare_player(self):

        # Generate player info
        scene_actor2id = {actor.get_name(): actor.get_id() for actor in self.scene.get_all_actors()}
        meta_actor2id = self.meta_data["actor"]
        meta2scene_actor = {}
        for key, value in meta_actor2id.items():
            if key not in scene_actor2id:
                pass
            else:
                meta2scene_actor[value] = scene_actor2id[key]

        # Generate articulation id mapping
        all_articulation_root = [robot.get_links()[0] for robot in self.scene.get_all_articulations()]
        scene_articulation2id = {actor.get_name(): actor.get_id() for actor in all_articulation_root}
        scene_articulation2dof = {r.get_links()[0].get_name(): r.dof for r in self.scene.get_all_articulations()}
        meta_articulation2id = self.meta_data["articulation"]
        meta_articulation2dof = self.meta_data["articulation_dof"]
        meta2scene_articulation = {}
        for key, value in meta_articulation2id.items():
            if key not in scene_articulation2id:
                print(f"Recorded articulation {key} not exists in the scene. Will skip it.")
            else:
                if meta_articulation2dof[key] == scene_articulation2dof[key]:
                    meta2scene_articulation[value] = scene_articulation2id[key]
                else:
                    print(
                        f"Recorded articulation {key} has {meta_articulation2dof[key]} dof while "
                        f"scene articulation has {scene_articulation2dof[key]}. Will skip it.")

        return meta2scene_actor, meta2scene_articulation

    def get_sim_data(self, item) -> Dict[str, Any]:
        sim_data = self.data[item]["simulation"]
        actor_data = sim_data["actor"]
        drive_data = sim_data["articulation_drive"]
        articulation_data = sim_data["articulation"]
        scene_actor_data = {self.meta2scene_actor[key]: value for key, value in actor_data.items() if
                            key in self.meta2scene_actor}
        scene_drive_data = {self.meta2scene_articulation[key]: value for key, value in drive_data.items() if
                            key in self.meta2scene_articulation}
        scene_articulation_data = {self.meta2scene_articulation[key]: value for key, value in articulation_data.items()
                                   if key in self.meta2scene_articulation}
        return dict(actor=scene_actor_data, articulation_drive=scene_drive_data, articulation=scene_articulation_data)

    def set_sim_data(self, item):
        data = self.get_sim_data(item)
        self.scene.unpack(data)

    def create_pick_place_table(self, table_height):
        # Build object table first
        builder = self.scene.create_actor_builder()
        table_thickness = 0.03

        # Top
        top_pose = sapien.Pose(np.array([lab.TABLE_ORIGIN[0], lab.TABLE_ORIGIN[1], -table_thickness / 2]))
        table_half_size = np.concatenate([lab.TABLE_XY_SIZE / 2, [table_thickness / 2]])
        # Leg
        table_visual_material = self.renderer.create_material()
        table_visual_material.set_metallic(0.0)
        table_visual_material.set_specular(0.3)
        table_visual_material.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
        table_visual_material.set_roughness(0.3)

        leg_size = np.array([0.025, 0.025, (table_height / 2 - table_half_size[2])])
        leg_height = -table_height / 2 - table_half_size[2]
        x = table_half_size[0] - 0.1
        y = table_half_size[1] - 0.1

        builder.add_box_visual(pose=top_pose, half_size=table_half_size, material=table_visual_material)
        builder.add_box_visual(pose=sapien.Pose([x, y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                               material=table_visual_material, name="leg0")
        builder.add_box_visual(pose=sapien.Pose([x, -y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                               material=table_visual_material, name="leg1")
        builder.add_box_visual(pose=sapien.Pose([-x, y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                               material=table_visual_material, name="leg2")
        builder.add_box_visual(pose=sapien.Pose([-x, -y + lab.TABLE_ORIGIN[1], leg_height]), half_size=leg_size,
                               material=table_visual_material, name="leg3")
        object_table = builder.build_static("object_table")

        # Build robot table
        robot_table_thickness = 0.025
        table_half_size = np.concatenate([lab.ROBOT_TABLE_XY_SIZE / 2, [robot_table_thickness / 2]])
        robot_table_offset = -lab.DESK2ROBOT_Z_AXIS
        table_height += robot_table_offset
        builder = self.scene.create_actor_builder()
        top_pose = sapien.Pose(
            np.array([lab.ROBOT2BASE.p[0] - table_half_size[0] + 0.08,
                      lab.ROBOT2BASE.p[1] - table_half_size[1] + 0.08,
                      -robot_table_thickness / 2 + robot_table_offset]))
        table_visual_material = self.renderer.create_material()
        table_visual_material.set_metallic(0.0)
        table_visual_material.set_specular(0.2)
        table_visual_material.set_base_color(np.array([0, 0, 0, 255]) / 255)
        table_visual_material.set_roughness(0.1)
        builder.add_box_visual(pose=top_pose, half_size=table_half_size, material=table_visual_material)
        robot_table = builder.build_static("robot_table")
        return object_table, robot_table

    def create_camera(self, position: np.ndarray, look_at_dir: np.ndarray, right_dir: np.ndarray, name: str,
                      resolution: Sequence[Union[float, int]], fov: float, mount_actor_name: str = None):
        if not len(resolution) == 2:
            raise ValueError(f"Resolution should be a 2d array, but now {len(resolution)} is given.")
        if mount_actor_name is not None:
            mount = [actor for actor in self.scene.get_all_actors() if actor.get_name() == mount_actor_name]
            if len(mount) == 0:
                raise ValueError(f"Camera mount {mount_actor_name} not found in the env.")
            if len(mount) > 1:
                raise ValueError(
                    f"Camera mount {mount_actor_name} name duplicates! To mount an camera on an actor,"
                    f" give the mount a unique name.")
            mount = mount[0]
            cam = self.scene.add_mounted_camera(name, mount, Pose(), width=resolution[0], height=resolution[1],
                                                fovy=fov, fovx=fov, near=0.1, far=10)
        else:
            # Construct camera pose
            look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
            right_dir = right_dir - np.sum(right_dir * look_at_dir).astype(np.float64) * look_at_dir
            right_dir = right_dir / np.linalg.norm(right_dir)
            up_dir = np.cross(look_at_dir, -right_dir)
            rot_mat_homo = np.stack([look_at_dir, -right_dir, up_dir, position], axis=1)
            pose_mat = np.concatenate([rot_mat_homo, np.array([[0, 0, 0, 1]])])
            pose_cam = sapien.Pose.from_transformation_matrix(pose_mat)
            cam = self.scene.add_camera(name, width=resolution[0], height=resolution[1], fovy=fov, near=0.1, far=10)
            cam.set_local_pose(pose_cam)

        self.cameras.update({name: cam})

    def create_camera_from_pose(self, pose: sapien.Pose, name: str, resolution: Sequence[Union[float, int]],
                                fov: float):
        if not len(resolution) == 2:
            raise ValueError(f"Resolution should be a 2d array, but now {len(resolution)} is given.")
        sapien2opencv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        sapien2opencv_quat = transforms3d.quaternions.mat2quat(sapien2opencv)
        pose_cam = pose * sapien.Pose(q=sapien2opencv_quat)
        cam = self.scene.add_camera(name, width=resolution[0], height=resolution[1], fovy=fov, near=0.1, far=10)
        cam.set_local_pose(pose_cam)
        self.cameras.update({name: cam})

    def setup_camera_from_config(self, config: Dict[str, Dict]):
        for cam_name, cfg in config.items():
            if cam_name in self.cameras.keys():
                raise ValueError(f"Camera {cam_name} already exists in the environment")
            if "mount_actor_name" in cfg:
                self.create_camera(None, None, None, name=cam_name, resolution=cfg["resolution"],
                                   fov=cfg["fov"], mount_actor_name=cfg["mount_actor_name"])
            else:
                if "position" in cfg:
                    self.create_camera(cfg["position"], cfg["look_at_dir"], cfg["right_dir"], cam_name,
                                       resolution=cfg["resolution"], fov=cfg["fov"])
                elif "pose" in cfg:
                    self.create_camera_from_pose(cfg["pose"], cam_name, resolution=cfg["resolution"], fov=cfg["fov"])

                else:
                    raise ValueError(f"Camera {cam_name} has no position or pose.")

    def setup_visual_obs_config(self, config: Dict[str, Dict]):
        for name, camera_cfg in config.items():
            if name not in self.cameras.keys():
                raise ValueError(
                    f"Camera {name} not created. Existing {len(self.cameras)} cameras: {self.cameras.keys()}")
            self.camera_infos[name] = {}
            banned_modality_set = {"point_cloud", "depth"}
            if len(banned_modality_set.intersection(set(camera_cfg.keys()))) == len(banned_modality_set):
                raise RuntimeError(f"Request both point_cloud and depth for same camera is not allowed. "
                                   f"Point cloud contains all information required by the depth.")

            # Add perturb for camera pose
            cam = self.cameras[name]
            if "pose_perturb_level" in camera_cfg:
                cam_pose_perturb = camera_cfg.pop("pose_perturb_level")
            else:
                cam_pose_perturb = None
            self.camera_pose_noise[name] = (cam_pose_perturb, cam.get_pose())

            for modality, cfg in camera_cfg.items():
                if modality == "point_cloud":
                    if "process_fn" not in cfg or "num_points" not in cfg:
                        raise RuntimeError(f"Missing process_fn or num_points in camera {name} point_cloud config.")

                self.camera_infos[name][modality] = cfg

        modality = []
        for camera_cfg in config.values():
            modality.extend(camera_cfg.keys())
        modality_set = set(modality)

    @classmethod
    def from_demo(cls, demo, robot_name, use_gui=False, **client_kwargs):
        meta_data = deepcopy(demo["meta_data"])
        data = demo["data"]

        env_params = meta_data["env_kwargs"]
        # Specify rendering device if the computing device is given
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"

        real_camera_cfg = {
            "relocate_view": dict(pose=lab.ROBOT2BASE * lab.CAM2ROBOT,
                                  fov=lab.fov, resolution=(224, 224))
        }
        render_player = cls(meta_data, data, robot_name=robot_name, use_gui=use_gui, **client_kwargs)
        render_player.setup_camera_from_config(real_camera_cfg)

        return render_player


if __name__ == '__main__':
    demo = np.load("sim/raw_data/pick_place_mustard_bottle_aug/demo_1_2.pickle", allow_pickle=True)
    render_player = RenderPlayer.from_demo(demo, robot_name="xarm6_allegro_modified_finger", use_gui=True)
    render_player.viewer.focus_camera(render_player.cameras["relocate_view"])

    for i in range(len(render_player.data["simulation"])):
        render_player.set_sim_data(i)
        render_player.viewer.render()
        print(i)
