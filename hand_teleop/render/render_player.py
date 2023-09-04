from functools import cached_property
from pathlib import Path
from typing import Optional, List, Dict, Sequence, Union

import numpy as np
import sapien.core as sapien
import transforms3d.quaternions
from sapien.core import Pose
from sapien.utils import Viewer

from hand_teleop.env.sim_env.constructor import get_engine_and_renderer, add_default_scene_light, download_maniskill
from hand_teleop.utils.random_utils import np_random


class RenderPlayer:
    def __init__(self, meta_data, use_gui=False, **renderer_kwargs):
        engine, renderer = get_engine_and_renderer(use_gui=use_gui, need_offscreen_render=True, no_rgb=False, **renderer_kwargs)
        self.use_gui = use_gui
        self.engine = engine
        self.renderer = renderer

        self.np_random = None
        self.viewer: Optional[Viewer] = None
        self.scene: Optional[sapien.Scene] = None
        self.robot: Optional[sapien.Articulation] = None
        self.init_state: Optional[Dict] = None
        self.robot_name = ""

        # Camera
        self.use_visual_obs = use_visual_obs
        self.use_offscreen_render = need_offscreen_render
        self.no_rgb = no_rgb and not use_gui
        self.cameras: Dict[str, sapien.CameraEntity] = {}

        self.seed()
        self.current_step = 0



        self.env_class = meta_data["env_class"]



        if self.env_class == "PickPlaceEnv":



