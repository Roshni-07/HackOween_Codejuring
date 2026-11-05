"""
sim/scene.py — the Halloween laboratory arena and the humanoid robot model.

Contents
  * Arena constants (room, walls, drop zone) and the colour PALETTE.
  * HUMANOID_URDF — native URDF string for the 16-link humanoid. It is written
    to a cache file and loaded with ``p.loadURDF`` (never with multi-link
    ``createMultiBody`` link arrays).
  * build_arena()       — slate floor + tile grid, 4 walls with paneling,
                          glowing orange trims, corner spider webs and bats.
  * build_drop_zone()   — orange platform with a flat jack-o'-lantern decal.
  * spawn_objects()     — the fetch objects (engine-internal identities only).
  * Lighting presets for the robot camera (TinyRenderer) and the GUI.

Hard rule: ZERO purple. ``is_purple`` / ``assert_palette_has_no_purple`` are
used at import time and by the tests to enforce it.

Coordinates: metres, z up, heading 0 = +x. The robot's URDF frame has +x
forward, +y left, z up, origin on the floor between the wheels.
"""

import colorsys
import math
import os

import pybullet as p

from .meshes import ASSET_DIR, Mesh, spawn_visual_mesh

# ---------------------------------------------------------------------------
# Arena geometry
# ---------------------------------------------------------------------------
ROOM_SIZE = 5.0
HALF_ROOM = ROOM_SIZE / 2.0            # wall centre lines at +-2.5
WALL_HALF_THICK = 0.1
WALL_INNER = HALF_ROOM - WALL_HALF_THICK   # inner wall faces at +-2.4
WALL_HEIGHT = 1.4

DROP_ZONE_POS = [1.8, 1.8, 0.0]
DROP_ZONE_HALF = 0.4                   # 0.8 m x 0.8 m square (in-zone test uses this)

# ---------------------------------------------------------------------------
# Palette (RGBA, 0..1). Warm Halloween lab — absolutely no purple/violet.
# ---------------------------------------------------------------------------
PALETTE = {
    # environment
    "floor_slate": (0.16, 0.17, 0.19, 1.0),
    "floor_grid": (0.10, 0.11, 0.12, 1.0),
    "wall": (0.24, 0.21, 0.19, 1.0),
    "wall_panel": (0.32, 0.27, 0.23, 1.0),
    "trim_glow": (1.00, 0.46, 0.02, 1.0),
    "web": (0.80, 0.78, 0.73, 1.0),
    "bat": (0.03, 0.03, 0.035, 1.0),
    "obstacle": (0.20, 0.19, 0.18, 1.0),
    # drop zone + jack-o'-lantern
    "zone": (1.00, 0.52, 0.06, 1.0),
    "zone_border": (1.00, 0.72, 0.22, 1.0),
    "pumpkin": (0.80, 0.30, 0.02, 1.0),
    "pumpkin_rib": (0.62, 0.22, 0.01, 1.0),
    "pumpkin_face": (0.05, 0.03, 0.02, 1.0),
    "pumpkin_stem": (0.16, 0.36, 0.08, 1.0),
    # robot
    "ivory": (0.93, 0.90, 0.82, 1.0),
    "graphite": (0.20, 0.21, 0.23, 1.0),
    "visor_black": (0.02, 0.02, 0.03, 1.0),
    "eye_cyan": (0.00, 0.90, 1.00, 1.0),
    "eye_alert": (1.00, 0.25, 0.05, 1.0),   # eyes flash this on collision
    "metal": (0.62, 0.64, 0.68, 1.0),
    "gripper_orange": (1.00, 0.42, 0.00, 1.0),
    # fetch objects
    "obj_orange": (1.00, 0.48, 0.05, 1.0),
    "obj_lime": (0.55, 0.92, 0.12, 1.0),
    "obj_red": (0.88, 0.08, 0.06, 1.0),
    "obj_teal": (0.00, 0.58, 0.56, 1.0),
    "obj_white": (0.95, 0.94, 0.90, 1.0),
}


def is_purple(rgba):
    """True for purple / violet / magenta hues (approx. 250°–335°)."""
    h, s, v = colorsys.rgb_to_hsv(float(rgba[0]), float(rgba[1]), float(rgba[2]))
    return s > 0.20 and v > 0.15 and 0.69 <= h <= 0.93


def assert_palette_has_no_purple():
    bad = [k for k, c in PALETTE.items() if is_purple(c)]
    if bad:
        raise ValueError("Purple colours are forbidden in this arena: %s" % bad)


assert_palette_has_no_purple()

# ---------------------------------------------------------------------------
# Fetch objects. Identities never leave the engine (no names in any API).
# Brief rules: 4-6 primitives, >=2 share a colour, >=2 share a shape.
#   shape pairs : two boxes (+red box), two cylinders
#   colour pair : red sphere + red box
#   "tallest"   : the white cylinder (0.40 m) is unambiguously the tallest
# ---------------------------------------------------------------------------
OBJECT_SPECS = [
    {"key": "orange_box", "shape": "box", "half_extents": (0.15, 0.15, 0.15),
     "color": PALETTE["obj_orange"], "pos": (1.5, 0.5)},
    {"key": "lime_box", "shape": "box", "half_extents": (0.15, 0.15, 0.15),
     "color": PALETTE["obj_lime"], "pos": (1.0, -1.0)},
    {"key": "red_sphere", "shape": "sphere", "radius": 0.15,
     "color": PALETTE["obj_red"], "pos": (1.8, -0.6)},
    {"key": "teal_cylinder", "shape": "cylinder", "radius": 0.12, "length": 0.30,
     "color": PALETTE["obj_teal"], "pos": (0.5, 1.5)},
    {"key": "white_cylinder", "shape": "cylinder", "radius": 0.10, "length": 0.40,
     "color": PALETTE["obj_white"], "pos": (-1.2, 1.0)},
    # 6th object: satisfies the brief's "at least two share a colour" rule.
    # Set INCLUDE_SHARED_COLOUR_OBJECT = False to go back to the five-object set.
    {"key": "red_box", "shape": "box", "half_extents": (0.13, 0.13, 0.13),
     "color": PALETTE["obj_red"], "pos": (-1.0, -1.3)},
]
INCLUDE_SHARED_COLOUR_OBJECT = True


def object_half_height(spec):
    if spec["shape"] == "box":
        return spec["half_extents"][2]
    if spec["shape"] == "sphere":
        return spec["radius"]
    return spec["length"] / 2.0


def object_half_width(spec):
    """Half the object's width across the robot's carry axis."""
    if spec["shape"] == "box":
        return max(spec["half_extents"][0], spec["half_extents"][1])
    return spec["radius"]


def object_footprint_radius(spec):
    """Horizontal bounding radius, used for spawn spacing."""
    if spec["shape"] == "box":
        hx, hy, _ = spec["half_extents"]
        return math.hypot(hx, hy)
    return spec["radius"]


def active_object_specs():
    specs = list(OBJECT_SPECS)
    if not INCLUDE_SHARED_COLOUR_OBJECT:
        specs = [s for s in specs if s["key"] != "red_box"]
    return specs


def spawn_objects():
    """Creates the fetch objects. Returns list of (body_id, spec)."""
    created = []
    for spec in active_object_specs():
        hz = object_half_height(spec)
        color = list(spec["color"])
        if spec["shape"] == "box":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(spec["half_extents"]))
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(spec["half_extents"]), rgbaColor=color)
        elif spec["shape"] == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=spec["radius"])
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=spec["radius"], rgbaColor=color)
        else:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=spec["radius"], height=spec["length"])
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=spec["radius"], length=spec["length"], rgbaColor=color)
        body = p.createMultiBody(1.0, col, vis, basePosition=[spec["pos"][0], spec["pos"][1], hz + 0.001])
        p.changeDynamics(
            body, -1, mass=1.0, lateralFriction=0.8, spinningFriction=0.1,
            rollingFriction=0.1, linearDamping=0.5, angularDamping=0.5,
        )
        created.append((body, spec))
    return created


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------
# Robot camera (TinyRenderer). Warm amber key light from above with a high
# ambient term so every object (and the white/teal ones in particular) stays
# crisp and true to colour.
CAMERA_LIGHTING = {
    "lightDirection": [0.35, -0.45, 1.0],
    "lightColor": [1.0, 0.86, 0.68],
    "lightDistance": 6.0,
    "shadow": 0,
    "lightAmbientCoeff": 0.62,
    "lightDiffuseCoeff": 0.55,
    "lightSpecularCoeff": 0.08,
}
# Pixels with no geometry (above the walls) are filled with this warm dark
# gradient (top row -> horizon) instead of the renderer's default white.
CAMERA_BACKGROUND_TOP = (14, 9, 6)
CAMERA_BACKGROUND_HORIZON = (46, 26, 12)

GUI_CONNECT_OPTIONS = (
    "--background_color_red=0.07 --background_color_green=0.045 --background_color_blue=0.03"
)


def configure_gui_lighting():
    """Warm key light + shadows for the GUI / demo-video view."""
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, lightPosition=[1.5, -2.0, 6.0])


# ---------------------------------------------------------------------------
# Humanoid URDF (16 links)
#   base_link (torso, ivory; capsule collision), left/right_wheel (continuous),
#   front/rear_caster, neck_link, head_link (shell), visor_link, left/right_eye,
#   left/right_shoulder (metal joint + upper arm), left/right_forearm,
#   left/right_gripper (orange).
# Only base_link has a collision shape (capsule r=0.22): links never
# self-collide and arm poses can never trigger false collisions.
# Kinematic constants used by sim/driver.py are mirrored in ROBOT_GEOMETRY.
# ---------------------------------------------------------------------------
ROBOT_GEOMETRY = {
    "collision_radius": 0.22,
    "wheel_radius": 0.08,
    "half_track": 0.205,
    "shoulder_x": 0.03,
    "shoulder_y": 0.20,
    "shoulder_z": 0.38,
    "upper_arm": 0.16,
    "forearm": 0.18,      # elbow -> gripper centre
    "neck_z": 0.41,
    "head_pitch_z": 0.46,
    "visor_x": 0.092,     # visor front face in base frame (head neutral)
    "eye_height": 0.51,
}


def _rgba(name):
    return "%.3f %.3f %.3f %.3f" % PALETTE[name]


HUMANOID_URDF = """<?xml version="1.0"?>
<robot name="hackoween_humanoid">
  <material name="ivory"><color rgba="{ivory}"/></material>
  <material name="graphite"><color rgba="{graphite}"/></material>
  <material name="visor_black"><color rgba="{visor_black}"/></material>
  <material name="eye_cyan"><color rgba="{eye_cyan}"/></material>
  <material name="metal"><color rgba="{metal}"/></material>
  <material name="gripper_orange"><color rgba="{gripper_orange}"/></material>

  <!-- 1. torso / chassis : the only collision geometry (capsule) -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.28"/>
      <mass value="10.0"/>
      <inertia ixx="0.30" ixy="0" ixz="0" iyy="0.30" iyz="0" izz="0.20"/>
    </inertial>
    <visual name="chassis_skirt">
      <origin xyz="0 0 0.13"/>
      <geometry><cylinder radius="0.19" length="0.08"/></geometry>
      <material name="graphite"/>
    </visual>
    <visual name="torso">
      <origin xyz="0 0 0.29"/>
      <geometry><cylinder radius="0.16" length="0.24"/></geometry>
      <material name="ivory"/>
    </visual>
    <visual name="chest_panel">
      <origin xyz="0.152 0 0.29"/>
      <geometry><box size="0.02 0.15 0.11"/></geometry>
      <material name="graphite"/>
    </visual>
    <visual name="shoulder_yoke">
      <origin xyz="0.03 0 0.38"/>
      <geometry><box size="0.12 0.36 0.05"/></geometry>
      <material name="graphite"/>
    </visual>
    <!-- Capsule spans z 0.03-0.51. Kept below the visor camera (0.5 m at
         x=0.12, where the capsule surface is at 0.47 m) so the camera and
         depth ray never start inside the robot's own collision shape. -->
    <collision>
      <origin xyz="0 0 0.27"/>
      <geometry><capsule radius="0.22" length="0.04"/></geometry>
    </collision>
  </link>

  <!-- 2-3. side drive wheels -->
  <link name="left_wheel">
    <inertial><mass value="0.4"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual name="tyre">
      <origin rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.08" length="0.045"/></geometry>
      <material name="graphite"/>
    </visual>
    <visual name="spoke">
      <geometry><box size="0.13 0.05 0.022"/></geometry>
      <material name="metal"/>
    </visual>
  </link>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/><child link="left_wheel"/>
    <origin xyz="0 0.205 0.08"/><axis xyz="0 1 0"/>
  </joint>

  <link name="right_wheel">
    <inertial><mass value="0.4"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual name="tyre">
      <origin rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.08" length="0.045"/></geometry>
      <material name="graphite"/>
    </visual>
    <visual name="spoke">
      <geometry><box size="0.13 0.05 0.022"/></geometry>
      <material name="metal"/>
    </visual>
  </link>
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/><child link="right_wheel"/>
    <origin xyz="0 -0.205 0.08"/><axis xyz="0 1 0"/>
  </joint>

  <!-- 4-5. passive casters -->
  <link name="front_caster">
    <inertial><mass value="0.05"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><geometry><sphere radius="0.03"/></geometry><material name="metal"/></visual>
  </link>
  <joint name="front_caster_joint" type="fixed">
    <parent link="base_link"/><child link="front_caster"/><origin xyz="0.13 0 0.03"/>
  </joint>
  <link name="rear_caster">
    <inertial><mass value="0.05"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><geometry><sphere radius="0.03"/></geometry><material name="metal"/></visual>
  </link>
  <joint name="rear_caster_joint" type="fixed">
    <parent link="base_link"/><child link="rear_caster"/><origin xyz="-0.13 0 0.03"/>
  </joint>

  <!-- 6. neck stem (yaw) -->
  <link name="neck_link">
    <inertial><mass value="0.1"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual>
      <origin xyz="0 0 0.025"/>
      <geometry><cylinder radius="0.035" length="0.05"/></geometry>
      <material name="metal"/>
    </visual>
  </link>
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/><child link="neck_link"/>
    <origin xyz="0 0 0.41"/><axis xyz="0 0 1"/>
    <limit lower="-1.3" upper="1.3" effort="20" velocity="4"/>
  </joint>

  <!-- 7. head shell (pitch; positive = look down) -->
  <link name="head_link">
    <inertial><origin xyz="0 0 0.045"/><mass value="0.4"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual name="shell">
      <origin xyz="0 0 0.045"/>
      <geometry><box size="0.17 0.22 0.13"/></geometry>
      <material name="ivory"/>
    </visual>
    <visual name="crown">
      <origin xyz="0 0 0.112"/>
      <geometry><box size="0.13 0.18 0.012"/></geometry>
      <material name="graphite"/>
    </visual>
    <visual name="left_ear_pod">
      <origin xyz="0 0.115 0.045" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.035" length="0.02"/></geometry>
      <material name="metal"/>
    </visual>
    <visual name="right_ear_pod">
      <origin xyz="0 -0.115 0.045" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.035" length="0.02"/></geometry>
      <material name="metal"/>
    </visual>
  </link>
  <joint name="head_joint" type="revolute">
    <parent link="neck_link"/><child link="head_link"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-0.6" upper="0.6" effort="10" velocity="4"/>
  </joint>

  <!-- 8. black visor screen -->
  <link name="visor_link">
    <inertial><mass value="0.02"/><inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/></inertial>
    <visual><geometry><box size="0.012 0.18 0.075"/></geometry><material name="visor_black"/></visual>
  </link>
  <joint name="visor_joint" type="fixed">
    <parent link="head_link"/><child link="visor_link"/><origin xyz="0.086 0 0.045"/>
  </joint>

  <!-- 9-10. cyan optical eyes -->
  <link name="left_eye">
    <inertial><mass value="0.005"/><inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/></inertial>
    <visual><geometry><sphere radius="0.016"/></geometry><material name="eye_cyan"/></visual>
  </link>
  <joint name="left_eye_joint" type="fixed">
    <parent link="visor_link"/><child link="left_eye"/><origin xyz="0.008 0.045 0.005"/>
  </joint>
  <link name="right_eye">
    <inertial><mass value="0.005"/><inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/></inertial>
    <visual><geometry><sphere radius="0.016"/></geometry><material name="eye_cyan"/></visual>
  </link>
  <joint name="right_eye_joint" type="fixed">
    <parent link="visor_link"/><child link="right_eye"/><origin xyz="0.008 -0.045 0.005"/>
  </joint>

  <!-- 11-13. left arm: metal shoulder joint + upper arm, forearm, gripper -->
  <link name="left_shoulder">
    <inertial><mass value="0.2"/><inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/></inertial>
    <visual name="joint_ball"><geometry><sphere radius="0.045"/></geometry><material name="metal"/></visual>
    <visual name="upper_arm">
      <origin xyz="0 0 -0.08"/>
      <geometry><cylinder radius="0.026" length="0.15"/></geometry>
      <material name="graphite"/>
    </visual>
  </link>
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/><child link="left_shoulder"/>
    <origin xyz="0.03 0.20 0.38"/><axis xyz="0 1 0"/>
    <limit lower="-2.8" upper="0.8" effort="30" velocity="4"/>
  </joint>
  <link name="left_forearm">
    <inertial><mass value="0.15"/><inertia ixx="0.0004" ixy="0" ixz="0" iyy="0.0004" iyz="0" izz="0.0004"/></inertial>
    <visual name="elbow"><geometry><sphere radius="0.03"/></geometry><material name="metal"/></visual>
    <visual name="forearm">
      <origin xyz="0 0 -0.075"/>
      <geometry><cylinder radius="0.024" length="0.14"/></geometry>
      <material name="ivory"/>
    </visual>
  </link>
  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder"/><child link="left_forearm"/>
    <origin xyz="0 0 -0.16"/><axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="0.3" effort="20" velocity="4"/>
  </joint>
  <link name="left_gripper">
    <inertial><mass value="0.05"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual name="palm">
      <origin xyz="0 0 -0.02"/>
      <geometry><box size="0.055 0.035 0.05"/></geometry>
      <material name="gripper_orange"/>
    </visual>
    <visual name="finger_front">
      <origin xyz="0.018 -0.012 -0.055"/>
      <geometry><box size="0.012 0.012 0.04"/></geometry>
      <material name="gripper_orange"/>
    </visual>
    <visual name="finger_back">
      <origin xyz="-0.018 -0.012 -0.055"/>
      <geometry><box size="0.012 0.012 0.04"/></geometry>
      <material name="gripper_orange"/>
    </visual>
  </link>
  <joint name="left_gripper_joint" type="fixed">
    <parent link="left_forearm"/><child link="left_gripper"/><origin xyz="0 0 -0.15"/>
  </joint>

  <!-- 14-16. right arm -->
  <link name="right_shoulder">
    <inertial><mass value="0.2"/><inertia ixx="0.0005" ixy="0" ixz="0" iyy="0.0005" iyz="0" izz="0.0005"/></inertial>
    <visual name="joint_ball"><geometry><sphere radius="0.045"/></geometry><material name="metal"/></visual>
    <visual name="upper_arm">
      <origin xyz="0 0 -0.08"/>
      <geometry><cylinder radius="0.026" length="0.15"/></geometry>
      <material name="graphite"/>
    </visual>
  </link>
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/><child link="right_shoulder"/>
    <origin xyz="0.03 -0.20 0.38"/><axis xyz="0 1 0"/>
    <limit lower="-2.8" upper="0.8" effort="30" velocity="4"/>
  </joint>
  <link name="right_forearm">
    <inertial><mass value="0.15"/><inertia ixx="0.0004" ixy="0" ixz="0" iyy="0.0004" iyz="0" izz="0.0004"/></inertial>
    <visual name="elbow"><geometry><sphere radius="0.03"/></geometry><material name="metal"/></visual>
    <visual name="forearm">
      <origin xyz="0 0 -0.075"/>
      <geometry><cylinder radius="0.024" length="0.14"/></geometry>
      <material name="ivory"/>
    </visual>
  </link>
  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_shoulder"/><child link="right_forearm"/>
    <origin xyz="0 0 -0.16"/><axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="0.3" effort="20" velocity="4"/>
  </joint>
  <link name="right_gripper">
    <inertial><mass value="0.05"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual name="palm">
      <origin xyz="0 0 -0.02"/>
      <geometry><box size="0.055 0.035 0.05"/></geometry>
      <material name="gripper_orange"/>
    </visual>
    <visual name="finger_front">
      <origin xyz="0.018 0.012 -0.055"/>
      <geometry><box size="0.012 0.012 0.04"/></geometry>
      <material name="gripper_orange"/>
    </visual>
    <visual name="finger_back">
      <origin xyz="-0.018 0.012 -0.055"/>
      <geometry><box size="0.012 0.012 0.04"/></geometry>
      <material name="gripper_orange"/>
    </visual>
  </link>
  <joint name="right_gripper_joint" type="fixed">
    <parent link="right_forearm"/><child link="right_gripper"/><origin xyz="0 0 -0.15"/>
  </joint>
</robot>
""".format(**{k: _rgba(k) for k in ("ivory", "graphite", "visor_black", "eye_cyan", "metal", "gripper_orange")})


def write_humanoid_urdf(directory=ASSET_DIR):
    """Writes HUMANOID_URDF to a cache file (loadURDF needs a path)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "hackoween_humanoid.urdf")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(HUMANOID_URDF)
    os.replace(tmp, path)
    return path


def load_humanoid(x=0.0, y=0.0, yaw_rad=0.0):
    """Native URDF load, fixed base (pose is driven kinematically by the driver)."""
    path = write_humanoid_urdf()
    orn = p.getQuaternionFromEuler([0, 0, yaw_rad])
    return p.loadURDF(path, basePosition=[x, y, 0.0], baseOrientation=orn, useFixedBase=True)


# ---------------------------------------------------------------------------
# Arena construction
# ---------------------------------------------------------------------------
def _walls():
    """(centre_x, centre_y, half_x, half_y) for the four walls."""
    h, t = HALF_ROOM, WALL_HALF_THICK
    return [
        (0.0, h, h, t),    # north (+y)
        (0.0, -h, h, t),   # south (-y)
        (h, 0.0, t, h),    # east  (+x)
        (-h, 0.0, t, h),   # west  (-x)
    ]


def _wall_frames():
    """For each wall's inner face: origin (left end, z=0), along-axis, inward normal, length."""
    w = WALL_INNER
    return [
        ((-w, w, 0.0), (1, 0, 0), (0, -1, 0), 2 * w),
        ((w, -w, 0.0), (-1, 0, 0), (0, 1, 0), 2 * w),
        ((w, w, 0.0), (0, -1, 0), (-1, 0, 0), 2 * w),
        ((-w, -w, 0.0), (0, 1, 0), (1, 0, 0), 2 * w),
    ]


def _wall_point(frame, u, z, offset):
    (ox, oy, _), a, n, _ = frame
    return (ox + a[0] * u + n[0] * offset, oy + a[1] * u + n[1] * offset, z)


def _wall_rect(mesh, frame, u0, u1, z0, z1, offset):
    n = frame[2]
    mesh.add_quad(
        _wall_point(frame, u0, z0, offset), _wall_point(frame, u1, z0, offset),
        _wall_point(frame, u1, z1, offset), _wall_point(frame, u0, z1, offset),
        normal_hint=n,
    )


def _bat_outline():
    """Bat silhouette as (x, y) points, star-shaped around (0, 0); wingspan ~0.76."""
    right = [  # (angle_deg, radius) from the body centre, going clockwise
        (88, 0.050), (72, 0.120), (55, 0.060), (38, 0.110), (27, 0.250),
        (19, 0.380), (11, 0.330), (4, 0.250), (-3, 0.290), (-11, 0.180),
        (-19, 0.200), (-34, 0.090), (-60, 0.075), (-90, 0.095),
    ]
    pts = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a, r in right]
    left = [(-x, y) for x, y in reversed(pts[:-1])]
    return pts + left


def _build_floor():
    col = p.createCollisionShape(p.GEOM_PLANE)
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[HALF_ROOM, HALF_ROOM, 0.005],
        visualFramePosition=[0, 0, -0.005], rgbaColor=list(PALETTE["floor_slate"]),
    )
    floor_id = p.createMultiBody(0, col, vis, basePosition=[0, 0, 0])
    p.changeDynamics(floor_id, -1, lateralFriction=1.0)

    grid = Mesh("floor_grid")
    up = (0, 0, 1)
    z = 0.002
    for i in range(-4, 5):
        c = i * 0.5
        grid.add_segment((c, -WALL_INNER, z), (c, WALL_INNER, z), 0.012, up)
        grid.add_segment((-WALL_INNER, c, z), (WALL_INNER, c, z), 0.012, up)
    spawn_visual_mesh(grid, PALETTE["floor_grid"])
    return floor_id


def _build_walls():
    ids = []
    for wx, wy, hx, hy in _walls():
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, WALL_HEIGHT / 2.0])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, WALL_HEIGHT / 2.0], rgbaColor=list(PALETTE["wall"]))
        ids.append(p.createMultiBody(0, col, vis, basePosition=[wx, wy, WALL_HEIGHT / 2.0]))
    return ids


def _build_wall_dressing():
    panels, trims, bats = Mesh("wall_panels"), Mesh("wall_trims"), Mesh("wall_bats")
    bat = _bat_outline()
    # (u fraction along wall, z centre, scale, tilt deg) — two bats per wall
    bat_layout = [
        [(0.28, 1.02, 0.72, 8), (0.66, 1.10, 0.52, -12)],
        [(0.35, 1.08, 0.62, -6), (0.74, 0.98, 0.46, 14)],
        [(0.22, 1.10, 0.50, 10), (0.58, 1.00, 0.70, -4)],
        [(0.40, 1.04, 0.60, 6), (0.80, 1.12, 0.44, -10)],
    ]
    panel_w, gap = 0.55, 0.10
    for wall_index, frame in enumerate(_wall_frames()):
        length = frame[3]
        count = int((length + gap) // (panel_w + gap))
        margin = (length - (count * (panel_w + gap) - gap)) / 2.0
        for k in range(count):
            u0 = margin + k * (panel_w + gap)
            _wall_rect(panels, frame, u0, u0 + panel_w, 0.16, 1.18, 0.004)
        # glowing trims: skirting band + top band
        _wall_rect(trims, frame, 0.0, length, 0.0, 0.07, 0.008)
        _wall_rect(trims, frame, 0.0, length, 1.26, 1.32, 0.008)
        for uf, zc, sc, tilt in bat_layout[wall_index]:
            t = math.radians(tilt)
            ring = []
            for bx, by in bat:
                rx = (bx * math.cos(t) - by * math.sin(t)) * sc
                ry = (bx * math.sin(t) + by * math.cos(t)) * sc
                ring.append(_wall_point(frame, uf * length + rx, zc + ry, 0.010))
            bats.add_fan(_wall_point(frame, uf * length, zc, 0.010), ring, frame[2])
    spawn_visual_mesh(panels, PALETTE["wall_panel"])
    spawn_visual_mesh(trims, PALETTE["trim_glow"])
    spawn_visual_mesh(bats, PALETTE["bat"])


def _build_spider_webs():
    """A classic corner web spanning each of the four upper corners."""
    web = Mesh("spider_webs")
    w, top, size = WALL_INNER - 0.002, WALL_HEIGHT - 0.01, 0.62
    room_centre = (0.0, 0.0, 0.7)
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx * w, sy * w, top)
            anchors = [(c[0] - sx * size, c[1], top), (c[0], c[1] - sy * size, top), (c[0], c[1], top - size)]
            n = _plane_normal(anchors, room_centre)
            hub = tuple(sum(a[i] for a in anchors) / 3.0 for i in range(3))
            perimeter = []
            for i in range(3):
                a, b = anchors[i], anchors[(i + 1) % 3]
                for t in (0.0, 1.0 / 3.0, 2.0 / 3.0):
                    perimeter.append(tuple(a[j] + (b[j] - a[j]) * t for j in range(3)))
            for r in perimeter:
                web.add_segment(hub, r, 0.007, n)
            web.add_polyline(anchors, 0.007, n, closed=True)
            for f in (0.22, 0.42, 0.62, 0.82):
                ring = [tuple(hub[j] + (r[j] - hub[j]) * f for j in range(3)) for r in perimeter]
                for i in range(len(ring)):
                    a, b = ring[i], ring[(i + 1) % len(ring)]
                    mid = tuple((a[j] + b[j]) / 2.0 for j in range(3))
                    sag = tuple(mid[j] + (hub[j] - mid[j]) * 0.10 for j in range(3))
                    web.add_polyline([a, sag, b], 0.005, n)
    spawn_visual_mesh(web, PALETTE["web"])


def _plane_normal(pts, towards):
    a, b, c = pts
    u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    d = (towards[0] - a[0], towards[1] - a[1], towards[2] - a[2])
    if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] < 0:
        n = (-n[0], -n[1], -n[2])
    length = math.sqrt(sum(x * x for x in n))
    return (n[0] / length, n[1] / length, n[2] / length)


def build_arena():
    """Floor, walls and all wall dressing. Returns dict of collidable ids."""
    floor_id = _build_floor()
    wall_ids = _build_walls()
    _build_wall_dressing()
    _build_spider_webs()
    return {"floor": floor_id, "walls": wall_ids}


def build_drop_zone(centre=DROP_ZONE_POS):
    """Orange platform + flat jack-o'-lantern decal. Visual only (no collision).

    The jack-o'-lantern is painted flat on the platform on purpose: a 3D
    pumpkin prop would be a round orange object on the floor, which the
    vision agent could mistake for a fetch target ("the round thing").
    """
    cx, cy = centre[0], centre[1]
    half = DROP_ZONE_HALF
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[half, half, 0.006], rgbaColor=list(PALETTE["zone"])
    )
    zone_id = p.createMultiBody(0, -1, vis, basePosition=[cx, cy, 0.006])

    upn = (0, 0, 1)
    border = Mesh("zone_border")
    bw, zb = 0.035, 0.014
    corners = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    inner = [(cx - half + bw, cy - half + bw), (cx + half - bw, cy - half + bw),
             (cx + half - bw, cy + half - bw), (cx - half + bw, cy + half - bw)]
    for i in range(4):
        j = (i + 1) % 4
        border.add_quad((corners[i][0], corners[i][1], zb), (corners[j][0], corners[j][1], zb),
                        (inner[j][0], inner[j][1], zb), (inner[i][0], inner[i][1], zb), upn)
    spawn_visual_mesh(border, PALETTE["zone_border"])

    # Decal frame: "up" points away from the arena centre, so the face reads
    # upright for a robot approaching from the middle of the room.
    d = math.hypot(cx, cy) or 1.0
    up = (cx / d, cy / d)
    right = (up[1], -up[0])

    def world(lx, ly, z):
        return (cx + right[0] * lx + up[0] * ly, cy + right[1] * lx + up[1] * ly, z)

    body, ribs, face, stem = Mesh("pumpkin_body"), Mesh("pumpkin_ribs"), Mesh("pumpkin_face"), Mesh("pumpkin_stem")
    ring = []
    for k in range(48):
        a = 2 * math.pi * k / 48
        lobe = 1.0 + 0.045 * math.cos(6 * a)
        ring.append(world(0.25 * lobe * math.cos(a), 0.205 * lobe * math.sin(a), 0.016))
    body.add_fan(world(0, 0, 0.016), ring, upn)
    for lx in (-0.16, -0.08, 0.08, 0.16):
        pts = []
        for k in range(9):
            ly = -0.19 + 0.38 * k / 8
            bulge = math.sqrt(max(0.0, 1 - (ly / 0.205) ** 2))
            pts.append(world(lx * bulge, ly * 0.97, 0.018))
        ribs.add_polyline(pts, 0.008, upn)
    for sgn in (-1, 1):
        face.add_triangle(world(sgn * 0.13, 0.02, 0.020), world(sgn * 0.04, 0.02, 0.020),
                          world(sgn * 0.085, 0.10, 0.020), upn)
    face.add_triangle(world(-0.025, -0.005, 0.020), world(0.025, -0.005, 0.020), world(0, 0.035, 0.020), upn)
    mouth = []
    xs = [0.15 - 0.3 * k / 12 for k in range(13)]   # upper edge, right -> left
    for x in xs:
        y = -0.03 - 0.03 * (1 - (x / 0.15) ** 2)
        if abs(abs(x) - 0.05) < 0.013:
            y -= 0.024   # teeth notches
        mouth.append(world(x, y, 0.020))
    for x in reversed(xs[1:-1]):                     # lower edge, left -> right
        mouth.append(world(x, -0.045 - 0.075 * (1 - (x / 0.15) ** 2), 0.020))
    face.add_fan(world(0, -0.09, 0.020), mouth, upn)
    stem.add_quad(world(-0.03, 0.19, 0.018), world(0.03, 0.19, 0.018),
                  world(0.045, 0.27, 0.018), world(-0.005, 0.28, 0.018), upn)
    for mesh, colour in ((body, "pumpkin"), (ribs, "pumpkin_rib"), (face, "pumpkin_face"), (stem, "pumpkin_stem")):
        spawn_visual_mesh(mesh, PALETTE[colour])
    return zone_id


def build_obstacle():
    """Internal partition wall for the stretch-goal obstacle mode."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.5])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.8, 0.5], rgbaColor=list(PALETTE["obstacle"]))
    body = p.createMultiBody(0, col, vis, basePosition=[0.9, 0.9, 0.5])
    trim = Mesh("obstacle_trim")
    for sx in (-1, 1):
        x = 0.9 + sx * 0.102
        n = (sx, 0, 0)
        trim.add_quad((x, 0.1, 0.0), (x, 1.7, 0.0), (x, 1.7, 0.06), (x, 0.1, 0.06), n)
        trim.add_quad((x, 0.1, 0.92), (x, 1.7, 0.92), (x, 1.7, 0.97), (x, 0.1, 0.97), n)
    spawn_visual_mesh(trim, PALETTE["trim_glow"])
    return body


OBSTACLE_FOOTPRINT = (0.9, 0.9, 0.1, 0.8)   # centre x, centre y, half x, half y

# Alias for preview.py compatibility
SceneManager = getattr(globals(), "SceneManager", None)
if SceneManager is None:
    import sys
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and name != "SceneManager" and ("Scene" in name or "Manager" in name):
            SceneManager = obj
            break
    if SceneManager is None:
        for name, obj in list(globals().items()):
            if isinstance(obj, type) and name != "SceneManager":
                SceneManager = obj
                break

# ==========================================
# Comprehensive Compatibility Aliases for preview.py
# ==========================================
def build_room(*args, **kwargs):
    return build_arena()

def spawn_robot(*args, **kwargs):
    return load_humanoid()

def spawn_drop_zone(*args, **kwargs):
    if args:
        return build_drop_zone(args[0])
    return build_drop_zone(DROP_ZONE_POS)

def spawn_objects_func(*args, **kwargs):
    return spawn_objects()

class SceneManager:
    def __init__(self, *args, **kwargs):
        pass
    
    @staticmethod
    def build_room(*args, **kwargs):
        return build_arena()

    @staticmethod
    def spawn_robot(*args, **kwargs):
        return load_humanoid()

    @staticmethod
    def spawn_drop_zone(*args, **kwargs):
        if args:
            return build_drop_zone(args[0])
        return build_drop_zone(DROP_ZONE_POS)

    @staticmethod
    def spawn_objects(*args, **kwargs):
        return spawn_objects()