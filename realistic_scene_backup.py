import mujoco
import mujoco.viewer
import time
import math


# ============================================================
# LARGE REALISTIC CITY DRONE SIMULATION
# ============================================================

xml = r"""
<mujoco model="large_realistic_city">

    <option gravity="0 0 0"/>

    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <headlight ambient="0.45 0.45 0.45"
                   diffuse="0.65 0.65 0.65"
                   specular="0.25 0.25 0.25"/>
    </visual>


    <worldbody>

        <!-- ================================================= -->
        <!-- SUN -->
        <!-- ================================================= -->

        <light name="sun"
               pos="-10 -15 30"
               dir="0.3 0.3 -1"
               directional="true"
               diffuse="1 1 1"
               specular="0.35 0.35 0.35"/>


        <light name="city_light"
               pos="20 10 20"
               dir="-0.5 -0.3 -1"
               directional="true"/>


        <!-- ================================================= -->
        <!-- HUGE GROUND -->
        <!-- ================================================= -->

        <geom name="ground"
              type="plane"
              size="50 40 0.1"
              rgba="0.38 0.40 0.35 1"/>


        <!-- ================================================= -->
        <!-- MAIN CITY ROADS -->
        <!-- ================================================= -->

        <!-- horizontal roads -->

        <geom type="box"
              pos="0 -12 0.03"
              size="45 3 0.03"
              rgba="0.10 0.10 0.11 1"/>

        <geom type="box"
              pos="0 0 0.03"
              size="45 3 0.03"
              rgba="0.10 0.10 0.11 1"/>

        <geom type="box"
              pos="0 12 0.03"
              size="45 3 0.03"
              rgba="0.10 0.10 0.11 1"/>


        <!-- vertical roads -->

        <geom type="box"
              pos="-18 0 0.03"
              size="3 35 0.03"
              rgba="0.10 0.10 0.11 1"/>

        <geom type="box"
              pos="-6 0 0.03"
              size="3 35 0.03"
              rgba="0.10 0.10 0.11 1"/>

        <geom type="box"
              pos="6 0 0.03"
              size="3 35 0.03"
              rgba="0.10 0.10 0.11 1"/>

        <geom type="box"
              pos="18 0 0.03"
              size="3 35 0.03"
              rgba="0.10 0.10 0.11 1"/>


        <!-- ================================================= -->
        <!-- ROAD CENTER LINES -->
        <!-- ================================================= -->

        <geom type="box" pos="-12 -12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="-8 -12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="-4 -12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="0 -12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="4 -12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="8 -12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="12 -12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>


        <geom type="box" pos="-12 0 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="-8 0 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="-4 0 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="0 0 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="4 0 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="8 0 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="12 0 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>


        <geom type="box" pos="-12 12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="-8 12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="-4 12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="0 12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="4 12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="8 12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>

        <geom type="box" pos="12 12 0.07"
              size="1.2 0.05 0.02"
              rgba="1 1 1 1"/>


        <!-- ================================================= -->
        <!-- BUILDINGS -->
        <!-- ================================================= -->

        <!-- Block 1 -->

        <body name="building1" pos="-12 -6 0">
            <geom type="box"
                  pos="0 0 4"
                  size="3 2.2 4"
                  rgba="0.48 0.50 0.53 1"/>

            <geom type="box"
                  pos="-1.4 -2.25 5.5"
                  size="0.45 0.04 0.55"
                  rgba="0.08 0.35 0.65 1"/>

            <geom type="box"
                  pos="0 -2.25 5.5"
                  size="0.45 0.04 0.55"
                  rgba="0.08 0.35 0.65 1"/>

            <geom type="box"
                  pos="1.4 -2.25 5.5"
                  size="0.45 0.04 0.55"
                  rgba="0.08 0.35 0.65 1"/>

            <geom type="box"
                  pos="-1.4 -2.25 3"
                  size="0.45 0.04 0.55"
                  rgba="0.08 0.35 0.65 1"/>

            <geom type="box"
                  pos="0 -2.25 3"
                  size="0.45 0.04 0.55"
                  rgba="0.08 0.35 0.65 1"/>

            <geom type="box"
                  pos="1.4 -2.25 3"
                  size="0.45 0.04 0.55"
                  rgba="0.08 0.35 0.65 1"/>
        </body>


        <!-- Block 2 -->

        <body name="building2" pos="0 -6 0">
            <geom type="box"
                  pos="0 0 6"
                  size="2.5 2.2 6"
                  rgba="0.42 0.45 0.49 1"/>

            <geom type="box"
                  pos="-1.1 -2.25 7"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.40 0.72 1"/>

            <geom type="box"
                  pos="0 -2.25 7"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.40 0.72 1"/>

            <geom type="box"
                  pos="1.1 -2.25 7"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.40 0.72 1"/>
        </body>


        <!-- Block 3 -->

        <body name="building3" pos="12 -6 0">
            <geom type="box"
                  pos="0 0 3"
                  size="3 2.2 3"
                  rgba="0.55 0.48 0.43 1"/>

            <geom type="box"
                  pos="-1.3 -2.25 4"
                  size="0.45 0.04 0.5"
                  rgba="0.10 0.38 0.68 1"/>

            <geom type="box"
                  pos="0 -2.25 4"
                  size="0.45 0.04 0.5"
                  rgba="0.10 0.38 0.68 1"/>

            <geom type="box"
                  pos="1.3 -2.25 4"
                  size="0.45 0.04 0.5"
                  rgba="0.10 0.38 0.68 1"/>
        </body>


        <!-- Block 4 -->

        <body name="building4" pos="-12 6 0">
            <geom type="box"
                  pos="0 0 5"
                  size="2.8 2.4 5"
                  rgba="0.46 0.48 0.51 1"/>

            <geom type="box"
                  pos="-1.2 2.45 6"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.40 0.70 1"/>

            <geom type="box"
                  pos="0 2.45 6"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.40 0.70 1"/>

            <geom type="box"
                  pos="1.2 2.45 6"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.40 0.70 1"/>
        </body>


        <!-- Block 5 -->

        <body name="building5" pos="0 6 0">
            <geom type="box"
                  pos="0 0 3.5"
                  size="2.2 2.4 3.5"
                  rgba="0.58 0.50 0.45 1"/>

            <geom type="box"
                  pos="-0.9 2.45 4.5"
                  size="0.35 0.04 0.45"
                  rgba="0.08 0.35 0.65 1"/>

            <geom type="box"
                  pos="0 2.45 4.5"
                  size="0.35 0.04 0.45"
                  rgba="0.08 0.35 0.65 1"/>

            <geom type="box"
                  pos="0.9 2.45 4.5"
                  size="0.35 0.04 0.45"
                  rgba="0.08 0.35 0.65 1"/>
        </body>


        <!-- Block 6 -->

        <body name="building6" pos="12 6 0">
            <geom type="box"
                  pos="0 0 6.5"
                  size="2.8 2.4 6.5"
                  rgba="0.40 0.44 0.48 1"/>

            <geom type="box"
                  pos="-1.2 2.45 8"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.42 0.72 1"/>

            <geom type="box"
                  pos="0 2.45 8"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.42 0.72 1"/>

            <geom type="box"
                  pos="1.2 2.45 8"
                  size="0.4 0.04 0.5"
                  rgba="0.08 0.42 0.72 1"/>
        </body>


        <!-- ================================================= -->
        <!-- TREES -->
        <!-- ================================================= -->

        <!-- Tree 1 -->

        <body name="tree1" pos="-20 -8 0">

            <geom type="cylinder"
                  pos="0 0 1.5"
                  size="0.35 1.5"
                  rgba="0.30 0.14 0.05 1"/>

            <geom type="capsule"
                  fromto="0 0 2 -0.8 0 3"
                  size="0.11"
                  rgba="0.25 0.11 0.04 1"/>

            <geom type="capsule"
                  fromto="0 0 2.3 0.8 0 3.2"
                  size="0.11"
                  rgba="0.25 0.11 0.04 1"/>

            <geom type="sphere"
                  pos="0 0 3.5"
                  size="1.3"
                  rgba="0.035 0.38 0.05 1"/>

            <geom type="sphere"
                  pos="0.8 0 3.2"
                  size="0.8"
                  rgba="0.04 0.46 0.06 1"/>

            <geom type="sphere"
                  pos="-0.8 0 3.2"
                  size="0.8"
                  rgba="0.04 0.43 0.05 1"/>
        </body>


        <!-- Tree 2 -->

        <body name="tree2" pos="-3 -16 0">

            <geom type="cylinder"
                  pos="0 0 1.4"
                  size="0.32 1.4"
                  rgba="0.30 0.14 0.05 1"/>

            <geom type="sphere"
                  pos="0 0 3.3"
                  size="1.25"
                  rgba="0.04 0.40 0.06 1"/>

            <geom type="sphere"
                  pos="0.75 0 3.1"
                  size="0.75"
                  rgba="0.05 0.48 0.07 1"/>

            <geom type="sphere"
                  pos="-0.75 0 3.1"
                  size="0.75"
                  rgba="0.04 0.44 0.06 1"/>
        </body>


        <!-- Tree 3 -->

        <body name="tree3" pos="20 -5 0">

            <geom type="cylinder"
                  pos="0 0 1.5"
                  size="0.35 1.5"
                  rgba="0.30 0.14 0.05 1"/>

            <geom type="sphere"
                  pos="0 0 3.6"
                  size="1.3"
                  rgba="0.04 0.40 0.05 1"/>

            <geom type="sphere"
                  pos="0.8 0 3.3"
                  size="0.8"
                  rgba="0.05 0.47 0.06 1"/>

            <geom type="sphere"
                  pos="-0.8 0 3.3"
                  size="0.8"
                  rgba="0.04 0.43 0.06 1"/>
        </body>


        <!-- Tree 4 -->

        <body name="tree4" pos="20 10 0">

            <geom type="cylinder"
                  pos="0 0 1.6"
                  size="0.35 1.6"
                  rgba="0.28 0.13 0.05 1"/>

            <geom type="sphere"
                  pos="0 0 3.7"
                  size="1.35"
                  rgba="0.035 0.40 0.05 1"/>

            <geom type="sphere"
                  pos="0.8 0 3.4"
                  size="0.8"
                  rgba="0.05 0.47 0.06 1"/>

            <geom type="sphere"
                  pos="-0.8 0 3.4"
                  size="0.8"
                  rgba="0.04 0.44 0.06 1"/>
        </body>


        <!-- ================================================= -->
        <!-- CONSTRUCTION SITE -->
        <!-- ================================================= -->

        <geom type="box"
              pos="15 12 0.12"
              size="4 3 0.12"
              rgba="0.55 0.38 0.18 1"/>


        <!-- Construction materials -->

        <geom type="box"
              pos="13 12 0.7"
              size="0.7 0.7 0.7"
              rgba="0.58 0.35 0.12 1"/>

        <geom type="box"
              pos="15 12 0.7"
              size="0.7 0.7 0.7"
              rgba="0.58 0.35 0.12 1"/>

        <geom type="box"
              pos="17 12 0.7"
              size="0.7 0.7 0.7"
              rgba="0.58 0.35 0.12 1"/>


        <!-- ================================================= -->
        <!-- LARGE CRANE -->
        <!-- ================================================= -->

        <body name="crane" pos="15 12 0">

            <!-- Tower -->

            <geom type="box"
                  pos="0 0 5"
                  size="0.3 0.3 5"
                  rgba="0.95 0.62 0.03 1"/>

            <!-- tower supports -->

            <geom type="capsule"
                  fromto="0 0 1 1.2 0 3"
                  size="0.08"
                  rgba="0.85 0.50 0.02 1"/>

            <geom type="capsule"
                  fromto="0 0 1 -1.2 0 3"
                  size="0.08"
                  rgba="0.85 0.50 0.02 1"/>

            <geom type="capsule"
                  fromto="0 0 3 1.2 0 5"
                  size="0.08"
                  rgba="0.85 0.50 0.02 1"/>

            <geom type="capsule"
                  fromto="0 0 3 -1.2 0 5"
                  size="0.08"
                  rgba="0.85 0.50 0.02 1"/>

            <!-- Main boom -->

            <geom type="box"
                  pos="4 0 9.7"
                  size="4 0.20 0.20"
                  rgba="1.0 0.65 0.02 1"/>

            <!-- Counter boom -->

            <geom type="box"
                  pos="-2 0 9.7"
                  size="1.8 0.20 0.20"
                  rgba="0.90 0.53 0.02 1"/>

            <!-- Boom support -->

            <geom type="capsule"
                  fromto="0 0 5 4 0 9.7"
                  size="0.09"
                  rgba="0.88 0.52 0.02 1"/>

            <!-- Cable -->

            <geom type="capsule"
                  fromto="7.5 0 9.5 7.5 0 5"
                  size="0.04"
                  rgba="0.08 0.08 0.08 1"/>

            <!-- Load -->

            <geom type="box"
                  pos="7.5 0 4.5"
                  size="0.55 0.55 0.55"
                  rgba="0.55 0.30 0.08 1"/>

        </body>


        <!-- ================================================= -->
        <!-- BIRDS -->
        <!-- ================================================= -->

        <body name="bird1" pos="-5 -2 10">

            <freejoint/>

            <geom type="ellipsoid"
                  size="0.28 0.13 0.13"
                  rgba="0.05 0.05 0.05 1"/>

            <geom type="capsule"
                  fromto="-0.1 0 0 -0.65 0.35 0"
                  size="0.045"
                  rgba="0.08 0.08 0.08 1"/>

            <geom type="capsule"
                  fromto="0.1 0 0 0.65 0.35 0"
                  size="0.045"
                  rgba="0.08 0.08 0.08 1"/>
        </body>


        <body name="bird2" pos="7 3 12">

            <freejoint/>

            <geom type="ellipsoid"
                  size="0.25 0.12 0.12"
                  rgba="0.06 0.06 0.06 1"/>

            <geom type="capsule"
                  fromto="-0.1 0 0 -0.6 0.32 0"
                  size="0.04"
                  rgba="0.08 0.08 0.08 1"/>

            <geom type="capsule"
                  fromto="0.1 0 0 0.6 0.32 0"
                  size="0.04"
                  rgba="0.08 0.08 0.08 1"/>
        </body>


        <body name="bird3" pos="16 -3 9">

            <freejoint/>

            <geom type="ellipsoid"
                  size="0.27 0.13 0.13"
                  rgba="0.05 0.05 0.05 1"/>

            <geom type="capsule"
                  fromto="-0.1 0 0 -0.65 0.35 0"
                  size="0.045"
                  rgba="0.08 0.08 0.08 1"/>

            <geom type="capsule"
                  fromto="0.1 0 0 0.65 0.35 0"
                  size="0.045"
                  rgba="0.08 0.08 0.08 1"/>
        </body>


        <!-- ================================================= -->
        <!-- TARGET -->
        <!-- ================================================= -->

        <geom name="target"
              pos="22 16 8"
              type="sphere"
              size="0.7"
              rgba="0.05 1 0.12 1"/>

        <geom type="cylinder"
              pos="22 16 0.08"
              size="1.7 0.05"
              rgba="0.05 1 0.12 1"/>

    </worldbody>

</mujoco>
"""


# ============================================================
# LOAD MUJOCO
# ============================================================

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


# ============================================================
# BODY IDs
# ============================================================

drone_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "drone"
)

bird1_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "bird1"
)

bird2_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "bird2"
)

bird3_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "bird3"
)


# ============================================================
# BIRD QPOS
# ============================================================

bird1_q = model.jnt_qposadr[model.body_jntadr[bird1_id]]
bird2_q = model.jnt_qposadr[model.body_jntadr[bird2_id]]
bird3_q = model.jnt_qposadr[model.body_jntadr[bird3_id]]


# ============================================================
# DRONE FLIGHT PATH
# ============================================================

waypoints = [
    (-22, -18, 8),
    (-18, -12, 8),
    (-12, -8, 8),
    (-6, -5, 8),
    (0, -3, 8),
    (6, 1, 8),
    (10, 5, 8),
    (15, 9, 8),
    (19, 13, 8),
    (22, 16, 8)
]


# ============================================================
# SET INITIAL DRONE POSITION
# ============================================================

data.qpos[0] = waypoints[0][0]
data.qpos[1] = waypoints[0][1]
data.qpos[2] = waypoints[0][2]

mujoco.mj_forward(model, data)


# ============================================================
# SMOOTH MOVEMENT FUNCTION
# ============================================================

def move_towards(current, target, speed):

    dx = target[0] - current[0]
    dy = target[1] - current[1]
    dz = target[2] - current[2]

    distance = math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )

    if distance < speed:
        return target

    return (
        current[0] + dx / distance * speed,
        current[1] + dy / distance * speed,
        current[2] + dz / distance * speed
    )


# ============================================================
# SIMULATION
# ============================================================

print()
print("============================================================")
print("           LARGE REALISTIC DRONE CITY")
print("============================================================")
print()
print("Drone       : BLUE")
print("Buildings   : CITY BLOCKS")
print("Trees       : GREEN")
print("Crane       : CONSTRUCTION AREA")
print("Birds       : MOVING")
print("Roads       : CITY GRID")
print("Target      : GREEN")
print()
print("Drone starts from the corner.")
print("Drone follows a smooth multi-waypoint flight path.")
print("============================================================")
print()


with mujoco.viewer.launch_passive(model, data) as viewer:

    # ========================================================
    # WIDE CAMERA
    # ========================================================

    viewer.cam.lookat[:] = [0, 0, 5]
    viewer.cam.distance = 32
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25


    current_waypoint = 1
    speed = 0.045

    start_time = time.time()


    while viewer.is_running():

        elapsed = time.time() - start_time


        # ====================================================
        # DRONE POSITION
        # ====================================================

        drone_pos = data.xpos[drone_id]

        current = (
            drone_pos[0],
            drone_pos[1],
            drone_pos[2]
        )


        # ====================================================
        # MOVE DRONE THROUGH WAYPOINTS
        # ====================================================

        if current_waypoint < len(waypoints):

            target_point = waypoints[current_waypoint]

            new_position = move_towards(
                current,
                target_point,
                speed
            )

            data.qpos[0] = new_position[0]
            data.qpos[1] = new_position[1]
            data.qpos[2] = new_position[2]


            distance_to_waypoint = math.sqrt(
                (new_position[0] - target_point[0]) ** 2 +
                (new_position[1] - target_point[1]) ** 2 +
                (new_position[2] - target_point[2]) ** 2
            )


            if distance_to_waypoint < 0.08:

                current_waypoint += 1

                print(
                    f"\nWaypoint {current_waypoint} reached"
                )


        # ====================================================
        # BIRD 1
        # ====================================================

        data.qpos[bird1_q] = (
            -5 + 4.0 * math.sin(elapsed * 0.45)
        )

        data.qpos[bird1_q + 1] = (
            -2 + 2.5 * math.cos(elapsed * 0.55)
        )

        data.qpos[bird1_q + 2] = (
            10 + 0.6 * math.sin(elapsed * 1.5)
        )


        # ====================================================
        # BIRD 2
        # ====================================================

        data.qpos[bird2_q] = (
            7 + 4.0 * math.sin(elapsed * 0.35)
        )

        data.qpos[bird2_q + 1] = (
            3 + 2.5 * math.cos(elapsed * 0.45)
        )

        data.qpos[bird2_q + 2] = (
            12 + 0.7 * math.sin(elapsed * 1.2)
        )


        # ====================================================
        # BIRD 3
        # ====================================================

        data.qpos[bird3_q] = (
            16 + 3.0 * math.sin(elapsed * 0.5)
        )

        data.qpos[bird3_q + 1] = (
            -3 + 3.0 * math.cos(elapsed * 0.4)
        )

        data.qpos[bird3_q + 2] = (
            9 + 0.5 * math.sin(elapsed * 1.7)
        )


        # ====================================================
        # UPDATE MUJOCO
        # ====================================================

        mujoco.mj_forward(model, data)


        # ====================================================
        # DISPLAY
        # ====================================================

        drone_pos = data.xpos[drone_id]

        target = waypoints[-1]

        distance_to_target = math.sqrt(
            (drone_pos[0] - target[0]) ** 2 +
            (drone_pos[1] - target[1]) ** 2 +
            (drone_pos[2] - target[2]) ** 2
        )


        print(
            f"\rDrone -> "
            f"X={drone_pos[0]:6.2f} | "
            f"Y={drone_pos[1]:6.2f} | "
            f"Z={drone_pos[2]:5.2f} | "
            f"Target Distance={distance_to_target:6.2f} m",
            end=""
        )


        # ====================================================
        # TARGET REACHED
        # ====================================================

        if distance_to_target < 0.6:

            print()
            print()
            print("============================================================")
            print("             🎯 TARGET REACHED!")
            print("============================================================")

            break


        viewer.sync()

        time.sleep(0.01)


print()
print("Simulation finished.")