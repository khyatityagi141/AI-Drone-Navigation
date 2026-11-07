import mujoco
import mujoco.viewer
import time
import math

xml = """
<mujoco>
    <option gravity="0 0 0"/>

    <worldbody>

        <light pos="0 0 8" directional="true"/>

        <!-- Ground -->
        <geom
            type="plane"
            size="20 20 0.1"
            rgba="0.3 0.3 0.3 1"/>

        <!-- ================= DRONE ================= -->
        <body name="drone" pos="0 0 3">

            <freejoint/>

            <!-- Body -->
            <geom
                type="box"
                size="0.5 0.35 0.12"
                rgba="0.1 0.3 0.9 1"/>

            <!-- Arms -->
            <geom
                type="capsule"
                fromto="-0.8 0 0 0.8 0 0"
                size="0.06"/>

            <geom
                type="capsule"
                fromto="0 -0.8 0 0 0.8 0"
                size="0.06"/>

            <!-- Propellers -->
            <geom
                type="cylinder"
                pos="-0.8 0 0.15"
                size="0.25 0.03"
                rgba="1 0 0 1"/>

            <geom
                type="cylinder"
                pos="0.8 0 0.15"
                size="0.25 0.03"
                rgba="1 0 0 1"/>

            <geom
                type="cylinder"
                pos="0 -0.8 0.15"
                size="0.25 0.03"
                rgba="0 1 0 1"/>

            <geom
                type="cylinder"
                pos="0 0.8 0.15"
                size="0.25 0.03"
                rgba="0 1 0 1"/>

        </body>


        <!-- ================= STATIC OBSTACLE ================= -->
        <geom
            pos="3 0 1"
            type="box"
            size="0.7 0.7 1"
            rgba="0.8 0.2 0.2 1"/>


        <!-- ================= DYNAMIC OBSTACLE ================= -->
        <body name="dynamic_obstacle" pos="6 2 1.5">

            <joint
                name="dynamic_slide"
                type="slide"
                axis="0 1 0"
                limited="true"
                range="-2 2"/>

            <geom
                type="box"
                size="0.8 0.8 1.5"
                rgba="1 0 0 1"/>

        </body>


        <!-- ================= TARGET ================= -->
        <geom
            pos="10 0 3"
            type="sphere"
            size="0.5"
            rgba="0.2 1 0.2 1"/>

    </worldbody>
</mujoco>
"""


# ============================================================
# CREATE MODEL
# ============================================================

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)


# ============================================================
# FIND DRONE
# ============================================================

drone_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "drone"
)


# ============================================================
# FIND DYNAMIC OBSTACLE JOINT
# ============================================================

joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "dynamic_slide"
)

joint_qpos = model.jnt_qposadr[joint_id]


# ============================================================
# START VIEWER
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    # SAME CAMERA AS YOUR WORKING CODE
    viewer.cam.lookat[:] = [4, 0, 2]
    viewer.cam.distance = 12
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20


    # Target
    target_x = 10
    target_y = 0
    target_z = 3

    start_z = 3

    # Time
    start_time = time.time()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while viewer.is_running():

        # ----------------------------------------------------
        # 1. DRONE MOVEMENT
        # SAME AS YOUR WORKING CODE
        # ----------------------------------------------------

        pos = data.xpos[drone_id]

        dx = target_x - pos[0]

        speed = 0.02

        if abs(dx) > 0.05:

            data.qpos[0] += speed

        # Keep same height
        data.qpos[2] = start_z


        # ----------------------------------------------------
        # 2. DYNAMIC OBSTACLE MOVEMENT
        # ----------------------------------------------------

        elapsed = time.time() - start_time

        # Move obstacle up/down in Y direction
        movement = 1.5 * math.sin(elapsed * 1.5)

        data.qpos[joint_qpos] = movement


        # ----------------------------------------------------
        # 3. UPDATE MUJOCO
        # ----------------------------------------------------

        mujoco.mj_forward(model, data)


        # ----------------------------------------------------
        # 4. PRINT POSITIONS
        # ----------------------------------------------------

        obstacle_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "dynamic_obstacle"
        )

        drone_position = data.xpos[drone_id]
        obstacle_position = data.xpos[obstacle_id]

        print(
            f"\rDrone X = {drone_position[0]:5.2f}   "
            f"Dynamic Obstacle Y = {obstacle_position[1]:5.2f}",
            end=""
        )


        # ----------------------------------------------------
        # 5. UPDATE VIEW
        # ----------------------------------------------------

        viewer.sync()

        time.sleep(0.01)