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

        <!-- DRONE -->
        <body name="drone" pos="0 0 3">
            <freejoint/>

            <geom
                type="box"
                size="0.5 0.35 0.12"
                rgba="0.1 0.3 0.9 1"/>

            <geom
                type="capsule"
                fromto="-0.8 0 0 0.8 0 0"
                size="0.06"/>

            <geom
                type="capsule"
                fromto="0 -0.8 0 0 0.8 0"
                size="0.06"/>
        </body>

        <!-- OBSTACLE 1 -->
        <geom
            name="obstacle1"
            pos="3 0 1"
            type="box"
            size="0.7 0.7 1"
            rgba="0.8 0.2 0.2 1"/>

        <!-- OBSTACLE 2 -->
        <geom
            name="obstacle2"
            pos="6 2 1.5"
            type="box"
            size="0.8 0.8 1.5"
            rgba="0.8 0.2 0.2 1"/>

        <!-- TARGET -->
        <geom
            pos="10 0 3"
            type="sphere"
            size="0.5"
            rgba="0.2 1 0.2 1"/>

    </worldbody>
</mujoco>
"""

# Create MuJoCo model
model = mujoco.MjModel.from_xml_string(xml)

# Create simulation data
data = mujoco.MjData(model)

# Find drone body ID
drone_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "drone"
)

# Find obstacle IDs
obstacle1_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "obstacle1"
)

obstacle2_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "obstacle2"
)

print("====================================")
print("   DRONE SENSOR SYSTEM STARTED")
print("====================================")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Camera settings
    viewer.cam.lookat[:] = [4, 0, 2]
    viewer.cam.distance = 12
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    start_time = time.time()

    # Target position
    target_x = 10

    while viewer.is_running():

        # --------------------------------
        # 1. DRONE POSITION
        # --------------------------------

        drone_pos = data.xpos[drone_id]

        # --------------------------------
        # 2. OBSTACLE POSITIONS
        # --------------------------------

        obstacle1_pos = data.geom_xpos[obstacle1_id]
        obstacle2_pos = data.geom_xpos[obstacle2_id]

        # --------------------------------
        # 3. DISTANCE SENSOR
        # --------------------------------

        distance1 = math.sqrt(
            (drone_pos[0] - obstacle1_pos[0]) ** 2
            + (drone_pos[1] - obstacle1_pos[1]) ** 2
            + (drone_pos[2] - obstacle1_pos[2]) ** 2
        )

        distance2 = math.sqrt(
            (drone_pos[0] - obstacle2_pos[0]) ** 2
            + (drone_pos[1] - obstacle2_pos[1]) ** 2
            + (drone_pos[2] - obstacle2_pos[2]) ** 2
        )

        # --------------------------------
        # 4. PRINT SENSOR DATA
        # --------------------------------

        if time.time() - start_time > 0.5:

            print(
                f"Drone Position: "
                f"X={drone_pos[0]:.2f}, "
                f"Y={drone_pos[1]:.2f}, "
                f"Z={drone_pos[2]:.2f}"
            )

            print(
                f"Sensor -> "
                f"Obstacle 1: {distance1:.2f} m | "
                f"Obstacle 2: {distance2:.2f} m"
            )

            print("------------------------------------")

            start_time = time.time()

        # --------------------------------
        # 5. MOVE DRONE TOWARDS TARGET
        # --------------------------------

        if data.qpos[0] < target_x:
            data.qpos[0] += 0.02

        # Keep drone at height 3
        data.qpos[2] = 3

        # Update simulation
        mujoco.mj_forward(model, data)

        # Update viewer
        viewer.sync()

        time.sleep(0.01)