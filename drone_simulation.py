import mujoco
import mujoco.viewer
import time

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

        <!-- OBSTACLE 1 -->
        <geom
            pos="3 0 1"
            type="box"
            size="0.7 0.7 1"
            rgba="0.8 0.2 0.2 1"/>

        <!-- OBSTACLE 2 -->
        <geom
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

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Drone body ID
drone_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "drone"
)

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat[:] = [4, 0, 2]
    viewer.cam.distance = 12
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    # Start position
    start_x = 0
    start_y = 0
    start_z = 3

    # Target position
    target_x = 10
    target_y = 0
    target_z = 3

    while viewer.is_running():

        # Current drone position
        pos = data.xpos[drone_id]

        # Direction towards target
        dx = target_x - pos[0]
        dy = target_y - pos[1]
        dz = target_z - pos[2]

        # Movement speed
        speed = 0.02

        # Move drone toward target
        if abs(dx) > 0.05:
            data.qpos[0] += speed

        # Keep drone at same height
        data.qpos[2] = start_z

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.01)