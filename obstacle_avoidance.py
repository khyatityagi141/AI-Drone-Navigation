import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>

    <option gravity="0 0 0"/>

    <worldbody>

        <light pos="0 0 10" directional="true"/>

        <geom
            type="plane"
            size="20 20 0.1"
            rgba="0.3 0.3 0.3 1"/>

        <!-- BLUE DRONE -->
        <body name="drone" pos="0 0 3">
            <geom
                type="box"
                size="0.5 0.35 0.12"
                rgba="0 0 1 1"/>
        </body>

        <!-- RED MOVING OBSTACLE -->
        <body name="moving_obstacle" pos="5 0 3">

            <joint
                name="move_y"
                type="slide"
                axis="0 1 0"/>

            <geom
                type="box"
                size="1 1 1"
                rgba="1 0 0 1"/>

        </body>

        <!-- GREEN TARGET -->
        <geom
            pos="10 0 3"
            type="sphere"
            size="0.5"
            rgba="0 1 0 1"/>

    </worldbody>

</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "move_y"
)

qpos_id = model.jnt_qposadr[joint_id]

print("RED OBSTACLE TEST STARTED")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Camera
    viewer.cam.lookat[:] = [5, 0, 3]
    viewer.cam.distance = 18
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10

    start = time.time()

    while viewer.is_running():

        t = time.time() - start

        # RED OBSTACLE MOVES Y DIRECTION
        y = 4 * __import__("math").sin(t)

        data.qpos[qpos_id] = y

        mujoco.mj_forward(model, data)

        print(f"\rRed obstacle Y = {y:.2f}", end="")

        viewer.sync()

        time.sleep(0.02)