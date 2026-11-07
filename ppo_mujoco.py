import mujoco
import mujoco.viewer
import time
import math
import numpy as np

from stable_baselines3 import PPO


# ============================================================
# MUJOCO MODEL
# ============================================================

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
            name="static_obstacle"
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
# CREATE MUJOCO MODEL
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

dynamic_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "dynamic_obstacle"
)


# ============================================================
# DYNAMIC OBSTACLE JOINT
# ============================================================

joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "dynamic_slide"
)

joint_qpos = model.jnt_qposadr[joint_id]


# ============================================================
# LOAD TRAINED PPO MODEL
# ============================================================

print()
print("==============================================")
print("       LOADING TRAINED PPO MODEL")
print("==============================================")

model_ppo = PPO.load("drone_ppo_model")

print("PPO model loaded successfully!")
print()


# ============================================================
# TARGET
# ============================================================

target = np.array(
    [10.0, 0.0, 3.0],
    dtype=np.float32
)


# ============================================================
# SETTINGS
# ============================================================

movement = 0.05

target_threshold = 0.5

start_time = time.time()


# ============================================================
# START VIEWER
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat[:] = [4, 0, 2]
    viewer.cam.distance = 12
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20


    print("==============================================")
    print("      PPO + MUJOCO DRONE SIMULATION")
    print("==============================================")
    print()
    print("Blue  = Drone")
    print("Red   = Obstacles")
    print("Green = Target")
    print()
    print("PPO is controlling the drone.")
    print("==============================================")
    print()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    for step in range(1000):

        if not viewer.is_running():
            break


        # ====================================================
        # 1. DRONE POSITION
        # ====================================================

        drone_pos = data.xpos[drone_id].copy()

        drone_x = drone_pos[0]
        drone_y = drone_pos[1]
        drone_z = drone_pos[2]


        # ====================================================
        # 2. MOVE DYNAMIC OBSTACLE
        # ====================================================

        elapsed = time.time() - start_time

        dynamic_movement = (
            1.5 * math.sin(elapsed * 1.5)
        )

        data.qpos[joint_qpos] = dynamic_movement

        mujoco.mj_forward(model, data)


        # ====================================================
        # 3. DYNAMIC OBSTACLE POSITION
        # ====================================================

        dynamic_pos = data.xpos[dynamic_id].copy()


        # ====================================================
        # 4. STATIC OBSTACLE
        # ====================================================

        static_pos = np.array(
            [3.0, 0.0, 1.0],
            dtype=np.float32
        )


        # ====================================================
        # 5. SENSOR DISTANCES
        # ====================================================

        static_distance = np.linalg.norm(
            drone_pos - static_pos
        )

        dynamic_distance = np.linalg.norm(
            drone_pos - dynamic_pos
        )


        # ====================================================
        # 6. CREATE PPO OBSERVATION
        # ====================================================

        observation = np.array([

            drone_x,
            drone_y,
            drone_z,

            target[0],
            target[1],
            target[2],

            static_distance,
            dynamic_distance

        ], dtype=np.float32)


        # ====================================================
        # 7. PPO SELECTS ACTION
        # ====================================================

        action, _states = model_ppo.predict(
            observation,
            deterministic=True
        )

        action = int(action)


        # ====================================================
        # 8. APPLY PPO ACTION
        # ====================================================

        if action == 0:

            # Forward
            data.qpos[0] += movement

        elif action == 1:

            # Left
            data.qpos[1] += movement

        elif action == 2:

            # Right
            data.qpos[1] -= movement

        elif action == 3:

            # Up
            data.qpos[2] += movement

        elif action == 4:

            # Down
            data.qpos[2] -= movement


        # ====================================================
        # 9. KEEP DRONE IN SAFE HEIGHT
        # ====================================================

        if data.qpos[2] < 1.5:

            data.qpos[2] = 1.5

        if data.qpos[2] > 5.0:

            data.qpos[2] = 5.0


        # ====================================================
        # 10. UPDATE MUJOCO
        # ====================================================

        mujoco.mj_forward(model, data)


        # ====================================================
        # 11. PRINT SENSOR + PPO ACTION
        # ====================================================

        action_names = {
            0: "FORWARD",
            1: "LEFT",
            2: "RIGHT",
            3: "UP",
            4: "DOWN"
        }

        print(
            f"\rStep={step:4d} | "
            f"X={data.qpos[0]:5.2f} "
            f"Y={data.qpos[1]:5.2f} "
            f"Z={data.qpos[2]:5.2f} | "
            f"Static={static_distance:5.2f}m | "
            f"Dynamic={dynamic_distance:5.2f}m | "
            f"PPO={action_names[action]:7s}",
            end=""
        )


        # ====================================================
        # 12. TARGET CHECK
        # ====================================================

        current_pos = np.array([
            data.qpos[0],
            data.qpos[1],
            data.qpos[2]
        ])

        distance_to_target = np.linalg.norm(
            current_pos - target
        )


        if distance_to_target < target_threshold:

            print()
            print()
            print("==============================================")
            print("           TARGET REACHED!")
            print("==============================================")
            print(
                f"Final Position: "
                f"{current_pos}"
            )
            print(
                f"Distance to Target: "
                f"{distance_to_target:.2f} m"
            )

            break


        # ====================================================
        # 13. UPDATE VIEW
        # ====================================================

        viewer.sync()

        time.sleep(0.01)


print()
print("Simulation finished.")