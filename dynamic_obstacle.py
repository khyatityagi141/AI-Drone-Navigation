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
# CREATE MODEL
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
# SETTINGS
# ============================================================

target_x = 10.0
target_y = 0.0
target_z = 3.0

speed = 0.02

# Distance at which drone starts avoiding
avoid_distance = 2.2

# Side position
safe_y = 2.0

# Avoidance state
avoiding = False

print()
print("==============================================")
print("   DRONE SENSOR + OBSTACLE AVOIDANCE")
print("==============================================")
print("Blue  = Drone")
print("Red   = Obstacles")
print("Green = Target")
print()
print("Sensor distance is active.")
print("Automatic avoidance is active.")
print("==============================================")
print()


# ============================================================
# SIMULATION
# ============================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    # SAME CAMERA AS WORKING VERSION
    viewer.cam.lookat[:] = [4, 0, 2]
    viewer.cam.distance = 12
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    start_time = time.time()

    while viewer.is_running():

        # ====================================================
        # 1. CURRENT DRONE POSITION
        # ====================================================

        drone_pos = data.xpos[drone_id]

        drone_x = drone_pos[0]
        drone_y = drone_pos[1]
        drone_z = drone_pos[2]


        # ====================================================
        # 2. MOVE DYNAMIC OBSTACLE
        # ====================================================

        elapsed = time.time() - start_time

        movement = 1.5 * math.sin(elapsed * 1.5)

        data.qpos[joint_qpos] = movement

        mujoco.mj_forward(model, data)


        # ====================================================
        # 3. GET DYNAMIC OBSTACLE POSITION
        # ====================================================

        dynamic_pos = data.xpos[dynamic_id]

        dynamic_x = dynamic_pos[0]
        dynamic_y = dynamic_pos[1]
        dynamic_z = dynamic_pos[2]


        # ====================================================
        # 4. STATIC OBSTACLE POSITION
        # ====================================================

        static_x = 3.0
        static_y = 0.0
        static_z = 1.0


        # ====================================================
        # 5. SENSOR DISTANCE
        # ====================================================

        static_distance = math.sqrt(
            (drone_x - static_x) ** 2 +
            (drone_y - static_y) ** 2 +
            (drone_z - static_z) ** 2
        )

        dynamic_distance = math.sqrt(
            (drone_x - dynamic_x) ** 2 +
            (drone_y - dynamic_y) ** 2 +
            (drone_z - dynamic_z) ** 2
        )


        # ====================================================
        # 6. CHECK OBSTACLE
        # ====================================================

        nearest_distance = min(
            static_distance,
            dynamic_distance
        )


        # ====================================================
        # 7. AUTOMATIC AVOIDANCE
        # ====================================================

        if nearest_distance < avoid_distance:

            avoiding = True


        if avoiding:

            # Move sideways
            if drone_y < safe_y:

                data.qpos[1] += 0.025

            else:

                # Once safely to the side,
                # continue forward
                data.qpos[0] += speed


            # Check if drone has passed obstacles
            if drone_x > 7.5:

                avoiding = False


        else:

            # Normal forward movement
            if drone_x < target_x - 0.05:

                data.qpos[0] += speed


        # ====================================================
        # 8. KEEP DRONE HEIGHT FIXED
        # ====================================================

        data.qpos[2] = target_z


        # ====================================================
        # 9. RETURN TOWARD TARGET Y
        # ====================================================

        if not avoiding and abs(drone_y) > 0.05:

            if drone_y > 0:

                data.qpos[1] -= 0.02

            else:

                data.qpos[1] += 0.02


        # ====================================================
        # 10. UPDATE SIMULATION
        # ====================================================

        mujoco.mj_forward(model, data)


        # ====================================================
        # 11. PRINT SENSOR DATA
        # ====================================================

        status = "AVOIDING" if avoiding else "MOVING"

        print(
            f"\rDrone X={drone_x:5.2f} "
            f"Y={drone_y:5.2f} | "
            f"Static={static_distance:5.2f}m | "
            f"Dynamic={dynamic_distance:5.2f}m | "
            f"{status}      ",
            end=""
        )


        # ====================================================
        # 12. TARGET CHECK
        # ====================================================

        if (
            drone_x >= target_x - 0.05
            and abs(drone_y) < 0.15
        ):

            print()
            print()
            print("==============================================")
            print("          TARGET REACHED!")
            print("==============================================")

            break


        # ====================================================
        # 13. UPDATE VIEW
        # ====================================================

        viewer.sync()

        time.sleep(0.01)