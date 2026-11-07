"""Procedural realistic-looking MuJoCo city scene for a drone navigation demo.

This is a hand-coded waypoint/obstacle-avoidance demo, NOT trained PPO inference.
It does not read or modify dynamic_obstacle.py or any PPO model.
Run from your simulation_3d folder's Python environment:
    python realistic_scene.py
"""
from pathlib import Path
import math
import time
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent

# City coordinates use X/Y for the ground plane and Z for height.
BUILDINGS = [
    (-20,-15,5.0,5.2,7.0,0), (-12,-15,4.5,5.0,10.0,1), (-5,-15,4.0,5.0,6.5,2),
    (5,-15,4.5,5.0,8.0,3), (12,-15,4.5,5.0,11.0,4), (20,-15,5.0,5.2,7.5,0),
    (-20,15,5.0,5.0,9.0,1), (-12,15,4.5,5.0,6.0,2), (-5,15,4.0,5.0,8.0,3),
    (5,15,4.0,5.0,6.5,4), (12,15,4.5,5.0,10.5,0), (20,15,5.0,5.0,8.0,2),
    (-25,-5,3.5,4.0,6.0,3), (-25,5,3.5,4.0,8.0,4),
    (25,-5,3.5,4.0,7.0,1), (25,5,3.5,4.0,9.0,0),
    (-15,23,4.5,4.0,7.0,2), (-7,23,4.0,4.0,9.0,0), (7,23,4.0,4.0,8.0,1), (15,23,4.5,4.0,10.0,3),
    (-15,-23,4.5,4.0,8.0,4), (-7,-23,4.0,4.0,6.0,1), (7,-23,4.0,4.0,9.0,2), (15,-23,4.5,4.0,7.0,0),
]
FACADE_COLORS = ["0.72 0.77 0.80 1", "0.82 0.75 0.65 1", "0.65 0.74 0.77 1", "0.80 0.82 0.78 1", "0.68 0.70 0.76 1"]
GLASS_COLORS = ["0.12 0.28 0.38 1", "0.18 0.36 0.45 1", "0.20 0.30 0.38 1"]


def geom(name, kind, pos, size, rgba, extra=""):
    return f'<geom name="{name}" type="{kind}" pos="{pos}" size="{size}" rgba="{rgba}" {extra}/>'

world = []
assets = '''
    <texture name="ground_tex" type="2d" builtin="checker" rgb1="0.30 0.32 0.32" rgb2="0.34 0.36 0.35" width="512" height="512"/>
    <material name="ground_mat" texture="ground_tex" texrepeat="24 24" reflectance="0.02"/>
'''

# Ground, roads, sidewalks and painted lane markings.
world.append('<geom name="ground" type="plane" pos="0 0 0" size="42 38 0.1" material="ground_mat" contype="1" conaffinity="1"/>')
world.append(geom('road_east_west','box','0 0 0.035','34 4.6 0.035','0.12 0.13 0.14 1','contype="0" conaffinity="0"'))
world.append(geom('road_north_south','box','0 0 0.04','4.5 32 0.04','0.12 0.13 0.14 1','contype="0" conaffinity="0"'))
# Sidewalk strips bordering the main streets.
for y in (-5.15,5.15):
    world.append(geom(f'sidewalk_x_{y}','box',f'0 {y} 0.075','34 0.48 0.075','0.62 0.63 0.60 1','contype="0" conaffinity="0"'))
for x in (-5.05,5.05):
    world.append(geom(f'sidewalk_y_{x}','box',f'{x} 0 0.08','0.45 32 0.08','0.62 0.63 0.60 1','contype="0" conaffinity="0"'))
for i, x in enumerate(np.arange(-32, 33, 4)):
    world.append(geom(f'lane_dash_{i}','box',f'{x} 0 0.079','1.25 0.07 0.012','0.96 0.84 0.35 1','contype="0" conaffinity="0"'))
for i, y in enumerate(np.arange(-28, 29, 4)):
    world.append(geom(f'cross_dash_{i}','box',f'0 {y} 0.082','0.07 1.2 0.012','0.96 0.84 0.35 1','contype="0" conaffinity="0"'))
# Crosswalk bars at the central intersection.
for i in range(7):
    xx = -2.0 + i * 0.65
    world.append(geom(f'crosswalk_n_{i}','box',f'{xx} 5.9 0.09','0.20 0.72 0.012','0.92 0.92 0.88 1','contype="0" conaffinity="0"'))
    world.append(geom(f'crosswalk_s_{i}','box',f'{xx} -5.9 0.09','0.20 0.72 0.012','0.92 0.92 0.88 1','contype="0" conaffinity="0"'))

# Buildings: main body, roof rim, horizontal floor bands, windows on four facades.
# They are generated procedurally, so this scene does not depend on poor external OBJ assets.
for bi, (x, y, w, d, h, style) in enumerate(BUILDINGS):
    col = FACADE_COLORS[style]
    world.append(geom(f'building_{bi}_body','box',f'{x} {y} {h/2:.3f}',f'{w/2} {d/2} {h/2:.3f}',col,'contype="1" conaffinity="1"'))
    # Roof cap and rooftop utility structure.
    world.append(geom(f'building_{bi}_roof_rim','box',f'{x} {y} {h+0.10:.3f}',f'{w/2+0.12} {d/2+0.12} 0.10','0.38 0.41 0.43 1','contype="0" conaffinity="0"'))
    world.append(geom(f'building_{bi}_roof_unit','box',f'{x-w*0.12:.2f} {y+d*0.08:.2f} {h+0.38:.2f}',f'{w*0.16:.2f} {d*0.17:.2f} 0.18','0.48 0.50 0.51 1','contype="0" conaffinity="0"'))
    floors = max(2, int(h // 2.4))
    # Thin floor bands around front and back.
    for fl in range(1, floors):
        z = fl * h / floors
        world.append(geom(f'b{bi}_band_front_{fl}','box',f'{x} {y-d/2-0.025:.3f} {z:.3f}',f'{w/2} 0.035 0.055','0.88 0.88 0.84 1','contype="0" conaffinity="0"'))
        world.append(geom(f'b{bi}_band_back_{fl}','box',f'{x} {y+d/2+0.025:.3f} {z:.3f}',f'{w/2} 0.035 0.055','0.88 0.88 0.84 1','contype="0" conaffinity="0"'))
    rows = max(2, int(h // 1.65))
    cols_front = max(2, int(w // 0.95))
    for row in range(rows):
        z = 0.85 + row * ((h-1.3)/max(1, rows-1))
        if z > h-0.35: continue
        for c in range(cols_front):
            wx = x - w*0.36 + c*(w*0.72/max(1,cols_front-1))
            glass = GLASS_COLORS[(bi+c+row) % len(GLASS_COLORS)]
            # Front and rear windows.
            world.append(geom(f'b{bi}_wf_{row}_{c}','box',f'{wx:.3f} {y-d/2-0.035:.3f} {z:.3f}','0.22 0.035 0.32',glass,'contype="0" conaffinity="0"'))
            world.append(geom(f'b{bi}_wr_{row}_{c}','box',f'{wx:.3f} {y+d/2+0.035:.3f} {z:.3f}','0.22 0.035 0.32',glass,'contype="0" conaffinity="0"'))
    cols_side = max(2, int(d // 1.0))
    for row in range(rows):
        z = 0.85 + row * ((h-1.3)/max(1, rows-1))
        if z > h-0.35: continue
        for c in range(cols_side):
            wy = y - d*0.34 + c*(d*0.68/max(1,cols_side-1))
            glass = GLASS_COLORS[(bi+c+row+1) % len(GLASS_COLORS)]
            world.append(geom(f'b{bi}_wl_{row}_{c}','box',f'{x-w/2-0.035:.3f} {wy:.3f} {z:.3f}','0.035 0.22 0.32',glass,'contype="0" conaffinity="0"'))
            world.append(geom(f'b{bi}_we_{row}_{c}','box',f'{x+w/2+0.035:.3f} {wy:.3f} {z:.3f}','0.035 0.22 0.32',glass,'contype="0" conaffinity="0"'))
    # A darker shopfront at street level gives some buildings a city-block feel.
    world.append(geom(f'b{bi}_shopfront','box',f'{x} {y-d/2-0.045:.3f} 0.62',f'{w*0.34:.2f} 0.045 0.42','0.10 0.19 0.23 1','contype="0" conaffinity="0"'))

# Trees: trunk and layered foliage, placed beside—not on—the central route.
tree_positions = [(-17,-8),(-10,-8),(10,-8),(17,-8),(-17,8),(-10,8),(10,8),(17,8),(-29,-10),(-29,10),(29,-10),(29,10),(-17,19),(17,19),(-17,-19),(17,-19)]
for i,(tx,ty) in enumerate(tree_positions):
    world.append(geom(f'tree_trunk_{i}','cylinder',f'{tx} {ty} 0.95','0.16 0.95','0.36 0.22 0.10 1','contype="0" conaffinity="0"'))
    world.append(geom(f'tree_crown_low_{i}','ellipsoid',f'{tx} {ty} 2.0','0.78 0.74 0.95','0.13 0.39 0.18 1','contype="0" conaffinity="0"'))
    world.append(geom(f'tree_crown_high_{i}','ellipsoid',f'{tx} {ty} 2.65','0.58 0.56 0.72','0.18 0.48 0.22 1','contype="0" conaffinity="0"'))

# Streetlights along the sidewalks.
for i, x in enumerate((-24,-16,-8,8,16,24)):
    for side in (-1,1):
        y = side*5.65
        world.append(geom(f'light_pole_{i}_{side}','cylinder',f'{x} {y} 2.1','0.07 2.1','0.18 0.19 0.20 1','contype="0" conaffinity="0"'))
        world.append(geom(f'light_arm_{i}_{side}','capsule',f'{x} {y-side*0.18} 4.1','0.06 0.26','0.18 0.19 0.20 1','contype="0" conaffinity="0"'))
        world.append(geom(f'light_bulb_{i}_{side}','sphere',f'{x} {y-side*0.36} 4.1','0.17','1.0 0.88 0.55 1','contype="0" conaffinity="0"'))

# Target platform and a bright vertical beacon.
world.append(geom('target_pad','cylinder','0 27 0.14','1.1 0.14','0.10 0.72 0.22 1','contype="0" conaffinity="0"'))
world.append(geom('target_beacon','cylinder','0 27 0.65','0.10 0.38','0.20 1.0 0.25 1','contype="0" conaffinity="0"'))

# XML body for a simple visible car that moves on the main road.
car_xml = '''
<body name="moving_car" mocap="true" pos="0 -1.55 0.58">
 <geom name="car_body" type="box" pos="0 0 0" size="1.15 0.55 0.24" rgba="0.82 0.12 0.08 1" contype="0" conaffinity="0"/>
 <geom name="car_roof" type="box" pos="-0.12 0 0.31" size="0.62 0.43 0.20" rgba="0.12 0.26 0.34 1" contype="0" conaffinity="0"/>
 <geom name="car_glass_front" type="box" pos="0.43 0 0.31" size="0.10 0.39 0.14" rgba="0.50 0.72 0.82 1" contype="0" conaffinity="0"/>
 <geom name="car_wheel_fl" type="cylinder" pos="0.68 0.52 -0.19" size="0.22 0.12" euler="1.5708 0 0" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
 <geom name="car_wheel_fr" type="cylinder" pos="0.68 -0.52 -0.19" size="0.22 0.12" euler="1.5708 0 0" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
 <geom name="car_wheel_rl" type="cylinder" pos="-0.68 0.52 -0.19" size="0.22 0.12" euler="1.5708 0 0" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
 <geom name="car_wheel_rr" type="cylinder" pos="-0.68 -0.52 -0.19" size="0.22 0.12" euler="1.5708 0 0" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0"/>
</body>'''

# Additional moving traffic, pedestrians, and birds are generated as mocap bodies.
extra_cars_xml = []
CAR_COLORS = ['0.08 0.32 0.85 1', '0.92 0.68 0.08 1', '0.12 0.55 0.28 1', '0.72 0.72 0.75 1', '0.10 0.10 0.12 1', '0.75 0.16 0.12 1']
for i, color in enumerate(CAR_COLORS):
    extra_cars_xml.append(f'''<body name="traffic_car_{i}" mocap="true" pos="{-25+i*8} {-2.8 if i%2==0 else 2.8} 0.48">
      <geom type="box" size="0.90 0.43 0.20" rgba="{color}" contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.05 0 0.25" size="0.48 0.32 0.15" rgba="0.22 0.42 0.52 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0.55 0.39 -0.15" size="0.16 0.09" euler="1.5708 0 0" rgba="0.04 0.04 0.04 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="-0.55 -0.39 -0.15" size="0.16 0.09" euler="1.5708 0 0" rgba="0.04 0.04 0.04 1" contype="0" conaffinity="0"/>
    </body>''')

pedestrians_xml = []
PED_COLORS = ['0.82 0.20 0.16 1', '0.10 0.30 0.82 1', '0.95 0.65 0.12 1', '0.18 0.55 0.30 1', '0.55 0.22 0.68 1', '0.90 0.90 0.86 1', '0.10 0.10 0.12 1', '0.15 0.62 0.68 1']
for i, color in enumerate(PED_COLORS):
    pedestrians_xml.append(f'''<body name="pedestrian_{i}" mocap="true" pos="{-18+i*5} {5.55 if i%2==0 else -5.55} 0.95">
      <geom type="capsule" fromto="0 0 -0.32 0 0 0.28" size="0.16" rgba="{color}" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0 0 0.48" size="0.14" rgba="0.72 0.52 0.36 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 -0.25 -0.08 0 -0.62" size="0.07" rgba="0.12 0.14 0.18 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 -0.25 0.08 0 -0.62" size="0.07" rgba="0.12 0.14 0.18 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0.12 -0.18 0 -0.10" size="0.055" rgba="{color}" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0.12 0.18 0 -0.10" size="0.055" rgba="{color}" contype="0" conaffinity="0"/>
    </body>''')

birds_xml = []
for i in range(9):
    birds_xml.append(f'''<body name="bird_{i}" mocap="true" pos="{-20+i*4} {8+(i%3)*4} {10+(i%4)*1.6}">
      <geom type="ellipsoid" size="0.20 0.075 0.075" rgba="0.12 0.13 0.15 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="-0.04 0 0  -0.42 {0.20 if i%2 else -0.20} 0.02" size="0.045" rgba="0.12 0.13 0.15 1" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0.04 0 0  0.42 {0.20 if i%2 else -0.20} 0.02" size="0.045" rgba="0.12 0.13 0.15 1" contype="0" conaffinity="0"/>
    </body>''')

drone_xml = '''
<body name="drone" mocap="true" pos="-28 -28 4">
 <geom name="drone_body" type="ellipsoid" pos="0 0 0" size="0.48 0.34 0.16" rgba="0.06 0.24 0.94 1" contype="0" conaffinity="0"/>
 <geom name="drone_nose" type="box" pos="0.40 0 0.02" size="0.20 0.13 0.10" rgba="0.08 0.15 0.60 1" contype="0" conaffinity="0"/>
 <geom name="drone_arm_1" type="capsule" fromto="-0.12 -0.12 0.02 -0.82 -0.82 0.02" size="0.055" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="drone_arm_2" type="capsule" fromto="-0.12 0.12 0.02 -0.82 0.82 0.02" size="0.055" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="drone_arm_3" type="capsule" fromto="0.12 -0.12 0.02 0.82 -0.82 0.02" size="0.055" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="drone_arm_4" type="capsule" fromto="0.12 0.12 0.02 0.82 0.82 0.02" size="0.055" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="rotor_1" type="cylinder" pos="-0.82 -0.82 0.12" size="0.32 0.035" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="rotor_2" type="cylinder" pos="-0.82 0.82 0.12" size="0.32 0.035" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="rotor_3" type="cylinder" pos="0.82 -0.82 0.12" size="0.32 0.035" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="rotor_4" type="cylinder" pos="0.82 0.82 0.12" size="0.32 0.035" rgba="0.10 0.10 0.12 1" contype="0" conaffinity="0"/>
 <geom name="drone_led" type="sphere" pos="0.53 0 0.06" size="0.09" rgba="1 0.08 0.06 1" contype="0" conaffinity="0"/>
</body>'''

xml = f'''<mujoco model="procedural_realistic_city_drone">
 <compiler angle="radian" autolimits="true"/>
 <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
 <visual><headlight diffuse="0.85 0.85 0.85" ambient="0.42 0.42 0.42" specular="0.22 0.22 0.22"/><global offwidth="1600" offheight="1000"/><quality shadowsize="2048"/></visual>
 <asset>{assets}</asset>
 <worldbody>
  <light pos="0 0 38" dir="0 0 -1" directional="true" diffuse="0.9 0.9 0.9"/>
  {''.join(world)}
  {car_xml}
  {"".join(extra_cars_xml)}
  {"".join(pedestrians_xml)}
  {"".join(birds_xml)}
  {drone_xml}
 </worldbody>
</mujoco>'''


def main():
    # Validate XML before launching the viewer; this gives clearer errors.
    ET.fromstring(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    drone_body_id = model.body('drone').id
    car_body_id = model.body('moving_car').id
    traffic_mocaps = [int(model.body_mocapid[model.body(f'traffic_car_{i}').id]) for i in range(len(extra_cars_xml))]
    pedestrian_mocaps = [int(model.body_mocapid[model.body(f'pedestrian_{i}').id]) for i in range(len(pedestrians_xml))]
    bird_mocaps = [int(model.body_mocapid[model.body(f'bird_{i}').id]) for i in range(len(birds_xml))]
    drone_mocap = int(model.body_mocapid[drone_body_id])
    car_mocap = int(model.body_mocapid[car_body_id])
    if drone_mocap < 0 or car_mocap < 0:
        raise RuntimeError('Drone and car must be mocap bodies.')

    route = [np.array(p, dtype=float) for p in [
        (-28,-28,4.0), (-28,0,4.0), (-12,0,4.0), (0,0,4.0), (0,14,4.5), (0,27,4.5)
    ]]
    drone_pos = route[0].copy()
    waypoint = 1
    speed = 0.105  # metres per simulation frame; smooth visible travel
    sim_t = 0.0
    last_log = 0.0
    arrived = False
    data.mocap_pos[drone_mocap] = drone_pos
    data.mocap_quat[drone_mocap] = [1,0,0,0]
    data.mocap_pos[car_mocap] = [0,-1.55,0.58]
    data.mocap_quat[car_mocap] = [1,0,0,0]
    mujoco.mj_forward(model, data)

    print('\nPROCEDURAL CITY DRONE DEMO')
    print(f'Generated {len(BUILDINGS)} detailed buildings, roads, trees, streetlights, 7 cars, 8 pedestrians and 9 flying birds.')
    print('Blue drone: (-28, -28, 4) -> green target: (0, 27, 4.5)')
    print('Navigation is waypoint-based with a simple moving-car altitude detour; this is not PPO inference.')
    print('Close the viewer window to stop.\n')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 3.0]
        viewer.cam.distance = 70
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while viewer.is_running():
            start = time.time()
            dt = model.opt.timestep
            sim_t += dt
            car_x = 9.0 * math.sin(sim_t * 0.42)
            data.mocap_pos[car_mocap] = [car_x, -1.55, 0.58]
            # Six extra cars move in both directions on the avenue.
            for i, mid in enumerate(traffic_mocaps):
                lane_y = -2.8 if i % 2 == 0 else 2.8
                direction = 1 if i % 2 == 0 else -1
                x = ((-27 + i*9 + direction*sim_t*2.4 + 27) % 54) - 27
                data.mocap_pos[mid] = [x, lane_y, 0.48]
                data.mocap_quat[mid] = [1, 0, 0, 0] if direction == 1 else [0, 0, 0, 1]
            # Pedestrians walk back and forth along both sidewalks.
            for i, mid in enumerate(pedestrian_mocaps):
                x = 28 * math.sin(sim_t * (0.12 + (i % 3)*0.025) + i*1.7)
                y = 5.55 if i % 2 == 0 else -5.55
                data.mocap_pos[mid] = [x, y, 0.95]
                data.mocap_quat[mid] = [1, 0, 0, 0]
            # Birds circle above the buildings in a loose flock.
            for i, mid in enumerate(bird_mocaps):
                angle = sim_t * 0.22 + i * (2*math.pi/len(bird_mocaps))
                bx = 15 * math.cos(angle) + 2 * math.sin(angle*1.7)
                by = 13 * math.sin(angle)
                bz = 11.5 + 1.5 * math.sin(angle*2 + i)
                data.mocap_pos[mid] = [bx, by, bz]
                yaw = angle + math.pi/2
                data.mocap_quat[mid] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]

            if not arrived:
                target = route[waypoint]
                delta = target - drone_pos
                dist = float(np.linalg.norm(delta))
                if dist < 0.38:
                    if waypoint < len(route)-1:
                        waypoint += 1
                        target = route[waypoint]
                        delta = target - drone_pos
                        dist = float(np.linalg.norm(delta))
                    else:
                        arrived = True
                        print('TARGET REACHED! Drone arrived at the green marker.')
                if not arrived and dist > 1e-8:
                    direction = delta / dist
                    next_pos = drone_pos + direction * min(speed, dist)
                    # Simple demonstration detour: climb if the moving car is nearby.
                    car_dist = float(np.linalg.norm(next_pos[:2] - np.array([car_x, -1.55])))
                    if car_dist < 4.5:
                        next_pos[2] = min(7.0, max(next_pos[2], 6.0))
                    else:
                        next_pos[2] += float(np.clip(target[2]-next_pos[2], -0.025, 0.025))
                    drone_pos = next_pos

            data.mocap_pos[drone_mocap] = drone_pos
            look_delta = route[waypoint][:2] - drone_pos[:2]
            if np.linalg.norm(look_delta) > 1e-5:
                yaw = math.atan2(look_delta[1], look_delta[0])
                data.mocap_quat[drone_mocap] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
            mujoco.mj_step(model, data)
            viewer.sync()
            if sim_t-last_log >= 2.0:
                last_log = sim_t
                remaining = float(np.linalg.norm(route[-1]-drone_pos))
                print(f'Drone=({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f}) | target distance={remaining:.1f} m')
            elapsed = time.time()-start
            if elapsed < dt:
                time.sleep(dt-elapsed)

if __name__ == '__main__':
    main()
