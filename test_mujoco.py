import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <worldbody>
        <light pos="0 0 5"/>

        <geom
            type="plane"
            size="10 10 0.1"/>

        <body pos="0 0 1">
            <freejoint/>

            <geom
                type="box"
                size="0.5 0.5 0.15"/>
        </body>
    </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()