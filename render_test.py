import mujoco

xml = """
<mujoco>
    <worldbody>
        <geom type="sphere" size="1"/>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)

print("RENDERER OK")