"""Emit a Blender Python scene for one fire.

The flame is a Mantaflow gas bake in a box around the fire. Each weather is
its own bake, because rain and snow cool the flame, add smoke and drag the
plume down; rain and snow are particles under gravity, drag and the wind
field. Ambient temperature and humidity set the buoyancy, the inflow heat
and the smoke. Each bake is rendered from one camera per viewpoint. Every
rendered frame also writes three masks: the flame grid, the smoke density
outside that grid, and the ash particles. The flame mask is not the emission
pass, so glowing embers stay out of it. Camera poses and looks are
computed here and written into the script as literals, so tests can check
them without launching Blender.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from firephantom.physics.fields import fuel_properties
from firephantom.physics.scenarios import ScenarioSpec

DEFAULT_FRAME_COUNT = 16
MAX_FRAME_COUNT = 48
# Bake frames skipped before the first rendered frame, so ignition is not a sample.
WARMUP_FRAMES = 12

_SCRIPT = '''"""Blender scene for the flame phantom.
Run with: blender --background --python scene.py
"""
# PARAMS_JSON=__PAYLOAD__
import json
import math
import os
import shutil

PARAMS = json.loads(r"""__PAYLOAD__""")


def _require_bpy():
    import bpy
    return bpy


def _clear(bpy):
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _box(bpy, name, xb):
    xmin, xmax, ymin, ymax, zmin, zmax = xb
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0
    sx = max((xmax - xmin) / 2.0, 1e-3)
    sy = max((ymax - ymin) / 2.0, 1e-3)
    sz = max((zmax - zmin) / 2.0, 1e-3)
    bpy.ops.mesh.primitive_cube_add(location=(cx, cy, cz), scale=(sx, sy, sz))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def _material(bpy, name, color, emission=0.0, alpha=1.0):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, alpha)
    emission_color = shader.inputs.get("Emission Color") or shader.inputs.get("Emission")
    if emission_color is not None:
        emission_color.default_value = (*color, 1.0)
    emission_strength = shader.inputs.get("Emission Strength")
    if emission_strength is not None:
        emission_strength.default_value = emission
    alpha_socket = shader.inputs.get("Alpha")
    if alpha_socket is not None:
        alpha_socket.default_value = alpha
    material.diffuse_color = (*color, alpha)
    if alpha < 1.0:
        try:
            material.surface_render_method = "DITHERED"
        except AttributeError:
            material.blend_method = "BLEND"
    return material


def _gas_material(bpy):
    """Blackbody flame plus gray smoke. Heat is not Kelvin, so it is mapped first."""
    material = bpy.data.materials.new(name="gas_volume")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 8.0
    volume.inputs["Density Attribute"].default_value = "density"
    volume.inputs["Absorption Color"].default_value = (0.55, 0.55, 0.55, 1.0)
    volume.inputs["Blackbody Tint"].default_value = (1.0, 0.8, 0.55, 1.0)
    # The default attribute multiplies this mapped Kelvin by the heat grid and
    # clips the flame to a white block. The MapRange below is the only scale.
    volume.inputs["Temperature Attribute"].default_value = ""
    # The smoke color grid is near black on a clear day when left at the
    # default. A floor keeps the plume gray if that grid is missing.
    color = nodes.new("ShaderNodeAttribute")
    color.attribute_name = "color"
    color.attribute_type = "GEOMETRY"
    floor = nodes.new("ShaderNodeVectorMath")
    floor.operation = "MAXIMUM"
    floor.inputs[1].default_value = (0.35, 0.35, 0.35)
    links.new(color.outputs["Color"], floor.inputs[0])
    links.new(floor.outputs["Vector"], volume.inputs["Color"])
    # Blender 4.5 exposes the same heat grid under both names.
    temperature = nodes.new("ShaderNodeAttribute")
    temperature.attribute_name = "temperature"
    temperature.attribute_type = "GEOMETRY"
    heat = nodes.new("ShaderNodeAttribute")
    heat.attribute_name = "heat"
    heat.attribute_type = "GEOMETRY"
    added = nodes.new("ShaderNodeMath")
    added.operation = "ADD"
    links.new(temperature.outputs["Fac"], added.inputs[0])
    links.new(heat.outputs["Fac"], added.inputs[1])
    mapped = nodes.new("ShaderNodeMapRange")
    mapped.clamp = True
    mapped.inputs["From Min"].default_value = 0.4
    mapped.inputs["From Max"].default_value = 2.4
    mapped.inputs["To Min"].default_value = 1300.0
    mapped.inputs["To Max"].default_value = 1900.0
    links.new(added.outputs["Value"], mapped.inputs["Value"])
    links.new(mapped.outputs["Result"], volume.inputs["Temperature"])
    flame = nodes.new("ShaderNodeAttribute")
    flame.attribute_name = "flame"
    flame.attribute_type = "GEOMETRY"
    gain = nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = 4.0
    links.new(flame.outputs["Fac"], gain.inputs[0])
    links.new(gain.outputs["Value"], volume.inputs["Blackbody Intensity"])
    # Cycles does not write shader AOVs from a volume, so the flame and smoke
    # masks are render passes. Only the ember surface writes an AOV.
    links.new(volume.outputs["Volume"], output.inputs["Volume"])
    return material


def _collider(bpy, obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type="FLUID")
    modifier = obj.modifiers["Fluid"]
    modifier.fluid_type = "EFFECTOR"
    modifier.effector_settings.effector_type = "COLLISION"
    modifier.effector_settings.surface_distance = 0.0


def _particle_hit(bpy, obj):
    """Stop rain and snow on the box. The fluid effector does not touch particles."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type="COLLISION")
    hit = obj.collision
    hit.permeability = 0.0
    hit.damping = 0.6
    hit.stickiness = 0.0
    hit.use_particle_kill = False


def _under_cover(x, y, z, boxes):
    """True when an inert box is overhead, so the drop is inside or under a lid."""
    for box in boxes:
        x0, x1, y0, y1, _z0, z1 = box
        if x0 <= x <= x1 and y0 <= y <= y1 and z < z1 - 0.01:
            return True
    return False


def _cull_precip(scene, _depsgraph=None):
    boxes = PARAMS.get("cover_boxes") or []
    if not boxes:
        return
    for obj in scene.objects:
        if not str(obj.name).startswith("precip_"):
            continue
        for system in obj.particle_systems:
            for particle in system.particles:
                loc = particle.location
                if _under_cover(loc.x, loc.y, loc.z, boxes):
                    particle.alive_state = "DEAD"


def _gas_domain(bpy):
    """Bake a box around the fire. The scenario domain is too large for the timeout."""
    box = PARAMS["gas_domain"]
    bpy.ops.mesh.primitive_cube_add(location=tuple(box["location"]), scale=tuple(box["scale"]))
    domain = bpy.context.active_object
    domain.name = "gas_domain"
    bpy.ops.object.modifier_add(type="FLUID")
    domain.modifiers["Fluid"].fluid_type = "DOMAIN"
    settings = domain.modifiers["Fluid"].domain_settings
    settings.domain_type = "GAS"
    sim = PARAMS["simulation"]
    settings.resolution_max = int(PARAMS["gas_domain"]["resolution"])
    settings.use_adaptive_domain = True
    settings.adapt_threshold = 0.002
    noise = int(sim["noise"])
    settings.use_noise = noise > 0
    if noise > 0:
        settings.noise_scale = noise
        settings.noise_strength = 0.5
    vorticity = float(sim["vorticity"])
    settings.vorticity = vorticity
    # The flame curls a little more than the smoke, in the default 0.22 : 0.35.
    settings.flame_vorticity = vorticity * 0.35 / 0.22
    ambient = PARAMS["ambient"]
    # Warmer ambient air is lighter: less buoyancy for the same heat.
    settings.alpha = float(ambient["alpha"])
    settings.beta = float(ambient["beta"])
    domain["base_burning_rate"] = float(sim["burning_rate"]) * float(ambient["burning_scale"])
    settings.burning_rate = domain["base_burning_rate"]
    settings.flame_ignition = 1.3
    settings.flame_max_temp = float(sim["flame_temp"])
    soot = float(PARAMS["soot_yield"])
    domain["base_flame_smoke"] = (0.35 + soot * 6.0) * float(sim["smoke"]) * float(ambient["smoke_scale"])
    settings.flame_smoke = domain["base_flame_smoke"]
    settings.flame_smoke_color = (0.45, 0.45, 0.45)
    settings.use_adaptive_timesteps = False
    settings.timesteps_min = 1
    settings.timesteps_max = 1
    settings.cache_type = "ALL"
    settings.cache_data_format = "UNI"
    settings.cache_frame_start = 1
    settings.cache_frame_end = int(PARAMS["warmup"]) + int(PARAMS["frame_count"])
    domain.data.materials.append(_gas_material(bpy))
    # The camera still sees the flame. Diffuse rays do not, so the lit floor
    # stays out of the flame mask.
    # Diffuse rays stay blind so the floor is not painted by the flame.
    # Volume scatter stays visible: that is the only way the smoke reaches the camera.
    domain.visible_diffuse = False
    domain.visible_glossy = False
    domain.visible_transmission = False
    domain.visible_volume_scatter = True
    domain.visible_shadow = False
    return domain


def _lean():
    """Legal wind is 0–100 m/s. Saturate so the plume leans and stays in the box."""
    return 2.2 * (1.0 - math.exp(-float(PARAMS["wind_speed"]) / 5.0))


def _inflow(bpy):
    origin = PARAMS["origin"]
    jitter = PARAMS["jitter"]
    ox, oy, oz = origin
    ox += float(jitter["dx"])
    oy += float(jitter["dy"])
    heat = max(0.4, min(float(PARAMS["hrr_kw"]) / 900.0, 3.5))
    lean = _lean()
    radius = 0.16 * (0.9 + 0.15 * float(PARAMS["plume_scale"]))
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=(ox, oy, oz + radius),
        segments=16,
        ring_count=8,
    )
    flow = bpy.context.active_object
    flow.name = "fire_inflow"
    flow.hide_render = True
    bpy.ops.object.modifier_add(type="FLUID")
    flow.modifiers["Fluid"].fluid_type = "FLOW"
    settings = flow.modifiers["Fluid"].flow_settings
    settings.flow_type = "BOTH"
    settings.flow_behavior = "INFLOW"
    # Humid air cools the plume before it leaves the burner. Smoke is not
    # emitted here: flame_smoke on the domain makes it when this fuel burns.
    flow["base_temperature"] = (1.7 + 0.45 * heat) * float(PARAMS["ambient"]["temperature_scale"])
    flow["base_fuel"] = 0.75 + 0.2 * heat
    settings.temperature = flow["base_temperature"]
    settings.fuel_amount = flow["base_fuel"]
    settings.density = 0.0
    settings.smoke_color = (0.45, 0.45, 0.45)
    settings.use_initial_velocity = True
    scale = float(jitter["velocity_scale"])
    settings.velocity_coord = (lean * scale, 0.0, 0.85 * scale)
    bpy.ops.object.empty_add(type="SINGLE_ARROW", location=(ox, oy, oz + 0.45))
    wind = bpy.context.active_object
    wind.name = "wind_field"
    bpy.ops.object.forcefield_toggle()
    wind.field.type = "WIND"
    wind.field.strength = lean * 2.0
    wind.rotation_euler = (0.0, math.pi / 2.0, 0.0)
    # Falling rain drags air down onto the plume top. Off until a weather sets it.
    bpy.ops.object.empty_add(type="SINGLE_ARROW", location=(ox, oy, oz + 2.0))
    drag = bpy.context.active_object
    drag.name = "precip_downdraft"
    bpy.ops.object.forcefield_toggle()
    drag.field.type = "WIND"
    drag.field.strength = 0.0
    drag.rotation_euler = (math.pi, 0.0, 0.0)
    return flow, drag


def _fit_visual(bpy, objects, package):
    """Move an imported glTF so its bounds sit on the fuel package."""
    from mathutils import Vector

    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        return
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]
    span = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    target = package["size"]
    obj.scale = tuple(
        obj.scale[axis] * target[axis] / max(span[axis], 1e-4) for axis in range(3)
    )
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = (
        (min(corner.x for corner in corners) + max(corner.x for corner in corners)) / 2.0,
        (min(corner.y for corner in corners) + max(corner.y for corner in corners)) / 2.0,
        (min(corner.z for corner in corners) + max(corner.z for corner in corners)) / 2.0,
    )
    shift = Vector(package["center"]) - Vector(center)
    obj.location = obj.location + shift
    obj.rotation_euler[2] = obj.rotation_euler[2] + float(package["yaw"])
    obj.name = "visual_" + package["name"]


# Multipliers of the weather being baked; _apply_weather sets them.
WEATHER = {"temperature_scale": 1.0, "fuel_scale": 1.0}


def _burn_amount(frame, full):
    """Cold, then one frame of half heat, then a full burn. full==1 is already hot."""
    if frame >= full:
        temperature, fuel = 2.0, 0.85
    elif frame >= full - 1:
        temperature, fuel = 1.1, 0.4
    else:
        temperature, fuel = 0.2, 0.02
    scale = float(PARAMS["ambient"]["temperature_scale"]) * float(WEATHER["temperature_scale"])
    return temperature * scale, fuel * float(WEATHER["fuel_scale"])


def _origin_amount(flow, frame, full):
    """Cold, one half-hot frame, then this burner's own full output."""
    if full <= 1 or frame >= full:
        temp_k, fuel_k = 1.0, 1.0
    elif frame >= full - 1:
        temp_k, fuel_k = 1.1 / 2.0, 0.4 / 0.85
    else:
        temp_k, fuel_k = 0.2 / 2.0, 0.02 / 0.85
    return (
        float(flow["base_temperature"]) * temp_k * float(WEATHER["temperature_scale"]),
        float(flow["base_fuel"]) * fuel_k * float(WEATHER["fuel_scale"]),
    )


def _key_origin(flow, full):
    """Keys the origin ramp. The handler repeats it when the bake skips frames."""
    settings = flow.modifiers["Fluid"].flow_settings
    settings.density = 0.0
    if full > 1:
        for mark in (max(1, full - 2), full - 1, full):
            temperature, fuel_amount = _origin_amount(flow, mark, full)
            settings.temperature = temperature
            settings.fuel_amount = fuel_amount
            settings.keyframe_insert(data_path="temperature", frame=mark)
            settings.keyframe_insert(data_path="fuel_amount", frame=mark)
    temperature, fuel_amount = _origin_amount(flow, 1, full)
    settings.temperature = temperature
    settings.fuel_amount = fuel_amount


def _key_front(settings, full):
    """Keys show the ramp. The handler repeats it when the bake skips frames."""
    if full > 1:
        for mark in (max(1, full - 2), full - 1, full):
            temperature, fuel_amount = _burn_amount(mark, full)
            settings.temperature = temperature
            settings.fuel_amount = fuel_amount
            settings.keyframe_insert(data_path="temperature", frame=mark)
            settings.keyframe_insert(data_path="fuel_amount", frame=mark)
    temperature, fuel_amount = _burn_amount(1, full)
    settings.temperature = temperature
    settings.fuel_amount = fuel_amount


def _place_fuels(bpy, origin):
    """One hidden inflow per slab. The origin burner is driven even with no packages."""
    # True marks the origin, whose full output is its own base temperature.
    drivers = [(origin, int(PARAMS["warmup"]), True)]
    for package in PARAMS.get("packages") or []:
        if not package.get("in_domain"):
            continue
        for index, front in enumerate(package.get("fronts") or []):
            bpy.ops.mesh.primitive_cube_add(location=tuple(front["center"]), scale=tuple(front["scale"]))
            flow = bpy.context.active_object
            flow.name = "fuel_" + package["name"] + "_" + str(index)
            flow.hide_render = True
            bpy.ops.object.modifier_add(type="FLUID")
            flow.modifiers["Fluid"].fluid_type = "FLOW"
            settings = flow.modifiers["Fluid"].flow_settings
            settings.flow_type = "BOTH"
            settings.flow_behavior = "INFLOW"
            settings.density = 0.0
            settings.smoke_color = (0.45, 0.45, 0.45)
            settings.use_initial_velocity = True
            settings.velocity_coord = (0.0, 0.0, 0.35)
            full = int(front["full_frame"])
            _key_front(settings, full)
            drivers.append((flow, full, False))
        path = package.get("asset_path") or ""
        if path and os.path.isfile(path):
            before = set(bpy.data.objects)
            try:
                bpy.ops.import_scene.gltf(filepath=path)
            except (RuntimeError, AttributeError):
                before = set()
            imported = [obj for obj in bpy.data.objects if obj not in before] if before else []
            if imported:
                _fit_visual(bpy, imported, package)

    # Blender passes (scene, depsgraph), so drivers must not be a default argument.
    def _spread(scene, _depsgraph=None):
        frame = scene.frame_current
        for obj, full, is_origin in drivers:
            flow_settings = obj.modifiers["Fluid"].flow_settings
            if is_origin:
                temperature, fuel_amount = _origin_amount(obj, frame, full)
            else:
                temperature, fuel_amount = _burn_amount(frame, full)
            flow_settings.temperature = temperature
            flow_settings.fuel_amount = fuel_amount

    bpy.app.handlers.frame_change_pre.append(_spread)
    return [(obj, full) for obj, full, is_origin in drivers if not is_origin]


def _apply_weather(bpy, look, gas, flow, downdraft, fronts, precipitation, snow_objects):
    """Set up the bake for one weather: flame, smoke, downdraft, ground and particles."""
    physics = look["physics"]
    WEATHER["temperature_scale"] = float(physics["temperature_scale"])
    WEATHER["fuel_scale"] = float(physics["fuel_scale"])
    settings = gas.modifiers["Fluid"].domain_settings
    settings.burning_rate = float(gas["base_burning_rate"]) * float(physics["burning_scale"])
    settings.flame_smoke = float(gas["base_flame_smoke"]) * float(physics["smoke_scale"])
    settings.flame_smoke_color = tuple(physics["smoke_color"])
    _key_origin(flow, int(PARAMS["warmup"]))
    inflow = flow.modifiers["Fluid"].flow_settings
    inflow.smoke_color = tuple(physics["smoke_color"])
    downdraft.field.strength = float(physics["downdraft"])
    for front, full in fronts:
        _key_front(front.modifiers["Fluid"].flow_settings, full)
    # A lid over the rain footprint keeps the ground dry. The flame still cools.
    _set_surface("dry" if look.get("ground_covered") else physics["surface"], snow_objects)
    for kind, modifier in precipitation.items():
        active = look["precipitation"] == kind
        modifier.show_render = active
        modifier.show_viewport = active


def _bake(bpy, gas, key):
    """Bake the gas into its own cache directory, one per weather."""
    settings = gas.modifiers["Fluid"].domain_settings
    cache_dir = os.path.abspath(os.path.join(os.getcwd(), "fluid_cache", key))
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)
    settings.cache_directory = cache_dir
    for obj in bpy.context.view_layer.objects:
        obj.select_set(False)
    gas.select_set(True)
    bpy.context.view_layer.objects.active = gas
    with bpy.context.temp_override(object=gas, active_object=gas, selected_objects=[gas]):
        baked = bpy.ops.fluid.bake_all()
    if "FINISHED" not in baked:
        raise RuntimeError("mantaflow bake failed: %s" % (baked,))
    return cache_dir


# Ground and wall materials whose color and roughness a weather changes.
SURFACES = []


def _textured(bpy, name, color, path):
    """Base color runs through a Hue/Saturation node, so rain can darken it."""
    material = _material(bpy, name, color)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    tint = nodes.new("ShaderNodeHueSaturation")
    tint.inputs["Color"].default_value = (*color, 1.0)
    if path and os.path.isfile(path):
        image = bpy.data.images.load(path)
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        links.new(texture.outputs["Color"], tint.inputs["Color"])
    links.new(tint.outputs["Color"], shader.inputs["Base Color"])
    roughness = shader.inputs["Roughness"]
    SURFACES.append({"tint": tint, "roughness": roughness, "dry": float(roughness.default_value)})
    return material


def _set_surface(surface, snow_objects):
    """Dry, wet (darker and glossy) or snow (white layer on the floor and on top of solids)."""
    for item in SURFACES:
        wet = surface == "wet"
        item["tint"].inputs["Value"].default_value = 0.55 if wet else 1.0
        item["roughness"].default_value = 0.12 if wet else item["dry"]
    for obj in snow_objects:
        obj.hide_render = surface != "snow"


def _snow_cover(bpy):
    """A thin white layer on the floor and on each solid top. Hidden until it snows."""
    domain = PARAMS["domain"]
    material = _material(bpy, "snow_material", (0.86, 0.88, 0.92))
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = 0.85
    covers = []
    bpy.ops.mesh.primitive_plane_add(
        size=max(domain[0], domain[1]) * 1.1,
        location=(domain[0] / 2.0, domain[1] / 2.0, 0.004),
    )
    covers.append(bpy.context.active_object)
    for item in PARAMS["obstacles"]:
        x0, x1, y0, y1, _z0, z1 = item["xb"]
        if z1 >= domain[2] - 0.01:
            continue
        covers.append(_box(bpy, "snow_" + item["name"], (x0, x1, y0, y1, z1, z1 + 0.03)))
    for obj in covers:
        obj.data.materials.append(material)
        obj.hide_render = True
    return covers


def _camera(bpy, pose):
    from mathutils import Vector

    bpy.ops.object.camera_add(location=tuple(pose["location"]))
    camera = bpy.context.active_object
    camera.name = "camera_" + pose["key"]
    camera.data.lens = float(pose["focal_mm"])
    camera.data.clip_start = 0.05
    camera.data.clip_end = max(PARAMS["domain"]) * 3.0
    camera.rotation_euler = (Vector(pose["target"]) - camera.location).to_track_quat("-Z", "Y").to_euler()
    return camera


def _light_drops(bpy, obj):
    """A soft sun that lights only the drops, so rain and snow read against a dark sky.

    Without it the flakes are lit by the dim world alone and show as dark
    specks. Light linking keeps the flame, the ground and the mask untouched.
    """
    name = "precip_light"
    lamp = bpy.data.objects.get(name)
    if lamp is None:
        data = bpy.data.lights.new(name, type="SUN")
        data.energy = 5.0
        data.angle = 1.2
        lamp = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(lamp)
        lamp.rotation_euler = (0.6, 0.0, 0.4)
        receivers = bpy.data.collections.new("precip_receivers")
        try:
            lamp.light_linking.receiver_collection = receivers
        except AttributeError:
            # No light linking in this Blender: light nothing rather than everything.
            data.energy = 0.0
    collection = lamp.light_linking.receiver_collection if hasattr(lamp, "light_linking") else None
    if collection is not None:
        collection.objects.link(obj)


def _drop_template(bpy, kind, motion):
    """The mesh each particle instances, parked far below the floor. Rain is a streak, snow a flake."""
    import bmesh

    mesh = bpy.data.meshes.new("drop_" + kind)
    bm = bmesh.new()
    if kind == "rain":
        # A vertical streak, leaning downwind about as far as the drops drift.
        drift = 0.73 * _lean() * float(motion["wind"])
        tilt = math.atan2(drift, float(motion["speed"]))
        bmesh.ops.create_cone(
            bm, cap_ends=True, segments=4, radius1=0.004, radius2=0.004, depth=0.3,
            matrix=__import__("mathutils").Matrix.Rotation(-tilt, 4, "Y"),
        )
    else:
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=0.009)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("drop_" + kind, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, -1000.0)
    _light_drops(bpy, obj)
    color = (0.7, 0.75, 0.82) if kind == "rain" else (0.92, 0.93, 0.95)
    alpha = 0.45 if kind == "rain" else 1.0
    material = _material(bpy, "drop_" + kind + "_material", color, alpha=alpha)
    _fade_near_camera(material)
    obj.data.materials.append(material)
    return obj


def _fade_near_camera(material):
    """Drops closer than about a meter and a half vanish, so none blots out the lens."""
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.get("Material Output")
    shader = nodes.get("Principled BSDF")
    camera = nodes.new("ShaderNodeCameraData")
    near = nodes.new("ShaderNodeMapRange")
    near.clamp = True
    near.inputs["From Min"].default_value = 1.0
    near.inputs["From Max"].default_value = 2.0
    near.inputs["To Min"].default_value = 0.0
    near.inputs["To Max"].default_value = 1.0
    links.new(camera.outputs["View Distance"], near.inputs["Value"])
    clear = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(near.outputs["Result"], mix.inputs["Fac"])
    links.new(clear.outputs["BSDF"], mix.inputs[1])
    links.new(shader.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def _precipitation(bpy, kind, motion):
    """Rain or snow as particles: launched down, pulled by gravity, slowed by drag and damping, pushed by wind.

    Blender does not simulate particles before frame 1, so there is no
    pre-roll. Instead they are born throughout the air over the fire and the
    cameras and live ``lifetime`` frames, which keeps the same density in
    every rendered frame after the warmup.
    """
    box = PARAMS["precip_box"]
    last = int(PARAMS["warmup"]) + int(PARAMS["frame_count"])
    lifetime = int(motion["lifetime"])
    top = float(box["center"][2])
    bpy.ops.mesh.primitive_cube_add(location=(float(box["center"][0]), float(box["center"][1]), top / 2.0))
    emitter = bpy.context.active_object
    emitter.name = "precip_" + kind
    emitter.scale = (float(box["size"][0]) / 2.0, float(box["size"][1]) / 2.0, top / 2.0)
    # Hiding the emitter itself would hide its particles too; hide only the box.
    emitter.show_instancer_for_render = False
    emitter.show_instancer_for_viewport = False
    emitter.visible_shadow = False
    bpy.ops.object.particle_system_add()
    modifier = emitter.modifiers[-1]
    system = emitter.particle_systems[-1]
    settings = system.settings
    settings.name = "precip_" + kind + "_settings"
    settings.frame_start = 1
    settings.frame_end = last
    settings.lifetime = lifetime
    # One particle step per rendered frame.
    settings.timestep = 1.0 / float(bpy.context.scene.render.fps)
    volume = float(box["size"][0]) * float(box["size"][1]) * top
    alive = volume * float(motion["per_m3"])
    settings.count = int(min(60000, max(300, alive * last / max(lifetime, 1))))
    settings.emit_from = "VOLUME"
    settings.distribution = "RAND"
    settings.normal_factor = 0.0
    settings.object_align_factor = (0.0, 0.0, -float(motion["speed"]))
    settings.factor_random = 0.1
    settings.physics_type = "NEWTON"
    settings.drag_factor = float(motion["drag"])
    settings.damping = float(motion["damping"])
    settings.brownian_factor = float(motion["brownian"])
    settings.mass = 1.0
    settings.effector_weights.gravity = float(motion["gravity"])
    settings.effector_weights.wind = float(motion["wind"])
    settings.render_type = "OBJECT"
    settings.instance_object = _drop_template(bpy, kind, motion)
    settings.particle_size = 1.0
    settings.size_random = 0.4
    # Store a fixed upright rotation; without one the cached frames come back
    # with random streak directions.
    settings.use_rotations = True
    settings.rotation_mode = "NONE"
    settings.phase_factor = 0.0
    settings.rotation_factor_random = 0.0
    modifier.show_render = False
    modifier.show_viewport = False
    return modifier


def _ember_material(bpy, name, color, emission):
    """A speck that shows in the picture and in the ember mask, not the flame mask."""
    material = _material(bpy, name, color, emission=emission)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    value = nodes.new("ShaderNodeValue")
    value.outputs[0].default_value = 1.0
    aov = nodes.new("ShaderNodeOutputAOV")
    aov.name = "ember"
    aov.aov_name = "ember"
    links.new(value.outputs["Value"], aov.inputs["Value"])
    return material


def _ember_template(bpy, kind):
    """One instanced speck, parked below the floor. Hot embers glow; ash is only gray."""
    import bmesh

    mesh = bpy.data.meshes.new("speck_" + kind)
    bm = bmesh.new()
    # Small enough to read as sparks and flakes at 256px, not as rocks.
    radius = 0.012 if kind == "ember" else 0.018
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("speck_" + kind, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, -1000.0)
    if kind == "ember":
        material = _ember_material(bpy, "ember_material", (1.0, 0.32, 0.05), 18.0)
    else:
        material = _ember_material(bpy, "ash_material", (0.28, 0.27, 0.25), 0.6)
    obj.data.materials.append(material)
    return obj


def _ember_system(bpy, emitter, name, count, rise, gravity, template):
    bpy.context.view_layer.objects.active = emitter
    emitter.select_set(True)
    bpy.ops.object.particle_system_add()
    system = emitter.particle_systems[-1]
    settings = system.settings
    settings.name = name
    warmup = int(PARAMS["warmup"])
    last = warmup + int(PARAMS["frame_count"])
    # Born with the first rendered frame, not during the cold warmup.
    settings.frame_start = warmup
    settings.frame_end = max(warmup, last)
    settings.lifetime = max(1, int(PARAMS["frame_count"]))
    settings.timestep = 1.0 / float(bpy.context.scene.render.fps)
    settings.count = int(count)
    settings.emit_from = "VOLUME"
    settings.distribution = "RAND"
    settings.normal_factor = 0.0
    settings.object_align_factor = (0.0, 0.0, float(rise))
    settings.factor_random = 0.65
    settings.physics_type = "NEWTON"
    settings.drag_factor = 0.18
    settings.damping = 0.02
    settings.brownian_factor = 0.45
    settings.mass = 1.0
    settings.effector_weights.gravity = float(gravity)
    settings.effector_weights.wind = 1.2
    settings.render_type = "OBJECT"
    settings.instance_object = template
    settings.particle_size = 0.7
    settings.size_random = 0.45
    settings.use_rotations = True
    settings.rotation_mode = "NONE"
    settings.phase_factor = 0.0
    settings.rotation_factor_random = 0.0


def _embers(bpy):
    """Ash and glowing embers born at the inflow, lifted, then pulled back down."""
    ox, oy, oz = PARAMS["origin"]
    bpy.ops.mesh.primitive_cube_add(location=(ox, oy, oz + 0.22), scale=(0.14, 0.14, 0.14))
    emitter = bpy.context.active_object
    emitter.name = "embers"
    # Hiding the emitter itself would hide its particles too.
    emitter.show_instancer_for_render = False
    emitter.show_instancer_for_viewport = False
    emitter.visible_shadow = False
    _ember_system(bpy, emitter, "embers_hot", 40, 2.2, 0.28, _ember_template(bpy, "ember"))
    _ember_system(bpy, emitter, "embers_ash", 32, 1.0, 0.9, _ember_template(bpy, "ash"))
    return emitter


def _step_particles(bpy, scene, first, last):
    """Particles are solved frame by frame. Walk the range once before jumping around."""
    for frame in range(first, last + 1):
        scene.frame_set(frame)


def _world(bpy):
    world = bpy.data.worlds.new("fire_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    return world.node_tree.nodes.get("Background")


def _apply_look(bpy, look, background, suns):
    if background is not None:
        background.inputs["Color"].default_value = (*look["world_color"], 1.0)
        background.inputs["Strength"].default_value = float(look["world_strength"])
    weather = look["weather"]
    factor = 1.0 if weather == "晴" else 0.0 if weather == "雨" else 0.4 / 2.2
    for sun in suns:
        if sun.get("custom"):
            energy = float(sun["base_energy"]) * factor
        else:
            energy = float(look["sun"])
        sun.data.energy = energy
        sun.hide_render = energy <= 0.0


def _file_mask(tree, output_dir, name, socket):
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.base_path = os.path.join(output_dir, "mask")
    output.format.file_format = "PNG"
    output.format.color_mode = "BW"
    output.format.color_depth = "8"
    tree.links.new(socket, output.inputs[0])
    return output


def _mask_outputs(bpy, scene, output_dir):
    """Flame from the emit pass, smoke from volume scatter, embers from their AOV.

    A volume shader AOV stays empty in Cycles, so the flame and the plume come
    from passes that the volume does write. Embers are a surface, and their AOV
    is punched out of the emit pass so a spark is not labeled as flame.
    """
    layer = bpy.context.view_layer
    layer.use_pass_emit = True
    layer.cycles.use_pass_volume_direct = True
    layer.cycles.use_pass_volume_indirect = True
    ember = layer.aovs.add()
    ember.name = "ember"
    ember.type = "VALUE"
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.layer = layer.name
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    emit = tree.nodes.new("CompositorNodeRGBToBW")
    tree.links.new(layers.outputs["Emit"], emit.inputs[0])
    added = tree.nodes.new("CompositorNodeMixRGB")
    added.blend_type = "ADD"
    added.inputs[0].default_value = 1.0
    tree.links.new(layers.outputs["VolumeDir"], added.inputs[1])
    tree.links.new(layers.outputs["VolumeInd"], added.inputs[2])
    smoke_luma = tree.nodes.new("CompositorNodeRGBToBW")
    tree.links.new(added.outputs[0], smoke_luma.inputs[0])
    # A spark only covers part of a pixel, so its AOV is well below 1 while its
    # emission is still bright. Any ember signal knocks that pixel out of the flame.
    hot = tree.nodes.new("CompositorNodeMath")
    hot.operation = "GREATER_THAN"
    hot.inputs[1].default_value = 0.06
    tree.links.new(layers.outputs["ember"], hot.inputs[0])
    inv = tree.nodes.new("CompositorNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    tree.links.new(hot.outputs[0], inv.inputs[1])
    flame = tree.nodes.new("CompositorNodeMath")
    flame.operation = "MULTIPLY"
    tree.links.new(emit.outputs[0], flame.inputs[0])
    tree.links.new(inv.outputs[0], flame.inputs[1])
    # Drop the bright core from the plume, then lift the linear scatter into 8-bit.
    clamped = tree.nodes.new("CompositorNodeMath")
    clamped.operation = "MINIMUM"
    clamped.inputs[1].default_value = 1.0
    tree.links.new(emit.outputs[0], clamped.inputs[0])
    opened = tree.nodes.new("CompositorNodeMath")
    opened.operation = "SUBTRACT"
    opened.inputs[0].default_value = 1.0
    tree.links.new(clamped.outputs[0], opened.inputs[1])
    smoke = tree.nodes.new("CompositorNodeMath")
    smoke.operation = "MULTIPLY"
    tree.links.new(smoke_luma.outputs[0], smoke.inputs[0])
    tree.links.new(opened.outputs[0], smoke.inputs[1])
    gain = tree.nodes.new("CompositorNodeMath")
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = 4.0
    tree.links.new(smoke.outputs[0], gain.inputs[0])
    return {
        "flame": _file_mask(tree, output_dir, "flame", flame.outputs[0]),
        "smoke": _file_mask(tree, output_dir, "smoke", gain.outputs[0]),
        "ember": _file_mask(tree, output_dir, "ember", layers.outputs["ember"]),
    }


def build_scene():
    """Create the scenario geometry, then bake the gas flame and render every view once per weather."""
    bpy = _require_bpy()
    _clear(bpy)
    domain = PARAMS["domain"]
    origin = PARAMS["origin"]
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=tuple(origin))
    bpy.context.active_object.name = "fire_origin"
    textures = PARAMS.get("surface_textures") or {}
    wall = _textured(bpy, "obstacle_material", (0.16, 0.15, 0.14), textures.get("obstacle") or "")
    tank = _textured(bpy, "tank_material", (0.35, 0.36, 0.38), textures.get("tank") or "")
    for item in PARAMS["obstacles"]:
        obj = _box(bpy, item["name"], item["xb"])
        material = tank if str(item["name"]).startswith("tank") else wall
        obj.data.materials.append(material)
        _collider(bpy, obj)
        _particle_hit(bpy, obj)
    floor = _textured(bpy, "floor_material", (0.035, 0.045, 0.05), textures.get("floor") or "")
    # The plane is the only floor. A slab on top of it hid the floor texture.
    bpy.ops.mesh.primitive_plane_add(
        size=max(domain[0], domain[1]) * 1.1,
        location=(domain[0] / 2.0, domain[1] / 2.0, 0.0),
    )
    bpy.context.active_object.data.materials.append(floor)
    scene = bpy.context.scene
    scene["scenario"] = PARAMS["scenario"]
    scene["wind_speed"] = PARAMS["wind_speed"]
    scene["temperature"] = PARAMS["temperature"]
    scene["humidity"] = PARAMS["humidity"]
    scene["fuel_type"] = PARAMS["fuel_type"]
    scene["hrr_kw"] = PARAMS["hrr_kw"]
    scene["soot_yield"] = PARAMS["soot_yield"]
    cameras = [(_camera(bpy, pose), pose) for pose in PARAMS["cameras"]]
    scene.camera = cameras[0][0]
    background = _world(bpy)
    suns = []
    if PARAMS.get("custom_lights"):
        for spec in PARAMS.get("lights") or []:
            if not spec.get("enabled", True):
                continue
            kind = spec["type"]
            bpy.ops.object.light_add(type=kind, location=tuple(spec["location"]))
            lamp = bpy.context.active_object
            lamp.name = "light_" + spec["id"]
            lamp.rotation_euler = tuple(spec["rotation"])
            data = lamp.data
            data.color = tuple(spec["color"])
            data.energy = float(spec["energy"])
            if kind == "SUN":
                lamp["custom"] = True
                lamp["base_energy"] = float(spec["energy"])
                data.angle = float(spec["size"])
                suns.append(lamp)
            elif kind == "AREA":
                data.shape = "RECTANGLE"
                data.size = float(spec["size"])
                data.size_y = float(spec["size_y"])
            elif kind == "SPOT":
                data.shadow_soft_size = float(spec["size"])
                data.spot_size = float(spec["spot_size"])
                data.spot_blend = float(spec["spot_blend"])
            else:
                data.shadow_soft_size = float(spec["size"])
    else:
        # A downward square leaves a hard patch on the floor and dark wall slabs.
        # A sphere above the fire falls off in every direction.
        fill_spec = PARAMS["fill_light"]
        bpy.ops.object.light_add(type="POINT", location=tuple(fill_spec["location"]))
        fill = bpy.context.active_object.data
        fill.energy = float(fill_spec["energy"])
        fill.color = (0.75, 0.8, 0.9)
        fill.shadow_soft_size = float(fill_spec["size"])
        bpy.ops.object.light_add(type="SUN", location=(origin[0], origin[1], domain[2]))
        sun = bpy.context.active_object
        sun.rotation_euler = (0.7, 0.2, 0.9)
        sun.data.color = (1.0, 0.96, 0.9)
        suns.append(sun)
    scene.render.fps = 12
    snow_objects = _snow_cover(bpy) if any(look["physics"]["surface"] == "snow" for look in PARAMS["looks"]) else []
    gas = _gas_domain(bpy)
    flow, downdraft = _inflow(bpy)
    fronts = _place_fuels(bpy, flow)
    _embers(bpy)
    # After the wind field exists, so the particles feel it.
    motions = {look["precipitation"]: look["physics"]["motion"] for look in PARAMS["looks"] if look["precipitation"]}
    scene.frame_start = 1
    precipitation = {kind: _precipitation(bpy, kind, motions[kind]) for kind in sorted(motions)}
    bpy.app.handlers.frame_change_pre.append(_cull_precip)
    warmup = int(PARAMS["warmup"])
    frame_count = int(PARAMS["frame_count"])
    scene.frame_end = warmup + frame_count
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(PARAMS["simulation"]["samples"])
    # Diffuse bounces stay off so the floor is not lit by the flame. Volume
    # bounces are what make the smoke visible, and max_bounces has to allow them.
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 2
    try:
        scene.cycles.use_denoising = True
    except AttributeError:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    output_dir = PARAMS["render_output"]
    if output_dir:
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
        masks = _mask_outputs(bpy, scene, output_dir)
        # Rain and snow change the flame, so every weather is its own bake.
        for look_index, look in enumerate(PARAMS["looks"]):
            _apply_weather(bpy, look, gas, flow, downdraft, fronts, precipitation, snow_objects)
            print("FIREPHANTOM bake %s %s" % (look["key"], look["weather"]), flush=True)
            cache_dir = _bake(bpy, gas, look["key"])
            _apply_look(bpy, look, background, suns)
            # Rain, snow and embers are solved one frame at a time. Stop short of
            # the first rendered frame, so setting it is a real frame change.
            if warmup > 1:
                _step_particles(bpy, scene, 1, warmup - 1)
            # The burner stays cold until this frame, so earlier frames are not a fire.
            for frame_index in range(frame_count):
                scene.frame_set(warmup + frame_index)
                for camera_index, (camera, _pose) in enumerate(cameras):
                    scene.camera = camera
                    stem = "v%d_l%d_f%03d" % (camera_index, look_index, frame_index)
                    scene.render.filepath = os.path.join(output_dir, "rgb", stem + ".png")
                    for name, node in masks.items():
                        node.file_slots[0].path = stem + "_" + name + "_"
                    bpy.ops.render.render(write_still=True)
            shutil.rmtree(cache_dir, ignore_errors=True)
    return scene


if __name__ == "__main__":
    try:
        build_scene()
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
'''


# Representative flame the camera frames. Poses stay near this height.
FLAME_HEIGHT = 1.15
SHAFT_HALF = 1.0
_SENSOR_MM = 36.0


def cut_shaft(
    boxes: list[tuple[str, tuple[float, float, float, float, float, float]]],
    shaft: tuple[float, float, float, float, float, float],
) -> list[tuple[str, tuple[float, float, float, float, float, float]]]:
    """Remove an axis-aligned shaft so a flame column is not sealed inside a solid."""
    pieces: list[tuple[str, tuple[float, float, float, float, float, float]]] = []
    for name, box in boxes:
        kept = _subtract_box(box, shaft)
        if len(kept) == 1 and kept[0] == box:
            pieces.append((name, box))
            continue
        for index, piece in enumerate(kept):
            pieces.append((f"{name}_{index}", piece))
    return pieces


def _subtract_box(
    box: tuple[float, float, float, float, float, float],
    shaft: tuple[float, float, float, float, float, float],
) -> list[tuple[float, float, float, float, float, float]]:
    ox0, ox1, oy0, oy1, oz0, oz1 = box
    sx0, sx1, sy0, sy1, sz0, sz1 = shaft
    ix0, ix1 = max(ox0, sx0), min(ox1, sx1)
    iy0, iy1 = max(oy0, sy0), min(oy1, sy1)
    iz0, iz1 = max(oz0, sz0), min(oz1, sz1)
    if ix0 >= ix1 - 1e-6 or iy0 >= iy1 - 1e-6 or iz0 >= iz1 - 1e-6:
        return [box]
    pieces = []
    if oz0 < iz0 - 1e-6:
        pieces.append((ox0, ox1, oy0, oy1, oz0, iz0))
    if iz1 < oz1 - 1e-6:
        pieces.append((ox0, ox1, oy0, oy1, iz1, oz1))
    if ox0 < ix0 - 1e-6:
        pieces.append((ox0, ix0, oy0, oy1, iz0, iz1))
    if ix1 < ox1 - 1e-6:
        pieces.append((ix1, ox1, oy0, oy1, iz0, iz1))
    if oy0 < iy0 - 1e-6:
        pieces.append((ix0, ix1, oy0, iy0, iz0, iz1))
    if iy1 < oy1 - 1e-6:
        pieces.append((ix0, ix1, iy1, oy1, iz0, iz1))
    return pieces


def visible_obstacles(spec: ScenarioSpec) -> list[dict[str, object]]:
    """Scenario solids with a vertical gap around the flame."""
    ox, oy, _oz = spec.origin
    _length, _width, height = spec.domain
    shaft = (
        ox - SHAFT_HALF,
        ox + SHAFT_HALF,
        oy - SHAFT_HALF,
        oy + SHAFT_HALF,
        0.0,
        height,
    )
    boxes = [(item.name, item.xb) for item in spec.obstacles]
    return [{"name": name, "xb": list(piece)} for name, piece in cut_shaft(boxes, shaft)]


def _point_in_box(point: np.ndarray, box: tuple[float, ...] | list[float], margin: float = 0.0) -> bool:
    x, y, z = (float(value) for value in point)
    x0, x1, y0, y1, z0, z1 = box
    return (x0 - margin) < x < (x1 + margin) and (y0 - margin) < y < (y1 + margin) and (z0 - margin) < z < (z1 + margin)


def point_in_solid(spec: ScenarioSpec, point) -> bool:
    """True when the point sits inside a solid, after the flame shaft is cut out."""
    location = np.asarray([float(value) for value in point], dtype=np.float64)
    return any(_point_in_box(location, tuple(item["xb"])) for item in visible_obstacles(spec))


def segment_hits_box(start: np.ndarray, end: np.ndarray, box: tuple[float, ...] | list[float]) -> bool:
    """True when the open segment from start to end enters the box before the target."""
    origin = np.asarray(start, dtype=np.float64)
    direction = np.asarray(end, dtype=np.float64) - origin
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        lo = float(box[axis * 2])
        hi = float(box[axis * 2 + 1])
        if abs(float(direction[axis])) < 1e-9:
            if origin[axis] < lo or origin[axis] > hi:
                return False
            continue
        inv = 1.0 / float(direction[axis])
        t1 = (lo - float(origin[axis])) * inv
        t2 = (hi - float(origin[axis])) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return t_min < 0.98 and t_max > 0.02


def _inside_domain(point: np.ndarray, domain: tuple[float, float, float], margin: float) -> bool:
    length, width, height = domain
    x, y, z = (float(value) for value in point)
    return margin < x < length - margin and margin < y < width - margin and margin < z < height - margin


def _focal_length(distance: float) -> float:
    """Lens that makes the flame about half the frame height on a 36 mm sensor."""
    flame_angle = 2.0 * math.atan((FLAME_HEIGHT / 2.0) / distance)
    fov = flame_angle / 0.5
    focal = (_SENSOR_MM / 2.0) / math.tan(fov / 2.0)
    return float(min(120.0, max(24.0, focal)))


def _frame_fraction(distance: float, focal_mm: float) -> float:
    flame_angle = 2.0 * math.atan((FLAME_HEIGHT / 2.0) / distance)
    fov = 2.0 * math.atan((_SENSOR_MM / 2.0) / focal_mm)
    return float(flame_angle / fov)


_VIEW_SEARCH = {
    # Level shots around the flame. This was the only camera before viewpoints moved it.
    "侧视": {"scales": (2.6, 3.4, 4.6, 6.5, 8.0), "pitches": (0.12, 0.28, 0.0, -0.55, -1.3, -2.6)},
    # Steep shots from above. Short distances keep the camera under a room ceiling.
    "俯视": {"scales": (1.6, 2.0, 2.6, 3.4, 4.6), "pitches": (-2.6, -2.0, -1.3)},
    # A standing observer: eye height fixed, close to the fire.
    "第一人称": {"scales": (1.8, 2.2, 2.6, 3.4), "eye_height": 1.6},
}


def _search_pose(spec: ScenarioSpec, viewpoint: str) -> dict[str, object] | None:
    boxes = [tuple(item["xb"]) for item in visible_obstacles(spec)]
    ox, oy, oz = spec.origin
    target = np.array([ox, oy, oz + FLAME_HEIGHT * 0.45], dtype=np.float64)
    length, width, _height = spec.domain
    center = np.array([length * 0.5, width * 0.5, float(target[2])], dtype=np.float64)
    away = center - target
    away[2] = 0.0
    if float(np.linalg.norm(away)) < 1e-3:
        away = np.array([1.0, 0.35, 0.0], dtype=np.float64)
    base = math.atan2(float(away[1]), float(away[0]))
    search = _VIEW_SEARCH[viewpoint]
    eye_height = search.get("eye_height")
    for scale in search["scales"]:
        distance = FLAME_HEIGHT * scale
        if eye_height is not None:
            rise = float(oz) + float(eye_height) - float(target[2])
            if abs(rise) >= distance:
                continue
            pitches = (-rise / math.sqrt(distance * distance - rise * rise),)
        else:
            pitches = search["pitches"]
        for pitch in pitches:
            for step in range(16):
                theta = base + step * (math.pi / 8.0)
                direction = np.array([math.cos(theta), math.sin(theta), pitch], dtype=np.float64)
                direction /= np.linalg.norm(direction)
                camera = target - direction * distance
                if not _inside_domain(camera, spec.domain, 0.2):
                    continue
                if any(_point_in_box(camera, box, margin=0.05) for box in boxes):
                    continue
                if any(segment_hits_box(camera, target, box) for box in boxes):
                    continue
                focal = _focal_length(distance)
                return {
                    "viewpoint": viewpoint,
                    "location": [float(value) for value in camera],
                    "target": [float(value) for value in target],
                    "focal_mm": focal,
                    "frame_fraction": _frame_fraction(distance, focal),
                }
    return None


def camera_pose(spec: ScenarioSpec, viewpoint: str = "侧视") -> dict[str, object]:
    """A camera inside the domain whose view ray reaches the flame.

    When the scenario has no room for the requested viewpoint, the side view
    is used and the pose says so in ``fallback``.
    """
    if viewpoint not in _VIEW_SEARCH:
        raise ValueError(f"未知视角 {viewpoint!r}")
    pose = _search_pose(spec, viewpoint)
    if pose is not None:
        return {**pose, "fallback": False}
    if viewpoint != "侧视":
        side = _search_pose(spec, "侧视")
        if side is not None:
            return {**side, "viewpoint": viewpoint, "fallback": True}
    raise RuntimeError(f"{spec.name} 里没有能看见火焰的相机位置")


def pose_from_override(viewpoint: str, override: dict) -> dict[str, object]:
    """A user camera. ``location``, ``target`` and ``focal_mm`` are scene meters and millimetres."""
    location = np.array([float(value) for value in override["location"]], dtype=np.float64)
    target = np.array([float(value) for value in override["target"]], dtype=np.float64)
    distance = float(np.linalg.norm(target - location))
    focal = float(override["focal_mm"])
    return {
        "viewpoint": viewpoint,
        "location": [float(value) for value in location],
        "target": [float(value) for value in target],
        "focal_mm": focal,
        "frame_fraction": _frame_fraction(max(distance, 1e-3), focal),
        "fallback": False,
    }


def _override_in_solid(spec: ScenarioSpec, pose: object) -> bool:
    location = pose.get("location") if isinstance(pose, dict) else None
    if not isinstance(location, (list, tuple)) or len(location) != 3:
        return False
    try:
        return point_in_solid(spec, location)
    except (TypeError, ValueError):
        return False


def camera_poses(
    spec: ScenarioSpec,
    viewpoints: list[str] | tuple[str, ...],
    overrides: dict | None = None,
) -> list[dict[str, object]]:
    """One pose per viewpoint. A stored override replaces the searched pose.

    A stored camera inside a wall or ceiling is ignored. The searched pose is
    used instead, so an old scene does not render from inside the solid.
    """
    custom = overrides or {}
    poses = []
    for viewpoint in viewpoints:
        pose = custom.get(viewpoint)
        if pose and _override_in_solid(spec, pose):
            pose = None
        poses.append(pose_from_override(viewpoint, pose) if pose else camera_pose(spec, viewpoint))
    return poses


# Outdoor light is kept dim next to the flame. A bright sky washes the
# blackbody volume out, and the flame mask comes from the emission pass anyway.
_SKY = {
    ("室内", "晴"): {"color": (0.015, 0.016, 0.018), "strength": 0.04, "sun": 0.0},
    ("室内", "雨"): {"color": (0.012, 0.014, 0.018), "strength": 0.035, "sun": 0.0},
    ("室内", "雪"): {"color": (0.02, 0.021, 0.024), "strength": 0.05, "sun": 0.0},
    ("室外", "晴"): {"color": (0.42, 0.56, 0.78), "strength": 0.35, "sun": 2.2},
    ("室外", "雨"): {"color": (0.30, 0.32, 0.35), "strength": 0.2, "sun": 0.0},
    ("室外", "雪"): {"color": (0.62, 0.64, 0.68), "strength": 0.3, "sun": 0.4},
}
_PRECIPITATION = {"晴": None, "雨": "rain", "雪": "snow"}


def look_settings(scene: str, weather: str) -> dict[str, object]:
    """World light and precipitation for one environment and weather."""
    try:
        sky = _SKY[(scene, weather)]
    except KeyError as exc:
        raise ValueError(f"未知环境或天气 {scene!r} {weather!r}") from exc
    return {
        "scene": scene,
        "weather": weather,
        "world_color": list(sky["color"]),
        "world_strength": float(sky["strength"]),
        "sun": float(sky["sun"]),
        "precipitation": _PRECIPITATION[weather],
        "physics": weather_physics(weather),
    }


# What each weather does to the flame and the ground. One bake per weather.
# Rain cools the inflow and burns less fuel, and the steam makes more, lighter
# smoke; falling drops drag the plume top down. Snow cools a little.
_WEATHER_PHYSICS = {
    "晴": {
        "temperature_scale": 1.0, "fuel_scale": 1.0, "burning_scale": 1.0, "smoke_scale": 1.0,
        "smoke_color": [0.45, 0.45, 0.45], "downdraft": 0.0, "surface": "dry",
    },
    "雨": {
        "temperature_scale": 0.75, "fuel_scale": 0.7, "burning_scale": 0.8, "smoke_scale": 1.6,
        "smoke_color": [0.68, 0.70, 0.72], "downdraft": 0.6, "surface": "wet",
    },
    "雪": {
        "temperature_scale": 0.9, "fuel_scale": 0.9, "burning_scale": 0.92, "smoke_scale": 1.2,
        "smoke_color": [0.55, 0.56, 0.58], "downdraft": 0.1, "surface": "snow",
    },
}
# Particle motion. ``speed`` is the launch speed (m/s) down from the emitter,
# ``wind`` is how much the wind field pushes them. ``damping`` takes that share of the velocity away each frame, which gives a
# terminal speed: about 7.4 * gravity m/s down at 0.1, so snow settles near
# 1.2 m/s and drifts with the wind instead of being flung out of the frame.
_PRECIPITATION_MOTION = {
    "rain": {"per_m3": 10.0, "speed": 7.0, "gravity": 1.0, "drag": 0.2, "damping": 0.0, "wind": 1.0, "brownian": 0.0},
    "snow": {"per_m3": 8.0, "speed": 1.2, "gravity": 0.16, "drag": 0.0, "damping": 0.1, "wind": 1.0, "brownian": 0.35},
}


def _rect_subtract(rects: list[tuple[float, float, float, float]], cover: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    """Cut one rectangle out of a list. Leftovers stay in the same order."""
    x0, x1, y0, y1 = cover
    kept = []
    for rx0, rx1, ry0, ry1 in rects:
        if rx1 <= x0 or rx0 >= x1 or ry1 <= y0 or ry0 >= y1:
            kept.append((rx0, rx1, ry0, ry1))
            continue
        if rx0 < x0:
            kept.append((rx0, x0, ry0, ry1))
        if rx1 > x1:
            kept.append((x1, rx1, ry0, ry1))
        ix0, ix1 = max(rx0, x0), min(rx1, x1)
        if ry0 < y0:
            kept.append((ix0, ix1, ry0, y0))
        if ry1 > y1:
            kept.append((ix0, ix1, y1, ry1))
    return [item for item in kept if item[1] - item[0] > 1e-4 and item[3] - item[2] > 1e-4]


def sky_covered(domain: tuple | list, obstacles: list | tuple, precip: dict) -> bool:
    """True when inert boxes flush with the site top cover the rain footprint.

    The footprint is clipped to the site. A ceiling with a flame shaft still
    counts: the shaft is cut later for the plume, and is not a skylight.
    """
    length, width, height = (float(value) for value in domain)
    center = precip["center"]
    size = precip["size"]
    x0 = max(float(center[0]) - float(size[0]) / 2.0, 0.0)
    x1 = min(float(center[0]) + float(size[0]) / 2.0, length)
    y0 = max(float(center[1]) - float(size[1]) / 2.0, 0.0)
    y1 = min(float(center[1]) + float(size[1]) / 2.0, width)
    if x1 - x0 <= 1e-3 or y1 - y0 <= 1e-3:
        return False
    rects = [(x0, x1, y0, y1)]
    for item in obstacles:
        role = item.role if hasattr(item, "role") else str(item.get("role") or item.get("kind") or "inert")
        if role != "inert":
            continue
        xb = item.xb if hasattr(item, "xb") else item["xb"]
        if float(xb[5]) < height - 0.05:
            continue
        rects = _rect_subtract(rects, (float(xb[0]), float(xb[1]), float(xb[2]), float(xb[3])))
        if not rects:
            return True
    return False


def environment_for_layout(domain: tuple | list, obstacles: list | tuple, precip: dict) -> str:
    """室内 when the rain of this bake lands on a lid, otherwise 室外."""
    return "室内" if sky_covered(domain, obstacles, precip) else "室外"


def precipitation_box(
    domain: tuple[float, float, float] | list[float],
    gas: dict,
    cameras: list[dict],
) -> dict[str, list[float]]:
    """Where rain and snow are emitted: over the fire and every camera, a little above both.

    Emitting over a whole 40 m forest from its 20 m sky spreads the drops
    thin and they never reach the frame in time.
    """
    length, width, height = (float(value) for value in domain)
    center = [float(value) for value in gas["location"]]
    half = [float(value) for value in gas["scale"]]
    xs = [center[0] - half[0], center[0] + half[0]] + [float(item["location"][0]) for item in cameras]
    ys = [center[1] - half[1], center[1] + half[1]] + [float(item["location"][1]) for item in cameras]
    top = max([center[2] + half[2]] + [float(item["location"][2]) for item in cameras])
    margin = 2.5
    x0, x1 = max(min(xs) - margin, -0.1 * length), min(max(xs) + margin, 1.1 * length)
    y0, y1 = max(min(ys) - margin, -0.1 * width), min(max(ys) + margin, 1.1 * width)
    z = min(top + 1.5, height * 1.15 + 0.5)
    return {"center": [(x0 + x1) / 2.0, (y0 + y1) / 2.0, z], "size": [x1 - x0, y1 - y0]}


def precipitation_lifetime(warmup: int) -> int:
    """Frames a drop lives. Born everywhere in the air, the count alive is steady once this many frames have passed.

    The warmup is not rendered, so a lifetime no longer than it keeps the
    density the same in every rendered frame.
    """
    return max(4, int(warmup))


def weather_physics(weather: str) -> dict[str, object]:
    """Multipliers on the clear-weather bake, the extra downdraft, and the ground surface."""
    try:
        physics = _WEATHER_PHYSICS[weather]
    except KeyError as exc:
        raise ValueError(f"未知天气 {weather!r}") from exc
    kind = _PRECIPITATION[weather]
    return {**physics, "smoke_color": list(physics["smoke_color"]), "motion": None if kind is None else dict(_PRECIPITATION_MOTION[kind])}


def ambient_physics(temperature: float, humidity: float) -> dict[str, float]:
    """Ambient air on the Mantaflow bake, as the field model in ``fields.py`` has it.

    Warmer air is lighter, so the hot plume is less buoyant (``beta``) and
    carries less dense smoke. Humid air cools the plume and suppresses smoke.
    """
    ambient_k = max(float(temperature) + 273.15, 200.0)
    density_ratio = 293.15 / ambient_k
    rh = float(humidity)
    return {
        "beta": round(min(max(density_ratio, 0.7), 1.5), 4),
        "alpha": round(min(max(density_ratio, 0.7), 1.5), 4),
        "temperature_scale": round(1.0 - 0.0022 * rh, 4),
        "smoke_scale": round(max(1.0 - 0.0035 * rh, 0.05) * density_ratio, 4),
        "burning_scale": round(1.0 - 0.0015 * rh, 4),
    }


def seed_jitter(seed: int | None) -> dict[str, float]:
    """Small inflow changes, so two seeds bake two different flames."""
    if seed is None:
        return {"dx": 0.0, "dy": 0.0, "velocity_scale": 1.0}
    rng = np.random.default_rng(int(seed))
    return {
        "dx": float(rng.uniform(-0.03, 0.03)),
        "dy": float(rng.uniform(-0.03, 0.03)),
        "velocity_scale": float(rng.uniform(0.9, 1.1)),
    }


def fill_light(
    domain: tuple[float, float, float] | list[float],
    origin: tuple[float, float, float] | list[float],
    obstacles: list[dict],
) -> dict[str, object]:
    """Point fill inside the domain, a short way above the fire.

    ``size`` is the sphere radius (``shadow_soft_size``), not a square.
    The center stays inside the floor plan and out of solids. A downward
    square of the same wattage leaves a hard rectangle on the floor.
    """
    length, width, height = (float(value) for value in domain)
    ox, oy, oz = (float(value) for value in origin)
    size = 1.0
    margin = 0.15
    # Keep the center this far from the outer boundary. The radius is shadow
    # softness, so it does not have to lie entirely inside a narrow room.
    inset = 0.35 + margin

    def clamp_axis(value: float, limit: float) -> float:
        if limit <= 2.0 * inset:
            return limit / 2.0
        return min(max(value, inset), limit - inset)

    def inside(x: float, y: float, z: float, xb: list | tuple) -> bool:
        x0, x1, y0, y1, z0, z1 = (float(value) for value in xb)
        pad = 0.02
        return (x0 - pad) < x < (x1 + pad) and (y0 - pad) < y < (y1 + pad) and (z0 - pad) < z < (z1 + pad)

    def blocked(x: float, y: float, z: float) -> bool:
        return any(inside(x, y, z, item["xb"]) for item in obstacles)

    cx = clamp_axis(ox, length)
    cy = clamp_axis(oy, width)
    cz = min(height - 0.45, oz + 1.8)
    cz = min(max(cz, oz + 0.55), max(oz + 0.2, height - 0.15))
    if blocked(cx, cy, cz):
        tops = [
            float(item["xb"][5])
            for item in obstacles
            if float(item["xb"][0]) < cx < float(item["xb"][1])
            and float(item["xb"][2]) < cy < float(item["xb"][3])
            and float(item["xb"][4]) <= cz
        ]
        if tops:
            raised = max(tops) + 0.35
            if raised < height - 0.15:
                cz = min(max(oz + 0.55, height - 0.45), raised)
        if blocked(cx, cy, cz):
            tx, ty = length * 0.5, width * 0.5
            for step in range(1, 17):
                t = step / 16.0
                x = clamp_axis(cx + (tx - cx) * t, length)
                y = clamp_axis(cy + (ty - cy) * t, width)
                if not blocked(x, y, cz):
                    cx, cy = x, y
                    break
    return {"location": [cx, cy, cz], "size": size, "energy": 20.0}


# Gap between the gas box and the floor plane (z = 0) or a wall it meets.
SOLID_CLEARANCE = 0.05
# Below the floor plane, wide enough to cover any gas box.
_FLOOR_SLAB = (-1.0e6, 1.0e6, -1.0e6, 1.0e6, -1.0, 0.0)


def clear_of_solids(box: dict, obstacles: list[dict]) -> dict:
    """Gas box pulled back from the floor and from walls that cross a face.

    A face of the gas box on or inside a solid makes Cycles lose the volume
    exit on rays that hit that surface. It then adds the flame glow over the
    whole empty-space block around them, as hard squares on the floor or wall.
    Only a solid that covers the whole face moves it. A partial solid, such as
    a ceiling with a shaft cut, would shrink the box around the plume.
    """
    location = [float(value) for value in box["location"]]
    scale = [float(value) for value in box["scale"]]
    lo = [location[axis] - scale[axis] for axis in range(3)]
    hi = [location[axis] + scale[axis] for axis in range(3)]
    eps = 1e-6
    solids = [_FLOOR_SLAB, *(tuple(float(value) for value in item["xb"]) for item in obstacles)]
    for solid in solids:
        for axis in range(3):
            others = [other for other in range(3) if other != axis]
            if not all(solid[2 * other] <= lo[other] + eps and solid[2 * other + 1] >= hi[other] - eps for other in others):
                continue
            s_lo, s_hi = solid[2 * axis], solid[2 * axis + 1]
            if s_lo - eps <= lo[axis] <= s_hi + eps:
                lo[axis] = s_hi + SOLID_CLEARANCE
            elif s_lo - eps <= hi[axis] <= s_hi + eps:
                hi[axis] = s_lo - SOLID_CLEARANCE
    if any(hi[axis] - lo[axis] < 0.5 for axis in range(3)):
        return dict(box)
    return {
        **box,
        "location": [(lo[axis] + hi[axis]) / 2.0 for axis in range(3)],
        "scale": [(hi[axis] - lo[axis]) / 2.0 for axis in range(3)],
    }


def render_key(camera_index: int, look_index: int, frame_index: int) -> str:
    """File stem for one rendered frame. Chinese labels stay out of file names."""
    return f"v{int(camera_index)}_l{int(look_index)}_f{int(frame_index):03d}"


def render_blender(
    spec: ScenarioSpec,
    wind_speed: float,
    temperature: float,
    humidity: float,
    fuel_type: str,
    render_output: Path | str | None = None,
    placements: list | None = None,
    layout: dict | None = None,
    viewpoints: list[str] | tuple[str, ...] = ("侧视",),
    looks: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (("室内", "晴"),),
    frame_count: int = DEFAULT_FRAME_COUNT,
    seed: int | None = None,
    camera_overrides: dict | None = None,
    lights: list | None = None,
    simulation: dict | None = None,
) -> str:
    from firephantom.physics.placements import prepare
    from firephantom.physics.simulation import resolved_simulation, warmup_frames

    fuel = fuel_properties(fuel_type)
    if layout is None:
        layout = prepare(spec, placements, fuel_type)
    obstacles = visible_obstacles(spec)
    cameras = [
        {"key": f"v{index}", **pose}
        for index, pose in enumerate(camera_poses(spec, list(viewpoints), camera_overrides))
    ]
    look_rows = []
    for index, (scene, weather) in enumerate(looks):
        row = {"key": f"l{index}", **look_settings(scene, weather)}
        row["ground_covered"] = scene == "室内"
        look_rows.append(row)
    box = precipitation_box(spec.domain, layout["gas_domain"], cameras)
    for row in look_rows:
        motion = row["physics"]["motion"]
        if motion:
            motion["lifetime"] = precipitation_lifetime(warmup_frames(simulation))
    params = {
        "scenario": spec.name,
        "chid": spec.chid,
        "wind_speed": float(wind_speed),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ambient": ambient_physics(temperature, humidity),
        "fuel_type": fuel_type,
        "hrr_kw": fuel["hrr_kw"],
        "soot_yield": fuel["soot_yield"],
        "plume_scale": float(spec.plume_scale),
        "domain": list(spec.domain),
        "origin": list(spec.origin),
        "obstacles": obstacles,
        "cover_boxes": [[float(value) for value in item.xb] for item in spec.obstacles if item.role == "inert"],
        "fill_light": None if lights is not None else fill_light(spec.domain, spec.origin, obstacles),
        "custom_lights": lights is not None,
        "lights": [] if lights is None else lights,
        "cameras": cameras,
        "looks": look_rows,
        "render_output": None if render_output is None else str(render_output),
        "warmup": warmup_frames(simulation),
        "simulation": resolved_simulation(simulation),
        "frame_count": int(frame_count),
        "jitter": seed_jitter(seed),
        "packages": layout["packages"],
        "gas_domain": clear_of_solids(layout["gas_domain"], obstacles),
        "precip_box": box,
        "surface_textures": layout["textures"],
        "spread_model": layout["spread_model"],
    }
    payload = json.dumps(params, ensure_ascii=False, sort_keys=True)
    return _SCRIPT.replace("__PAYLOAD__", payload)
