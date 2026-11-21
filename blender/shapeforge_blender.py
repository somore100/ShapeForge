"""ShapeForge for Blender: type a prompt, get a 3D model in your scene.

This add-on is a thin client. Blender's bundled Python cannot (and should not)
host PyTorch, so the add-on starts ShapeForge's own persistent Shap-E worker
(engine/server.py) with YOUR Python install, sends it prompts, and imports the
finished mesh. Shap-E is loaded once and stays warm between generations.

Setup
  1. Edit > Preferences > Add-ons > Install from Disk... > this file, enable it.
  2. In the add-on preferences, check the three paths (they default to the usual
     ShapeForge layout).
  3. 3D Viewport > press N > "ShapeForge" tab > type a prompt > Generate.

Needs a working ShapeForge checkout and Shap-E setup (see the ShapeForge README).
"""

bl_info = {
    "name": "ShapeForge",
    "author": "domore100",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar > ShapeForge",
    "description": "Generate 3D models from text with a local Shap-E engine",
    "category": "3D View",
}

import atexit
import collections
import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

PERCENT = re.compile(r"(\d+)%\|")

# Import axes as (forward, up) describing the generated file's orientation.
ORIENTATION_AXES = {"Z": ("Y", "Z"), "Y": ("NEGATIVE_Z", "Y")}
COLLECTION_NAME = "ShapeForge"


# ---------------------------------------------------------------- preferences

def _default_python():
    candidate = Path.home() / ".local" / "python3.10" / "bin" / "python3"
    return str(candidate) if candidate.exists() else "python3"


def _default_shapeforge_dir():
    return str(Path.home() / "shap-e" / "ShapeForge")


def _default_shap_e_home():
    return str(Path.home() / "shap-e")


def _addon_prefs(context=None):
    context = context or bpy.context
    return context.preferences.addons[__name__].preferences


def _expand(path):
    return Path(bpy.path.abspath(path)).expanduser()


def check_setup(prefs):
    """Return a list of human-readable problems (empty = ready)."""
    problems = []
    python = prefs.python_path.strip()
    if not python or not (os.path.isfile(str(_expand(python)))
                          or _which(python)):
        problems.append(f"Python not found: {python or '(empty)'}")
    if not (_expand(prefs.shapeforge_dir) / "engine" / "server.py").is_file():
        problems.append("ShapeForge folder has no engine/server.py")
    if not _expand(prefs.shap_e_home).is_dir():
        problems.append("Shap-E folder not found")
    return problems


def _which(name):
    from shutil import which
    return which(name)


class ShapeForgePreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    python_path: bpy.props.StringProperty(
        name="Python",
        description="Python that can run Shap-E (the one with PyTorch + CUDA)",
        subtype="FILE_PATH", default=_default_python(),
    )
    shapeforge_dir: bpy.props.StringProperty(
        name="ShapeForge folder",
        description="Folder containing engine/server.py",
        subtype="DIR_PATH", default=_default_shapeforge_dir(),
    )
    shap_e_home: bpy.props.StringProperty(
        name="Shap-E folder",
        description="Your Shap-E checkout (holds shap_e_model_cache)",
        subtype="DIR_PATH", default=_default_shap_e_home(),
    )
    output_dir: bpy.props.StringProperty(
        name="Output folder",
        description="Where generated meshes are kept "
                    "(empty = <ShapeForge folder>/outputs/blender)",
        subtype="DIR_PATH", default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "python_path")
        layout.prop(self, "shapeforge_dir")
        layout.prop(self, "shap_e_home")
        layout.prop(self, "output_dir")
        problems = check_setup(self)
        box = layout.box()
        if problems:
            box.alert = True
            for problem in problems:
                box.label(text=problem, icon="ERROR")
        else:
            box.label(text="Setup looks good.", icon="CHECKMARK")


# --------------------------------------------------------------------- engine

class EngineError(Exception):
    pass


def _clean_env(shap_e_home):
    """Environment for the worker: undo anything Blender injected."""
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"):
        env.pop(key, None)
    ld = env.get("LD_LIBRARY_PATH")
    if ld:
        kept = [p for p in ld.split(os.pathsep) if "blender" not in p.lower()]
        if kept:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    env["PYTHONPATH"] = shap_e_home
    env["SHAP_E_HOME"] = shap_e_home
    env["SHAPE_E_HOME"] = shap_e_home
    return env


class Engine:
    """Owns the worker process. Only poll() touches Blender state."""

    def __init__(self):
        self.proc = None
        self.events = queue.Queue()
        self.stderr_tail = collections.deque(maxlen=12)
        self.state = "stopped"      # stopped | starting | ready | busy
        self.job = None             # job queued or running (None = idle)
        self.job_sent = False
        self.status = "Engine stopped."
        self.error = False
        self.progress = 0.0
        self.kill_at = None
        self.next_id = 0
        self.on_done = None

    # -- lifecycle ---------------------------------------------------------

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, prefs):
        problems = check_setup(prefs)
        if problems:
            raise EngineError("; ".join(problems))
        python = prefs.python_path.strip()
        python = str(_expand(python)) if os.path.isfile(str(_expand(python))) \
            else python
        server = _expand(prefs.shapeforge_dir) / "engine" / "server.py"
        home = str(_expand(prefs.shap_e_home))

        self.events = queue.Queue()
        self.stderr_tail.clear()
        try:
            self.proc = subprocess.Popen(
                [python, str(server)], cwd=home, env=_clean_env(home),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except OSError as error:
            self.proc = None
            raise EngineError(f"Could not start Python: {error}")
        for target in (self._read_stdout, self._read_stderr):
            threading.Thread(
                target=target, args=(self.proc, self.events), daemon=True
            ).start()
        self.state = "starting"
        self.status = "Starting engine (loading Shap-E, one time)..."
        self.error = False
        self.progress = 0.0

    def stop(self, force=False):
        proc, self.proc = self.proc, None
        self.state = "stopped"
        self.job = None
        self.job_sent = False
        self.progress = 0.0
        self.kill_at = None
        self.status = "Engine stopped."
        self.error = False
        if proc is None or proc.poll() is not None:
            return
        try:
            if not force:
                proc.stdin.write('{"cmd": "quit"}\n')
                proc.stdin.flush()
        except Exception:
            pass
        if force:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        else:
            threading.Thread(
                target=self._reap, args=(proc,), daemon=True
            ).start()

    @staticmethod
    def _reap(proc):
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # -- reader threads (never touch bpy) ------------------------------------

    @staticmethod
    def _read_stdout(proc, events):
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.put(("event", json.loads(raw)))
            except ValueError:
                pass
        events.put(("exit", proc))

    def _read_stderr(self, proc, events):
        for line in proc.stderr:
            line = line.rstrip()
            if not line:
                continue
            match = PERCENT.search(line)
            if match:
                events.put(("progress", int(match.group(1))))
            elif "FutureWarning" not in line and not line.lstrip().startswith("def "):
                self.stderr_tail.append(line)

    # -- jobs ----------------------------------------------------------------

    def submit(self, job, prefs):
        if self.job is not None:
            raise EngineError("A generation is already running.")
        if not self.alive():
            self.start(prefs)
        self.next_id += 1
        job["id"] = self.next_id
        self.job = job
        self.job_sent = False
        self.progress = 0.0
        self.error = False
        if self.state == "ready":
            self._send()
        else:
            self.status = "Starting engine (loading Shap-E, one time)..."

    def _send(self):
        job = self.job
        line = json.dumps({
            "id": job["id"], "prompt": job["prompt"], "output": job["out_dir"],
        })
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, AttributeError):
            self._died("Engine connection lost.")
            return
        self.job_sent = True
        self.state = "busy"
        self.status = "Generating..."

    # -- main-thread event pump ----------------------------------------------

    def poll(self):
        changed = False
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "progress":
                if self.state == "busy":
                    self.progress = payload / 100.0
                    self.status = f"Sampling {payload}%"
            elif kind == "event":
                self._handle(payload)
            elif kind == "exit" and payload is self.proc:
                self._died("Engine stopped unexpectedly.")
        return changed

    def _handle(self, event):
        name = event.get("event")
        if name == "log":
            if self.state == "starting":
                self.status = event.get("message", self.status)
        elif name == "ready":
            self.state = "ready"
            self.status = "Engine ready."
            if self.job is not None and not self.job_sent:
                self._send()
        elif name == "start":
            self.state = "busy"
        elif name == "fatal":
            first = (event.get("error") or "").strip().splitlines()
            self._died("Engine failed: " + (first[-1] if first else "unknown"))
        elif name == "done":
            job, self.job = self.job, None
            self.job_sent = False
            self.progress = 0.0
            self.state = "ready"
            self.status = "Engine ready."
            if job is None:
                return
            if event.get("ok"):
                self.error = False
                try:
                    if self.on_done:
                        self.on_done(job, event)
                except Exception as error:
                    traceback.print_exc()
                    self.status = f"Import failed: {error}"
                    self.error = True
            else:
                self.status = f"Generation failed: {event.get('error')}"
                self.error = True

    def _died(self, message):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
        tail = " | ".join(list(self.stderr_tail)[-2:])
        self.state = "stopped"
        self.job = None
        self.job_sent = False
        self.progress = 0.0
        self.status = message + (f" ({tail})" if tail else "")
        self.error = True


ENGINE = Engine()
atexit.register(lambda: ENGINE.stop(force=True))


def _redraw():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    except Exception:
        pass


def _tick():
    if ENGINE.poll():
        _redraw()
    if ENGINE.proc is None and ENGINE.events.empty():
        return None
    return 0.15


def _ensure_timer():
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.1, persistent=True)


# ------------------------------------------------------------ mesh processing

def remove_small_islands(mesh, fraction):
    """Delete edge-connected pieces smaller than fraction of the largest."""
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    seen = [False] * len(bm.faces)
    islands = []
    for start in bm.faces:
        if seen[start.index]:
            continue
        seen[start.index] = True
        stack, island = [start], []
        while stack:
            face = stack.pop()
            island.append(face)
            for edge in face.edges:
                for other in edge.link_faces:
                    if not seen[other.index]:
                        seen[other.index] = True
                        stack.append(other)
        islands.append(island)

    removed = 0
    if len(islands) > 1:
        biggest = max(len(island) for island in islands)
        doomed = []
        for island in islands:
            if len(island) < fraction * biggest:
                removed += 1
                doomed.extend(island)
        if doomed:
            bmesh.ops.delete(bm, geom=doomed, context="FACES")
            loose = [v for v in bm.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return removed


def _color_material(layer_name):
    name = f"ShapeForge Colors ({layer_name})"
    material = bpy.data.materials.get(name)
    if material is not None:
        return material
    material = bpy.data.materials.new(name)
    try:
        material.use_nodes = True   # needed before 5.0; removed in 5.0
    except AttributeError:
        pass
    tree = material.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    colors = tree.nodes.new("ShaderNodeVertexColor")
    colors.layer_name = layer_name
    colors.location = (-300, 200)
    if bsdf is not None:
        tree.links.new(colors.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.6
    return material


def _shapeforge_collection(scene):
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION_NAME)
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)
    return collection


def import_result(job, event):
    """Import the generated mesh and apply the job's settings."""
    s = job["settings"]
    scene = bpy.data.scenes.get(job["scene"]) or bpy.context.scene
    ply = event.get("ply")
    if not ply or not os.path.isfile(ply):
        raise RuntimeError("generated.ply was not found")

    active = bpy.context.view_layer.objects.active
    if active is not None and active.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass

    forward, up = ORIENTATION_AXES[s["orientation"]]
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(
        filepath=ply, forward_axis=forward, up_axis=up, merge_verts=True,
    )
    created = [o for o in bpy.data.objects if o not in before]
    if not created or created[-1].type != "MESH":
        raise RuntimeError("Blender did not create a mesh object")
    obj = created[-1]
    mesh = obj.data

    # The importer stores the axis conversion as an object rotation. Bake it
    # into the mesh so size, grounding and the origin are computed upright.
    mesh.transform(obj.rotation_euler.to_matrix().to_4x4())
    mesh.update()
    obj.rotation_euler = (0, 0, 0)

    if s["remove_fragments"]:
        remove_small_islands(mesh, s["fragment_percent"] / 100.0)

    # Scale to the requested size, then put the origin at the base centre.
    count = len(mesh.vertices)
    coords = np.empty(count * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(-1, 3)
    low, high = coords.min(axis=0), coords.max(axis=0)
    extent = float(np.max(high - low))
    factor = 1.0
    if s["target_size"] > 0 and extent > 0:
        factor = s["target_size"] / extent
    centre = (low + high) / 2
    coords = (coords - np.array([centre[0], centre[1], low[2]])) * factor
    mesh.vertices.foreach_set("co", coords.ravel())
    mesh.update()

    obj.location = scene.cursor.location if s["at_cursor"] else Vector((0, 0, 0))

    if s["shade_smooth"]:
        mesh.shade_smooth()

    if s["use_colors"] and len(mesh.color_attributes):
        layer = mesh.color_attributes[0].name
        mesh.materials.clear()
        mesh.materials.append(_color_material(layer))

    if s["decimate_faces"] > 0 and len(mesh.polygons) > s["decimate_faces"]:
        modifier = obj.modifiers.new("ShapeForge Decimate", "DECIMATE")
        modifier.ratio = max(0.01, s["decimate_faces"] / len(mesh.polygons))

    obj.name = job["name"]
    mesh.name = job["name"]
    obj["shapeforge_prompt"] = job["prompt"]
    obj["shapeforge_folder"] = job["out_dir"]

    collection = _shapeforge_collection(scene)
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)

    view_layer = bpy.context.view_layer
    for other in view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj
    ENGINE.status = f"Imported '{obj.name}' ({len(mesh.polygons):,} faces)."


ENGINE.on_done = import_result


# ------------------------------------------------------------------ settings

class ShapeForgeSettings(bpy.types.PropertyGroup):
    prompt: bpy.props.StringProperty(
        name="Prompt", description="What to generate", default="")
    orientation: bpy.props.EnumProperty(
        name="Source up axis",
        description="Which way is up in the generated mesh. If models arrive "
                    "lying on their side, switch this",
        items=[("Z", "Z-up (as generated)", "Use as generated"),
               ("Y", "Y-up (rotate)", "Rotate from Y-up to Blender's Z-up")],
        default="Z")
    target_size: bpy.props.FloatProperty(
        name="Size", description="Largest dimension of the model (0 = keep "
        "the generated size)", default=2.0, min=0.0, soft_max=100.0,
        unit="LENGTH")
    at_cursor: bpy.props.BoolProperty(
        name="Place at 3D cursor", default=True)
    shade_smooth: bpy.props.BoolProperty(name="Shade smooth", default=True)
    use_colors: bpy.props.BoolProperty(
        name="Vertex-color material",
        description="Shap-E meshes carry vertex colors; build a material that "
                    "shows them (visible in Material Preview and Render)",
        default=True)
    remove_fragments: bpy.props.BoolProperty(
        name="Remove stray fragments", default=True)
    fragment_percent: bpy.props.IntProperty(
        name="Under", description="Delete pieces smaller than this share of "
        "the largest piece", default=5, min=1, max=50, subtype="PERCENTAGE")
    decimate_faces: bpy.props.IntProperty(
        name="Decimate to faces", description="Adds a Decimate modifier "
        "targeting this face count (0 = off)", default=0, min=0,
        max=500000)


def _snapshot(settings):
    return {
        "orientation": settings.orientation,
        "target_size": settings.target_size,
        "at_cursor": settings.at_cursor,
        "shade_smooth": settings.shade_smooth,
        "use_colors": settings.use_colors,
        "remove_fragments": settings.remove_fragments,
        "fragment_percent": settings.fragment_percent,
        "decimate_faces": settings.decimate_faces,
    }


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "model"


def _output_root(prefs):
    if prefs.output_dir.strip():
        return _expand(prefs.output_dir)
    return _expand(prefs.shapeforge_dir) / "outputs" / "blender"


# ----------------------------------------------------------------- operators

class SHAPEFORGE_OT_generate(bpy.types.Operator):
    bl_idname = "shapeforge.generate"
    bl_label = "Generate"
    bl_description = "Generate a 3D model from the prompt and import it"

    @classmethod
    def poll(cls, context):
        return ENGINE.job is None

    def execute(self, context):
        settings = context.scene.shapeforge
        prompt = " ".join(settings.prompt.split())
        if not prompt:
            self.report({"ERROR"}, "Type a prompt first.")
            return {"CANCELLED"}
        prefs = _addon_prefs(context)
        slug = _slug(prompt)
        folder = _output_root(prefs) / f"{time.strftime('%Y%m%d_%H%M%S')}_{slug}"
        job = {
            "prompt": prompt, "name": prompt[:40], "out_dir": str(folder),
            "scene": context.scene.name, "settings": _snapshot(settings),
        }
        try:
            ENGINE.submit(job, prefs)
        except EngineError as error:
            ENGINE.status = str(error)
            ENGINE.error = True
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        _ensure_timer()
        _redraw()
        return {"FINISHED"}


class SHAPEFORGE_OT_start_engine(bpy.types.Operator):
    bl_idname = "shapeforge.start_engine"
    bl_label = "Start Engine"
    bl_description = "Load Shap-E now so the first generation is fast"

    @classmethod
    def poll(cls, context):
        return not ENGINE.alive()

    def execute(self, context):
        try:
            ENGINE.start(_addon_prefs(context))
        except EngineError as error:
            ENGINE.status = str(error)
            ENGINE.error = True
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        _ensure_timer()
        return {"FINISHED"}


class SHAPEFORGE_OT_stop_engine(bpy.types.Operator):
    bl_idname = "shapeforge.stop_engine"
    bl_label = "Stop Engine"
    bl_description = "Stop the worker and free its GPU memory"

    @classmethod
    def poll(cls, context):
        return ENGINE.proc is not None

    def execute(self, context):
        ENGINE.stop(force=ENGINE.job is not None)
        _redraw()
        return {"FINISHED"}


# --------------------------------------------------------------------- panels

def _progress_text(fraction, width=20):
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class SHAPEFORGE_PT_main(bpy.types.Panel):
    bl_label = "ShapeForge"
    bl_idname = "SHAPEFORGE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ShapeForge"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.shapeforge
        layout.prop(settings, "prompt", text="")

        row = layout.row()
        row.scale_y = 1.5
        row.operator("shapeforge.generate", icon="PLAY")

        box = layout.box()
        box.alert = ENGINE.error
        box.label(text=ENGINE.status, icon="ERROR" if ENGINE.error else "INFO")
        if ENGINE.state == "busy":
            box.label(text=_progress_text(ENGINE.progress))


class SHAPEFORGE_PT_import(bpy.types.Panel):
    bl_label = "Import options"
    bl_idname = "SHAPEFORGE_PT_import"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ShapeForge"
    bl_parent_id = "SHAPEFORGE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.shapeforge
        layout.prop(settings, "target_size")
        layout.prop(settings, "at_cursor")
        layout.prop(settings, "orientation", text="Up axis")
        layout.prop(settings, "shade_smooth")
        layout.prop(settings, "use_colors")
        layout.prop(settings, "remove_fragments")
        sub = layout.row()
        sub.enabled = settings.remove_fragments
        sub.prop(settings, "fragment_percent")
        layout.prop(settings, "decimate_faces")


class SHAPEFORGE_PT_engine(bpy.types.Panel):
    bl_label = "Engine"
    bl_idname = "SHAPEFORGE_PT_engine"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ShapeForge"
    bl_parent_id = "SHAPEFORGE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"State: {ENGINE.state}")
        row = layout.row(align=True)
        row.operator("shapeforge.start_engine", icon="PLAY")
        row.operator("shapeforge.stop_engine", icon="PAUSE")
        layout.label(text="Stopping frees the GPU memory (~3 GB).")


CLASSES = (
    ShapeForgePreferences,
    ShapeForgeSettings,
    SHAPEFORGE_OT_generate,
    SHAPEFORGE_OT_start_engine,
    SHAPEFORGE_OT_stop_engine,
    SHAPEFORGE_PT_main,
    SHAPEFORGE_PT_import,
    SHAPEFORGE_PT_engine,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.shapeforge = bpy.props.PointerProperty(
        type=ShapeForgeSettings)


def unregister():
    ENGINE.stop(force=True)
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    del bpy.types.Scene.shapeforge
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
