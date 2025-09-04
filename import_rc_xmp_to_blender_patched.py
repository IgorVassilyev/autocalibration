# import_rc_xmp_to_blender.py (patched)
# Fixes: "'tuple' object does not support item assignment" when setting 3x3 into 4x4
# Usage: Text Editor → Open → set FOLDER → Run Script

import os
import math
import xml.etree.ElementTree as ET
from mathutils import Matrix, Vector
import bpy
import traceback

# ---- CONFIG ----
FOLDER = r"C:\\Users\\admin\\PycharmProjects\\autocalibration\\data"   # e.g. r"D:\my_project\xmp"
# ----------------

RC_NS = {"rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#", "xcr":"http://www.capturingreality.com/ns/xcr/1.1#"}

def _floats(s):
    return [float(x) for x in str(s).strip().split()] if s is not None else []

def parse_rc_xmp(path):
    tree = ET.parse(path)
    root = tree.getroot()
    desc = root.find(".//rdf:Description", RC_NS)
    if desc is None:
        return None
    pos = _floats(desc.findtext("xcr:Position", default="", namespaces=RC_NS))
    rot = _floats(desc.findtext("xcr:Rotation", default="", namespaces=RC_NS))
    dist = _floats(desc.findtext("xcr:DistortionCoeficients", default="", namespaces=RC_NS))
    attrs = {k.split('}')[1]: v for k, v in desc.attrib.items() if k.startswith('{'+RC_NS['xcr']+'}')}
    return {"path": path, "name": os.path.splitext(os.path.basename(path))[0],
            "position": pos, "rotation": rot, "dist": dist, "attrs": attrs}

def to_blender_cam_matrix(R_w2cv_3x3, C_world_vec3):
    # OpenCV: +X right, +Y down, +Z forward
    # Blender cam: +X right, +Y up, +Z backward (looks along -Z)
    R_bcam2cv = Matrix(((1,0,0),(0,-1,0),(0,0,-1)))
    R_cv2bcam = R_bcam2cv.transposed()

    # world -> blender_cam
    R_w2bcam = R_cv2bcam @ R_w2cv_3x3

    # Blender needs object matrix (local->world): invert rotation to get blender_cam->world
    R_bcam2w = R_w2bcam.transposed()  # rotation, so inverse == transpose

    # Build 4x4 safely
    M = R_bcam2w.to_4x4()
    M.translation = Vector(C_world_vec3)
    return M

def ensure_collection(name="RealityCaptureCams"):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def create_camera(cam_data, collection):
    name = cam_data["name"]
    pos = cam_data["position"]
    rot = cam_data["rotation"]
    attrs = cam_data["attrs"]

    if len(pos) != 3 or len(rot) != 9:
        print(f"[WARN] Skipping {name}: bad Position/Rotation formats (pos={len(pos)} rot={len(rot)})")
        return None

    C_world = Vector(pos)
    R_w2cv = Matrix((rot[0:3], rot[3:6], rot[6:9]))

    cam_data_block = bpy.data.cameras.new(name=name+"_DATA")
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data_block)
    collection.objects.link(cam_obj)

    # Extrinsics
    cam_obj.matrix_world = to_blender_cam_matrix(R_w2cv, C_world)

    # Intrinsics
    f35 = float(attrs.get("FocalLength35mm", 0.0) or 0.0)
    if f35 > 0:
        cam_data_block.sensor_fit = 'HORIZONTAL'
        cam_data_block.sensor_width = 36.0
        cam_data_block.lens = f35

    # Principal point (heuristic; RC's U/V may be normalized differently in some exports)
    try:
        ppu = float(attrs.get("PrincipalPointU", ""))
        ppv = float(attrs.get("PrincipalPointV", ""))
        # If values are small (<0.05), treat as offset from center in normalized sensor coords
        if abs(ppu) < 0.05 and abs(ppv) < 0.05:
            cam_data_block.shift_x = ppu
            cam_data_block.shift_y = -ppv
        else:
            # assume [0..1] range
            cam_data_block.shift_x = (ppu - 0.5)
            cam_data_block.shift_y = -(ppv - 0.5)
    except Exception:
        pass

    cam_obj["RC_attrs"] = attrs
    cam_obj["RC_distortion"] = cam_data["dist"]
    return cam_obj

def import_folder(folder):
    if not folder:
        folder = os.path.dirname(bpy.data.filepath)
    if not folder or not os.path.isdir(folder):
        raise RuntimeError("Please set FOLDER to a valid directory that contains .xmp files")

    xmp_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".xmp")]
    if not xmp_paths:
        raise RuntimeError(f"No .xmp files found in: {folder}")

    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0

    coll = ensure_collection()
    imported = []
    for p in sorted(xmp_paths):
        try:
            data = parse_rc_xmp(p)
            if data:
                cam = create_camera(data, coll)
                if cam:
                    imported.append(cam.name)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            print(traceback.format_exc())

    print(f"Imported {len(imported)} cameras into collection '{coll.name}'")
    return imported

if __name__ == "__main__":
    import_folder(FOLDER)
