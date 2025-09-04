#!/usr/bin/env python3
"""
Единый импортер камер и ArUco маркеров для Blender
================================================

Объединяет функциональность:
1. Импорт камер из XMP файлов RealityCapture (из прикрепленного скрипта)
2. Импорт ArUco маркеров из результатов триангуляции

Использование:
1. Установите константы FOLDER и MARKERS_FILE в начале скрипта
2. Запустите в Text Editor Blender
"""

import os
import math
import json
import xml.etree.ElementTree as ET
from mathutils import Matrix, Vector
import bpy
import traceback

# ---- CONFIG ----
FOLDER = r"C:\Users\admin\PycharmProjects\autocalibration\data"  # Папка с XMP файлами
MARKERS_FILE = r"C:\Users\admin\PycharmProjects\autocalibration\results\blender_aruco_markers.json"  # Файл с маркерами
# ----------------

RC_NS = {"rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#", "xcr":"http://www.capturingreality.com/ns/xcr/1.1#"}

def _floats(s):
    """Парсинг чисел из строки"""
    return [float(x) for x in str(s).strip().split()] if s is not None else []

def parse_rc_xmp(path):
    """Парсинг XMP файла RealityCapture"""
    tree = ET.parse(path)
    root = tree.getroot()
    desc = root.find(".//rdf:Description", RC_NS)
    if desc is None:
        return None
    
    pos = _floats(desc.findtext("xcr:Position", default="", namespaces=RC_NS))
    rot = _floats(desc.findtext("xcr:Rotation", default="", namespaces=RC_NS))
    dist = _floats(desc.findtext("xcr:DistortionCoeficients", default="", namespaces=RC_NS))
    attrs = {k.split('}')[1]: v for k, v in desc.attrib.items() if k.startswith('{'+RC_NS['xcr']+'}')}
    
    return {
        "path": path, 
        "name": os.path.splitext(os.path.basename(path))[0],
        "position": pos, 
        "rotation": rot, 
        "dist": dist, 
        "attrs": attrs
    }

def to_blender_cam_matrix(R_w2cv_3x3, C_world_vec3):
    """Преобразование матрицы камеры RealityCapture в Blender"""
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

def ensure_collection(name):
    """Создание или получение коллекции"""
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def create_camera(cam_data, collection):
    """Создание объекта камеры в Blender"""
    name = cam_data["name"]
    pos = cam_data["position"]
    rot = cam_data["rotation"]
    attrs = cam_data["attrs"]

    if len(pos) != 3 or len(rot) != 9:
        print(f"[WARN] Пропускаем камеру {name}: неправильный формат Position/Rotation (pos={len(pos)} rot={len(rot)})")
        return None

    C_world = Vector(pos)
    R_w2cv = Matrix((rot[0:3], rot[3:6], rot[6:9]))

    # Создание объекта камеры
    cam_data_block = bpy.data.cameras.new(name=name+"_DATA")
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data_block)
    collection.objects.link(cam_obj)

    # Внешние параметры
    cam_obj.matrix_world = to_blender_cam_matrix(R_w2cv, C_world)

    # Внутренние параметры
    f35 = float(attrs.get("FocalLength35mm", 0.0) or 0.0)
    if f35 > 0:
        cam_data_block.sensor_fit = 'HORIZONTAL'
        cam_data_block.sensor_width = 36.0
        cam_data_block.lens = f35

    # Главная точка (Principal point)
    try:
        ppu = float(attrs.get("PrincipalPointU", ""))
        ppv = float(attrs.get("PrincipalPointV", ""))
        # Если значения малы (<0.05), рассматриваем как смещение от центра
        if abs(ppu) < 0.05 and abs(ppv) < 0.05:
            cam_data_block.shift_x = ppu
            cam_data_block.shift_y = -ppv
        else:
            # предполагаем диапазон [0..1]
            cam_data_block.shift_x = (ppu - 0.5)
            cam_data_block.shift_y = -(ppv - 0.5)
    except Exception:
        pass

    # Сохраняем дополнительные данные
    cam_obj["RC_attrs"] = attrs
    cam_obj["RC_distortion"] = cam_data["dist"]
    
    return cam_obj

def create_marker(marker_id, marker_data, collection, settings):
    """Создание объекта ArUco маркера в Blender"""
    name = f"ArUco_Marker_{marker_id:02d}"
    position = marker_data['position']
    confidence = marker_data['confidence']
    quality = marker_data.get('quality', 'unknown')
    
    # Создание Empty объекта
    bpy.ops.object.empty_add(
        type=settings['empty_type'],
        location=tuple(position)
    )
    
    marker_obj = bpy.context.active_object
    marker_obj.name = name
    
    # Размер в зависимости от качества
    if settings['size_by_quality']:
        if quality == 'high':
            size = settings['marker_size']
        elif quality == 'medium':
            size = settings['marker_size'] * 0.8
        else:
            size = settings['marker_size'] * 0.6
    else:
        size = settings['marker_size']
    
    marker_obj.empty_display_size = size
    
    # Цвет в зависимости от качества
    if settings['color_by_quality']:
        if quality == 'high':
            marker_obj.color = (0.0, 1.0, 0.0, 1.0)  # Зеленый
        elif quality == 'medium':
            marker_obj.color = (1.0, 1.0, 0.0, 1.0)  # Желтый
        else:
            marker_obj.color = (1.0, 0.5, 0.0, 1.0)  # Оранжевый
    
    # Кастомные свойства
    marker_obj["aruco_id"] = marker_id
    marker_obj["confidence"] = confidence
    marker_obj["quality"] = quality
    marker_obj["triangulated_position"] = position
    
    # Перемещение в коллекцию
    collection.objects.link(marker_obj)
    
    # Убираем из Scene Collection
    if marker_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(marker_obj)
    
    return marker_obj

def import_cameras(folder):
    """Импорт камер из XMP файлов"""
    print("🎥 Импорт камер из XMP файлов...")
    
    if not folder or not os.path.isdir(folder):
        print(f"[ERROR] Неверная папка: {folder}")
        return []

    xmp_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".xmp")]
    if not xmp_paths:
        print(f"[ERROR] XMP файлы не найдены в: {folder}")
        return []

    coll = ensure_collection("RealityCapture_Cameras")
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

    print(f"   Импортировано камер: {len(imported)} в коллекцию '{coll.name}'")
    return imported

def import_markers(markers_file):
    """Импорт ArUco маркеров из JSON файла"""
    print("🏷️  Импорт ArUco маркеров...")
    
    if not markers_file or not os.path.exists(markers_file):
        print(f"[ERROR] Файл с маркерами не найден: {markers_file}")
        return []
    
    try:
        with open(markers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Ошибка чтения файла маркеров: {e}")
        return []
    
    markers_data = data.get('markers', {})
    if not markers_data:
        print("[ERROR] Данные маркеров не найдены в файле")
        return []
    
    # Настройки отображения маркеров
    marker_settings = {
        'empty_type': 'PLAIN_AXES',
        'marker_size': 0.1,
        'size_by_quality': True,
        'color_by_quality': True
    }
    
    coll = ensure_collection("ArUco_Markers")
    imported = []
    high_quality_count = 0
    
    for marker_name, marker_info in markers_data.items():
        try:
            marker_id = marker_info['id']
            confidence = marker_info['confidence']
            quality = marker_info.get('quality', 'unknown')
            
            # Создание маркера
            marker_obj = create_marker(marker_id, marker_info, coll, marker_settings)
            imported.append(marker_obj.name)
            
            if quality == 'high':
                high_quality_count += 1
                
        except Exception as e:
            print(f"[ERROR] Ошибка создания маркера {marker_name}: {e}")
            continue
    
    print(f"   Импортировано маркеров: {len(imported)} (высокого качества: {high_quality_count})")
    print(f"   в коллекцию '{coll.name}'")
    return imported

def setup_scene():
    """Настройка сцены для корректного отображения"""
    # Настройка единиц измерения
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0
    
    # Очистка выделения
    bpy.ops.object.select_all(action='DESELECT')

def clear_existing_data():
    """Очистка существующих камер и маркеров"""
    print("🧹 Очистка существующих данных...")
    
    # Удаляем объекты камер
    cameras_to_delete = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    for cam in cameras_to_delete:
        bpy.data.objects.remove(cam, do_unlink=True)
    
    # Удаляем объекты маркеров
    markers_to_delete = [obj for obj in bpy.data.objects if obj.name.startswith('ArUco_Marker_')]
    for marker in markers_to_delete:
        bpy.data.objects.remove(marker, do_unlink=True)
    
    # Удаляем коллекции
    collections_to_remove = ['RealityCapture_Cameras', 'ArUco_Markers']
    for coll_name in collections_to_remove:
        if coll_name in bpy.data.collections:
            coll = bpy.data.collections[coll_name]
            bpy.data.collections.remove(coll)
    
    print("   Существующие данные очищены")

def main():
    """Главная функция импорта"""
    print("="*60)
    print("🚀 ЕДИНЫЙ ИМПОРТЕР КАМЕР И ARUCO МАРКЕРОВ")
    print("="*60)
    print(f"📂 Папка XMP: {FOLDER}")
    print(f"🏷️  Файл маркеров: {MARKERS_FILE}")
    print("="*60)
    
    try:
        # Настройка сцены
        setup_scene()
        
        # Очистка существующих данных
        clear_existing_data()
        
        # Импорт камер
        imported_cameras = import_cameras(FOLDER)
        
        # Импорт маркеров
        imported_markers = import_markers(MARKERS_FILE)
        
        # Итоговая статистика
        print("\n✅ ИМПОРТ ЗАВЕРШЕН УСПЕШНО!")
        print(f"   🎥 Импортировано камер: {len(imported_cameras)}")
        print(f"   🏷️  Импортировано маркеров: {len(imported_markers)}")
        
        if imported_cameras or imported_markers:
            print("\n📋 КОЛЛЕКЦИИ BLENDER:")
            if imported_cameras:
                print("   • RealityCapture_Cameras - камеры из XMP")
            if imported_markers:
                print("   • ArUco_Markers - триангулированные маркеры")
                print("     🟢 Зеленые - высокое качество")
                print("     🟡 Желтые - среднее качество")
                print("     🟠 Оранжевые - низкое качество")
        else:
            print("\n⚠️  Данные не найдены для импорта")
            
    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()