#!/usr/bin/env python3
"""
Главный скрипт полного пайплайна автокалибровки ArUco маркеров
=============================================================

Обрабатывает данные и создает скрипт для Blender:
1. Загрузка параметров камер из XMP файлов
2. Конвертация в OpenCV формат  
3. Детекция ArUco маркеров (DICT_4X4_1000, ID 1-13)
4. 3D триангуляция маркеров
5. Создание Blender-скрипта для импорта камер и маркеров

ВАЖНО: Этот файл НЕ содержит Blender-импорты (bpy, mathutils)!
       Blender-скрипт создается динамически как текстовый файл.

Использование:
    python main.py
"""

import os
import sys
import json
import time

# Импорт наших модулей
try:
    from xmp_parser import SimpleXMPParser
    from xmp_to_opencv import convert_cameras_to_opencv
    from aruco_detector import SimpleArUcoDetector
    from triangulation import triangulate_markers, prepare_blender_export
    from config import CURRENT_IMAGE_SIZE
except ImportError as e:
    print(f"❌ Ошибка импорта модулей: {e}")
    print("Убедитесь, что все файлы проекта находятся в одной директории:")
    print("  - xmp_parser.py, xmp_to_opencv.py, aruco_detector.py")
    print("  - triangulation.py, config.py")
    sys.exit(1)


def validate_input_data(data_dir: str) -> bool:
    """Валидация входных данных"""
    if not os.path.exists(data_dir):
        print(f"❌ Директория не найдена: {data_dir}")
        return False
    
    # Поиск XMP файлов
    xmp_files = [f for f in os.listdir(data_dir) if f.endswith('.xmp')]
    if not xmp_files:
        print(f"❌ XMP файлы не найдены в {data_dir}")
        return False
    
    # Поиск изображений
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for f in os.listdir(data_dir):
        if any(f.lower().endswith(ext) for ext in image_extensions):
            image_files.append(f)
    
    if not image_files:
        print(f"❌ Изображения не найдены в {data_dir}")
        return False
    
    # Проверка соответствия
    xmp_ids = {os.path.splitext(f)[0] for f in xmp_files}
    image_ids = {os.path.splitext(f)[0] for f in image_files}
    common_ids = xmp_ids & image_ids
    
    if len(common_ids) < 3:
        print(f"❌ Недостаточно пар XMP-изображение: {len(common_ids)} < 3")
        return False
    
    print(f"✅ Найдено {len(xmp_files)} XMP файлов и {len(image_files)} изображений")
    print(f"   Совпадающих пар: {len(common_ids)}")
    return True


def load_cameras(data_dir: str):
    """Этап 1: Загрузка параметров камер из XMP файлов"""
    print("\n🔧 Этап 1: Загрузка параметров камер")
    
    parser = SimpleXMPParser(enable_logging=False)
    xmp_cameras = parser.load_all_cameras(data_dir)
    
    if not xmp_cameras:
        raise ValueError("Не удалось загрузить камеры")
    
    print(f"   Загружено камер: {len(xmp_cameras)}")
    return xmp_cameras


def convert_cameras(xmp_cameras):
    """Этап 2: Конвертация параметров камер в OpenCV формат"""
    print("🔄 Этап 2: Конвертация в OpenCV формат")
    
    opencv_cameras = convert_cameras_to_opencv(xmp_cameras, CURRENT_IMAGE_SIZE)
    
    if not opencv_cameras:
        raise ValueError("Ошибка конвертации камер")
    
    print(f"   Конвертировано камер: {len(opencv_cameras)}")
    return opencv_cameras


def detect_markers(data_dir: str):
    """Этап 3: Детекция ArUco маркеров"""
    print("🎯 Этап 3: Детекция ArUco маркеров (ID 1-13)")
    
    detector = SimpleArUcoDetector(enable_logging=False, filter_6x6=True)
    marker_detections = detector.detect_markers_in_directory(data_dir)
    
    if not marker_detections:
        raise ValueError("Маркеры не найдены")
    
    # Анализ результатов
    total_detections = sum(len(detections) for detections in marker_detections.values())
    unique_markers = set()
    for detections in marker_detections.values():
        unique_markers.update(detections.keys())
    
    # Подсчет маркеров для триангуляции
    marker_frequency = {}
    for detections in marker_detections.values():
        for marker_id in detections.keys():
            marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
    
    triangulatable_markers = sum(1 for freq in marker_frequency.values() if freq >= 3)
    found_markers = sorted(list(unique_markers))
    
    print(f"   Найдено маркеров: {found_markers}")
    print(f"   Всего детекций: {total_detections}")
    print(f"   Готовых для триангуляции: {triangulatable_markers}")
    
    return marker_detections


def triangulate_all_markers(opencv_cameras, marker_detections):
    """Этап 4: 3D триангуляция маркеров"""
    print("🔺 Этап 4: 3D триангуляция маркеров")
    
    # Отладочная информация
    print(f"   Камер с параметрами: {len(opencv_cameras)}")
    print(f"   Камер с детекциями: {len(marker_detections)}")
    
    # Проверим совместимость данных
    camera_ids_opencv = set(opencv_cameras.keys())
    camera_ids_detections = set(marker_detections.keys())
    common_cameras = camera_ids_opencv & camera_ids_detections
    print(f"   Общих камер: {len(common_cameras)}")
    
    if len(common_cameras) == 0:
        raise ValueError("Нет общих камер между OpenCV параметрами и детекциями")
    
    triangulated_markers = triangulate_markers(
        opencv_cameras,
        marker_detections,
        min_cameras=3,
        max_reprojection_error=200.0
    )
    
    if not triangulated_markers:
        # Более детальная диагностика
        print("   🔍 Диагностика проблемы:")
        for camera_id in common_cameras:
            detections = marker_detections[camera_id]
            print(f"     Камера {camera_id}: {len(detections)} маркеров")
            for marker_id in detections.keys():
                print(f"       Маркер {marker_id}")
        raise ValueError("Не удалось триангулировать маркеры")
    
    # Анализ результатов
    high_confidence = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7)
    avg_error = sum(m.reprojection_error for m in triangulated_markers.values()) / len(triangulated_markers)
    triangulated_ids = sorted(list(triangulated_markers.keys()))
    
    print(f"   Триангулировано маркеров: {triangulated_ids}")
    print(f"   Высокого качества: {high_confidence}/{len(triangulated_markers)}")
    print(f"   Средняя ошибка: {avg_error:.2f} пикс")
    
    return triangulated_markers


def create_blender_files(triangulated_markers, output_dir: str, data_dir: str):
    """Этап 5: Подготовка данных для Blender"""
    print("🎨 Этап 5: Подготовка для Blender")
    
    # Подготовка данных для маркеров
    blender_data = prepare_blender_export(triangulated_markers)
    
    # Добавляем метаданные
    blender_data['metadata']['created_by'] = 'ArUco Autocalibration Pipeline'
    blender_data['metadata']['format_version'] = '1.0'
    blender_data['metadata']['blender_script_compatible'] = True
    
    # Сохранение данных для маркеров
    blender_file = os.path.join(output_dir, 'blender_aruco_markers.json')
    with open(blender_file, 'w', encoding='utf-8') as f:
        json.dump(blender_data, f, indent=2, ensure_ascii=False)
    
    # Создание единого Blender скрипта
    blender_script = create_unified_blender_script(output_dir, data_dir, blender_file)
    
    # Статистика для отладки
    stats = {
        'triangulation_stats': {
            'total_markers': len(triangulated_markers),
            'high_quality': sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7),
            'medium_quality': sum(1 for m in triangulated_markers.values() if 0.5 <= m.triangulation_confidence < 0.7),
            'low_quality': sum(1 for m in triangulated_markers.values() if m.triangulation_confidence < 0.5),
            'avg_reprojection_error': sum(m.reprojection_error for m in triangulated_markers.values()) / len(triangulated_markers),
            'marker_details': {
                str(marker_id): {
                    'position': list(result.position_3d),
                    'confidence': result.triangulation_confidence,
                    'error': result.reprojection_error,
                    'cameras': result.camera_ids
                }
                for marker_id, result in triangulated_markers.items()
            }
        }
    }
    
    stats_file = os.path.join(output_dir, 'triangulation_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Создание инструкции по использованию
    instructions = f"""# ArUco + Камеры - Инструкция по импорту в Blender

## Быстрый старт:

1. **Откройте Blender**
2. **Откройте Scripting workspace** (вверху Blender)
3. **Откройте скрипт:** Text → Open → выберите `{os.path.basename(blender_script)}`
4. **Настройте пути в начале скрипта:**
   ```python
   FOLDER = r"{os.path.abspath(data_dir)}"
   MARKERS_FILE = r"{os.path.abspath(blender_file)}"
   ```
5. **Запустите:** Text → Run Script (или Alt+P)

## Что будет импортировано:

### 🎥 Камеры (RealityCapture_Cameras):
- Позиции и ориентации из XMP файлов
- Внутренние параметры (фокусное расстояние, главная точка)
- Дополнительные данные как кастомные свойства

### 🏷️ ArUco Маркеры (ArUco_Markers):
- 3D позиции из триангуляции
- Цветовая кодировка по качеству:
  - 🟢 **Зеленые** - высокое качество (confidence ≥ 0.7)
  - 🟡 **Желтые** - среднее качество (confidence 0.5-0.7)  
  - 🟠 **Оранжевые** - низкое качество (confidence < 0.5)

## Решение проблем:

- **Не найдены XMP файлы:** Проверьте путь FOLDER
- **Не найден файл маркеров:** Проверьте путь MARKERS_FILE  
- **Пустая сцена:** Убедитесь что пути корректны и файлы существуют
- **Ошибки в консоли:** Смотрите System Console (Window → Toggle System Console)

---

💡 **Совет:** Используйте только маркеры высокого качества (зеленые) для наиболее точных результатов.

📍 **Координатная система:** Данные импортируются в абсолютной системе координат RealityCapture.
"""
    
    instructions_file = os.path.join(output_dir, 'blender_import_instructions.txt')
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"   💾 Данные маркеров: {blender_file}")
    print(f"   📜 Blender скрипт: {blender_script}")
    print(f"   📖 Инструкции: {instructions_file}")
    print(f"   📊 Статистика: {stats_file}")
    
    return blender_file, blender_script, instructions_file, stats_file


def create_unified_blender_script(output_dir: str, data_dir: str, markers_file: str) -> str:
    """Создание единого Blender скрипта для импорта"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Единый импортер камер и ArUco маркеров для Blender
================================================

Этот скрипт должен запускаться ТОЛЬКО внутри Blender!

НАСТРОЙТЕ ПУТИ НИЖЕ ПЕРЕД ЗАПУСКОМ!
"""

import os
import json
import xml.etree.ElementTree as ET

# Blender-специфичные импорты (работают только в Blender!)
from mathutils import Matrix, Vector
import bpy
import traceback

# ---- НАСТРОЙТЕ ЭТИ ПУТИ! ----
FOLDER = r"{os.path.abspath(data_dir)}"  # Папка с XMP файлами
MARKERS_FILE = r"{os.path.abspath(markers_file)}"  # Файл с маркерами
# -----------------------------

RC_NS = {{"rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#", "xcr":"http://www.capturingreality.com/ns/xcr/1.1#"}}

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
    attrs = {{k.split('}}')[1]: v for k, v in desc.attrib.items() if k.startswith('{{'+RC_NS['xcr']+'}}')}}
    return {{"path": path, "name": os.path.splitext(os.path.basename(path))[0],
            "position": pos, "rotation": rot, "dist": dist, "attrs": attrs}}

def to_blender_cam_matrix(R_w2cv_3x3, C_world_vec3):
    R_bcam2cv = Matrix(((1,0,0),(0,-1,0),(0,0,-1)))
    R_cv2bcam = R_bcam2cv.transposed()
    R_w2bcam = R_cv2bcam @ R_w2cv_3x3
    R_bcam2w = R_w2bcam.transposed()
    M = R_bcam2w.to_4x4()
    M.translation = Vector(C_world_vec3)
    return M

def ensure_collection(name):
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
        print(f"[WARN] Пропускаем {{name}}: неправильный формат позиции/поворота")
        return None
    
    C_world = Vector(pos)
    R_w2cv = Matrix((rot[0:3], rot[3:6], rot[6:9]))
    
    cam_data_block = bpy.data.cameras.new(name=name+"_DATA")
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data_block)
    collection.objects.link(cam_obj)
    
    cam_obj.matrix_world = to_blender_cam_matrix(R_w2cv, C_world)
    
    f35 = float(attrs.get("FocalLength35mm", 0.0) or 0.0)
    if f35 > 0:
        cam_data_block.sensor_fit = 'HORIZONTAL'
        cam_data_block.sensor_width = 36.0
        cam_data_block.lens = f35
    
    try:
        ppu = float(attrs.get("PrincipalPointU", ""))
        ppv = float(attrs.get("PrincipalPointV", ""))
        if abs(ppu) < 0.05 and abs(ppv) < 0.05:
            cam_data_block.shift_x = ppu
            cam_data_block.shift_y = -ppv
        else:
            cam_data_block.shift_x = (ppu - 0.5)
            cam_data_block.shift_y = -(ppv - 0.5)
    except Exception:
        pass
    
    cam_obj["RC_attrs"] = attrs
    cam_obj["RC_distortion"] = cam_data["dist"]
    return cam_obj

def create_marker(marker_id, marker_data, collection):
    name = f"ArUco_Marker_{{marker_id:02d}}"
    position = marker_data['position']
    confidence = marker_data['confidence']
    quality = marker_data.get('quality', 'unknown')
    
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=tuple(position))
    marker_obj = bpy.context.active_object
    marker_obj.name = name
    
    # Размер по качеству
    if quality == 'high':
        size = 0.1
    elif quality == 'medium':
        size = 0.08
    else:
        size = 0.06
    marker_obj.empty_display_size = size
    
    # Цвет по качеству
    if quality == 'high':
        marker_obj.color = (0.0, 1.0, 0.0, 1.0)  # Зеленый
    elif quality == 'medium':
        marker_obj.color = (1.0, 1.0, 0.0, 1.0)  # Желтый
    else:
        marker_obj.color = (1.0, 0.5, 0.0, 1.0)  # Оранжевый
    
    # Свойства
    marker_obj["aruco_id"] = marker_id
    marker_obj["confidence"] = confidence
    marker_obj["quality"] = quality
    marker_obj["triangulated_position"] = position
    
    # Перемещение в коллекцию
    collection.objects.link(marker_obj)
    if marker_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(marker_obj)
    
    return marker_obj

def import_cameras(folder):
    print("🎥 Импорт камер...")
    if not folder or not os.path.isdir(folder):
        print(f"[ERROR] Папка не найдена: {{folder}}")
        return []
    
    xmp_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".xmp")]
    if not xmp_paths:
        print(f"[ERROR] XMP файлы не найдены в: {{folder}}")
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
            print(f"[ERROR] {{p}}: {{e}}")
    
    print(f"   Импортировано камер: {{len(imported)}}")
    return imported

def import_markers(markers_file):
    print("🏷️ Импорт маркеров...")
    if not markers_file or not os.path.exists(markers_file):
        print(f"[ERROR] Файл маркеров не найден: {{markers_file}}")
        return []
    
    try:
        with open(markers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Ошибка чтения файла: {{e}}")
        return []
    
    markers_data = data.get('markers', {{}})
    if not markers_data:
        print("[ERROR] Данные маркеров не найдены")
        return []
    
    coll = ensure_collection("ArUco_Markers")
    imported = []
    high_quality = 0
    
    for marker_name, marker_info in markers_data.items():
        try:
            marker_id = marker_info['id']
            quality = marker_info.get('quality', 'unknown')
            marker_obj = create_marker(marker_id, marker_info, coll)
            imported.append(marker_obj.name)
            if quality == 'high':
                high_quality += 1
        except Exception as e:
            print(f"[ERROR] Ошибка создания маркера {{marker_name}}: {{e}}")
    
    print(f"   Импортировано маркеров: {{len(imported)}} (высокого качества: {{high_quality}})")
    return imported

def clear_existing():
    print("🧹 Очистка...")
    # Удаляем камеры и маркеры
    objects_to_remove = []
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' or obj.name.startswith('ArUco_Marker_'):
            objects_to_remove.append(obj)
    
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Удаляем коллекции
    for coll_name in ['RealityCapture_Cameras', 'ArUco_Markers']:
        if coll_name in bpy.data.collections:
            bpy.data.collections.remove(bpy.data.collections[coll_name])

def main():
    print("="*50)
    print("🚀 ЕДИНЫЙ ИМПОРТЕР КАМЕР И МАРКЕРОВ")
    print("="*50)
    print(f"📂 XMP папка: {{FOLDER}}")
    print(f"🏷️ Файл маркеров: {{MARKERS_FILE}}")
    print("="*50)
    
    # Настройка сцены
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.ops.object.select_all(action='DESELECT')
    
    try:
        clear_existing()
        cameras = import_cameras(FOLDER)
        markers = import_markers(MARKERS_FILE)
        
        print(f"\\n✅ ИМПОРТ ЗАВЕРШЕН!")
        print(f"   🎥 Камер: {{len(cameras)}}")
        print(f"   🏷️ Маркеров: {{len(markers)}}")
        
        if cameras or markers:
            print("\\n📋 КОЛЛЕКЦИИ:")
            if cameras:
                print("   • RealityCapture_Cameras - камеры из XMP")
            if markers:
                print("   • ArUco_Markers - маркеры (🟢высокое 🟡среднее 🟠низкое качество)")
        else:
            print("\\n⚠️ Проверьте пути к файлам!")
            
    except Exception as e:
        print(f"\\n💥 ОШИБКА: {{e}}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    # Сохранение скрипта
    script_path = os.path.join(output_dir, 'blender_import_cameras_and_markers.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    return script_path


def main():
    """Главная функция запуска пайплайна"""
    
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    
    print("🚀 ArUco Автокалибровка - Полный пайплайн")
    print("=" * 50)
    print("От XMP файлов до единого Blender импорта")
    print("=" * 50)
    
    # Создание выходной директории
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Валидация данных
        if not validate_input_data(DATA_DIR):
            return 1
        
        # Этап 1: Загрузка камер
        xmp_cameras = load_cameras(DATA_DIR)
        
        # Этап 2: Конвертация камер
        opencv_cameras = convert_cameras(xmp_cameras)
        
        # Этап 3: Детекция маркеров
        marker_detections = detect_markers(DATA_DIR)
        
        # Этап 4: Триангуляция
        triangulated_markers = triangulate_all_markers(opencv_cameras, marker_detections)
        
        # Этап 5: Подготовка для Blender
        blender_file, blender_script, instructions_file, stats_file = create_blender_files(
            triangulated_markers, OUTPUT_DIR, DATA_DIR
        )
        
        # Финальный результат
        execution_time = time.time() - start_time
        
        print(f"\n🎉 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print(f"⏱️  Время выполнения: {execution_time:.1f} сек")
        print(f"🎨 Триангулировано маркеров: {len(triangulated_markers)}")
        
        # Детальная статистика по качеству
        high_quality = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7)
        medium_quality = sum(1 for m in triangulated_markers.values() if 0.5 <= m.triangulation_confidence < 0.7)
        low_quality = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence < 0.5)
        
        print(f"   🟢 Высокого качества: {high_quality}")
        print(f"   🟡 Среднего качества: {medium_quality}")
        print(f"   🟠 Низкого качества: {low_quality}")
        
        print(f"📂 Результаты в: {OUTPUT_DIR}")
        print(f"")
        print(f"🎬 Следующие шаги:")
        print(f"   1. Откройте Blender")
        print(f"   2. Откройте Scripting workspace")
        print(f"   3. Загрузите скрипт: {os.path.basename(blender_script)}")
        print(f"   4. Проверьте пути в начале скрипта")
        print(f"   5. Запустите скрипт (Alt+P)")
        print(f"")
        print(f"📋 Что будет импортировано:")
        print(f"   • RealityCapture_Cameras - {len(xmp_cameras)} камер из XMP")
        print(f"   • ArUco_Markers - {len(triangulated_markers)} триангулированных маркеров")
        print(f"")
        print(f"💡 Рекомендация: {high_quality} маркеров высокого качества готовы к использованию")
        
        return 0
        
    except Exception as e:
        print(f"💥 Ошибка пайплайна: {e}")
        return 1


if __name__ == "__main__":
    exit(main())