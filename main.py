#!/usr/bin/env python3
"""
Главный скрипт полного пайплайна автокалибровки ArUco маркеров
=============================================================

Полный пайплайн от XMP файлов до 3D координат для Blender:
1. Загрузка параметров камер из XMP файлов
2. Конвертация в OpenCV формат
3. Детекция ArUco маркеров (DICT_4X4_1000, ID 1-13)
4. 3D триангуляция маркеров
5. Подготовка данных для экспорта в Blender

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
        max_reprojection_error=2.0
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


def create_blender_files(triangulated_markers, output_dir: str):
    """Этап 5: Подготовка данных для экспорта в Blender"""
    print("🎨 Этап 5: Подготовка для Blender")
    
    # Подготовка данных
    blender_data = prepare_blender_export(triangulated_markers)
    
    # Сохранение данных для Blender
    blender_file = os.path.join(output_dir, 'blender_aruco_markers.json')
    with open(blender_file, 'w', encoding='utf-8') as f:
        json.dump(blender_data, f, indent=2, ensure_ascii=False)
    
    # Создание скрипта для Blender
    blender_script = f'''import bpy
import json
import os

def import_aruco_markers():
    """Импорт ArUco маркеров в Blender как Empty объекты"""
    
    # ИСПРАВЛЕННЫЙ ПОИСК ФАЙЛА
    # Ищем файл в нескольких местах
    possible_paths = [
        # 1. В той же папке где находится этот скрипт
        os.path.join(os.path.dirname(__file__), "blender_aruco_markers.json") if __name__ != "__main__" else None,
        
        # 2. В директории blend файла (если файл сохранен)
        os.path.join(os.path.dirname(bpy.data.filepath), "blender_aruco_markers.json") if bpy.data.filepath else None,
        
        # 3. В текущей рабочей директории
        os.path.join(os.getcwd(), "blender_aruco_markers.json"),
        
        # 4. В папке results от текущей директории
        os.path.join(os.getcwd(), "results", "blender_aruco_markers.json"),
        
        # 5. На рабочем столе (если пользователь туда скопировал)
        os.path.join(os.path.expanduser("~"), "Desktop", "blender_aruco_markers.json"),
        
        # 6. Абсолютный путь к проекту
        r"C:\\Users\\admin\\PycharmProjects\\autocalibration\\results\\blender_aruco_markers.json"
    ]
    
    # Ищем файл
    data_file = None
    for path in possible_paths:
        if path and os.path.exists(path):
            data_file = path
            break
    
    if not data_file:
        print("❌ Файл blender_aruco_markers.json не найден!")
        print("🔍 Искали в следующих местах:")
        for path in possible_paths:
            if path:
                print(f"   - {{path}}")
        print("\\n💡 Решение:")
        print("   1. Скопируйте blender_aruco_markers.json в папку с .blend файлом")
        print("   2. Или скопируйте в папку Blender")
        print("   3. Или измените путь в строке 21 этого скрипта")
        return
    
    print(f"✅ Найден файл: {{data_file}}")
    
    # Загрузка данных
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {{e}}")
        return
    
    # Очистка существующих ArUco маркеров
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith('ArUco_Marker_'):
            obj.select_set(True)
    bpy.ops.object.delete()
    
    # Создание коллекции для маркеров
    collection_name = "ArUco_Markers"
    if collection_name in bpy.data.collections:
        bpy.data.collections.remove(bpy.data.collections[collection_name])
    
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    
    # Импорт маркеров
    markers_data = data['markers']
    total_markers = len(markers_data)
    high_quality_count = 0
    
    print(f"🎯 Импортируем {{total_markers}} ArUco маркеров...")
    
    for marker_name, marker_info in markers_data.items():
        marker_id = marker_info['id']
        position = marker_info['position']
        confidence = marker_info['confidence']
        quality = marker_info['quality']
        
        # Создание Empty объекта
        bpy.ops.object.empty_add(
            type='PLAIN_AXES',
            location=(position[0], position[1], position[2])
        )
        
        empty_obj = bpy.context.active_object
        empty_obj.name = f"ArUco_Marker_{{marker_id:02d}}"
        
        # Размер в зависимости от качества
        if quality == 'high':
            empty_obj.empty_display_size = 0.1
            high_quality_count += 1
        elif quality == 'medium':
            empty_obj.empty_display_size = 0.08
        else:
            empty_obj.empty_display_size = 0.06
        
        # Цвет в зависимости от качества
        if quality == 'high':
            empty_obj.color = (0.0, 1.0, 0.0, 1.0)  # Зеленый
        elif quality == 'medium':
            empty_obj.color = (1.0, 1.0, 0.0, 1.0)  # Желтый
        else:
            empty_obj.color = (1.0, 0.5, 0.0, 1.0)  # Оранжевый
        
        # Добавление кастомных свойств
        empty_obj["aruco_id"] = marker_id
        empty_obj["confidence"] = confidence
        empty_obj["quality"] = quality
        
        # ИСПРАВЛЕННОЕ добавление в коллекцию
        # Убираем из Scene Collection только если объект там есть
        if empty_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(empty_obj)
        
        # Добавляем в нашу коллекцию
        collection.objects.link(empty_obj)
    
    print(f"✅ Импорт завершен!")
    print(f"   📊 Всего маркеров: {{total_markers}}")
    print(f"   ⭐ Высокого качества: {{high_quality_count}}")
    print(f"   📁 Коллекция: {{collection_name}}")
    print(f"")
    print(f"🎨 Цветовая схема:")
    print(f"   🟢 Зеленый - высокое качество (confidence ≥ 0.7)")
    print(f"   🟡 Желтый - среднее качество (confidence 0.5-0.7)")
    print(f"   🟠 Оранжевый - низкое качество (confidence < 0.5)")
    print(f"")
    print(f"💡 Совет: Маркеры имеют кастомные свойства с данными о триангуляции")

# Запуск импорта
if __name__ == "__main__":
    import_aruco_markers()
'''
    
    script_file = os.path.join(output_dir, 'import_aruco_markers.py')
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(blender_script)
    
    print(f"   💾 Данные маркеров: {blender_file}")
    print(f"   💾 Скрипт Blender: {script_file}")
    
    return blender_file, script_file


def main():
    """Главная функция запуска пайплайна"""
    
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    
    print("🚀 ArUco Автокалибровка - Полный пайплайн")
    print("=" * 50)
    print("От XMP файлов до 3D координат в Blender")
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
        blender_file, script_file = create_blender_files(triangulated_markers, OUTPUT_DIR)
        
        # Финальный результат
        execution_time = time.time() - start_time
        
        print(f"\n🎉 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print(f"⏱️  Время выполнения: {execution_time:.1f} сек")
        print(f"🎨 Триангулировано маркеров: {len(triangulated_markers)}")
        print(f"📂 Результаты в: {OUTPUT_DIR}")
        print(f"")
        print(f"🎬 Следующий шаг - Blender:")
        print(f"   1. Откройте Blender")
        print(f"   2. Scripting → Open → {script_file}")
        print(f"   3. Run Script")
        print(f"   4. Маркеры появятся как Empty объекты")
        
        return 0
        
    except Exception as e:
        print(f"💥 Ошибка пайплайна: {e}")
        return 1


if __name__ == "__main__":
    exit(main())