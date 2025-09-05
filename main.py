#!/usr/bin/env python3
"""
Главный скрипт полного пайплайна автокалибровки ArUco маркеров
=============================================================

Обрабатывает данные и создает aruco_marker.json:
1. Загрузка параметров камер из XMP файлов
2. Конвертация в OpenCV формат  
3. Детекция ArUco маркеров (DICT_4X4_1000, ID 1-13)
4. 3D триангуляция маркеров
5. Создание aruco_marker.json для Blender

РЕЗУЛЬТАТ: Файл aruco_marker.json с триангулированными маркерами

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
    from triangulation import triangulate_markers
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
        raise ValueError("Не удалось триангулировать маркеры")
    
    # Анализ результатов
    high_confidence = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7)
    avg_error = sum(m.reprojection_error for m in triangulated_markers.values()) / len(triangulated_markers)
    triangulated_ids = sorted(list(triangulated_markers.keys()))
    
    print(f"   Триангулировано маркеров: {triangulated_ids}")
    print(f"   Высокого качества: {high_confidence}/{len(triangulated_markers)}")
    print(f"   Средняя ошибка: {avg_error:.2f} пикс")
    
    return triangulated_markers


def create_blender_files(triangulated_markers, opencv_cameras, xmp_cameras, output_dir: str, data_dir: str):
    """Этап 5: Создание aruco_marker.json"""
    print("🎨 Этап 5: Создание aruco_marker.json")
    
    # Подготовка данных для маркеров
    blender_data = prepare_blender_export(triangulated_markers)
    
    # Сохранение JSON файла
    json_file = os.path.join(output_dir, 'aruco_marker.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(blender_data, f, indent=2, ensure_ascii=False)
    
    # Статистика
    high_quality_markers = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7)
    
    print(f"   💾 JSON файл: {json_file}")
    print(f"   🏷️ Маркеров высокого качества: {high_quality_markers}/{len(triangulated_markers)}")
    print(f"   📊 Размер файла: {os.path.getsize(json_file) // 1024:.1f} KB")
    
    return json_file


def prepare_blender_export(triangulated_markers) -> dict:
    """Подготовка данных маркеров для экспорта в JSON"""
    
    # Подсчитываем маркеры высокого качества
    high_confidence_count = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7)
    medium_confidence_count = sum(1 for m in triangulated_markers.values() if 0.5 <= m.triangulation_confidence < 0.7)
    low_confidence_count = len(triangulated_markers) - high_confidence_count - medium_confidence_count
    
    blender_data = {
        'metadata': {
            'total_markers': len(triangulated_markers),
            'high_confidence_markers': high_confidence_count,
            'medium_confidence_markers': medium_confidence_count,
            'low_confidence_markers': low_confidence_count,
            'coordinate_system': 'realitycapture_absolute',
            'created_by': 'ArUco Autocalibration Pipeline',
            'format_version': '1.0'
        },
        'markers': {}
    }
    
    for marker_id, result in triangulated_markers.items():
        quality = 'high' if result.triangulation_confidence >= 0.7 else 'medium' if result.triangulation_confidence >= 0.5 else 'low'
        
        blender_data['markers'][f'marker_{marker_id}'] = {
            'id': marker_id,
            'position': list(result.position_3d),
            'confidence': result.triangulation_confidence,
            'quality': quality,
            'reprojection_error': result.reprojection_error,
            'observations_count': result.observations_count,
            'camera_ids': result.camera_ids
        }
    
    return blender_data


def main():
    """Главная функция запуска пайплайна"""
    
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    
    print("🚀 ArUco Автокалибровка - Полный пайплайн")
    print("=" * 50)
    print("От XMP файлов до aruco_marker.json")
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
        
        # Этап 5: Создание JSON файла
        json_file = create_blender_files(
            triangulated_markers, opencv_cameras, xmp_cameras, OUTPUT_DIR, DATA_DIR
        )
        
        # Финальный результат
        execution_time = time.time() - start_time
        
        print(f"\n🎉 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print(f"⏱️  Время выполнения: {execution_time:.1f} сек")
        print(f"🎨 Триангулировано маркеров: {len(triangulated_markers)}")
        
        # Детальная статистика по качеству
        high_quality_markers = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7)
        medium_quality_markers = sum(1 for m in triangulated_markers.values() if 0.5 <= m.triangulation_confidence < 0.7)
        low_quality_markers = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence < 0.5)
        
        print(f"\n📊 СТАТИСТИКА КАЧЕСТВА:")
        print(f"   🏷️ Маркеры - 🟢 {high_quality_markers}  🟡 {medium_quality_markers}  🟠 {low_quality_markers}")
        
        print(f"\n📂 Результат: {OUTPUT_DIR}")
        print(f"   💾 {os.path.basename(json_file)} - данные триангулированных маркеров")
        print(f"")
        print(f"🎯 Содержимое JSON:")
        print(f"   • metadata - информация о триангуляции")
        print(f"   • markers - 3D позиции маркеров с метаданными")
        print(f"")
        print(f"📋 Структура маркера:")
        print(f"   • id - номер маркера (1-13)")
        print(f"   • position - [X, Y, Z] координаты в метрах")
        print(f"   • confidence - уверенность триангуляции (0-1)")
        print(f"   • quality - 'high'/'medium'/'low'")
        print(f"   • reprojection_error - ошибка в пикселях")
        print(f"   • observations_count - количество камер")
        print(f"   • camera_ids - список ID камер")
        print(f"")
        
        # Рекомендации по качеству
        if high_quality_markers >= 8:
            print(f"✅ Отличное качество! {high_quality_markers} маркеров высокого качества")
        elif high_quality_markers >= 5:
            print(f"⚠️  Хорошее качество. {high_quality_markers} маркеров высокого качества")
        else:
            print(f"⚠️  Ограниченное качество. Только {high_quality_markers} маркеров высокого качества")
        
        print(f"\n💡 JSON готов для использования в других приложениях!")
        
        return 0
        
    except Exception as e:
        print(f"💥 Ошибка пайплайна: {e}")
        return 1


if __name__ == "__main__":
    exit(main())