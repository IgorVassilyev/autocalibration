#!/usr/bin/env python3
"""
Главный скрипт полного пайплайна автокалибровки ArUco маркеров
=============================================================

Полный пайплайн от XMP файлов до 3D координат для Blender:
1. Загрузка параметров камер из XMP файлов
2. Конвертация в OpenCV формат
3. Детекция ArUco маркеров (DICT_4X4_1000, ID 1-13)
4. 3D триангуляция маркеров
5. Подготовка данных для импорта через Blender addon

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
        min_cameras=0,
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
    """Этап 5: Подготовка данных для Blender addon"""
    print("🎨 Этап 5: Подготовка для Blender addon")
    
    # Подготовка данных
    blender_data = prepare_blender_export(triangulated_markers)
    
    # Добавляем метаданные для addon
    blender_data['metadata']['created_by'] = 'ArUco Autocalibration Pipeline'
    blender_data['metadata']['format_version'] = '1.0'
    blender_data['metadata']['addon_compatible'] = True
    
    # Сохранение данных для Blender addon
    blender_file = os.path.join(output_dir, 'blender_aruco_markers.json')
    with open(blender_file, 'w', encoding='utf-8') as f:
        json.dump(blender_data, f, indent=2, ensure_ascii=False)
    
    # Создание инструкции по установке addon
    addon_instructions = """
# ArUco Markers Blender Addon - Инструкция по установке

## Установка addon:

1. **Сохраните файл addon:**
   - Сохраните код addon как `aruco_importer.py`
   - Можно скачать с репозитория проекта

2. **Установите в Blender:**
   - Откройте Blender
   - Edit → Preferences → Add-ons
   - Install... → выберите `aruco_importer.py`
   - Поставьте галочку для активации addon

3. **Использование:**
   - В 3D Viewport нажмите N (боковая панель)
   - Появится вкладка "ArUco"
   - Нажмите "Auto Find" или выберите файл blender_aruco_markers.json
   - Нажмите "Import Markers"

## Настройки addon:

- **Размер маркеров**: Размер Empty объектов
- **Размер по качеству**: Автоматический размер по уверенности триангуляции
- **Цвет по качеству**: Цветовая схема по качеству
- **Фильтры качества**: Импорт только высококачественных маркеров
- **Настройки коллекции**: Управление организацией маркеров

## Цветовая схема:

- 🟢 **Зеленый** - высокое качество (confidence ≥ 0.7)
- 🟡 **Желтый** - среднее качество (confidence 0.5-0.7)  
- 🟠 **Оранжевый** - низкое качество (confidence < 0.5)

## Дополнительные функции:

- **Preview Meshes**: Создание mesh представления маркеров
- **Информация о маркерах**: Просмотр свойств выбранных маркеров
- **Автопоиск файлов**: Addon автоматически ищет файл данных

## Кастомные свойства маркеров:

Каждый импортированный маркер содержит:
- `aruco_id`: ID маркера
- `confidence`: Уверенность триангуляции (0-1)
- `quality`: Качество ('high', 'medium', 'low')
- `triangulated_position`: 3D координаты

---

💡 **Совет**: Рекомендуется импортировать только маркеры высокого качества (confidence ≥ 0.7) для наиболее точных результатов.
"""
    
    instructions_file = os.path.join(output_dir, 'blender_addon_instructions.md')
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(addon_instructions)
    
    # Сохраняем также статистику для отладки
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
    
    print(f"   💾 Данные для addon: {blender_file}")
    print(f"   📖 Инструкции: {instructions_file}")
    print(f"   📊 Статистика: {stats_file}")
    
    return blender_file, instructions_file, stats_file


def main():
    """Главная функция запуска пайплайна"""
    
    DATA_DIR = "data"
    OUTPUT_DIR = "results"
    
    print("🚀 ArUco Автокалибровка - Полный пайплайн")
    print("=" * 50)
    print("От XMP файлов до Blender addon импорта")
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
        
        # Этап 5: Подготовка для Blender addon
        blender_file, instructions_file, stats_file = create_blender_files(triangulated_markers, OUTPUT_DIR)
        
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
        print(f"   1. Установите Blender addon (см. {os.path.basename(instructions_file)})")
        print(f"   2. В Blender: N → ArUco → Import Markers")
        print(f"   3. Маркеры появятся как Empty объекты с цветовой кодировкой")
        print(f"")
        print(f"💡 Рекомендация: Импортируйте только маркеры высокого качества")
        print(f"   для наиболее точных результатов (≥{high_quality} маркеров доступно)")
        
        return 0
        
    except Exception as e:
        print(f"💥 Ошибка пайплайна: {e}")
        return 1


if __name__ == "__main__":
    exit(main())