#!/usr/bin/env python3
"""
Главный скрипт пайплайна автокалибровки ArUco маркеров
======================================================

Интегрированный пайплайн от XMP файлов до подготовки данных для триангуляции:
1. Загрузка параметров камер из XMP файлов
2. Конвертация в OpenCV формат
3. Детекция ArUco маркеров на изображениях DICT_4X4_1000
4. Подготовка данных для 3D триангуляции

Использование:
    python main.py --data_dir data --output_dir results
    python main.py --data_dir data --output_dir results --detect_only
    python main.py --help
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, Optional
import logging

# Импорт наших модулей
try:
    from xmp_parser import SimpleXMPParser
    from xmp_to_opencv import convert_cameras_to_opencv
    from aruco_detector import SimpleArUcoDetector, detect_all_markers_in_directory
    from config import CURRENT_IMAGE_SIZE
except ImportError as e:
    print(f"❌ Ошибка импорта модулей: {e}")
    print("Убедитесь, что все файлы проекта находятся в одной директории:")
    print("  - xmp_parser.py")
    print("  - xmp_to_opencv.py") 
    print("  - aruco_detector.py")
    print("  - config.py")
    sys.exit(1)


class AutoCalibrationPipeline:
    """
    Главный класс интеграционного пайплайна автокалибровки
    """
    
    def __init__(self, data_dir: str, output_dir: str, 
                 enable_logging: bool = True, save_intermediate: bool = True):
        """
        Инициализация пайплайна
        
        Parameters:
        -----------
        data_dir : str
            Директория с XMP файлами и изображениями
        output_dir : str
            Директория для сохранения результатов
        enable_logging : bool
            Включить детальное логирование
        save_intermediate : bool
            Сохранять промежуточные результаты
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.enable_logging = enable_logging
        self.save_intermediate = save_intermediate
        
        # Создание выходной директории
        os.makedirs(output_dir, exist_ok=True)
        
        # Настройка логирования
        if enable_logging:
            self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        
        # Промежуточные результаты
        self.xmp_cameras = {}
        self.opencv_cameras = {}
        self.marker_detections = {}
        
        # Статистика выполнения
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'stages_completed': [],
            'stages_failed': [],
            'total_execution_time': 0
        }
    
    def _setup_logging(self):
        """Настройка системы логирования"""
        log_file = os.path.join(self.output_dir, 'pipeline.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def validate_input_data(self) -> bool:
        """
        Валидация входных данных
        
        Returns:
        --------
        bool
            True если данные валидны
        """
        print("🔍 Валидация входных данных...")
        
        # Проверка существования директории
        if not os.path.exists(self.data_dir):
            print(f"❌ Директория не найдена: {self.data_dir}")
            return False
        
        # Поиск XMP файлов
        xmp_files = [f for f in os.listdir(self.data_dir) if f.endswith('.xmp')]
        if not xmp_files:
            print(f"❌ XMP файлы не найдены в {self.data_dir}")
            return False
        
        # Поиск изображений
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for f in os.listdir(self.data_dir):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                image_files.append(f)
        
        if not image_files:
            print(f"❌ Изображения не найдены в {self.data_dir}")
            return False
        
        # Проверка соответствия XMP файлов и изображений
        xmp_ids = {os.path.splitext(f)[0] for f in xmp_files}
        image_ids = {os.path.splitext(f)[0] for f in image_files}
        
        missing_images = xmp_ids - image_ids
        missing_xmp = image_ids - xmp_ids
        
        if missing_images:
            print(f"⚠️  Нет изображений для XMP: {missing_images}")
        
        if missing_xmp:
            print(f"⚠️  Нет XMP для изображений: {missing_xmp}")
        
        common_ids = xmp_ids & image_ids
        if len(common_ids) < 3:
            print(f"❌ Недостаточно пар XMP-изображение: {len(common_ids)} < 3")
            return False
        
        print(f"✅ Валидация пройдена:")
        print(f"   XMP файлов: {len(xmp_files)}")
        print(f"   Изображений: {len(image_files)}")
        print(f"   Совпадающих пар: {len(common_ids)}")
        
        return True
    
    def stage_1_load_cameras(self) -> bool:
        """
        Этап 1: Загрузка параметров камер из XMP файлов
        """
        stage_name = "load_cameras"
        print("\n🔧 ЭТАП 1: Загрузка параметров камер")
        print("-" * 40)
        
        try:
            parser = SimpleXMPParser(enable_logging=self.enable_logging)
            self.xmp_cameras = parser.load_all_cameras(self.data_dir)
            
            if not self.xmp_cameras:
                raise ValueError("Не удалось загрузить камеры")
            
            # Краткая статистика
            stats = parser.get_summary_stats()
            print(f"📊 Статистика загруженных камер:")
            print(f"   Всего камер: {stats['total_cameras']}")
            print(f"   Диапазон фокусных расстояний: {stats['focal_length_range'][0]:.1f}-{stats['focal_length_range'][1]:.1f}mm")
            print(f"   Модели дисторсии: {', '.join(stats['distortion_models'])}")
            print(f"   Системы координат: {', '.join(stats['coordinate_systems'])}")
            
            # Сохранение промежуточных результатов
            if self.save_intermediate:
                output_path = os.path.join(self.output_dir, 'stage1_xmp_cameras.json')
                parser.export_summary_report(os.path.join(self.output_dir, 'stage1_xmp_report.txt'))
                print(f"💾 Отчет XMP камер: {os.path.join(self.output_dir, 'stage1_xmp_report.txt')}")
            
            self.pipeline_stats['stages_completed'].append(stage_name)
            print(f"✅ Этап 1 завершен успешно")
            return True
            
        except Exception as e:
            self.pipeline_stats['stages_failed'].append(stage_name)
            print(f"❌ Этап 1 неудачен: {e}")
            return False
    
    def stage_2_convert_cameras(self) -> bool:
        """
        Этап 2: Конвертация параметров камер в OpenCV формат
        """
        stage_name = "convert_cameras"
        print("\n🔄 ЭТАП 2: Конвертация в OpenCV формат")
        print("-" * 40)
        
        try:
            self.opencv_cameras = convert_cameras_to_opencv(
                self.xmp_cameras, 
                CURRENT_IMAGE_SIZE
            )
            
            if not self.opencv_cameras:
                raise ValueError("Ошибка конвертации камер")
            
            # Показываем пример конвертации одной камеры
            first_camera_id = next(iter(self.opencv_cameras))
            first_camera = self.opencv_cameras[first_camera_id]
            
            print(f"📷 Пример конвертации камеры {first_camera_id}:")
            print(f"   Размер изображения: {CURRENT_IMAGE_SIZE[0]}×{CURRENT_IMAGE_SIZE[1]} пикселей")
            print(f"   Фокусные расстояния: fx={first_camera['fx']:.1f}, fy={first_camera['fy']:.1f} пикс")
            print(f"   Главная точка: cx={first_camera['cx']:.1f}, cy={first_camera['cy']:.1f} пикс")
            
            warnings = first_camera.get('conversion_warnings', [])
            if warnings:
                print(f"   ⚠️  Предупреждения: {'; '.join(warnings)}")
            else:
                print(f"   ✅ Конвертация без предупреждений")
            
            # Сохранение промежуточных результатов
            if self.save_intermediate:
                output_path = os.path.join(self.output_dir, 'stage2_opencv_cameras.json')
                # Преобразуем numpy массивы в списки для JSON
                json_data = {}
                for cam_id, cam_data in self.opencv_cameras.items():
                    json_data[cam_id] = {
                        'camera_matrix': cam_data['camera_matrix'].tolist(),
                        'distortion_coeffs': cam_data['distortion_coeffs'].tolist(),
                        'fx': float(cam_data['fx']),
                        'fy': float(cam_data['fy']),
                        'cx': float(cam_data['cx']),
                        'cy': float(cam_data['cy']),
                        'position': cam_data['position'].tolist(),
                        'rotation': cam_data['rotation'].tolist(),
                        'image_size': cam_data['image_size'],
                        'conversion_warnings': cam_data.get('conversion_warnings', [])
                    }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                print(f"💾 Параметры OpenCV камер: {output_path}")
            
            self.pipeline_stats['stages_completed'].append(stage_name)
            print(f"✅ Этап 2 завершен успешно")
            return True
            
        except Exception as e:
            self.pipeline_stats['stages_failed'].append(stage_name)
            print(f"❌ Этап 2 неудачен: {e}")
            return False
    
    def stage_3_detect_markers(self, create_marked_images: bool = False) -> bool:
        """
        Этап 3: Детекция ArUco маркеров DICT_4X4_1000
        """
        stage_name = "detect_markers"
        print("\n🎯 ЭТАП 3: Детекция ArUco маркеров (DICT_4X4_1000)")
        print("-" * 40)
        
        try:
            # Подготовка путей для сохранения
            detection_json = os.path.join(self.output_dir, 'stage3_marker_detections.json')
            marked_images_dir = os.path.join(self.output_dir, 'marked_images') if create_marked_images else None
            
            # Запуск детекции
            self.marker_detections = detect_all_markers_in_directory(
                directory=self.data_dir,
                output_file=detection_json,
                create_images=create_marked_images,
                images_output_dir=marked_images_dir
            )
            
            if not self.marker_detections:
                raise ValueError("Маркеры не найдены")
            
            # Анализ результатов детекции
            total_detections = sum(len(detections) for detections in self.marker_detections.values())
            unique_markers = set()
            for detections in self.marker_detections.values():
                unique_markers.update(detections.keys())
            
            # Подсчет маркеров пригодных для триангуляции
            marker_frequency = {}
            for detections in self.marker_detections.values():
                for marker_id in detections.keys():
                    marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
            
            triangulatable_markers = sum(1 for freq in marker_frequency.values() if freq >= 3)
            
            print(f"\n🎯 АНАЛИЗ РЕЗУЛЬТАТОВ ДЕТЕКЦИИ:")
            print(f"   Всего детекций: {total_detections}")
            print(f"   Уникальных маркеров: {len(unique_markers)}")
            print(f"   Готовых для триангуляции (≥3 камер): {triangulatable_markers}")
            
            # Проверка достаточности для триангуляции
            if triangulatable_markers < 3:
                print(f"   ⚠️  ВНИМАНИЕ: Мало маркеров для триангуляции!")
                print(f"   Рекомендуется минимум 3 маркера на ≥3 камерах")
            elif triangulatable_markers >= 8:
                print(f"   ✅ ОТЛИЧНО: Достаточно маркеров для надежной триангуляции")
            else:
                print(f"   ⚠️  ХОРОШО: Достаточно для базовой триангуляции")
            
            if create_marked_images:
                print(f"💾 Изображения с отмеченными маркерами: {marked_images_dir}")
            
            self.pipeline_stats['stages_completed'].append(stage_name)
            print(f"✅ Этап 3 завершен успешно")
            return True
            
        except Exception as e:
            self.pipeline_stats['stages_failed'].append(stage_name)
            print(f"❌ Этап 3 неудачен: {e}")
            return False
    
    def stage_4_prepare_triangulation_data(self) -> bool:
        """
        Этап 4: Подготовка данных для 3D триангуляции
        """
        stage_name = "prepare_triangulation"
        print("\n📐 ЭТАП 4: Подготовка данных для триангуляции")
        print("-" * 40)
        
        try:
            # Структура данных для триангуляции
            triangulation_data = {
                'metadata': {
                    'pipeline_version': '1.0',
                    'image_size': CURRENT_IMAGE_SIZE,
                    'aruco_dictionary': 'DICT_4X4_1000',
                    'total_cameras': len(self.opencv_cameras),
                    'total_detections': sum(len(d) for d in self.marker_detections.values()),
                    'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'cameras': {},
                'markers': {}
            }
            
            # Подготовка данных камер
            for camera_id, opencv_data in self.opencv_cameras.items():
                triangulation_data['cameras'][camera_id] = {
                    'camera_matrix': opencv_data['camera_matrix'].tolist(),
                    'distortion_coeffs': opencv_data['distortion_coeffs'].tolist(),
                    'position': opencv_data['position'].tolist(),
                    'rotation': opencv_data['rotation'].tolist(),
                    'fx': float(opencv_data['fx']),
                    'fy': float(opencv_data['fy']),
                    'cx': float(opencv_data['cx']),
                    'cy': float(opencv_data['cy']),
                    'image_size': opencv_data['image_size']
                }
            
            # Группировка детекций по маркерам
            for camera_id, detections in self.marker_detections.items():
                if camera_id not in self.opencv_cameras:
                    print(f"   ⚠️  Пропускаем камеру {camera_id}: нет параметров камеры")
                    continue
                    
                for marker_id, detection in detections.items():
                    if marker_id not in triangulation_data['markers']:
                        triangulation_data['markers'][marker_id] = {
                            'observations': [],
                            'cameras_count': 0
                        }
                    
                    triangulation_data['markers'][marker_id]['observations'].append({
                        'camera_id': camera_id,
                        'center': detection.center,
                        'corners': detection.corners,
                        'area': detection.area
                    })
                    triangulation_data['markers'][marker_id]['cameras_count'] += 1
            
            # Анализ готовности для триангуляции
            print(f"📊 АНАЛИЗ ГОТОВНОСТИ ДЛЯ ТРИАНГУЛЯЦИИ:")
            
            markers_by_camera_count = {}
            for marker_id, marker_data in triangulation_data['markers'].items():
                cam_count = marker_data['cameras_count']
                if cam_count not in markers_by_camera_count:
                    markers_by_camera_count[cam_count] = []
                markers_by_camera_count[cam_count].append(marker_id)
            
            for cam_count in sorted(markers_by_camera_count.keys(), reverse=True):
                markers = markers_by_camera_count[cam_count]
                status = "✅" if cam_count >= 3 else "⚠️" if cam_count >= 2 else "❌"
                print(f"   {cam_count} камер: {len(markers)} маркеров {status} {markers}")
            
            # Подсчет готовых для триангуляции
            ready_for_triangulation = sum(
                len(markers) for cam_count, markers in markers_by_camera_count.items() 
                if cam_count >= 3
            )
            
            triangulation_data['metadata']['triangulatable_markers'] = ready_for_triangulation
            
            # Сохранение данных для триангуляции
            if self.save_intermediate:
                output_path = os.path.join(self.output_dir, 'stage4_triangulation_ready.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(triangulation_data, f, indent=2, ensure_ascii=False)
                print(f"💾 Данные для триангуляции: {output_path}")
            
            print(f"\n🎯 ИТОГОВАЯ ГОТОВНОСТЬ:")
            print(f"   Маркеров готовых для триангуляции: {ready_for_triangulation}")
            
            if ready_for_triangulation >= 8:
                print(f"   ✅ ОТЛИЧНО! Готово для высококачественной триангуляции")
            elif ready_for_triangulation >= 5:
                print(f"   ⚠️  ХОРОШО! Готово для стандартной триангуляции")
            elif ready_for_triangulation >= 3:
                print(f"   ⚠️  МИНИМАЛЬНО! Возможна базовая триангуляция")
            else:
                print(f"   ❌ НЕДОСТАТОЧНО для надежной триангуляции")
                print(f"   Рекомендуется улучшить качество детекции или добавить маркеры")
            
            self.pipeline_stats['stages_completed'].append(stage_name)
            print(f"✅ Этап 4 завершен успешно")
            return True
            
        except Exception as e:
            self.pipeline_stats['stages_failed'].append(stage_name)
            print(f"❌ Этап 4 неудачен: {e}")
            return False
    
    def create_final_report(self) -> None:
        """Создание финального отчета о выполнении пайплайна"""
        
        report_path = os.path.join(self.output_dir, 'pipeline_report.md')
        
        # Статистика времени выполнения
        execution_time = self.pipeline_stats['total_execution_time']
        
        # Анализ результатов
        total_cameras = len(self.opencv_cameras)
        total_detections = sum(len(d) for d in self.marker_detections.values()) if self.marker_detections else 0
        unique_markers = len(set().union(*[d.keys() for d in self.marker_detections.values()])) if self.marker_detections else 0
        
        # Готовность для триангуляции
        marker_frequency = {}
        if self.marker_detections:
            for detections in self.marker_detections.values():
                for marker_id in detections.keys():
                    marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
        
        triangulatable_markers = sum(1 for freq in marker_frequency.values() if freq >= 3)
        
        # Содержание отчета
        report_content = f"""# Отчет пайплайна автокалибровки ArUco маркеров

## Общая информация

- **Дата выполнения**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Входная директория**: {self.data_dir}
- **Выходная директория**: {self.output_dir}
- **Время выполнения**: {execution_time:.2f} секунд
- **Словарь ArUco**: DICT_4X4_1000

## Результаты этапов

### ✅ Успешно завершенные этапы
{chr(10).join([f"- {stage}" for stage in self.pipeline_stats['stages_completed']])}

### ❌ Неудачные этапы
{chr(10).join([f"- {stage}" for stage in self.pipeline_stats['stages_failed']]) if self.pipeline_stats['stages_failed'] else "Все этапы выполнены успешно"}

## Статистика данных

### Камеры
- Загружено XMP файлов: {len(self.xmp_cameras)}
- Конвертировано камер: {total_cameras}
- Размер изображений: {CURRENT_IMAGE_SIZE[0]}×{CURRENT_IMAGE_SIZE[1]} пикселей

### ArUco маркеры
- Обработано изображений: {len(self.marker_detections) if self.marker_detections else 0}
- Всего детекций: {total_detections}
- Уникальных маркеров найдено: {unique_markers}
- **Готовых для триангуляции: {triangulatable_markers}**

## Готовность для следующих этапов

{"### ✅ ОТЛИЧНО! Готово для триангуляции" if triangulatable_markers >= 8 else "### ⚠️ ДОСТАТОЧНО для базовой триангуляции" if triangulatable_markers >= 3 else "### ❌ НЕДОСТАТОЧНО для триангуляции"}

{f"Найдено {triangulatable_markers} маркеров видимых на ≥3 камерах. Это достаточно для надежной 3D триангуляции." if triangulatable_markers >= 8 else f"Найдено {triangulatable_markers} маркеров видимых на ≥3 камерах. Можно выполнить базовую триангуляцию." if triangulatable_markers >= 3 else f"Найдено только {triangulatable_markers} маркеров видимых на ≥3 камерах. Недостаточно для надежной триангуляции."}

## Выходные файлы

### Промежуточные результаты
- `stage1_xmp_report.txt` - Подробный отчет по XMP файлам
- `stage2_opencv_cameras.json` - Параметры камер в OpenCV формате  
- `stage3_marker_detections.json` - Результаты детекции маркеров
- `stage4_triangulation_ready.json` - Данные готовые для триангуляции

### Дополнительные файлы
- `marked_images/` - Изображения с отмеченными маркерами (если создавались)
- `pipeline.log` - Подробный лог выполнения
- `pipeline_report.md` - Этот отчет

## Следующие шаги

{f"1. ✅ Данные готовы для 3D триангуляции" if triangulatable_markers >= 3 else "1. ❌ Улучшите качество детекции маркеров"}
{f"2. ✅ Можно переходить к реализации триангуляции" if triangulatable_markers >= 3 else "2. ❌ Добавьте больше маркеров или улучшите их качество"}
{f"3. ✅ После триангуляции - экспорт для Blender" if triangulatable_markers >= 3 else "3. ❌ Повторите детекцию с улучшенными параметрами"}

## Техническая информация

- **Размеры изображений**: {CURRENT_IMAGE_SIZE[0]}×{CURRENT_IMAGE_SIZE[1]} пикселей
- **Система координат**: RealityCapture absolute
- **Готово для триангуляции**: {"Да" if triangulatable_markers >= 3 else "Нет"}

---
*Отчет создан автоматически пайплайном автокалибровки ArUco маркеров*
"""
        
        # Сохранение отчета
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 Финальный отчет создан: {report_path}")
    
    def run_detection_pipeline(self, create_marked_images: bool = False) -> bool:
        """
        Запуск пайплайна детекции (этапы 1-4)
        
        Parameters:
        -----------
        create_marked_images : bool
            Создавать ли изображения с отмеченными маркерами
            
        Returns:
        --------
        bool
            True если пайплайн выполнен успешно
        """
        self.pipeline_stats['start_time'] = time.time()
        
        print("🚀 ЗАПУСК ПАЙПЛАЙНА ДЕТЕКЦИИ ARUCO МАРКЕРОВ")
        print("=" * 80)
        print(f"📂 Входные данные: {self.data_dir}")
        print(f"📂 Результаты: {self.output_dir}")
        print(f"🔍 Словарь ArUco: DICT_4X4_1000")
        print("=" * 80)
        
        try:
            # Валидация входных данных
            if not self.validate_input_data():
                return False
            
            # Этап 1: Загрузка камер
            if not self.stage_1_load_cameras():
                return False
            
            # Этап 2: Конвертация камер
            if not self.stage_2_convert_cameras():
                return False
            
            # Этап 3: Детекция маркеров
            if not self.stage_3_detect_markers(create_marked_images):
                return False
            
            # Этап 4: Подготовка данных для триангуляции
            if not self.stage_4_prepare_triangulation_data():
                return False
            
            # Финализация
            self.pipeline_stats['end_time'] = time.time()
            self.pipeline_stats['total_execution_time'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            )
            
            self.create_final_report()
            
            print(f"\n🎉 ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
            print(f"⏱️  Время выполнения: {self.pipeline_stats['total_execution_time']:.2f} секунд")
            print(f"📂 Все результаты сохранены в: {self.output_dir}")
            
            return True
            
        except Exception as e:
            self.pipeline_stats['end_time'] = time.time()
            self.pipeline_stats['total_execution_time'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            )
            
            print(f"💥 КРИТИЧЕСКАЯ ОШИБКА ПАЙПЛАЙНА: {e}")
            self.create_final_report()
            return False


def main():
    """Главная функция с CLI интерфейсом"""
    
    parser = argparse.ArgumentParser(
        description="Пайплайн автокалибровки ArUco маркеров (DICT_4X4_1000)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --data_dir data --output_dir results
  python main.py --data_dir data --output_dir results --create_images
  python main.py --data_dir data --output_dir results --verbose --create_images
        """
    )
    
    # Основные параметры
    parser.add_argument(
        '--data_dir', '--data',
        default='data',
        help='Директория с XMP файлами и изображениями (по умолчанию: data)'
    )
    
    parser.add_argument(
        '--output_dir', '--output',
        default='results',
        help='Директория для сохранения результатов (по умолчанию: results)'
    )
    
    # Опции выполнения
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробное логирование'
    )
    
    parser.add_argument(
        '--no_intermediate',
        action='store_true',
        help='Не сохранять промежуточные результаты'
    )
    
    parser.add_argument(
        '--create_images',
        action='store_true',
        help='Создать изображения с отмеченными маркерами'
    )
    
    args = parser.parse_args()
    
    # Проверка входной директории
    if not os.path.exists(args.data_dir):
        print(f"❌ Директория не найдена: {args.data_dir}")
        return 1
    
    # Создание и запуск пайплайна
    pipeline = AutoCalibrationPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        enable_logging=args.verbose,
        save_intermediate=not args.no_intermediate
    )
    
    # Запуск пайплайна детекции
    success = pipeline.run_detection_pipeline(create_marked_images=args.create_images)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())