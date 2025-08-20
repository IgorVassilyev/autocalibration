#!/usr/bin/env python3
"""
ArUco детектор на основе рабочего test.py с DICT_4X4_1000
=========================================================

Простой и надежный детектор ArUco маркеров, адаптированный под словарь DICT_4X4_1000.
Основан на проверенном коде из test.py с добавлением функций для интеграции.

Использование:
    from aruco_detector import SimpleArUcoDetector
    
    detector = SimpleArUcoDetector()
    results = detector.detect_markers_in_directory("data")
"""

import cv2
import os
import glob
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MarkerDetection:
    """Структура для хранения информации о детектированном маркере"""
    marker_id: int
    center: Tuple[float, float]
    corners: List[List[float]]  # 4 угла [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    area: float
    

class SimpleArUcoDetector:
    """
    Простой детектор ArUco маркеров
    Основан на рабочем коде из test.py, адаптирован под DICT_4X4_1000
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Инициализация детектора
        
        Parameters:
        -----------
        enable_logging : bool
            Включить вывод информации о процессе
        """
        self.enable_logging = enable_logging
        
        # Используем DICT_4X4_1000 как в test.py
        self.dictionary_type = cv2.aruco.DICT_4X4_1000
        
        # Инициализация детектора (точно как в test.py)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Статистика
        self.detection_stats = {
            'total_images': 0,
            'images_with_markers': 0,
            'total_markers_found': 0,
            'unique_marker_ids': set(),
            'failed_images': []
        }
        
        if self.enable_logging:
            print(f"🔧 ArUco детектор инициализирован с словарем: DICT_4X4_1000")
    
    def detect_markers_in_image(self, image_path: str) -> Dict[int, MarkerDetection]:
        """
        Детекция маркеров в одном изображении (основано на test.py)
        
        Parameters:
        -----------
        image_path : str
            Путь к изображению
            
        Returns:
        --------
        Dict[int, MarkerDetection]
            Словарь детектированных маркеров {marker_id: MarkerDetection}
        """
        try:
            # Загрузка изображения (как в test.py)
            img = cv2.imread(image_path)
            if img is None:
                if self.enable_logging:
                    print(f"[!] Не удалось прочитать {image_path}")
                self.detection_stats['failed_images'].append(image_path)
                return {}
            
            # Конвертация в серый (как в test.py)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Поиск маркеров (как в test.py)
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            corners, ids, _ = detector.detectMarkers(gray)
            
            # Обработка результатов
            detections = {}
            
            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids.flatten()):
                    # Извлечение углов маркера
                    marker_corners = corners[i].reshape(4, 2)
                    
                    # Вычисление центра
                    center_x = float(np.mean(marker_corners[:, 0]))
                    center_y = float(np.mean(marker_corners[:, 1]))
                    
                    # Вычисление площади
                    area = float(cv2.contourArea(marker_corners))
                    
                    # Создание объекта детекции
                    detection = MarkerDetection(
                        marker_id=int(marker_id),
                        center=(center_x, center_y),
                        corners=marker_corners.tolist(),
                        area=area
                    )
                    
                    detections[int(marker_id)] = detection
                
                # Обновление статистики
                self.detection_stats['total_markers_found'] += len(detections)
                self.detection_stats['unique_marker_ids'].update(detections.keys())
                self.detection_stats['images_with_markers'] += 1
                
                if self.enable_logging:
                    filename = os.path.basename(image_path)
                    marker_ids = list(detections.keys())
                    print(f"[OK] {filename}: найдено {len(marker_ids)} маркер(ов) {marker_ids}")
            else:
                if self.enable_logging:
                    filename = os.path.basename(image_path)
                    print(f"[..] {filename}: маркеры не найдены")
            
            self.detection_stats['total_images'] += 1
            return detections
            
        except Exception as e:
            if self.enable_logging:
                print(f"[!] Ошибка обработки {image_path}: {e}")
            self.detection_stats['failed_images'].append(image_path)
            return {}
    
    def detect_markers_in_directory(self, directory: str) -> Dict[str, Dict[int, MarkerDetection]]:
        """
        Детекция маркеров во всех изображениях директории
        
        Parameters:
        -----------
        directory : str
            Путь к директории с изображениями
            
        Returns:
        --------
        Dict[str, Dict[int, MarkerDetection]]
            Результаты {camera_id: {marker_id: MarkerDetection}}
        """
        if self.enable_logging:
            print(f"🔍 Поиск изображений в {directory}")
        
        # Поиск изображений (как в test.py)
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        
        for ext in image_extensions:
            pattern = os.path.join(directory, ext)
            images.extend(glob.glob(pattern))
            # Также в верхнем регистре
            pattern_upper = os.path.join(directory, ext.upper())
            images.extend(glob.glob(pattern_upper))
        
        images = sorted(list(set(images)))  # убрать дубли и отсортировать
        
        if not images:
            if self.enable_logging:
                print(f"❌ Изображения не найдены в {directory}")
            return {}
        
        if self.enable_logging:
            print(f"Найдено {len(images)} изображений")
        
        # Обработка каждого изображения
        all_detections = {}
        
        for image_path in images:
            # Получаем camera_id из имени файла (без расширения)
            filename = os.path.basename(image_path)
            camera_id = os.path.splitext(filename)[0]
            
            # Детекция маркеров
            detections = self.detect_markers_in_image(image_path)
            all_detections[camera_id] = detections
        
        # Финальная статистика
        if self.enable_logging:
            self._print_detection_summary(all_detections)
        
        return all_detections
    
    def _print_detection_summary(self, all_detections: Dict[str, Dict[int, MarkerDetection]]) -> None:
        """Печать сводки результатов детекции"""
        
        print(f"\n{'='*60}")
        print("📊 СВОДКА ДЕТЕКЦИИ МАРКЕРОВ")
        print(f"{'='*60}")
        
        # Общая статистика
        total_cameras = len(all_detections)
        cameras_with_markers = len([d for d in all_detections.values() if d])
        total_detections = sum(len(detections) for detections in all_detections.values())
        unique_markers = len(self.detection_stats['unique_marker_ids'])
        
        print(f"🎥 Обработано камер: {total_cameras}")
        print(f"✅ Камер с маркерами: {cameras_with_markers}")
        print(f"🏷️  Всего детекций: {total_detections}")
        print(f"🔢 Уникальных маркеров: {unique_markers}")
        
        if total_cameras > 0:
            success_rate = (cameras_with_markers / total_cameras) * 100
            avg_markers = total_detections / cameras_with_markers if cameras_with_markers > 0 else 0
            print(f"📈 Успешность: {success_rate:.1f}%")
            print(f"📊 Среднее маркеров на камеру: {avg_markers:.1f}")
        
        # Список всех найденных маркеров
        if unique_markers > 0:
            sorted_markers = sorted(self.detection_stats['unique_marker_ids'])
            print(f"🆔 ID маркеров: {sorted_markers}")
        
        # Частота обнаружения каждого маркера
        marker_frequency = {}
        for detections in all_detections.values():
            for marker_id in detections.keys():
                marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
        
        if marker_frequency:
            print(f"\n📊 ЧАСТОТА ОБНАРУЖЕНИЯ:")
            for marker_id in sorted(marker_frequency.keys()):
                frequency = marker_frequency[marker_id]
                percentage = (frequency / total_cameras) * 100
                triangulatable = "✅" if frequency >= 3 else "⚠️" if frequency >= 2 else "❌"
                print(f"   Маркер {marker_id:2d}: {frequency:2d}/{total_cameras} камер ({percentage:5.1f}%) {triangulatable}")
        
        # Камеры без маркеров
        failed_cameras = [cam_id for cam_id, detections in all_detections.items() if not detections]
        if failed_cameras:
            print(f"\n⚠️  Камеры без маркеров: {failed_cameras}")
        
        # Оценка готовности для триангуляции
        triangulatable_markers = sum(1 for freq in marker_frequency.values() if freq >= 3)
        print(f"\n🎯 ГОТОВНОСТЬ ДЛЯ 3D ТРИАНГУЛЯЦИИ:")
        print(f"   Маркеров видимых на ≥3 камерах: {triangulatable_markers}")
        
        if triangulatable_markers >= 8:
            print("   ✅ Отлично! Достаточно для надежной триангуляции")
        elif triangulatable_markers >= 5:
            print("   ⚠️  Хорошо. Достаточно для базовой триангуляции")
        elif triangulatable_markers >= 3:
            print("   ⚠️  Минимально. Результат может быть неточным")
        else:
            print("   ❌ Недостаточно для триангуляции")
    
    def save_results_to_json(self, detections: Dict[str, Dict[int, MarkerDetection]], 
                           output_path: str) -> None:
        """
        Сохранение результатов в JSON файл
        
        Parameters:
        -----------
        detections : dict
            Результаты детекции
        output_path : str
            Путь для сохранения JSON файла
        """
        # Подготовка данных для JSON
        json_data = {
            'metadata': {
                'detector_version': 'simple_dict_4x4_1000',
                'dictionary': 'DICT_4X4_1000',
                'total_cameras': len(detections),
                'cameras_with_markers': len([d for d in detections.values() if d]),
                'unique_markers': len(self.detection_stats['unique_marker_ids']),
                'total_detections': sum(len(d) for d in detections.values())
            },
            'cameras': {}
        }
        
        for camera_id, camera_detections in detections.items():
            json_data['cameras'][camera_id] = {}
            
            for marker_id, detection in camera_detections.items():
                json_data['cameras'][camera_id][str(marker_id)] = {
                    'center': detection.center,
                    'corners': detection.corners,
                    'area': detection.area
                }
        
        # Сохранение
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        if self.enable_logging:
            print(f"💾 Результаты сохранены в {output_path}")
    
    def get_detection_statistics(self) -> Dict:
        """Получение статистики детекции"""
        stats = self.detection_stats.copy()
        stats['unique_marker_ids'] = sorted(list(stats['unique_marker_ids']))
        return stats
    
    def create_output_images(self, directory: str, output_dir: str) -> None:
        """
        Создание изображений с отмеченными маркерами (как в test.py)
        
        Parameters:
        -----------
        directory : str
            Директория с исходными изображениями
        output_dir : str
            Директория для сохранения результатов
        """
        # Создание выходной директории
        os.makedirs(output_dir, exist_ok=True)
        
        if self.enable_logging:
            print(f"🎨 Создание изображений с отмеченными маркерами...")
        
        # Поиск изображений
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        
        for ext in image_extensions:
            pattern = os.path.join(directory, ext)
            images.extend(glob.glob(pattern))
        
        images = sorted(list(set(images)))
        
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                if self.enable_logging:
                    print(f"[!] Не удалось прочитать {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Поиск маркеров (как в test.py)
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            corners, ids, _ = detector.detectMarkers(gray)

            # Отрисовка (как в test.py)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img, corners, ids)

            # Сохраняем результат (как в test.py)
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, img)
        
        if self.enable_logging:
            print(f"✅ Изображения с маркерами сохранены в {output_dir}")


# Удобные функции для совместимости

def detect_markers_simple(image_path: str) -> Dict[int, Tuple[float, float]]:
    """
    Простая функция детекции одного изображения
    
    Parameters:
    -----------
    image_path : str
        Путь к изображению
        
    Returns:
    --------
    Dict[int, Tuple[float, float]]
        Словарь {marker_id: (center_x, center_y)}
    """
    detector = SimpleArUcoDetector(enable_logging=False)
    detections = detector.detect_markers_in_image(image_path)
    
    # Преобразование к простому формату
    return {
        marker_id: detection.center 
        for marker_id, detection in detections.items()
    }


def detect_all_markers_in_directory(directory: str = "data", 
                                   output_file: str = "detection_results.json",
                                   create_images: bool = False,
                                   images_output_dir: str = "output") -> Dict:
    """
    Основная функция для детекции маркеров в директории
    
    Parameters:
    -----------
    directory : str
        Директория с изображениями
    output_file : str
        Файл для сохранения результатов
    create_images : bool
        Создавать ли изображения с отмеченными маркерами
    images_output_dir : str
        Директория для сохранения изображений с маркерами
        
    Returns:
    --------
    dict
        Результаты детекции
    """
    print("🚀 ДЕТЕКЦИЯ ARUCO МАРКЕРОВ (DICT_4X4_1000)")
    print("=" * 50)
    print(f"📂 Директория: {directory}")
    print(f"💾 Результаты будут сохранены в: {output_file}")
    if create_images:
        print(f"🎨 Изображения с маркерами в: {images_output_dir}")
    print("=" * 50)
    
    # Создание детектора
    detector = SimpleArUcoDetector(enable_logging=True)
    
    # Детекция
    detections = detector.detect_markers_in_directory(directory)
    
    # Сохранение результатов
    if detections:
        detector.save_results_to_json(detections, output_file)
        
        # Создание изображений с маркерами если запрошено
        if create_images:
            detector.create_output_images(directory, images_output_dir)
    
    # Возврат результатов
    return detections


def main():
    """Функция для прямого запуска модуля"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Детекция ArUco маркеров DICT_4X4_1000 (на основе test.py)"
    )
    
    parser.add_argument(
        '--input', '-i',
        default='data',
        help='Директория с изображениями (по умолчанию: data)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='detection_results.json',
        help='Файл для сохранения результатов (по умолчанию: detection_results.json)'
    )
    
    parser.add_argument(
        '--create_images',
        action='store_true',
        help='Создать изображения с отмеченными маркерами'
    )
    
    parser.add_argument(
        '--images_output',
        default='output',
        help='Директория для изображений с маркерами (по умолчанию: output)'
    )
    
    args = parser.parse_args()
    
    # Проверка входной директории
    if not os.path.exists(args.input):
        print(f"❌ Директория не найдена: {args.input}")
        return 1
    
    # Запуск детекции
    detections = detect_all_markers_in_directory(
        directory=args.input,
        output_file=args.output,
        create_images=args.create_images,
        images_output_dir=args.images_output
    )
    
    if detections:
        print(f"\n✅ Детекция завершена успешно!")
        
        # Проверка готовности для триангуляции
        marker_frequency = {}
        for camera_detections in detections.values():
            for marker_id in camera_detections.keys():
                marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
        
        triangulatable_markers = sum(1 for freq in marker_frequency.values() if freq >= 3)
        
        print(f"\n🎯 Готовность для 3D триангуляции:")
        print(f"   Маркеров видимых на ≥3 камерах: {triangulatable_markers}")
        
        if triangulatable_markers >= 5:
            print("   ✅ Готово для следующего этапа триангуляции!")
        elif triangulatable_markers >= 3:
            print("   ⚠️  Минимально достаточно для триангуляции")
        else:
            print("   ❌ Недостаточно для надежной триангуляции")
            
        return 0
    else:
        print(f"\n❌ Маркеры не найдены")
        return 1


if __name__ == "__main__":
    exit(main())
