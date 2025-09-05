#!/usr/bin/env python3
"""
ArUco детектор с жесткой фильтрацией маркеров ID > 13
=========================================================

Детектор ArUco маркеров 4x4, который работает ТОЛЬКО с маркерами ID от 1 до 13.
Все маркеры с ID > 13 полностью игнорируются как ложные срабатывания.

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
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


# ЖЕСТКОЕ ОГРАНИЧЕНИЕ - ТОЛЬКО МАРКЕРЫ 1-13
MAX_VALID_MARKER_ID = 13


@dataclass
class MarkerDetection:
    """Структура для хранения информации о детектированном маркере"""
    marker_id: int
    center: Tuple[float, float]
    corners: List[List[float]]  # 4 угла [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    area: float
    

class SimpleArUcoDetector:
    """
    Детектор ArUco маркеров 4x4 с жесткой фильтрацией ID > 13
    """
    
    def __init__(self, enable_logging: bool = True, 
                 filter_6x6: bool = True,
                 min_marker_perimeter_rate: float = 0.03,
                 max_marker_perimeter_rate: float = 4.0):
        """
        Инициализация детектора
        
        Parameters:
        -----------
        enable_logging : bool
            Включить вывод информации о процессе
        filter_6x6 : bool
            Фильтровать 6x6 маркеры (по умолчанию True)
        min_marker_perimeter_rate : float
            Минимальный периметр маркера относительно размера изображения
        max_marker_perimeter_rate : float
            Максимальный периметр маркера относительно размера изображения
        """
        self.enable_logging = enable_logging
        self.filter_6x6 = filter_6x6
        
        # Используем DICT_4X4_1000 для целевых маркеров
        self.dictionary_4x4 = cv2.aruco.DICT_4X4_1000
        self.aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(self.dictionary_4x4)
        
        # Также загружаем словарь 6x6 для фильтрации
        if self.filter_6x6:
            self.dictionary_6x6 = cv2.aruco.DICT_6X6_250
            self.aruco_dict_6x6 = cv2.aruco.getPredefinedDictionary(self.dictionary_6x6)
        
        # Настройка параметров детектора для более строгой детекции
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Более строгие параметры для уменьшения ложных срабатываний
        self.parameters.minMarkerPerimeterRate = min_marker_perimeter_rate
        self.parameters.maxMarkerPerimeterRate = max_marker_perimeter_rate
        self.parameters.polygonalApproxAccuracyRate = 0.03
        self.parameters.minCornerDistanceRate = 0.05
        self.parameters.minDistanceToBorder = 3
        
        # Параметры для улучшения качества детекции
        self.parameters.minOtsuStdDev = 5.0
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        
        # Параметры для фильтрации шума
        self.parameters.perspectiveRemovePixelPerCell = 4
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
        
        # Параметры проверки битов маркера
        self.parameters.maxErroneousBitsInBorderRate = 0.35
        self.parameters.errorCorrectionRate = 0.6
        
        # Параметры для corner refinement
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 30
        self.parameters.cornerRefinementMinAccuracy = 0.1
        
        # Статистика
        self.detection_stats = {
            'total_images': 0,
            'images_with_markers': 0,
            'total_markers_found': 0,
            'unique_marker_ids': set(),
            'failed_images': [],
            'filtered_6x6_count': 0,
            'filtered_6x6_ids': set()
        }
        
        if self.enable_logging:
            print(f"ArUco детектор инициализирован")
            print(f"   Целевой словарь: DICT_4X4_1000")
            print(f"   ТОЛЬКО маркеры с ID 1-{MAX_VALID_MARKER_ID}")
            if self.filter_6x6:
                print(f"   Фильтрация 6x6: ВКЛЮЧЕНА (DICT_6X6_250)")
            print(f"   Строгие параметры детекции: ВКЛЮЧЕНЫ")
    
    def _detect_6x6_markers(self, gray_image: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Детекция 6x6 маркеров для последующей фильтрации
        
        Parameters:
        -----------
        gray_image : np.ndarray
            Изображение в градациях серого
            
        Returns:
        --------
        Set[Tuple[int, int]]
            Множество центров 6x6 маркеров для исключения
        """
        if not self.filter_6x6:
            return set()
        
        try:
            # Создаем детектор для 6x6
            detector_6x6 = cv2.aruco.ArucoDetector(self.aruco_dict_6x6, self.parameters)
            corners_6x6, ids_6x6, _ = detector_6x6.detectMarkers(gray_image)
            
            excluded_regions = set()
            
            if ids_6x6 is not None and len(ids_6x6) > 0:
                for i, marker_id in enumerate(ids_6x6.flatten()):
                    # Получаем центр 6x6 маркера
                    marker_corners = corners_6x6[i].reshape(4, 2)
                    center_x = int(np.mean(marker_corners[:, 0]))
                    center_y = int(np.mean(marker_corners[:, 1]))
                    
                    # Вычисляем размер маркера
                    width = np.max(marker_corners[:, 0]) - np.min(marker_corners[:, 0])
                    height = np.max(marker_corners[:, 1]) - np.min(marker_corners[:, 1])
                    
                    # Добавляем регион для исключения (с запасом)
                    margin = max(width, height) * 0.2  # 20% запас
                    excluded_regions.add((center_x, center_y, margin))
                    
                    self.detection_stats['filtered_6x6_count'] += 1
                    self.detection_stats['filtered_6x6_ids'].add(int(marker_id))
                    
                    if self.enable_logging:
                        print(f"   [6x6] Обнаружен 6x6 маркер ID={marker_id} в ({center_x}, {center_y})")
            
            return excluded_regions
            
        except Exception as e:
            if self.enable_logging:
                print(f"   [!] Ошибка при детекции 6x6: {e}")
            return set()
    
    def _is_in_excluded_region(self, center: Tuple[float, float], 
                              excluded_regions: Set[Tuple[int, int, float]]) -> bool:
        """
        Проверка, находится ли маркер в исключенной области (где есть 6x6)
        
        Parameters:
        -----------
        center : Tuple[float, float]
            Центр проверяемого маркера
        excluded_regions : Set[Tuple[int, int, float]]
            Множество исключенных областей (x, y, radius)
            
        Returns:
        --------
        bool
            True если маркер в исключенной области
        """
        for ex_x, ex_y, margin in excluded_regions:
            distance = np.sqrt((center[0] - ex_x)**2 + (center[1] - ex_y)**2)
            if distance < margin:
                return True
        return False
    
    def _validate_4x4_marker(self, corners: np.ndarray, marker_id: int) -> bool:
        """
        Дополнительная валидация 4x4 маркера
        
        Parameters:
        -----------
        corners : np.ndarray
            Углы маркера
        marker_id : int
            ID маркера
            
        Returns:
        --------
        bool
            True если маркер прошел валидацию
        """
        # Проверка соотношения сторон (должен быть близок к квадрату)
        corners_2d = corners.reshape(4, 2)
        
        # Вычисляем длины сторон
        side1 = np.linalg.norm(corners_2d[1] - corners_2d[0])
        side2 = np.linalg.norm(corners_2d[2] - corners_2d[1])
        side3 = np.linalg.norm(corners_2d[3] - corners_2d[2])
        side4 = np.linalg.norm(corners_2d[0] - corners_2d[3])
        
        # Средняя длина стороны
        avg_side = (side1 + side2 + side3 + side4) / 4
        
        # Проверка что все стороны примерно равны (допуск 30%)
        for side in [side1, side2, side3, side4]:
            if abs(side - avg_side) / avg_side > 0.3:
                return False
        
        # Проверка площади (не слишком маленький)
        area = cv2.contourArea(corners_2d)
        if area < 100:  # Минимальная площадь в пикселях
            return False
        
        # Проверка выпуклости
        if not cv2.isContourConvex(corners_2d.astype(np.float32)):
            return False
        
        return True
    
    def detect_markers_in_image(self, image_path: str) -> Dict[int, MarkerDetection]:
        """
        Детекция 4x4 маркеров с ID от 1 до 13
        
        Parameters:
        -----------
        image_path : str
            Путь к изображению
            
        Returns:
        --------
        Dict[int, MarkerDetection]
            Словарь детектированных 4x4 маркеров {marker_id: MarkerDetection}
        """
        try:
            # Загрузка изображения
            img = cv2.imread(image_path)
            if img is None:
                if self.enable_logging:
                    print(f"[!] Не удалось прочитать {image_path}")
                self.detection_stats['failed_images'].append(image_path)
                return {}
            
            # Конвертация в серый
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Сначала находим 6x6 маркеры для исключения
            excluded_regions = self._detect_6x6_markers(gray)
            
            # Теперь ищем 4x4 маркеры
            detector_4x4 = cv2.aruco.ArucoDetector(self.aruco_dict_4x4, self.parameters)
            corners_4x4, ids_4x4, _ = detector_4x4.detectMarkers(gray)
            
            # КРИТИЧЕСКАЯ ФИЛЬТРАЦИЯ: сразу отбрасываем все ID > 13
            if ids_4x4 is not None and len(ids_4x4) > 0:
                valid_indices = []
                for i, marker_id in enumerate(ids_4x4.flatten()):
                    if marker_id <= MAX_VALID_MARKER_ID:
                        valid_indices.append(i)
                
                # Оставляем только валидные маркеры
                if valid_indices:
                    ids_4x4 = np.array([ids_4x4[i] for i in valid_indices])
                    corners_4x4 = [corners_4x4[i] for i in valid_indices]
                else:
                    ids_4x4 = None
                    corners_4x4 = []
            
            # Обработка только валидных маркеров
            detections = {}
            
            if ids_4x4 is not None and len(ids_4x4) > 0:
                for i, marker_id in enumerate(ids_4x4.flatten()):
                    marker_id_int = int(marker_id)
                    
                    # Извлечение углов маркера
                    marker_corners = corners_4x4[i].reshape(4, 2)
                    
                    # Вычисление центра
                    center_x = float(np.mean(marker_corners[:, 0]))
                    center_y = float(np.mean(marker_corners[:, 1]))
                    center = (center_x, center_y)
                    
                    # Проверка, не находится ли маркер в области 6x6
                    if self._is_in_excluded_region(center, excluded_regions):
                        continue
                    
                    # Дополнительная валидация 4x4 маркера
                    if not self._validate_4x4_marker(corners_4x4[i], marker_id_int):
                        continue
                    
                    # Вычисление площади
                    area = float(cv2.contourArea(marker_corners))
                    
                    # Создание объекта детекции
                    detection = MarkerDetection(
                        marker_id=marker_id_int,
                        center=center,
                        corners=marker_corners.tolist(),
                        area=area
                    )
                    
                    detections[marker_id_int] = detection
                
                # Обновление статистики
                if detections:
                    self.detection_stats['total_markers_found'] += len(detections)
                    self.detection_stats['unique_marker_ids'].update(detections.keys())
                    self.detection_stats['images_with_markers'] += 1
                
                if self.enable_logging:
                    filename = os.path.basename(image_path)
                    if detections:
                        marker_ids = list(detections.keys())
                        print(f"[OK] {filename}: найдено {len(marker_ids)} маркер(ов) 4x4: {marker_ids}")
            else:
                if self.enable_logging:
                    filename = os.path.basename(image_path)
                    print(f"[..] {filename}: валидные маркеры 4x4 не найдены")
            
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
            print(f"Поиск изображений в {directory}")
        
        # Поиск изображений
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
                print(f"Изображения не найдены в {directory}")
            return {}
        
        if self.enable_logging:
            print(f"Найдено {len(images)} изображений")
            print(f"Детекция ТОЛЬКО маркеров с ID 1-{MAX_VALID_MARKER_ID}...")
            print("-" * 50)
        
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
        print("СВОДКА ДЕТЕКЦИИ МАРКЕРОВ")
        print(f"{'='*60}")
        
        # Общая статистика
        total_cameras = len(all_detections)
        cameras_with_markers = len([d for d in all_detections.values() if d])
        total_detections = sum(len(detections) for detections in all_detections.values())
        unique_markers = len(self.detection_stats['unique_marker_ids'])
        
        print(f"Обработано камер: {total_cameras}")
        print(f"Камер с маркерами 4x4: {cameras_with_markers}")
        print(f"Всего детекций 4x4: {total_detections}")
        print(f"Уникальных маркеров 4x4: {unique_markers}")
        print(f"Допустимый диапазон ID: 1-{MAX_VALID_MARKER_ID}")
        
        # Статистика фильтрации
        if self.filter_6x6 and self.detection_stats['filtered_6x6_count'] > 0:
            print(f"\nФИЛЬТРАЦИЯ 6x6:")
            print(f"   Обнаружено и отфильтровано 6x6 маркеров: {self.detection_stats['filtered_6x6_count']}")
        
        if total_cameras > 0:
            success_rate = (cameras_with_markers / total_cameras) * 100
            avg_markers = total_detections / cameras_with_markers if cameras_with_markers > 0 else 0
            print(f"\nУспешность детекции 4x4: {success_rate:.1f}%")
            print(f"Среднее маркеров 4x4 на камеру: {avg_markers:.1f}")
        
        # Список всех найденных 4x4 маркеров
        if unique_markers > 0:
            sorted_markers = sorted(self.detection_stats['unique_marker_ids'])
            print(f"\nID найденных маркеров: {sorted_markers}")
        
        # Частота обнаружения каждого маркера
        marker_frequency = {}
        for detections in all_detections.values():
            for marker_id in detections.keys():
                marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
        
        if marker_frequency:
            print(f"\nЧАСТОТА ОБНАРУЖЕНИЯ 4x4:")
            for marker_id in sorted(marker_frequency.keys()):
                frequency = marker_frequency[marker_id]
                percentage = (frequency / total_cameras) * 100
                triangulatable = "OK" if frequency >= 3 else "WARN" if frequency >= 2 else "NO"
                print(f"   Маркер {marker_id:2d}: {frequency:2d}/{total_cameras} камер ({percentage:5.1f}%) {triangulatable}")
        
        # Камеры без маркеров
        failed_cameras = [cam_id for cam_id, detections in all_detections.items() if not detections]
        if failed_cameras:
            print(f"\nКамеры без маркеров 4x4: {failed_cameras}")
        
        # Оценка готовности для триангуляции
        triangulatable_markers = sum(1 for freq in marker_frequency.values() if freq >= 3)
        print(f"\nГОТОВНОСТЬ ДЛЯ 3D ТРИАНГУЛЯЦИИ:")
        print(f"   Маркеров видимых на ≥3 камерах: {triangulatable_markers}")
        
        if triangulatable_markers >= 8:
            print("   Отлично! Достаточно для надежной триангуляции")
        elif triangulatable_markers >= 5:
            print("   Хорошо. Достаточно для базовой триангуляции")
        elif triangulatable_markers >= 3:
            print("   Минимально. Результат может быть неточным")
        else:
            print("   Недостаточно для триангуляции")
    
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
                'detector_version': 'strict_4x4_only_1_to_13',
                'dictionary': 'DICT_4X4_1000',
                'valid_id_range': f'1-{MAX_VALID_MARKER_ID}',
                'filter_6x6': self.filter_6x6,
                'total_cameras': len(detections),
                'cameras_with_markers': len([d for d in detections.values() if d]),
                'unique_markers_4x4': len(self.detection_stats['unique_marker_ids']),
                'total_detections_4x4': sum(len(d) for d in detections.values())
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
            print(f"Результаты сохранены в {output_path}")
    
    def get_detection_statistics(self) -> Dict:
        """Получение статистики детекции"""
        stats = self.detection_stats.copy()
        stats['unique_marker_ids'] = sorted(list(stats['unique_marker_ids']))
        stats['filtered_6x6_ids'] = sorted(list(stats['filtered_6x6_ids']))
        return stats
    
    def create_output_images(self, directory: str, output_dir: str) -> None:
        """
        Создание изображений с отмеченными маркерами (ТОЛЬКО ID 1-13)
        
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
            print(f" Создание изображений с отмеченными маркерами...")
        
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

            # Детекция 6x6 маркеров (для визуализации)
            if self.filter_6x6:
                detector_6x6 = cv2.aruco.ArucoDetector(self.aruco_dict_6x6, self.parameters)
                corners_6x6, ids_6x6, _ = detector_6x6.detectMarkers(gray)
                
                # Отрисовка 6x6 красным цветом
                if ids_6x6 is not None:
                    for i in range(len(ids_6x6)):
                        cv2.drawContours(img, [corners_6x6[i].astype(int)], -1, (0, 0, 255), 2)
                        # Подпись ID
                        center = np.mean(corners_6x6[i].reshape(4, 2), axis=0).astype(int)
                        cv2.putText(img, f"6x6:{ids_6x6[i][0]}", 
                                  tuple(center - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Детекция 4x4 маркеров
            detector_4x4 = cv2.aruco.ArucoDetector(self.aruco_dict_4x4, self.parameters)
            corners_4x4, ids_4x4, _ = detector_4x4.detectMarkers(gray)

            # ФИЛЬТРУЕМ И РИСУЕМ ТОЛЬКО МАРКЕРЫ С ID 1-13
            if ids_4x4 is not None:
                valid_corners = []
                valid_ids = []
                
                for i, marker_id in enumerate(ids_4x4.flatten()):
                    if marker_id <= MAX_VALID_MARKER_ID:
                        valid_corners.append(corners_4x4[i])
                        valid_ids.append([marker_id])
                
                # Отрисовка только валидных 4x4 зеленым цветом
                if valid_corners:
                    cv2.aruco.drawDetectedMarkers(img, valid_corners, 
                                                 np.array(valid_ids), 
                                                 borderColor=(0, 255, 0))

            # Сохраняем результат
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, img)
        
        if self.enable_logging:
            print(f" Изображения с маркерами сохранены в {output_dir}")
            print(f"    Зеленые рамки - маркеры 4x4 с ID 1-{MAX_VALID_MARKER_ID}")
            print(f"    Красные рамки - маркеры 6x6 (отфильтрованные)")


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
    print("ДЕТЕКЦИЯ ARUCO МАРКЕРОВ 4x4 (DICT_4X4_1000)")
    print(f"    СТРОГО ID 1-{MAX_VALID_MARKER_ID}")
    print(f"   с автоматической фильтрацией 6x6 маркеров")
    print("=" * 50)
    print(f" Директория: {directory}")
    print(f" Результаты будут сохранены в: {output_file}")
    if create_images:
        print(f" Изображения с маркерами в: {images_output_dir}")
    print("=" * 50)
    
    # Создание детектора с жесткой фильтрацией
    detector = SimpleArUcoDetector(enable_logging=True, filter_6x6=True)
    
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
        description=f"Детекция ArUco маркеров 4x4 (ТОЛЬКО ID 1-{MAX_VALID_MARKER_ID})"
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
    
    parser.add_argument(
        '--no_filter_6x6',
        action='store_true',
        help='Отключить фильтрацию 6x6 маркеров'
    )
    
    args = parser.parse_args()
    
    # Проверка входной директории
    if not os.path.exists(args.input):
        print(f"Директория не найдена: {args.input}")
        return 1
    
    print("ДЕТЕКЦИЯ ARUCO МАРКЕРОВ 4x4")
    print(f"    ТОЛЬКО ID от 1 до {MAX_VALID_MARKER_ID}")
    print("=" * 50)
    
    # Создание детектора
    detector = SimpleArUcoDetector(
        enable_logging=True, 
        filter_6x6=not args.no_filter_6x6
    )
    
    # Запуск детекции
    detections = detector.detect_markers_in_directory(args.input)
    
    if detections:
        # Сохранение результатов
        detector.save_results_to_json(detections, args.output)
        
        # Создание изображений если запрошено
        if args.create_images:
            detector.create_output_images(args.input, args.images_output)
        
        print(f"\nДетекция завершена успешно!")
        
        # Проверка готовности для триангуляции
        marker_frequency = {}
        for camera_detections in detections.values():
            for marker_id in camera_detections.keys():
                marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
        
        triangulatable_markers = sum(1 for freq in marker_frequency.values() if freq >= 3)
        
        print(f"\nГотовность для 3D триангуляции:")
        print(f"   Маркеров видимых на ≥3 камерах: {triangulatable_markers}")
        
        if triangulatable_markers >= 5:
            print(" Готово для следующего этапа триангуляции!")
        elif triangulatable_markers >= 3:
            print("  Минимально достаточно для триангуляции")
        else:
            print(" Недостаточно для надежной триангуляции")
            
        return 0
    else:
        print(f"\nМаркеры не найдены")
        return 1


if __name__ == "__main__":
    exit(main())
    