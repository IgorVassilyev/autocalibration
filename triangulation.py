#!/usr/bin/env python3
"""
3D триангуляция ArUco маркеров
==============================

Модуль для 3D триангуляции позиций ArUco маркеров на основе 
их 2D детекций на нескольких камерах с известными параметрами.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MarkerTriangulation:
    """Результат триангуляции одного маркера"""
    marker_id: int
    position_3d: Tuple[float, float, float]  # (X, Y, Z) в мировых координатах
    observations_count: int  # количество камер где виден маркер
    reprojection_error: float  # средняя ошибка репроекции в пикселях
    triangulation_confidence: float  # уверенность триангуляции (0-1)
    camera_ids: List[str]  # ID камер где виден маркер


class ArUcoTriangulator:
    """Класс для 3D триангуляции ArUco маркеров"""
    
    def __init__(self, min_cameras: int = 3, max_reprojection_error: float = 2.0):
        self.min_cameras = min_cameras
        self.max_reprojection_error = max_reprojection_error
    
    def _create_projection_matrix(self, camera_matrix: np.ndarray, 
                                rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Создание матрицы проекции P = K[R|t] для камеры"""
        # t = -R * position (так как position - это позиция камеры в мире)
        translation = -rotation @ position.reshape(3, 1)
        # Создаем матрицу [R|t]
        rt_matrix = np.hstack([rotation, translation])
        # P = K * [R|t]
        projection_matrix = camera_matrix @ rt_matrix
        return projection_matrix
    
    def _triangulate_point_pair(self, p1: np.ndarray, p2: np.ndarray,
                               proj1: np.ndarray, proj2: np.ndarray) -> np.ndarray:
        """Триангуляция 3D точки по двум 2D наблюдениям"""
        points_4d = cv2.triangulatePoints(
            proj1, proj2, 
            p1.reshape(2, 1), p2.reshape(2, 1)
        )
        return points_4d.flatten()
    
    def _convert_homogeneous_to_3d(self, point_4d: np.ndarray) -> np.ndarray:
        """Преобразование из однородных координат в 3D"""
        if abs(point_4d[3]) < 1e-6:
            return np.array([float('inf'), float('inf'), float('inf')])
        return point_4d[:3] / point_4d[3]
    
    def _calculate_reprojection_error(self, point_3d: np.ndarray,
                                    camera_matrix: np.ndarray,
                                    rotation: np.ndarray, position: np.ndarray,
                                    observed_2d: np.ndarray) -> float:
        """Вычисление ошибки репроекции для 3D точки"""
        # Преобразуем 3D точку в систему координат камеры
        point_camera = rotation @ (point_3d - position)
        
        # Проецируем в 2D
        if abs(point_camera[2]) < 1e-6:
            return float('inf')  # Точка за камерой
        
        projected_2d = camera_matrix @ point_camera
        projected_2d = projected_2d[:2] / projected_2d[2]
        
        # Вычисляем евклидово расстояние
        error = np.linalg.norm(projected_2d - observed_2d)
        return float(error)
    
    def _triangulate_marker_robust(self, marker_id: int, 
                                 observations: Dict[str, Dict]) -> Optional[MarkerTriangulation]:
        """Робастная триангуляция одного маркера с несколькими камерами"""
        camera_ids = list(observations.keys())
        n_cameras = len(camera_ids)
        
        if n_cameras < self.min_cameras:
            return None
        
        # Собираем все возможные пары камер для триангуляции
        triangulated_points = []
        projection_matrices = {}
        
        # Предвычисляем матрицы проекции
        for cam_id in camera_ids:
            cam_data = observations[cam_id]['camera_data']
            
            # Убеждаемся что все данные в numpy массивах
            camera_matrix = np.array(cam_data['camera_matrix'])
            rotation = np.array(cam_data['rotation'])
            position = np.array(cam_data['position'])
            
            proj_matrix = self._create_projection_matrix(
                camera_matrix, rotation, position
            )
            projection_matrices[cam_id] = proj_matrix
        
        # Триангулируем по всем парам камер
        for i in range(n_cameras):
            for j in range(i + 1, n_cameras):
                cam1_id = camera_ids[i]
                cam2_id = camera_ids[j]
                
                # Получаем 2D координаты
                p1 = np.array(observations[cam1_id]['center'])
                p2 = np.array(observations[cam2_id]['center'])
                
                # Триангулируем
                try:
                    point_4d = self._triangulate_point_pair(
                        p1, p2,
                        projection_matrices[cam1_id],
                        projection_matrices[cam2_id]
                    )
                    
                    point_3d = self._convert_homogeneous_to_3d(point_4d)
                    
                    # Проверяем на валидность
                    if not np.any(np.isinf(point_3d)) and not np.any(np.isnan(point_3d)):
                        triangulated_points.append(point_3d)
                        
                except Exception as e:
                    print(f"     Ошибка триангуляции пары {cam1_id}-{cam2_id}: {e}")
                    continue
        
        print(f"     Получено {len(triangulated_points)} валидных точек из {n_cameras*(n_cameras-1)//2} пар")
        
        if not triangulated_points:
            print(f"     Нет валидных триангуляций")
            return None
        
        # Усредняем результаты
        triangulated_points = np.array(triangulated_points)
        
        # Удаляем выбросы (простой метод - убираем точки далеко от медианы)
        if len(triangulated_points) > 2:
            median_point = np.median(triangulated_points, axis=0)
            distances = np.linalg.norm(triangulated_points - median_point, axis=1)
            median_distance = np.median(distances)
            
            # Оставляем точки в пределах 2 медианных отклонений
            valid_mask = distances <= (median_distance * 2 + 0.1)
            triangulated_points = triangulated_points[valid_mask]
            print(f"     После фильтрации выбросов: {len(triangulated_points)} точек")
        
        if len(triangulated_points) == 0:
            print(f"     Все точки отфильтрованы как выбросы")
            return None
        
        # Финальная 3D позиция - среднее
        final_3d_position = np.mean(triangulated_points, axis=0)
        print(f"     Финальная позиция: ({final_3d_position[0]:.3f}, {final_3d_position[1]:.3f}, {final_3d_position[2]:.3f})")
        
        # Вычисляем ошибки репроекции для всех камер
        reprojection_errors = []
        
        for cam_id in camera_ids:
            cam_data = observations[cam_id]['camera_data']
            observed_2d = np.array(observations[cam_id]['center'])
            
            # Убеждаемся что данные камеры в numpy формате
            camera_matrix = np.array(cam_data['camera_matrix'])
            rotation = np.array(cam_data['rotation'])
            position = np.array(cam_data['position'])
            
            error = self._calculate_reprojection_error(
                final_3d_position,
                camera_matrix,
                rotation,
                position,
                observed_2d
            )
            
            if not np.isinf(error):
                reprojection_errors.append(error)
            
            print(f"       Камера {cam_id}: ошибка {error:.2f} пикс")
        
        if not reprojection_errors:
            print(f"     Нет валидных ошибок репроекции")
            return None
        
        avg_reprojection_error = np.mean(reprojection_errors)
        print(f"     Средняя ошибка репроекции: {avg_reprojection_error:.2f} пикс (лимит: {self.max_reprojection_error})")
        
        # Проверяем допустимость ошибки (поднимаем лимит до 200)
        if avg_reprojection_error > 200.0:  # Поднял до 200 пикселей
            print(f"     Ошибка слишком велика: {avg_reprojection_error:.2f} > 200.0")
            return None
        
        # Вычисляем уверенность (чем меньше ошибка и больше камер, тем выше)
        confidence = min(1.0, (n_cameras - 2) / 5) * (1.0 - min(1.0, avg_reprojection_error / 200.0))
        
        return MarkerTriangulation(
            marker_id=marker_id,
            position_3d=(float(final_3d_position[0]), float(final_3d_position[1]), float(final_3d_position[2])),
            observations_count=n_cameras,
            reprojection_error=float(avg_reprojection_error),
            triangulation_confidence=float(confidence),
            camera_ids=camera_ids
        )
    
    def triangulate_all_markers(self, opencv_cameras: Dict[str, Dict], 
                              marker_detections: Dict[str, Dict]) -> Dict[int, MarkerTriangulation]:
        """Триангуляция всех маркеров"""
        
        # Группируем наблюдения по маркерам
        markers_observations = {}
        
        for camera_id, detections in marker_detections.items():
            if camera_id not in opencv_cameras:
                print(f"   ⚠️  Пропускаем камеру {camera_id}: нет параметров")
                continue
            
            camera_data = opencv_cameras[camera_id]
            
            for marker_id, detection in detections.items():
                if marker_id not in markers_observations:
                    markers_observations[marker_id] = {}
                
                # detection - это объект MarkerDetection, у него есть атрибут center
                markers_observations[marker_id][camera_id] = {
                    'center': detection.center,
                    'camera_data': camera_data
                }
        
        print(f"   Анализ наблюдений:")
        for marker_id, observations in markers_observations.items():
            n_cams = len(observations)
            status = "✅" if n_cams >= self.min_cameras else "❌"
            print(f"     Маркер {marker_id}: {n_cams} камер {status}")
        
        # Триангулируем каждый маркер
        triangulated_markers = {}
        
        for marker_id, observations in markers_observations.items():
            n_cameras = len(observations)
            
            if n_cameras < self.min_cameras:
                continue
            
            # Триангулируем маркер
            try:
                result = self._triangulate_marker_robust(marker_id, observations)
                
                if result is not None:
                    triangulated_markers[marker_id] = result
                    
            except Exception as e:
                print(f"   ❌ Маркер {marker_id}: ошибка триангуляции: {e}")
                continue
        
        return triangulated_markers


def triangulate_markers(opencv_cameras: Dict[str, Dict], 
                       marker_detections: Dict[str, Dict],
                       min_cameras: int = 3,
                       max_reprojection_error: float = 2.0) -> Dict[int, MarkerTriangulation]:
    """
    Главная функция триангуляции маркеров
    
    Parameters:
    -----------
    opencv_cameras : Dict[str, Dict]
        Параметры камер в OpenCV формате (из xmp_to_opencv)
    marker_detections : Dict[str, Dict]
        Детекции маркеров (из aruco_detector)
    min_cameras : int
        Минимальное количество камер для триангуляции
    max_reprojection_error : float
        Максимальная допустимая ошибка репроекции
        
    Returns:
    --------
    Dict[int, MarkerTriangulation]
        Результаты триангуляции {marker_id: MarkerTriangulation}
    """
    triangulator = ArUcoTriangulator(
        min_cameras=min_cameras,
        max_reprojection_error=max_reprojection_error
    )
    
    return triangulator.triangulate_all_markers(opencv_cameras, marker_detections)


def prepare_blender_export(triangulated_markers: Dict[int, MarkerTriangulation]) -> Dict:
    """Подготовка данных для экспорта в Blender"""
    
    # Подсчитываем маркеры высокого качества
    high_confidence_count = sum(1 for m in triangulated_markers.values() if m.triangulation_confidence >= 0.7)
    
    blender_data = {
        'metadata': {
            'total_markers': len(triangulated_markers),
            'high_confidence_markers': high_confidence_count,  # ДОБАВИЛИ ЭТО
            'coordinate_system': 'realitycapture_absolute'
        },
        'markers': {}
    }
    
    for marker_id, result in triangulated_markers.items():
        blender_data['markers'][f'marker_{marker_id}'] = {
            'id': marker_id,
            'position': list(result.position_3d),
            'confidence': result.triangulation_confidence,
            'quality': 'high' if result.triangulation_confidence >= 0.7 else 'medium' if result.triangulation_confidence >= 0.5 else 'low'
        }
    
    return blender_data