#!/usr/bin/env python3
"""
Экспортер камер для Blender
===========================

Преобразует данные камер из RealityCapture XMP в формат для Blender Camera объектов.
"""

import numpy as np
import math
from typing import Dict, Tuple


class CameraExporter:
    """Класс для экспорта камер в Blender формат"""
    
    def __init__(self):
        """Инициализация экспортера"""
        pass
    
    def _convert_position_to_blender(self, position: np.ndarray) -> Tuple[float, float, float]:
        """
        Конвертация позиции из RealityCapture в Blender координаты
        
        На основе анализа данных:
        RealityCapture: Y вверх, Z назад (глубина), X вправо
        Blender: Z вверх, Y назад, X вправо
        
        Преобразование: X→X, Y→Z, Z→-Y
        """
        x_rc, y_rc, z_rc = position
        
        x_blender = float(x_rc)   # X остается X
        y_blender = float(-z_rc)  # Z становится -Y (назад в Blender)  
        z_blender = float(y_rc)   # Y становится Z (вверх в Blender)
        
        return (x_blender, y_blender, z_blender)
    
    def _convert_rotation_matrix_to_blender(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Конвертация матрицы поворота RealityCapture в Euler углы Blender
        
        Применяет преобразование координат RC → Blender и учитывает
        что камеры в Blender смотрят в -Z направлении
        """
        # Матрица преобразования координат RC → Blender  
        # X→X, Y→Z, Z→-Y
        coord_transform = np.array([
            [1,  0,  0],   # X остается X
            [0,  0, -1],   # Z становится -Y
            [0,  1,  0]    # Y становится Z
        ])
        
        # Применяем преобразование координат
        blender_rotation = coord_transform @ rotation_matrix @ coord_transform.T
        
        # В RealityCapture камеры смотрят в -Z, в Blender тоже в -Z
        # Но из-за перестановки осей нужна корректировка
        # Поворот на 180° вокруг Z для правильной ориентации
        correction_rotation = np.array([
            [-1,  0,  0],  # Поворот на 180° вокруг Z
            [ 0, -1,  0],
            [ 0,  0,  1]
        ])
        
        blender_rotation = correction_rotation @ blender_rotation
        
        # Извлекаем Euler углы
        sy = math.sqrt(blender_rotation[0,0]**2 + blender_rotation[1,0]**2)
        
        if sy > 1e-6:
            x = math.atan2(blender_rotation[2,1], blender_rotation[2,2])
            y = math.atan2(-blender_rotation[2,0], sy)
            z = math.atan2(blender_rotation[1,0], blender_rotation[0,0])
        else:
            x = math.atan2(-blender_rotation[1,2], blender_rotation[1,1])
            y = math.atan2(-blender_rotation[2,0], sy)
            z = 0
        
        return (x, y, z)
    
    def _calculate_blender_camera_params(self, focal_length_35mm: float, 
                                       image_size: tuple = (2592, 1944)) -> dict:
        """
        Вычисление параметров Blender Camera объекта
        """
        image_width, image_height = image_size
        aspect_ratio = image_width / image_height
        
        # Настройки для Blender Camera
        if aspect_ratio >= 1.0:
            sensor_fit = 'HORIZONTAL'
            sensor_width = 36.0  # мм (стандартный полнокадровый сенсор)
        else:
            sensor_fit = 'VERTICAL'
            sensor_width = 36.0 / aspect_ratio
        
        # Размер отображения камеры
        display_size = max(0.3, min(2.0, 50.0 / focal_length_35mm))
        
        return {
            'lens': focal_length_35mm,
            'sensor_width': sensor_width,
            'sensor_fit': sensor_fit,
            'display_size': display_size,
            'clip_start': 0.1,
            'clip_end': 1000.0,
        }
    
    def _determine_camera_quality(self, xmp_data: dict) -> str:
        """Определение качества калибровки камеры"""
        validation = xmp_data.get('validation', {})
        
        if not validation.get('is_valid', True):
            return 'low'
        
        warnings_count = len(validation.get('warnings', []))
        if warnings_count == 0:
            return 'high'
        elif warnings_count <= 2:
            return 'medium'
        else:
            return 'low'
    
    def export_single_camera(self, camera_id: str, xmp_data: dict, 
                           image_size: tuple = (2592, 1944)) -> dict:
        """Экспорт одной камеры в Blender формат"""
        
        # Извлекаем данные из XMP
        position = np.array(xmp_data['position'])
        rotation = np.array(xmp_data['rotation'])
        focal_length = xmp_data['focal_length']
        
        # Конвертируем в Blender координаты
        blender_position = self._convert_position_to_blender(position)
        blender_rotation = self._convert_rotation_matrix_to_blender(rotation)
        
        # Вычисляем параметры Blender Camera
        camera_params = self._calculate_blender_camera_params(focal_length, image_size)
        
        # Определяем качество
        quality = self._determine_camera_quality(xmp_data)
        
        # Подготавливаем экспортные данные
        return {
            'id': camera_id,
            'position': list(blender_position),
            'rotation_euler': list(blender_rotation),
            'quality': quality,
            'camera_params': camera_params,
            'focal_length_35mm': focal_length,
            'image_size': list(image_size),
            'used_in_texturing': xmp_data['in_texturing'],
            'used_in_meshing': xmp_data['in_meshing'],
            'realitycapture_version': xmp_data['realitycapture_version'],
            'calibration_prior': xmp_data['calibration_prior'],
            'pose_prior': xmp_data['pose_prior'],
            'validation_warnings': xmp_data['validation'].get('warnings', []),
        }
    
    def export_all_cameras(self, xmp_cameras: Dict[str, dict], 
                         image_size: tuple = (2592, 1944)) -> Dict[str, dict]:
        """Экспорт всех камер в Blender формат"""
        
        exported_cameras = {}
        
        print("🎥 Экспорт камер для Blender:")
        
        for camera_id, xmp_data in xmp_cameras.items():
            try:
                camera_export = self.export_single_camera(camera_id, xmp_data, image_size)
                exported_cameras[camera_id] = camera_export
                
                pos = camera_export['position']
                quality = camera_export['quality']
                focal = camera_export['focal_length_35mm']
                
                quality_icon = {'high': '✅', 'medium': '⚠️', 'low': '❌'}[quality]
                print(f"   {camera_id}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                      f"f={focal:.1f}mm {quality_icon}")
                
            except Exception as e:
                print(f"   ❌ Ошибка экспорта {camera_id}: {e}")
                continue
        
        print(f"   Экспортировано Camera объектов: {len(exported_cameras)}")
        return exported_cameras


def prepare_blender_export(triangulated_markers: dict, xmp_cameras: Dict[str, dict],
                          image_size: tuple = (2592, 1944)) -> dict:
    """
    Главная функция подготовки экспорта камер + маркеров для Blender
    """
    # Экспортируем камеры
    camera_exporter = CameraExporter()
    exported_cameras = camera_exporter.export_all_cameras(xmp_cameras, image_size)
    
    # Подготавливаем маркеры
    blender_markers = {}
    high_confidence_markers = 0
    
    for marker_id, result in triangulated_markers.items():
        quality = 'high' if result.triangulation_confidence >= 0.7 else 'medium' if result.triangulation_confidence >= 0.5 else 'low'
        
        blender_markers[f'marker_{marker_id}'] = {
            'id': marker_id,
            'position': list(result.position_3d),
            'confidence': result.triangulation_confidence,
            'quality': quality,
            'reprojection_error': result.reprojection_error,
            'observations_count': result.observations_count,
        }
        
        if quality == 'high':
            high_confidence_markers += 1
    
    # Статистика по качеству камер
    camera_quality_stats = {
        'high': sum(1 for cam in exported_cameras.values() if cam['quality'] == 'high'),
        'medium': sum(1 for cam in exported_cameras.values() if cam['quality'] == 'medium'),
        'low': sum(1 for cam in exported_cameras.values() if cam['quality'] == 'low')
    }
    
    # Границы сцены
    all_positions = []
    for camera_data in exported_cameras.values():
        all_positions.append(camera_data['position'])
    for marker_data in blender_markers.values():
        all_positions.append(marker_data['position'])
    
    if all_positions:
        all_positions = np.array(all_positions)
        bounds_min = all_positions.min(axis=0).tolist()
        bounds_max = all_positions.max(axis=0).tolist()
        bounds_center = all_positions.mean(axis=0).tolist()
        bounds_size = float(np.linalg.norm(np.array(bounds_max) - np.array(bounds_min)))
    else:
        bounds_min = bounds_max = bounds_center = [0, 0, 0]
        bounds_size = 0
    
    # Комбинированные данные
    return {
        'metadata': {
            'format_version': '2.0',
            'created_by': 'ArUco Autocalibration Pipeline with Cameras',
            'coordinate_system': 'realitycapture_to_blender_converted',
            'has_cameras': True,
            'has_markers': True,
            'cameras_total': len(exported_cameras),
            'cameras_quality': camera_quality_stats,
            'markers_total': len(triangulated_markers),
            'markers_high_quality': high_confidence_markers,
            'scene_bounds': {
                'min': bounds_min,
                'max': bounds_max,
                'center': bounds_center,
                'size': bounds_size
            }
        },
        'cameras': exported_cameras,
        'markers': blender_markers
    }