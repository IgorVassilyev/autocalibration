#!/usr/bin/env python3
"""
ArUco One-Click Addon - Полный автоматический пайплайн в одной кнопке
===================================================================

Аддон для Blender, который выполняет весь пайплайн обработки ArUco маркеров:
1. Загружает XMP файлы камер из папки data/
2. Конвертирует параметры камер в OpenCV формат
3. Детектирует ArUco маркеры на изображениях (ID 1-13)
4. Триангулирует 3D позиции маркеров
5. Импортирует камеры и маркеры в Blender

ВСЁ В ОДНОЙ КНОПКЕ!

Установка:
1. Установите OpenCV в Blender Python: pip install opencv-python
2. Сохраните как aruco_oneclick_addon.py
3. Blender → Edit → Preferences → Add-ons → Install
4. Активируйте "ArUco One-Click Addon"

Использование:
1. Поместите XMP файлы и изображения в папку data/
2. Нажмите "СТАРТ"
3. Готово!
"""

bl_info = {
    "name": "ArUco One-Click Addon",
    "author": "ArUco Autocalibration Project",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport → Sidebar → ArUco OneClick",
    "description": "Complete ArUco pipeline in one click - from images to 3D markers",
    "category": "Import-Export",
}

import bpy
import os
import json
import glob
import time
import xml.etree.ElementTree as ET
import traceback
from mathutils import Matrix, Vector
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy.types import Panel, Operator, PropertyGroup

# Проверка OpenCV
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
    print("OpenCV is available for ArUco detection")
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not found. ArUco detection will not work.")

# =============================================================================
# ВСТРОЕННЫЕ МОДУЛИ ИЗ ПРОЕКТА
# =============================================================================

class EmbeddedXMPParser:
    """Встроенный XMP парсер (из xmp_parser.py)"""
    
    def __init__(self):
        self.RC_NS = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xcr": "http://www.capturingreality.com/ns/xcr/1.1#"
        }
    
    def _safe_floats(self, text, expected_count=None):
        if not text or text.strip() == "":
            return []
        try:
            values = [float(x) for x in str(text).strip().split()]
            if expected_count and len(values) != expected_count:
                return []
            return values
        except:
            return []
    
    def parse_xmp_file(self, xmp_path):
        try:
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            desc = root.find(".//rdf:Description", self.RC_NS)
            if desc is None:
                return None
            
            # Позиция и поворот
            pos_elem = desc.find("xcr:Position", self.RC_NS)
            rot_elem = desc.find("xcr:Rotation", self.RC_NS)
            dist_elem = desc.find("xcr:DistortionCoeficients", self.RC_NS)
            
            position = self._safe_floats(pos_elem.text if pos_elem is not None else "", 3)
            rotation = self._safe_floats(rot_elem.text if rot_elem is not None else "", 9)
            distortion = self._safe_floats(dist_elem.text if dist_elem is not None else "")
            
            if len(position) != 3 or len(rotation) != 9:
                return None
            
            # Дополняем дисторсию до 6 коэффициентов
            while len(distortion) < 6:
                distortion.append(0.0)
            
            # Атрибуты
            attrs = {}
            xcr_prefix = '{' + self.RC_NS['xcr'] + '}'
            for key, value in desc.attrib.items():
                if key.startswith(xcr_prefix):
                    attr_name = key[len(xcr_prefix):]
                    attrs[attr_name] = value
            
            return {
                'name': os.path.splitext(os.path.basename(xmp_path))[0],
                'position': position,
                'rotation': rotation,
                'focal_length': float(attrs.get('FocalLength35mm', 35.0)),
                'principal_point_u': float(attrs.get('PrincipalPointU', 0.0)),
                'principal_point_v': float(attrs.get('PrincipalPointV', 0.0)),
                'aspect_ratio': float(attrs.get('AspectRatio', 1.0)),
                'distortion': distortion,
                'in_texturing': attrs.get('InTexturing', '1') == '1',
                'in_meshing': attrs.get('InMeshing', '1') == '1',
                'validation': {'is_valid': True, 'warnings': [], 'errors': []}
            }
            
        except Exception as e:
            print(f"Error parsing {xmp_path}: {e}")
            return None
    
    def load_all_cameras(self, directory):
        cameras = {}
        if not os.path.exists(directory):
            return cameras
        
        xmp_files = [f for f in os.listdir(directory) if f.lower().endswith('.xmp')]
        for xmp_file in sorted(xmp_files):
            xmp_path = os.path.join(directory, xmp_file)
            data = self.parse_xmp_file(xmp_path)
            if data:
                cameras[data['name']] = data
        
        return cameras

class EmbeddedXMPToOpenCV:
    """Встроенный XMP→OpenCV конвертер (из xmp_to_opencv.py)"""
    
    def __init__(self, sensor_width_35mm=36.0):
        self.sensor_width_35mm = sensor_width_35mm
    
    def convert_single_camera(self, camera_id, xmp_data, image_size):
        image_width, image_height = image_size
        
        focal_length_35mm = xmp_data['focal_length']
        principal_point_u = xmp_data['principal_point_u']
        principal_point_v = xmp_data['principal_point_v']
        aspect_ratio = xmp_data['aspect_ratio']
        distortion_coeffs = xmp_data['distortion']
        
        # Преобразование фокусного расстояния в пиксели
        fx_pixels = (focal_length_35mm / self.sensor_width_35mm) * image_width
        fy_pixels = fx_pixels * aspect_ratio
        
        # Главная точка в пиксели
        cx_pixels = image_width / 2 + principal_point_u * (image_width / 2)
        cy_pixels = image_height / 2 + principal_point_v * (image_height / 2)
        
        # Матрица камеры
        if OPENCV_AVAILABLE:
            camera_matrix = np.array([
                [fx_pixels, 0.0, cx_pixels],
                [0.0, fy_pixels, cy_pixels],
                [0.0, 0.0, 1.0]
            ])
            distortion_array = np.array(distortion_coeffs)
            position = np.array(xmp_data['position'])
            rotation = np.array(xmp_data['rotation']).reshape(3, 3)
        else:
            camera_matrix = [[fx_pixels, 0.0, cx_pixels],
                           [0.0, fy_pixels, cy_pixels],
                           [0.0, 0.0, 1.0]]
            distortion_array = distortion_coeffs
            position = xmp_data['position']
            rotation = [xmp_data['rotation'][i:i+3] for i in range(0, 9, 3)]
        
        return {
            'camera_matrix': camera_matrix,
            'distortion_coeffs': distortion_array,
            'fx': fx_pixels,
            'fy': fy_pixels,
            'cx': cx_pixels,
            'cy': cy_pixels,
            'image_size': image_size,
            'position': position,
            'rotation': rotation,
            'original_focal_35mm': focal_length_35mm,
            'conversion_warnings': []
        }
    
    def convert_all_cameras(self, xmp_cameras, image_size):
        opencv_cameras = {}
        for camera_id, xmp_data in xmp_cameras.items():
            try:
                opencv_params = self.convert_single_camera(camera_id, xmp_data, image_size)
                opencv_cameras[camera_id] = opencv_params
            except Exception as e:
                print(f"Error converting camera {camera_id}: {e}")
        return opencv_cameras

class EmbeddedArUcoDetector:
    """Встроенный ArUco детектор (из aruco_detector.py)"""
    
    def __init__(self):
        if not OPENCV_AVAILABLE:
            self.aruco_dict = None
            self.parameters = None
            return
        
        self.dictionary = cv2.aruco.DICT_4X4_1000
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Настройки детектора
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.polygonalApproxAccuracyRate = 0.03
        self.parameters.minCornerDistanceRate = 0.05
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    def detect_markers_in_image(self, image_path):
        if not OPENCV_AVAILABLE:
            return {}
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {}
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            corners, ids, _ = detector.detectMarkers(gray)
            
            detections = {}
            
            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids.flatten()):
                    # ТОЛЬКО маркеры с ID 1-13
                    if 1 <= marker_id <= 13:
                        marker_corners = corners[i].reshape(4, 2)
                        center_x = float(np.mean(marker_corners[:, 0]))
                        center_y = float(np.mean(marker_corners[:, 1]))
                        area = float(cv2.contourArea(marker_corners))
                        
                        # Создаем объект детекции (как MarkerDetection)
                        detection = type('MarkerDetection', (), {
                            'marker_id': int(marker_id),
                            'center': (center_x, center_y),
                            'corners': marker_corners.tolist(),
                            'area': area
                        })()
                        
                        detections[int(marker_id)] = detection
            
            return detections
            
        except Exception as e:
            print(f"Error detecting markers in {image_path}: {e}")
            return {}
    
    def detect_markers_in_directory(self, directory):
        all_detections = {}
        
        if not OPENCV_AVAILABLE:
            print("OpenCV not available - cannot detect markers")
            return all_detections
        
        # Поиск изображений
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        
        for ext in image_extensions:
            pattern = os.path.join(directory, ext)
            images.extend(glob.glob(pattern))
            pattern_upper = os.path.join(directory, ext.upper())
            images.extend(glob.glob(pattern_upper))
        
        images = sorted(list(set(images)))
        
        print(f"Detecting ArUco markers in {len(images)} images...")
        
        for image_path in images:
            filename = os.path.basename(image_path)
            camera_id = os.path.splitext(filename)[0]
            
            detections = self.detect_markers_in_image(image_path)
            all_detections[camera_id] = detections
            
            if detections:
                marker_ids = list(detections.keys())
                print(f"  {filename}: found markers {marker_ids}")
        
        return all_detections

class EmbeddedTriangulator:
    """Встроенный триангулятор (из triangulation.py)"""
    
    def __init__(self, min_cameras=3, max_reprojection_error=200.0):
        self.min_cameras = min_cameras
        self.max_reprojection_error = max_reprojection_error
    
    def triangulate_markers(self, opencv_cameras, marker_detections):
        if not OPENCV_AVAILABLE:
            print("OpenCV not available - cannot triangulate")
            return {}
        
        # Группируем наблюдения по маркерам
        markers_observations = {}
        
        for camera_id, detections in marker_detections.items():
            if camera_id not in opencv_cameras:
                continue
            
            camera_data = opencv_cameras[camera_id]
            
            for marker_id, detection in detections.items():
                if marker_id not in markers_observations:
                    markers_observations[marker_id] = {}
                
                markers_observations[marker_id][camera_id] = {
                    'center': detection.center,
                    'camera_data': camera_data
                }
        
        print(f"Triangulating markers...")
        triangulated_markers = {}
        
        for marker_id, observations in markers_observations.items():
            n_cameras = len(observations)
            
            if n_cameras < self.min_cameras:
                print(f"  Marker {marker_id}: only {n_cameras} cameras, need {self.min_cameras}")
                continue
            
            try:
                result = self._triangulate_marker_robust(marker_id, observations)
                if result is not None:
                    triangulated_markers[marker_id] = result
                    print(f"  Marker {marker_id}: triangulated with {n_cameras} cameras")
                
            except Exception as e:
                print(f"  Marker {marker_id}: triangulation error: {e}")
        
        return triangulated_markers
    
    def _triangulate_marker_robust(self, marker_id, observations):
        if not OPENCV_AVAILABLE:
            return None
        
        camera_ids = list(observations.keys())
        n_cameras = len(camera_ids)
        
        # Собираем матрицы проекции
        projection_matrices = {}
        for cam_id in camera_ids:
            cam_data = observations[cam_id]['camera_data']
            camera_matrix = np.array(cam_data['camera_matrix'])
            rotation = np.array(cam_data['rotation'])
            position = np.array(cam_data['position'])
            
            # t = -R * position
            translation = -rotation @ position.reshape(3, 1)
            rt_matrix = np.hstack([rotation, translation])
            proj_matrix = camera_matrix @ rt_matrix
            projection_matrices[cam_id] = proj_matrix
        
        # Триангулируем по всем парам
        triangulated_points = []
        
        for i in range(n_cameras):
            for j in range(i + 1, n_cameras):
                cam1_id = camera_ids[i]
                cam2_id = camera_ids[j]
                
                p1 = np.array(observations[cam1_id]['center'])
                p2 = np.array(observations[cam2_id]['center'])
                
                try:
                    points_4d = cv2.triangulatePoints(
                        projection_matrices[cam1_id],
                        projection_matrices[cam2_id],
                        p1.reshape(2, 1),
                        p2.reshape(2, 1)
                    )
                    
                    point_3d = points_4d[:3] / points_4d[3]
                    
                    if not np.any(np.isinf(point_3d)) and not np.any(np.isnan(point_3d)):
                        triangulated_points.append(point_3d.flatten())
                        
                except Exception:
                    continue
        
        if not triangulated_points:
            return None
        
        # Усредняем результаты
        final_3d_position = np.mean(triangulated_points, axis=0)
        
        # Вычисляем ошибки репроекции
        reprojection_errors = []
        for cam_id in camera_ids:
            cam_data = observations[cam_id]['camera_data']
            observed_2d = np.array(observations[cam_id]['center'])
            
            camera_matrix = np.array(cam_data['camera_matrix'])
            rotation = np.array(cam_data['rotation'])
            position = np.array(cam_data['position'])
            
            # Репроекция
            point_camera = rotation @ (final_3d_position - position)
            if abs(point_camera[2]) > 1e-6:
                projected_2d = camera_matrix @ point_camera
                projected_2d = projected_2d[:2] / projected_2d[2]
                error = np.linalg.norm(projected_2d - observed_2d)
                reprojection_errors.append(error)
        
        if not reprojection_errors:
            return None
        
        avg_error = np.mean(reprojection_errors)
        if avg_error > self.max_reprojection_error:
            return None
        
        # Уверенность триангуляции
        confidence = min(1.0, (n_cameras - 2) / 5) * (1.0 - min(1.0, avg_error / 200.0))
        
        # Создаем результат (как MarkerTriangulation)
        result = type('MarkerTriangulation', (), {
            'marker_id': marker_id,
            'position_3d': (float(final_3d_position[0]), float(final_3d_position[1]), float(final_3d_position[2])),
            'observations_count': n_cameras,
            'reprojection_error': float(avg_error),
            'triangulation_confidence': float(confidence),
            'camera_ids': camera_ids
        })()
        
        return result

# =============================================================================
# ОСНОВНОЙ ПРОЦЕССОР
# =============================================================================

class ArUcoOneClickProcessor:
    """Процессор полного пайплайна в одной кнопке"""
    
    def __init__(self):
        self.image_size = (2592, 1944)  # Можно настроить
    
    def process_full_pipeline(self, data_directory):
        """Выполнение полного пайплайна"""
        results = {
            'success': False,
            'cameras_loaded': 0,
            'cameras_converted': 0,
            'images_processed': 0,
            'markers_detected': 0,
            'markers_triangulated': 0,
            'high_quality_markers': 0,
            'execution_time': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            print("🚀 Starting ArUco One-Click Pipeline...")
            print(f"Data directory: {data_directory}")
            
            # Этап 1: Загрузка XMP камер
            print("\n📷 Stage 1: Loading XMP cameras...")
            parser = EmbeddedXMPParser()
            xmp_cameras = parser.load_all_cameras(data_directory)
            
            if not xmp_cameras:
                raise Exception("No XMP cameras loaded")
            
            results['cameras_loaded'] = len(xmp_cameras)
            print(f"Loaded {len(xmp_cameras)} cameras")
            
            # Этап 2: Конвертация в OpenCV
            print("\n🔄 Stage 2: Converting to OpenCV format...")
            converter = EmbeddedXMPToOpenCV()
            opencv_cameras = converter.convert_all_cameras(xmp_cameras, self.image_size)
            
            if not opencv_cameras:
                raise Exception("No cameras converted to OpenCV format")
            
            results['cameras_converted'] = len(opencv_cameras)
            print(f"Converted {len(opencv_cameras)} cameras")
            
            # Этап 3: Детекция ArUco маркеров
            print("\n🎯 Stage 3: Detecting ArUco markers...")
            detector = EmbeddedArUcoDetector()
            marker_detections = detector.detect_markers_in_directory(data_directory)
            
            if not marker_detections:
                raise Exception("No marker detections")
            
            # Подсчет статистики детекций
            total_detections = sum(len(detections) for detections in marker_detections.values())
            unique_markers = set()
            for detections in marker_detections.values():
                unique_markers.update(detections.keys())
            
            results['images_processed'] = len(marker_detections)
            results['markers_detected'] = len(unique_markers)
            print(f"Processed {len(marker_detections)} images, found {len(unique_markers)} unique markers")
            
            # Этап 4: Триангуляция маркеров
            print("\n📐 Stage 4: Triangulating markers...")
            triangulator = EmbeddedTriangulator()
            triangulated_markers = triangulator.triangulate_markers(opencv_cameras, marker_detections)
            
            if not triangulated_markers:
                raise Exception("No markers triangulated")
            
            results['markers_triangulated'] = len(triangulated_markers)
            
            # Подсчет высококачественных маркеров
            high_quality = sum(1 for m in triangulated_markers.values() 
                             if m.triangulation_confidence >= 0.7)
            results['high_quality_markers'] = high_quality
            
            print(f"Triangulated {len(triangulated_markers)} markers ({high_quality} high quality)")
            
            # Сохранение результатов для импорта в Blender
            self.processed_cameras = xmp_cameras
            self.processed_markers = triangulated_markers
            
            results['success'] = True
            results['execution_time'] = time.time() - start_time
            
            print(f"\n✅ Pipeline completed successfully in {results['execution_time']:.1f} seconds!")
            
        except Exception as e:
            error_msg = str(e)
            results['errors'].append(error_msg)
            results['execution_time'] = time.time() - start_time
            print(f"\n❌ Pipeline failed: {error_msg}")
            traceback.print_exc()
        
        return results

# =============================================================================
# BLENDER ИМПОРТЕР
# =============================================================================

class BlenderImporter:
    """Импортер результатов в Blender"""
    
    def create_blender_camera_matrix(self, rotation_flat, position):
        """Создание матрицы камеры для Blender"""
        try:
            # Преобразование из RealityCapture в Blender
            R_rc = Matrix([
                [rotation_flat[0], rotation_flat[1], rotation_flat[2]],
                [rotation_flat[3], rotation_flat[4], rotation_flat[5]],
                [rotation_flat[6], rotation_flat[7], rotation_flat[8]]
            ])
            
            # OpenCV -> Blender camera преобразование
            R_cv2bcam = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
            R_bcam2cv = R_cv2bcam.transposed()
            
            R_w2bcam = R_bcam2cv @ R_rc
            R_bcam2w = R_w2bcam.transposed()
            
            matrix_4x4 = R_bcam2w.to_4x4()
            matrix_4x4.translation = Vector(position)
            
            return matrix_4x4
            
        except Exception as e:
            print(f"Error creating camera matrix: {e}")
            return Matrix()
    
    def ensure_collection(self, name):
        """Создание или получение коллекции"""
        coll = bpy.data.collections.get(name)
        if not coll:
            coll = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(coll)
        return coll
    
    def import_cameras(self, xmp_cameras, props):
        """Импорт камер в Blender"""
        cameras_collection = self.ensure_collection("ArUco_Cameras")
        imported_count = 0
        
        for camera_id, cam_data in xmp_cameras.items():
            try:
                # Создание камеры
                cam_data_block = bpy.data.cameras.new(name=f"{camera_id}_CAM")
                cam_obj = bpy.data.objects.new(name=f"Cam_{camera_id}", object_data=cam_data_block)
                
                cameras_collection.objects.link(cam_obj)
                
                # Трансформация
                matrix = self.create_blender_camera_matrix(cam_data['rotation'], cam_data['position'])
                cam_obj.matrix_world = matrix
                
                # Параметры камеры
                focal_35mm = cam_data['focal_length']
                if 10.0 <= focal_35mm <= 300.0:
                    cam_data_block.sensor_fit = 'HORIZONTAL'
                    cam_data_block.sensor_width = 36.0
                    cam_data_block.lens = focal_35mm
                
                cam_data_block.display_size = props.camera_size
                
                # Сохранение данных
                cam_obj["ArUco_Camera"] = True
                cam_obj["focal_length"] = focal_35mm
                cam_obj["camera_id"] = camera_id
                
                imported_count += 1
                
            except Exception as e:
                print(f"Error importing camera {camera_id}: {e}")
        
        return imported_count
    
    def import_markers(self, triangulated_markers, props):
        """Импорт маркеров в Blender"""
        markers_collection = self.ensure_collection("ArUco_Markers")
        imported_count = 0
        
        for marker_id, result in triangulated_markers.items():
            try:
                confidence = result.triangulation_confidence
                
                # Фильтрация по качеству
                if props.filter_by_quality and confidence < props.min_quality_filter:
                    continue
                
                # Определение качества
                if confidence >= props.quality_threshold_high:
                    quality = 'high'
                elif confidence >= props.quality_threshold_medium:
                    quality = 'medium'
                else:
                    quality = 'low'
                
                # Размер
                if props.size_by_quality:
                    if quality == 'high':
                        size = props.marker_size
                    elif quality == 'medium':
                        size = props.marker_size * 0.8
                    else:
                        size = props.marker_size * 0.6
                else:
                    size = props.marker_size
                
                # Цвет
                if props.color_by_quality:
                    if quality == 'high':
                        color = (0.0, 1.0, 0.0, 1.0)  # Зеленый
                    elif quality == 'medium':
                        color = (1.0, 1.0, 0.0, 1.0)  # Желтый
                    else:
                        color = (1.0, 0.5, 0.0, 1.0)  # Оранжевый
                else:
                    color = (1.0, 1.0, 1.0, 1.0)
                
                # Создание объекта
                bpy.ops.object.empty_add(
                    type=props.marker_type,
                    location=tuple(result.position_3d)
                )
                
                marker_obj = bpy.context.active_object
                marker_obj.name = f"ArUco_{marker_id:02d}"
                marker_obj.empty_display_size = size
                marker_obj.color = color
                
                # Перемещение в коллекцию
                markers_collection.objects.link(marker_obj)
                if marker_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(marker_obj)
                
                # Сохранение данных
                marker_obj["ArUco_Marker"] = True
                marker_obj["marker_id"] = marker_id
                marker_obj["confidence"] = confidence
                marker_obj["quality"] = quality
                marker_obj["observations"] = result.observations_count
                marker_obj["reprojection_error"] = result.reprojection_error
                
                imported_count += 1
                
            except Exception as e:
                print(f"Error importing marker {marker_id}: {e}")
        
        return imported_count
    
    def clear_existing(self):
        """Очистка существующих ArUco объектов"""
        objects_to_remove = []
        for obj in bpy.data.objects:
            if obj.get("ArUco_Camera") or obj.get("ArUco_Marker"):
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Удаление коллекций
        for coll_name in ['ArUco_Cameras', 'ArUco_Markers']:
            if coll_name in bpy.data.collections:
                bpy.data.collections.remove(bpy.data.collections[coll_name])

# =============================================================================
# СВОЙСТВА АДДОНА
# =============================================================================

class ArUcoOneClickProperties(PropertyGroup):
    """Свойства аддона одной кнопки"""
    
    # Автоматически найденная папка
    data_folder: StringProperty(
        name="Папка данных",
        description="Папка с XMP файлами и изображениями",
        default="",
        subtype='DIR_PATH'
    )
    
    # Настройки процессинга
    image_width: bpy.props.IntProperty(
        name="Ширина изображения",
        description="Ширина изображений в пикселях",
        default=2592,
        min=100,
        max=10000
    )
    
    image_height: bpy.props.IntProperty(
        name="Высота изображения",
        description="Высота изображений в пикселях",
        default=1944,
        min=100,
        max=10000
    )
    
    # Настройки отображения камер
    camera_size: FloatProperty(
        name="Размер камер",
        description="Размер отображения камер",
        default=0.3,
        min=0.01,
        max=2.0
    )
    
    # Настройки маркеров
    marker_type: EnumProperty(
        name="Тип маркеров",
        items=[
            ('PLAIN_AXES', 'Оси', 'Простые оси'),
            ('ARROWS', 'Стрелки', 'Стрелки'),
            ('SINGLE_ARROW', 'Стрелка', 'Одна стрелка'),
            ('CIRCLE', 'Круг', 'Круг'),
            ('CUBE', 'Куб', 'Куб'),
            ('SPHERE', 'Сфера', 'Сфера'),
        ],
        default='PLAIN_AXES'
    )
    
    marker_size: FloatProperty(
        name="Размер маркеров",
        description="Базовый размер маркеров",
        default=0.1,
        min=0.001,
        max=1.0
    )
    
    size_by_quality: BoolProperty(
        name="Размер по качеству",
        description="Изменять размер в зависимости от качества",
        default=True
    )
    
    color_by_quality: BoolProperty(
        name="Цвет по качеству",
        description="Раскрашивать по качеству",
        default=True
    )
    
    # Пороги качества
    quality_threshold_high: FloatProperty(
        name="Порог высокого качества",
        default=0.7,
        min=0.0,
        max=1.0
    )
    
    quality_threshold_medium: FloatProperty(
        name="Порог среднего качества",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    # Фильтрация
    filter_by_quality: BoolProperty(
        name="Фильтровать по качеству",
        description="Импортировать только качественные маркеры",
        default=False
    )
    
    min_quality_filter: FloatProperty(
        name="Минимальное качество",
        default=0.3,
        min=0.0,
        max=1.0
    )
    
    # Управление
    clear_existing: BoolProperty(
        name="Очистить существующие",
        description="Удалить предыдущие результаты",
        default=True
    )

# =============================================================================
# ОПЕРАТОРЫ
# =============================================================================

class ARUCO_OT_oneclick_process(Operator):
    """Основная операция - обработка всего пайплайна в одной кнопке"""
    
    bl_idname = "aruco.oneclick_process"
    bl_label = "СТАРТ"
    bl_description = "Выполнить весь пайплайн: XMP → детекция → триангуляция → импорт в Blender"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.aruco_oneclick_props
        
        # Проверка OpenCV
        if not OPENCV_AVAILABLE:
            self.report({'ERROR'}, "OpenCV не установлен! Установите: pip install opencv-python")
            return {'CANCELLED'}
        
        # Автопоиск папки данных если не задана
        if not props.data_folder or not os.path.exists(props.data_folder):
            data_folder = self.auto_find_data_folder()
            if data_folder:
                props.data_folder = data_folder
                self.report({'INFO'}, f"Найдена папка данных: {data_folder}")
            else:
                self.report({'ERROR'}, "Папка с данными не найдена! Укажите папку data/ с XMP и изображениями")
                return {'CANCELLED'}
        
        try:
            # Очистка существующих объектов
            if props.clear_existing:
                importer = BlenderImporter()
                importer.clear_existing()
            
            # Запуск процессинга
            processor = ArUcoOneClickProcessor()
            processor.image_size = (props.image_width, props.image_height)
            
            self.report({'INFO'}, "Запуск обработки...")
            
            results = processor.process_full_pipeline(props.data_folder)
            
            if not results['success']:
                error_msg = '; '.join(results['errors']) if results['errors'] else "Unknown error"
                self.report({'ERROR'}, f"Ошибка обработки: {error_msg}")
                return {'CANCELLED'}
            
            # Импорт результатов в Blender (всегда и камеры и маркеры)
            importer = BlenderImporter()
            imported_cameras = 0
            imported_markers = 0
            
            # Импортируем камеры если есть
            if hasattr(processor, 'processed_cameras'):
                imported_cameras = importer.import_cameras(processor.processed_cameras, props)
            
            # Импортируем маркеры если есть
            if hasattr(processor, 'processed_markers'):
                imported_markers = importer.import_markers(processor.processed_markers, props)
            
            # Настройка сцены
            bpy.context.scene.unit_settings.system = 'METRIC'
            bpy.context.scene.unit_settings.scale_length = 1.0
            
            # Результат
            message = (f"✅ Готово! Время: {results['execution_time']:.1f}с, "
                      f"Камер: {imported_cameras}, Маркеров: {imported_markers} "
                      f"(качественных: {results['high_quality_markers']})")
            
            self.report({'INFO'}, message)
            print(f"\nArUco One-Click: {message}")
            
            return {'FINISHED'}
            
        except Exception as e:
            error_msg = f"Критическая ошибка: {str(e)}"
            self.report({'ERROR'}, error_msg)
            print(f"ArUco One-Click Error: {error_msg}")
            traceback.print_exc()
            return {'CANCELLED'}
    
    def auto_find_data_folder(self):
        """Автопоиск папки с данными"""
        search_paths = []
        
        # Относительно blend файла
        if bpy.data.filepath:
            blend_dir = os.path.dirname(bpy.data.filepath)
            search_paths.append(blend_dir)
            search_paths.append(os.path.dirname(blend_dir))
        
        # Текущая рабочая директория
        search_paths.append(os.getcwd())
        
        for base_path in search_paths:
            data_path = os.path.join(base_path, "data")
            if os.path.exists(data_path):
                # Проверяем наличие XMP файлов
                try:
                    xmp_files = [f for f in os.listdir(data_path) if f.lower().endswith('.xmp')]
                    if xmp_files:
                        print(f"Found data folder with {len(xmp_files)} XMP files: {data_path}")
                        return data_path
                except:
                    pass
        
        return None

class ARUCO_OT_select_data_folder(Operator):
    """Выбор папки с данными"""
    
    bl_idname = "aruco.select_data_folder"
    bl_label = "Выбрать папку"
    bl_description = "Выбрать папку с XMP файлами и изображениями"
    
    filepath: StringProperty(subtype="DIR_PATH")
    
    def execute(self, context):
        context.scene.aruco_oneclick_props.data_folder = self.filepath
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class ARUCO_OT_auto_find_data(Operator):
    """Автопоиск папки с данными"""
    
    bl_idname = "aruco.auto_find_data"
    bl_label = "Автопоиск"
    bl_description = "Автоматически найти папку data/"
    
    def execute(self, context):
        props = context.scene.aruco_oneclick_props
        
        # Поиск папки
        search_paths = []
        
        if bpy.data.filepath:
            blend_dir = os.path.dirname(bpy.data.filepath)
            search_paths.append(blend_dir)
            search_paths.append(os.path.dirname(blend_dir))
        
        search_paths.append(os.getcwd())
        
        found = False
        
        for base_path in search_paths:
            data_path = os.path.join(base_path, "data")
            if os.path.exists(data_path):
                try:
                    xmp_files = [f for f in os.listdir(data_path) if f.lower().endswith('.xmp')]
                    image_files = []
                    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
                        pattern = os.path.join(data_path, f"*.{ext}")
                        image_files.extend(glob.glob(pattern))
                        pattern = os.path.join(data_path, f"*.{ext.upper()}")
                        image_files.extend(glob.glob(pattern))
                    
                    if xmp_files and image_files:
                        props.data_folder = data_path
                        found = True
                        self.report({'INFO'}, f"Найдена папка: {len(xmp_files)} XMP, {len(image_files)} изображений")
                        break
                except:
                    pass
        
        if not found:
            self.report({'WARNING'}, "Папка data/ не найдена. Укажите вручную")
        
        return {'FINISHED'}

class ARUCO_OT_clear_aruco(Operator):
    """Очистка ArUco объектов"""
    
    bl_idname = "aruco.clear_aruco"
    bl_label = "Очистить"
    bl_description = "Удалить все ArUco камеры и маркеры"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        importer = BlenderImporter()
        importer.clear_existing()
        self.report({'INFO'}, "ArUco объекты удалены")
        return {'FINISHED'}

# =============================================================================
# ИНТЕРФЕЙС
# =============================================================================

class ARUCO_PT_oneclick_main(Panel):
    """Главная панель аддона одной кнопки"""
    
    bl_label = "ArUco One-Click"
    bl_idname = "ARUCO_PT_oneclick_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco OneClick"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_oneclick_props
        
        # Заголовок
        box = layout.box()
        box.label(text="🎯 ArUco Pipeline в одной кнопке", icon='IMPORT')
        
        if not OPENCV_AVAILABLE:
            box.label(text="⚠️ OpenCV не установлен!", icon='ERROR')
            box.label(text="Установите: pip install opencv-python")
            return
        
        # Папка данных
        box = layout.box()
        box.label(text="📁 Папка с данными:", icon='FILE_FOLDER')
        
        row = box.row(align=True)
        row.prop(props, "data_folder", text="")
        row.operator("aruco.select_data_folder", text="", icon='FILEBROWSER')
        
        box.operator("aruco.auto_find_data", icon='VIEWZOOM')
        
        # Статус папки
        if props.data_folder and os.path.exists(props.data_folder):
            try:
                xmp_files = [f for f in os.listdir(props.data_folder) if f.lower().endswith('.xmp')]
                
                image_files = []
                for ext in ['jpg', 'jpeg', 'png', 'bmp']:
                    pattern = os.path.join(props.data_folder, f"*.{ext}")
                    image_files.extend(glob.glob(pattern))
                    pattern = os.path.join(props.data_folder, f"*.{ext.upper()}")
                    image_files.extend(glob.glob(pattern))
                
                if xmp_files and image_files:
                    box.label(text=f"✅ XMP: {len(xmp_files)}, Изображений: {len(image_files)}", icon='CHECKMARK')
                    can_process = True
                else:
                    box.label(text="❌ Нет XMP файлов или изображений", icon='ERROR')
                    can_process = False
            except:
                box.label(text="❌ Ошибка чтения папки", icon='ERROR')
                can_process = False
        else:
            box.label(text="❌ Папка не найдена", icon='ERROR')
            can_process = False
        
        # Главная кнопка
        layout.separator()
        row = layout.row()
        row.scale_y = 3.0
        
        if can_process and OPENCV_AVAILABLE:
            row.operator("aruco.oneclick_process", icon='PLAY')
        else:
            row.enabled = False
            if not OPENCV_AVAILABLE:
                row.operator("aruco.oneclick_process", text="❌ Установите OpenCV", icon='ERROR')
            else:
                row.operator("aruco.oneclick_process", text="❌ Укажите папку данных", icon='ERROR')
        
        # Очистка
        layout.separator()
        layout.operator("aruco.clear_aruco", icon='TRASH')

class ARUCO_PT_oneclick_settings(Panel):
    """Панель настроек"""
    
    bl_label = "⚙️ Настройки"
    bl_idname = "ARUCO_PT_oneclick_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco OneClick"
    bl_parent_id = "ARUCO_PT_oneclick_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_oneclick_props
        
        # Размеры изображений
        box = layout.box()
        box.label(text="🖼️ Изображения:")
        row = box.row(align=True)
        row.prop(props, "image_width")
        row.prop(props, "image_height")
        
        # Камеры
        box = layout.box()
        box.label(text="📷 Камеры:")
        box.prop(props, "camera_size")
        
        # Маркеры
        box = layout.box()
        box.label(text="🎯 Маркеры:")
        box.prop(props, "marker_type")
        box.prop(props, "marker_size")
        box.prop(props, "size_by_quality")
        box.prop(props, "color_by_quality")
        
        # Качество
        if props.color_by_quality or props.size_by_quality:
            col = box.column(align=True)
            col.prop(props, "quality_threshold_high")
            col.prop(props, "quality_threshold_medium")
        
        # Фильтрация
        box.separator()
        box.prop(props, "filter_by_quality")
        if props.filter_by_quality:
            box.prop(props, "min_quality_filter")
        
        # Очистка
        box = layout.box()
        box.label(text="🧹 Очистка:")
        box.prop(props, "clear_existing")

class ARUCO_PT_oneclick_info(Panel):
    """Информационная панель"""
    
    bl_label = "ℹ️ Информация"
    bl_idname = "ARUCO_PT_oneclick_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco OneClick"
    bl_parent_id = "ARUCO_PT_oneclick_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Статус OpenCV
        box = layout.box()
        if OPENCV_AVAILABLE:
            box.label(text="✅ OpenCV готов к работе", icon='CHECKMARK')
        else:
            box.label(text="❌ OpenCV не установлен", icon='ERROR')
            box.label(text="Установите в Blender Python:")
            box.label(text="pip install opencv-python")
        
        # Статистика сцены
        aruco_cameras = sum(1 for obj in bpy.data.objects if obj.get("ArUco_Camera"))
        aruco_markers = sum(1 for obj in bpy.data.objects if obj.get("ArUco_Marker"))
        
        if aruco_cameras > 0 or aruco_markers > 0:
            box = layout.box()
            box.label(text="📊 В сцене:")
            if aruco_cameras > 0:
                box.label(text=f"📷 Камер: {aruco_cameras}")
            if aruco_markers > 0:
                box.label(text=f"🎯 Маркеров: {aruco_markers}")
                
                # Качество маркеров
                high = sum(1 for obj in bpy.data.objects if obj.get("ArUco_Marker") and obj.get("quality") == "high")
                medium = sum(1 for obj in bpy.data.objects if obj.get("ArUco_Marker") and obj.get("quality") == "medium")
                low = aruco_markers - high - medium
                
                if high > 0:
                    box.label(text=f"🟢 Высокое качество: {high}")
                if medium > 0:
                    box.label(text=f"🟡 Среднее качество: {medium}")
                if low > 0:
                    box.label(text=f"🟠 Низкое качество: {low}")
        
        # Инструкция
        box = layout.box()
        box.label(text="📋 Использование:")
        col = box.column(align=True)
        col.label(text="1. Поместите XMP и изображения в папку data/")
        col.label(text="2. Нажмите 'Автопоиск' или выберите папку")
        col.label(text="3. Нажмите 'СТАРТ'")
        col.label(text="4. Ждите завершения обработки")
        col.label(text="5. Наслаждайтесь результатом!")

# =============================================================================
# РЕГИСТРАЦИЯ
# =============================================================================

classes = [
    # Свойства
    ArUcoOneClickProperties,
    
    # Операторы
    ARUCO_OT_oneclick_process,
    ARUCO_OT_select_data_folder,
    ARUCO_OT_auto_find_data,
    ARUCO_OT_clear_aruco,
    
    # Панели
    ARUCO_PT_oneclick_main,
    ARUCO_PT_oneclick_settings,
    ARUCO_PT_oneclick_info,
]

def register():
    """Регистрация аддона"""
    print("🚀 Registering ArUco One-Click Addon...")
    
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.aruco_oneclick_props = bpy.props.PointerProperty(
        type=ArUcoOneClickProperties
    )
    
    print("✅ ArUco One-Click Addon registered successfully!")

def unregister():
    """Отмена регистрации аддона"""
    print("🔻 Unregistering ArUco One-Click Addon...")
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    if hasattr(bpy.types.Scene, 'aruco_oneclick_props'):
        del bpy.types.Scene.aruco_oneclick_props
    
    print("✅ ArUco One-Click Addon unregistered.")

if __name__ == "__main__":
    register()