#!/usr/bin/env python3
"""
ArUco Complete Addon - Финальная версия
=======================================

Полнофункциональный аддон для импорта камер, маркеров и вычисления проектора:
1. Импорт камер из XMP файлов RealityCapture
2. Импорт маркеров из JSON файла как Empty объекты
3. Вычисление позиции проектора по маркерам
4. Создание проектора в Blender (камера/свет/empty)

Использование:
1. Запустите main.py для создания JSON с маркерами
2. Установите аддон в Blender
3. Используйте автопоиск или укажите пути вручную
4. Импортируйте данные и вычислите проектор
"""

bl_info = {
    "name": "ArUco Complete Addon",
    "author": "ArUco Autocalibration Project", 
    "version": (2, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport → Sidebar → ArUco Complete",
    "description": "Complete ArUco workflow: cameras, markers, and projector positioning",
    "category": "Import-Export",
}

import bpy
import os
import json
import xml.etree.ElementTree as ET
from mathutils import Matrix, Vector
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy.types import Panel, Operator, PropertyGroup
from bpy_extras.io_utils import ImportHelper
import traceback
import mathutils

# =============================================================================
# XMP ПАРСЕР
# =============================================================================

class SimpleXMPParser:
    """Простой парсер XMP файлов"""
    
    def __init__(self):
        self.RC_NS = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", 
            "xcr": "http://www.capturingreality.com/ns/xcr/1.1#"
        }
    
    def _floats(self, s):
        """Безопасное преобразование в список чисел"""
        if not s or s.strip() == "":
            return []
        try:
            return [float(x) for x in str(s).strip().split()]
        except:
            return []
    
    def parse_xmp_file(self, xmp_path):
        """Парсинг одного XMP файла"""
        try:
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            
            desc = root.find(".//rdf:Description", self.RC_NS)
            if desc is None:
                return None
            
            # Извлечение данных
            pos_elem = desc.find("xcr:Position", self.RC_NS)
            rot_elem = desc.find("xcr:Rotation", self.RC_NS)
            
            pos = self._floats(pos_elem.text if pos_elem is not None else "")
            rot = self._floats(rot_elem.text if rot_elem is not None else "")
            
            # Атрибуты
            attrs = {}
            for key, value in desc.attrib.items():
                if key.startswith('{' + self.RC_NS['xcr'] + '}'):
                    attr_name = key.split('}')[1]
                    attrs[attr_name] = value
            
            # Валидация
            if len(pos) != 3 or len(rot) != 9:
                return None
            
            return {
                'name': os.path.splitext(os.path.basename(xmp_path))[0],
                'position': pos,
                'rotation': rot,
                'focal_length': float(attrs.get('FocalLength35mm', 35.0)),
                'principal_point_u': float(attrs.get('PrincipalPointU', 0.0)),
                'principal_point_v': float(attrs.get('PrincipalPointV', 0.0)),
                'attrs': attrs
            }
            
        except Exception as e:
            print(f"Error parsing {xmp_path}: {e}")
            return None
    
    def load_all_cameras(self, directory):
        """Загрузка всех камер из папки"""
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

# =============================================================================
# ВЫЧИСЛЕНИЕ ПОЗИЦИИ ПРОЕКТОРА
# =============================================================================

class ProjectorCalculator:
    """Класс для вычисления позиции проектора по маркерам"""
    
    def calculate_projector_position(self, markers_data, settings):
        """
        Вычисление позиции проектора
        
        Args:
            markers_data: данные маркеров из JSON
            settings: настройки вычисления
        
        Returns:
            dict с данными проектора или None
        """
        try:
            # Фильтруем маркеры по качеству
            quality_markers = self._filter_markers_by_quality(markers_data, settings)
            
            if len(quality_markers) < 1:
                print(f"Недостаточно качественных маркеров")
                return None
            
            # Извлекаем 3D позиции
            positions = [Vector(marker['position']) for marker in quality_markers]
            
            # Вычисляем центроид
            centroid = self._calculate_centroid(positions)
            
            # Выбор метода размещения проектора
            method = settings.get('method', 'CENTROID')
            distance = settings.get('projector_distance', 2.0)
            
            if method == 'CENTROID':
                # Простой метод - прямо над центроидом
                projector_pos = Vector((centroid.x, centroid.y, centroid.z + distance))
                plane_normal = Vector((0, 0, 1))
                
            elif method == 'FRONT':
                # Спереди от центроида (по -Y)
                projector_pos = Vector((centroid.x, centroid.y - distance, centroid.z))
                plane_normal = Vector((0, 1, 0))
                
            elif method == 'BACK':
                # Сзади от центроида (по +Y)
                projector_pos = Vector((centroid.x, centroid.y + distance, centroid.z))
                plane_normal = Vector((0, -1, 0))
                
            elif method == 'CUSTOM':
                # Пользовательские координаты
                custom_offset = settings.get('custom_offset', [0, 0, 2])
                projector_pos = centroid + Vector(custom_offset)
                plane_normal = Vector((0, 0, 1))
                
            elif method == 'PLANE_FIT':
                # Подгонка плоскости к маркерам
                plane_normal, plane_center = self._fit_plane_to_points(positions)
                direction = plane_normal
                if settings.get('projector_side', 'front') == 'back':
                    direction = -direction
                projector_pos = plane_center + direction * distance
                
            else:
                # По умолчанию - над центроидом
                projector_pos = Vector((centroid.x, centroid.y, centroid.z + distance))
                plane_normal = Vector((0, 0, 1))
            
            # Вычисление ориентации
            projector_rotation = self._calculate_projector_orientation(
                projector_pos, centroid, plane_normal
            )
            
            result = {
                'position': projector_pos,
                'rotation': projector_rotation,
                'centroid': centroid,
                'plane_normal': plane_normal,
                'markers_used': len(quality_markers),
                'method': method,
                'marker_positions': positions
            }
            
            return result
            
        except Exception as e:
            print(f"Ошибка вычисления позиции проектора: {e}")
            return None
    
    def _filter_markers_by_quality(self, markers_data, settings):
        """Фильтрация маркеров по качеству"""
        quality_markers = []
        min_quality = settings.get('min_quality', 0.5)
        
        for marker_name, marker_info in markers_data.items():
            confidence = marker_info.get('confidence', 0.0)
            if confidence >= min_quality:
                quality_markers.append(marker_info)
        
        # Сортируем по уверенности (лучшие первыми)
        quality_markers.sort(key=lambda m: m.get('confidence', 0.0), reverse=True)
        
        return quality_markers
    
    def _calculate_centroid(self, positions):
        """Вычисление центроида точек"""
        if not positions:
            return Vector((0, 0, 0))
        
        centroid = Vector((0, 0, 0))
        for pos in positions:
            centroid += pos
        centroid /= len(positions)
        
        return centroid
    
    def _fit_plane_to_points(self, positions):
        """Подгонка плоскости к точкам"""
        if len(positions) < 3:
            return Vector((0, 0, 1)), self._calculate_centroid(positions)
        
        # Центрируем точки
        centroid = self._calculate_centroid(positions)
        centered_points = [pos - centroid for pos in positions]
        
        # Простой метод: находим нормаль через произведение векторов
        if len(centered_points) >= 2:
            # Находим два наиболее удаленных вектора
            max_distance = 0
            best_v1, best_v2 = centered_points[0], centered_points[1] if len(centered_points) > 1 else centered_points[0]
            
            for i, v1 in enumerate(centered_points):
                for j, v2 in enumerate(centered_points[i+1:], i+1):
                    distance = (v1 - v2).length
                    if distance > max_distance:
                        max_distance = distance
                        best_v1, best_v2 = v1, v2
            
            normal = best_v1.cross(best_v2)
            
            if normal.length > 1e-6:
                normal = normal.normalized()
                # Убеждаемся что нормаль смотрит "вверх"
                if normal.z < 0:
                    normal = -normal
            else:
                normal = Vector((0, 0, 1))
        else:
            normal = Vector((0, 0, 1))
        
        return normal, centroid
    
    def _calculate_projector_orientation(self, projector_pos, target_pos, up_vector):
        """Вычисление ориентации проектора (смотрит на цель)"""
        # Направление от проектора к цели
        direction = (target_pos - projector_pos).normalized()
        
        if direction.length < 0.001:
            direction = Vector((0, 0, -1))
        
        # Определяем up вектор
        world_up = Vector((0, 0, 1))
        if abs(direction.dot(world_up)) > 0.99:
            up = Vector((0, 1, 0))
        else:
            up = (world_up - direction * direction.dot(world_up)).normalized()
        
        right = direction.cross(up).normalized()
        up = right.cross(direction).normalized()
        
        # Для Blender камеры инвертируем направление
        forward = -direction
        
        rotation_matrix = Matrix((
            (right.x,   right.y,   right.z),
            (up.x,      up.y,      up.z),
            (forward.x, forward.y, forward.z)
        ))
        
        return rotation_matrix.to_euler()

# =============================================================================
# СВОЙСТВА АДДОНА
# =============================================================================

class ArUcoCompleteProperties(PropertyGroup):
    """Свойства полного аддона"""
    
    # Пути к данным
    xmp_folder: StringProperty(
        name="Папка XMP",
        description="Папка с XMP файлами RealityCapture",
        default="",
        subtype='DIR_PATH'
    )
    
    markers_json: StringProperty(
        name="Файл маркеров JSON",
        description="JSON файл с маркерами (создается main.py)",
        default="",
        subtype='FILE_PATH'
    )
    
    # Настройки импорта
    import_cameras: BoolProperty(
        name="Импортировать камеры",
        description="Импортировать камеры из XMP файлов",
        default=True
    )
    
    import_markers: BoolProperty(
        name="Импортировать маркеры", 
        description="Импортировать маркеры из JSON файла",
        default=True
    )
    
    import_projector: BoolProperty(
        name="Вычислить проектор",
        description="Автоматически вычислить и создать проектор",
        default=True
    )
    
    # Настройки отображения
    marker_size: FloatProperty(
        name="Размер маркеров",
        description="Размер Empty объектов маркеров",
        default=0.1,
        min=0.01, max=1.0
    )
    
    size_by_quality: BoolProperty(
        name="Размер по качеству",
        description="Изменять размер в зависимости от качества",
        default=True
    )
    
    color_by_quality: BoolProperty(
        name="Цвет по качеству",
        description="Раскрашивать маркеры по качеству",
        default=True
    )
    
    clear_existing: BoolProperty(
        name="Очистить существующие",
        description="Удалить предыдущие данные перед импортом",
        default=True
    )
    
    # Настройки проектора
    projector_method: EnumProperty(
        name="Метод размещения",
        description="Метод определения позиции проектора",
        items=[
            ('CENTROID', 'Над центроидом', 'Простое размещение над центром маркеров'),
            ('CUSTOM', 'Пользовательский', 'Пользовательское смещение от центроида'),
            ('PLANE_FIT', 'По плоскости', 'Подгонка плоскости к маркерам'),
            ('FRONT', 'Спереди', 'Спереди от центра маркеров'),
            ('BACK', 'Сзади', 'Сзади от центра маркеров'),
        ],
        default='CENTROID'
    )
    
    projector_distance: FloatProperty(
        name="Расстояние",
        description="Расстояние проектора от центроида/плоскости",
        default=2.0,
        min=0.1,
        max=10.0
    )
    
    projector_side: EnumProperty(
        name="Сторона проектора",
        description="С какой стороны плоскости разместить проектор",
        items=[
            ('front', 'Спереди', 'Проектор перед плоскостью маркеров'),
            ('back', 'Сзади', 'Проектор за плоскостью маркеров'),
        ],
        default='front'
    )
    
    # Пользовательское смещение
    custom_offset_x: FloatProperty(
        name="Смещение X",
        description="Пользовательское смещение по X от центроида",
        default=0.0,
        min=-10.0,
        max=10.0
    )
    
    custom_offset_y: FloatProperty(
        name="Смещение Y", 
        description="Пользовательское смещение по Y от центроида",
        default=0.0,
        min=-10.0,
        max=10.0
    )
    
    custom_offset_z: FloatProperty(
        name="Смещение Z",
        description="Пользовательское смещение по Z от центроида", 
        default=2.0,
        min=-10.0,
        max=10.0
    )
    
    projector_min_quality: FloatProperty(
        name="Мин. качество маркеров",
        description="Минимальное качество маркеров для вычисления проектора",
        default=0.5,
        min=0.0,
        max=1.0
    )
    
    projector_size: FloatProperty(
        name="Размер проектора",
        description="Размер отображения проектора",
        default=0.5,
        min=0.1,
        max=2.0
    )
    
    projector_type: EnumProperty(
        name="Тип проектора",
        description="Тип объекта для проектора",
        items=[
            ('CAMERA', 'Камера', 'Использовать объект камеры'),
            ('LIGHT_SPOT', 'Прожектор', 'Использовать spot light'),
            ('EMPTY', 'Empty', 'Простой Empty объект'),
        ],
        default='CAMERA'
    )
    
    # Визуализация
    create_plane_visual: BoolProperty(
        name="Показать плоскость",
        description="Создать визуализацию плоскости маркеров (только для метода по плоскости)",
        default=True
    )

# =============================================================================
# ОПЕРАТОРЫ
# =============================================================================

class ARUCO_OT_complete_import(Operator):
    """Полный импорт камер, маркеров и проектора"""
    
    bl_idname = "aruco.complete_import"
    bl_label = "Импортировать всё"
    bl_description = "Импорт камер, маркеров и вычисление проектора"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.aruco_complete_props
        
        try:
            # Очистка существующих данных
            if props.clear_existing:
                self.clear_existing()
            
            imported_cameras = 0
            imported_markers = 0
            projector_created = False
            
            # Импорт камер
            if props.import_cameras and props.xmp_folder and os.path.exists(props.xmp_folder):
                imported_cameras = self.import_cameras(props.xmp_folder)
            
            # Импорт маркеров
            markers_data = None
            if props.import_markers and props.markers_json and os.path.exists(props.markers_json):
                imported_markers, markers_data = self.import_markers(props.markers_json, props)
            
            # Вычисление проектора
            if props.import_projector and markers_data:
                projector_created = self.calculate_and_create_projector(markers_data, props)
            
            # Результат
            result_msg = f"Импортировано: {imported_cameras} камер, {imported_markers} маркеров"
            if projector_created:
                result_msg += ", проектор создан"
            
            self.report({'INFO'}, result_msg)
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка импорта: {str(e)}")
            traceback.print_exc()
            return {'CANCELLED'}
    
    def clear_existing(self):
        """Очистка существующих данных"""
        # Удаляем объекты
        objects_to_remove = []
        for obj in bpy.data.objects:
            if (obj.type == 'CAMERA' or 
                obj.name.startswith('ArUco_Marker_') or 
                obj.name.startswith('ArUco_Projector') or
                obj.name == 'ArUco_Markers_Plane'):
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Удаляем коллекции
        for coll_name in ['RealityCapture_Cameras', 'ArUco_Markers', 'ArUco_Projectors']:
            if coll_name in bpy.data.collections:
                bpy.data.collections.remove(bpy.data.collections[coll_name])
    
    def ensure_collection(self, name):
        """Создание или получение коллекции"""
        coll = bpy.data.collections.get(name)
        if not coll:
            coll = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(coll)
        return coll
    
    def to_blender_cam_matrix(self, R_w2cv_3x3, C_world_vec3):
        """Преобразование матрицы камеры в Blender"""
        try:
            # Преобразование координатных систем
            R_bcam2cv = Matrix(((1,0,0), (0,-1,0), (0,0,-1)))
            R_cv2bcam = R_bcam2cv.transposed()
            
            R_w2bcam = R_cv2bcam @ R_w2cv_3x3
            R_bcam2w = R_w2bcam.transposed()
            
            M = R_bcam2w.to_4x4()
            M.translation = Vector(C_world_vec3)
            
            return M
        except:
            return Matrix()
    
    def import_cameras(self, xmp_folder):
        """Импорт камер из XMP файлов"""
        parser = SimpleXMPParser()
        cameras = parser.load_all_cameras(xmp_folder)
        
        if not cameras:
            return 0
        
        cameras_collection = self.ensure_collection("RealityCapture_Cameras")
        imported_count = 0
        
        for camera_id, cam_data in cameras.items():
            try:
                # Создание камеры
                cam_data_block = bpy.data.cameras.new(camera_id + "_CAM")
                cam_obj = bpy.data.objects.new(camera_id, cam_data_block)
                
                cameras_collection.objects.link(cam_obj)
                
                # Матрица трансформации
                pos = Vector(cam_data['position'])
                rot_list = cam_data['rotation']
                rot = Matrix([rot_list[i:i+3] for i in range(0, 9, 3)])
                
                cam_obj.matrix_world = self.to_blender_cam_matrix(rot, pos)
                
                # Параметры камеры
                focal_35mm = cam_data['focal_length']
                if focal_35mm > 0:
                    cam_data_block.sensor_fit = 'HORIZONTAL'
                    cam_data_block.sensor_width = 36.0
                    cam_data_block.lens = focal_35mm
                
                # Principal point
                ppu = cam_data['principal_point_u']
                ppv = cam_data['principal_point_v']
                
                if abs(ppu) < 0.1 and abs(ppv) < 0.1:
                    cam_data_block.shift_x = ppu
                    cam_data_block.shift_y = -ppv
                
                # Сохраняем данные
                cam_obj["RC_camera_data"] = str(cam_data['attrs'])
                
                imported_count += 1
                
            except Exception as e:
                print(f"Error creating camera {camera_id}: {e}")
        
        return imported_count
    
    def import_markers(self, json_file, props):
        """Импорт маркеров из JSON файла"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return 0, None
        
        markers_data = data.get('markers', {})
        if not markers_data:
            return 0, None
        
        markers_collection = self.ensure_collection("ArUco_Markers")
        imported_count = 0
        
        for marker_name, marker_info in markers_data.items():
            try:
                marker_id = marker_info['id']
                position = marker_info['position']
                confidence = marker_info['confidence']
                quality = marker_info.get('quality', 'unknown')
                
                # Определение размера и цвета
                if props.size_by_quality:
                    if quality == 'high':
                        size = props.marker_size
                    elif quality == 'medium':
                        size = props.marker_size * 0.8
                    else:
                        size = props.marker_size * 0.6
                else:
                    size = props.marker_size
                
                if props.color_by_quality:
                    if quality == 'high':
                        color = (0.0, 1.0, 0.0, 1.0)  # Зеленый
                    elif quality == 'medium':
                        color = (1.0, 1.0, 0.0, 1.0)  # Желтый
                    else:
                        color = (1.0, 0.5, 0.0, 1.0)  # Оранжевый
                else:
                    color = (1.0, 1.0, 1.0, 1.0)  # Белый
                
                # Создание Empty объекта
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=tuple(position))
                marker_obj = bpy.context.active_object
                marker_obj.name = f"ArUco_Marker_{marker_id:02d}"
                marker_obj.empty_display_size = size
                marker_obj.color = color
                
                # Кастомные свойства
                marker_obj["aruco_id"] = marker_id
                marker_obj["confidence"] = confidence
                marker_obj["quality"] = quality
                marker_obj["triangulated_position"] = position
                
                # Перемещение в коллекцию
                markers_collection.objects.link(marker_obj)
                if marker_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(marker_obj)
                
                imported_count += 1
                
            except Exception as e:
                print(f"Error creating marker {marker_name}: {e}")
        
        return imported_count, markers_data
    
    def calculate_and_create_projector(self, markers_data, props):
        """Вычисление позиции проектора и создание объекта"""
        try:
            # Настройки для вычисления
            calc_settings = {
                'method': props.projector_method,
                'projector_distance': props.projector_distance,
                'projector_side': props.projector_side,
                'min_quality': props.projector_min_quality,
                'custom_offset': [props.custom_offset_x, props.custom_offset_y, props.custom_offset_z]
            }
            
            # Вычисляем позицию проектора
            calculator = ProjectorCalculator()
            projector_data = calculator.calculate_projector_position(markers_data, calc_settings)
            
            if not projector_data:
                print("Не удалось вычислить позицию проектора")
                return False
            
            # Создаем объект проектора
            return self.create_projector_object(projector_data, props)
            
        except Exception as e:
            print(f"Ошибка создания проектора: {e}")
            traceback.print_exc()
            return False
    
    def create_projector_object(self, projector_data, props):
        """Создание объекта проектора в Blender"""
        try:
            projectors_collection = self.ensure_collection("ArUco_Projectors")
            
            position = projector_data['position']
            rotation_euler = projector_data['rotation']
            
            # Создание визуализации плоскости если нужно
            if props.create_plane_visual and projector_data['method'] == 'PLANE_FIT':
                self.create_plane_visualization(projector_data, projectors_collection)
            
            if props.projector_type == 'CAMERA':
                # Создаем камеру-проектор
                proj_cam_data = bpy.data.cameras.new("ArUco_Projector_CAM")
                proj_obj = bpy.data.objects.new("ArUco_Projector", proj_cam_data)
                
                # Настройки камеры (широкий угол для проектора)
                proj_cam_data.lens = 20.0  # Широкий угол
                proj_cam_data.display_size = props.projector_size
                proj_cam_data.show_limits = True
                
            elif props.projector_type == 'LIGHT_SPOT':
                # Создаем spot light
                proj_light_data = bpy.data.lights.new("ArUco_Projector_Light", 'SPOT')
                proj_obj = bpy.data.objects.new("ArUco_Projector", proj_light_data)
                
                # Настройки света
                proj_light_data.energy = 100
                proj_light_data.spot_size = 1.0  # Широкий конус
                proj_light_data.show_cone = True
                
            else:  # EMPTY
                # Создаем Empty объект
                bpy.ops.object.empty_add(type='SINGLE_ARROW', location=tuple(position))
                proj_obj = bpy.context.active_object
                proj_obj.name = "ArUco_Projector"
                proj_obj.empty_display_size = props.projector_size
                proj_obj.color = (1.0, 0.0, 1.0, 1.0)  # Магента для проектора
            
            # Устанавливаем позицию и поворот
            proj_obj.location = position
            proj_obj.rotation_euler = rotation_euler
            
            # Перемещаем в коллекцию
            projectors_collection.objects.link(proj_obj)
            if proj_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(proj_obj)
            
            # Сохраняем данные вычислений
            proj_obj["projector_method"] = projector_data['method']
            proj_obj["markers_used"] = projector_data['markers_used']
            proj_obj["projector_distance"] = props.projector_distance
            proj_obj["centroid"] = projector_data['centroid'][:]
            
            return True
            
        except Exception as e:
            print(f"Ошибка создания объекта проектора: {e}")
            traceback.print_exc()
            return False
    
    def create_plane_visualization(self, projector_data, collection):
        """Создание визуализации плоскости маркеров"""
        try:
            plane_center = projector_data['centroid']
            marker_positions = projector_data['marker_positions']
            
            # Вычисляем размер плоскости
            max_distance = 0
            for pos in marker_positions:
                distance = (pos - plane_center).length
                max_distance = max(max_distance, distance)
            
            plane_size = max_distance * 2.0
            
            # Создаем mesh плоскости
            bpy.ops.mesh.primitive_plane_add(size=plane_size, location=tuple(plane_center))
            plane_obj = bpy.context.active_object
            plane_obj.name = "ArUco_Markers_Plane"
            
            # Полупрозрачный синий материал
            mat = bpy.data.materials.new(name="ArUco_Plane_Material")
            mat.use_nodes = True
            if mat.node_tree:
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                
                bsdf_node.inputs['Base Color'].default_value = (0.2, 0.5, 1.0, 1.0)
                bsdf_node.inputs['Alpha'].default_value = 0.3
                
                mat.node_tree.links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
                mat.blend_method = 'BLEND'
            
            plane_obj.data.materials.append(mat)
            
            # Перемещаем в коллекцию
            collection.objects.link(plane_obj)
            if plane_obj.name in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.unlink(plane_obj)
            
        except Exception as e:
            print(f"Ошибка создания визуализации плоскости: {e}")

class ARUCO_OT_calculate_projector_only(Operator):
    """Отдельное вычисление проектора"""
    
    bl_idname = "aruco.calculate_projector_only"
    bl_label = "Пересчитать проектор"
    bl_description = "Пересчитать позицию проектора по существующим маркерам"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.aruco_complete_props
        
        if not props.markers_json or not os.path.exists(props.markers_json):
            self.report({'ERROR'}, "Укажите файл с маркерами")
            return {'CANCELLED'}
        
        try:
            # Загружаем данные маркеров
            with open(props.markers_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            markers_data = data.get('markers', {})
            if not markers_data:
                self.report({'ERROR'}, "Данные маркеров не найдены")
                return {'CANCELLED'}
            
            # Удаляем существующие проекторы
            projectors_to_remove = [obj for obj in bpy.data.objects 
                                   if (obj.name.startswith('ArUco_Projector') or 
                                       obj.name == 'ArUco_Markers_Plane')]
            for obj in projectors_to_remove:
                bpy.data.objects.remove(obj, do_unlink=True)
            
            # Создаем новый проектор
            importer = ARUCO_OT_complete_import()
            success = importer.calculate_and_create_projector(markers_data, props)
            
            if success:
                self.report({'INFO'}, f"Проектор пересчитан методом {props.projector_method}")
            else:
                self.report({'ERROR'}, "Ошибка пересчета проектора")
                return {'CANCELLED'}
            
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка: {str(e)}")
            return {'CANCELLED'}

class ARUCO_OT_auto_find_files(Operator):
    """Автопоиск файлов"""
    
    bl_idname = "aruco.auto_find_files_complete"
    bl_label = "Автопоиск"
    bl_description = "Автоматически найти data/ и results/aruco_marker.json"
    
    def execute(self, context):
        props = context.scene.aruco_complete_props
        
        # Поиск относительно текущего blend файла или рабочей директории
        search_paths = []
        
        if bpy.data.filepath:
            blend_dir = os.path.dirname(bpy.data.filepath)
            search_paths.append(blend_dir)
            search_paths.append(os.path.dirname(blend_dir))
        
        search_paths.append(os.getcwd())
        
        found_data = False
        found_json = False
        
        for base_path in search_paths:
            # Поиск папки data
            data_path = os.path.join(base_path, "data")
            if os.path.exists(data_path) and not found_data:
                xmp_files = [f for f in os.listdir(data_path) if f.lower().endswith('.xmp')]
                if xmp_files:
                    props.xmp_folder = data_path
                    found_data = True
            
            # Поиск JSON файла
            json_candidates = [
                os.path.join(base_path, "results", "aruco_marker.json"),
                os.path.join(base_path, "aruco_marker.json"),
            ]
            
            for json_path in json_candidates:
                if os.path.exists(json_path) and not found_json:
                    props.markers_json = json_path
                    found_json = True
                    break
        
        # Сообщение о результатах
        if found_data and found_json:
            self.report({'INFO'}, "Найдены папка data и файл маркеров")
        elif found_data:
            self.report({'WARNING'}, "Найдена папка data, но не найден JSON с маркерами")
        elif found_json:
            self.report({'WARNING'}, "Найден JSON с маркерами, но не найдена папка data")
        else:
            self.report({'ERROR'}, "Файлы не найдены. Укажите пути вручную")
        
        return {'FINISHED'}

class ARUCO_OT_select_markers_file(Operator, ImportHelper):
    """Выбор файла с маркерами"""
    
    bl_idname = "aruco.select_markers_file_complete"
    bl_label = "Выбрать файл маркеров"
    bl_description = "Выбрать JSON файл с маркерами"
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        context.scene.aruco_complete_props.markers_json = self.filepath
        return {'FINISHED'}

# =============================================================================
# ИНТЕРФЕЙС
# =============================================================================

class ARUCO_PT_complete_main_panel(Panel):
    """Главная панель полного аддона"""
    
    bl_label = "ArUco Complete"
    bl_idname = "ARUCO_PT_complete_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco Complete"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_complete_props
        
        # Заголовок
        layout.label(text="🎯 ArUco Complete Pipeline", icon='IMPORT')
        
        # Автопоиск
        layout.operator("aruco.auto_find_files_complete", icon='VIEWZOOM')
        layout.separator()
        
        # Пути к данным
        box = layout.box()
        box.label(text="📁 Данные:")
        
        # XMP папка
        row = box.row(align=True)
        row.prop(props, "xmp_folder", text="")
        if props.xmp_folder and os.path.exists(props.xmp_folder):
            xmp_count = len([f for f in os.listdir(props.xmp_folder) 
                           if f.lower().endswith('.xmp')])
            box.label(text=f"✅ XMP файлов: {xmp_count}")
        else:
            box.label(text="❌ Папка XMP не найдена", icon='ERROR')
        
        # JSON файл
        row = box.row(align=True)
        row.prop(props, "markers_json", text="")
        row.operator("aruco.select_markers_file_complete", text="", icon='FILEBROWSER')
        
        if props.markers_json and os.path.exists(props.markers_json):
            try:
                with open(props.markers_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    markers_count = len(data.get('markers', {}))
                box.label(text=f"✅ Найдено маркеров: {markers_count}")
            except:
                box.label(text="❌ Ошибка чтения JSON", icon='ERROR')
        else:
            box.label(text="❌ JSON файл не найден", icon='ERROR')
        
        # Что импортировать
        box = layout.box()
        box.label(text="Импортировать:")
        box.prop(props, "import_cameras")
        box.prop(props, "import_markers")
        box.prop(props, "import_projector")
        
        # Кнопка импорта
        layout.separator()
        row = layout.row()
        row.scale_y = 2.0
        
        can_import = ((props.import_cameras and props.xmp_folder and os.path.exists(props.xmp_folder)) or 
                     (props.import_markers and props.markers_json and os.path.exists(props.markers_json)))
        
        if can_import:
            row.operator("aruco.complete_import", icon='IMPORT')
        else:
            row.enabled = False
            row.operator("aruco.complete_import", text="Укажите файлы", icon='ERROR')
        
        # Отдельная кнопка для пересчета проектора
        if props.markers_json and os.path.exists(props.markers_json):
            layout.separator()
            layout.operator("aruco.calculate_projector_only", icon='FILE_REFRESH')

class ARUCO_PT_complete_projector_panel(Panel):
    """Панель настроек проектора"""
    
    bl_label = "⚙️ Настройки проектора"
    bl_idname = "ARUCO_PT_complete_projector_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco Complete"
    bl_parent_id = "ARUCO_PT_complete_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_complete_props
        
        # Основные настройки
        box = layout.box()
        box.label(text="🎥 Проектор:")
        box.prop(props, "projector_type")
        box.prop(props, "projector_size")
        
        box.separator()
        box.prop(props, "projector_method")
        
        # Настройки в зависимости от метода
        if props.projector_method == 'CUSTOM':
            col = box.column(align=True)
            col.prop(props, "custom_offset_x")
            col.prop(props, "custom_offset_y")
            col.prop(props, "custom_offset_z")
        elif props.projector_method in ['CENTROID', 'FRONT', 'BACK']:
            box.prop(props, "projector_distance")
        elif props.projector_method == 'PLANE_FIT':
            box.prop(props, "projector_distance")
            box.prop(props, "projector_side")
            box.prop(props, "create_plane_visual")
        
        # Фильтрация маркеров
        box = layout.box()
        box.label(text="📏 Качество:")
        box.prop(props, "projector_min_quality")
        
        # Показываем информацию о готовности
        if props.markers_json and os.path.exists(props.markers_json):
            try:
                with open(props.markers_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                markers_data = data.get('markers', {})
                
                quality_markers = [m for m in markers_data.values() 
                                 if m.get('confidence', 0) >= props.projector_min_quality]
                
                box.label(text=f"Маркеров для вычисления: {len(quality_markers)}")
                
                if len(quality_markers) >= 1:
                    box.label(text="✅ Готов для вычисления", icon='CHECKMARK')
                else:
                    box.label(text="❌ Недостаточно маркеров!", icon='ERROR')
                    
            except:
                box.label(text="Ошибка анализа маркеров", icon='ERROR')

class ARUCO_PT_complete_settings_panel(Panel):
    """Панель настроек отображения"""
    
    bl_label = "Настройки отображения"
    bl_idname = "ARUCO_PT_complete_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco Complete"
    bl_parent_id = "ARUCO_PT_complete_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_complete_props
        
        # Настройки маркеров
        box = layout.box()
        box.label(text="🎯 Маркеры:")
        box.prop(props, "marker_size")
        box.prop(props, "size_by_quality")
        box.prop(props, "color_by_quality")
        
        # Очистка
        box = layout.box()
        box.label(text="🧹 Очистка:")
        box.prop(props, "clear_existing")

class ARUCO_PT_complete_info_panel(Panel):
    """Информационная панель"""
    
    bl_label = "ℹ️ Информация"
    bl_idname = "ARUCO_PT_complete_info_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco Complete"
    bl_parent_id = "ARUCO_PT_complete_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Статистика сцены
        cameras_count = sum(1 for obj in bpy.data.objects if obj.type == 'CAMERA')
        markers_count = sum(1 for obj in bpy.data.objects if obj.name.startswith('ArUco_Marker_'))
        projectors_count = sum(1 for obj in bpy.data.objects if obj.name.startswith('ArUco_Projector'))
        plane_exists = any(obj.name == 'ArUco_Markers_Plane' for obj in bpy.data.objects)
        
        if cameras_count > 0 or markers_count > 0 or projectors_count > 0:
            box = layout.box()
            box.label(text="📊 В сцене:")
            if cameras_count > 0:
                box.label(text=f"📷 Камер: {cameras_count}")
            if markers_count > 0:
                box.label(text=f"🎯 Маркеров: {markers_count}")
            if projectors_count > 0:
                box.label(text=f"🎥 Проекторов: {projectors_count}")
            if plane_exists:
                box.label(text="🔷 Плоскость маркеров: есть")
            
            # Качество маркеров
            if markers_count > 0:
                high_quality = sum(1 for obj in bpy.data.objects 
                                 if obj.name.startswith('ArUco_Marker_') 
                                 and obj.get('quality') == 'high')
                medium_quality = sum(1 for obj in bpy.data.objects 
                                   if obj.name.startswith('ArUco_Marker_') 
                                   and obj.get('quality') == 'medium')
                low_quality = markers_count - high_quality - medium_quality
                
                if high_quality > 0:
                    box.label(text=f"🟢 Высокого качества: {high_quality}")
                if medium_quality > 0:
                    box.label(text=f"🟡 Среднего качества: {medium_quality}")
                if low_quality > 0:
                    box.label(text=f"🟠 Низкого качества: {low_quality}")
        
        # Инструкции
        box = layout.box()
        box.label(text="📋 Использование:")
        col = box.column(align=True)
        col.label(text="1. Запустите main.py для JSON с маркерами")
        col.label(text="2. Нажмите 'Автопоиск' или укажите пути")
        col.label(text="3. Выберите что импортировать")
        col.label(text="4. Нажмите 'Импортировать всё'")
        col.label(text="5. При необходимости пересчитайте проектор")
        
        box.separator()
        col = box.column(align=True)
        col.label(text="🎨 Методы размещения проектора:")
        col.label(text="• Над центроидом - простой и надежный")
        col.label(text="• Пользовательский - точное управление")
        col.label(text="• По плоскости - для сложных случаев")
        col.label(text="• Спереди/Сзади - альтернативные позиции")

# =============================================================================
# РЕГИСТРАЦИЯ
# =============================================================================

classes = [
    ArUcoCompleteProperties,
    ARUCO_OT_complete_import,
    ARUCO_OT_calculate_projector_only,
    ARUCO_OT_auto_find_files,
    ARUCO_OT_select_markers_file,
    ARUCO_PT_complete_main_panel,
    ARUCO_PT_complete_projector_panel,
    ARUCO_PT_complete_settings_panel,
    ARUCO_PT_complete_info_panel,
]

def register():
    """Регистрация аддона"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.aruco_complete_props = bpy.props.PointerProperty(
        type=ArUcoCompleteProperties
    )
    
    print("ArUco Complete Addon зарегистрирован")

def unregister():
    """Отмена регистрации аддона"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.aruco_complete_props
    
    print("ArUco Complete Addon удален")

if __name__ == "__main__":
    register()