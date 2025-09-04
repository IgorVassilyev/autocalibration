#!/usr/bin/env python3
"""
ArUco Simple Addon - Аддон без OpenCV
====================================

Простой аддон для импорта камер и маркеров в Blender:
1. Импорт камер из XMP файлов (без зависимостей)
2. Импорт маркеров из готового JSON файла (созданного main.py)

Установка:
1. Сохраните как aruco_addon_simple.py
2. Blender → Edit → Preferences → Add-ons → Install
3. Активируйте "ArUco Simple Addon"

Использование:
- Сначала запустите main.py для создания JSON с маркерами
- Затем используйте аддон для импорта в Blender
"""

bl_info = {
    "name": "ArUco Simple Addon",
    "author": "ArUco Autocalibration Project", 
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport → Sidebar → ArUco Simple",
    "description": "Import cameras from XMP and markers from JSON (no OpenCV required)",
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

# =============================================================================
# ПАРСЕР XMP (БЕЗ ЗАВИСИМОСТЕЙ)
# =============================================================================

class SimpleXMPParser:
    """Простой парсер XMP файлов без зависимостей"""
    
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
# СВОЙСТВА АДДОНА
# =============================================================================

class ArUcoSimpleProperties(PropertyGroup):
    """Свойства простого аддона"""
    
    # Пути
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
    
    # Настройки отображения
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

# =============================================================================
# ОПЕРАТОРЫ
# =============================================================================

class ARUCO_OT_simple_import(Operator):
    """Простой импорт камер и маркеров"""
    
    bl_idname = "aruco.simple_import"
    bl_label = "Импортировать"
    bl_description = "Импорт камер из XMP и маркеров из JSON"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.aruco_simple_props
        
        try:
            # Очистка существующих данных
            if props.clear_existing:
                self.clear_existing()
            
            imported_cameras = 0
            imported_markers = 0
            
            # Импорт камер
            if props.import_cameras and props.xmp_folder and os.path.exists(props.xmp_folder):
                imported_cameras = self.import_cameras(props.xmp_folder)
            
            # Импорт маркеров
            if props.import_markers and props.markers_json and os.path.exists(props.markers_json):
                imported_markers = self.import_markers(props.markers_json, props)
            
            # Результат
            self.report({'INFO'}, 
                f"Импортировано: {imported_cameras} камер, {imported_markers} маркеров"
            )
            
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
            if obj.type == 'CAMERA' or obj.name.startswith('ArUco_Marker_'):
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Удаляем коллекции
        for coll_name in ['RealityCapture_Cameras', 'ArUco_Markers']:
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
            return 0
        
        markers_data = data.get('markers', {})
        if not markers_data:
            return 0
        
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
        
        return imported_count

class ARUCO_OT_select_markers_file(Operator, ImportHelper):
    """Выбор файла с маркерами"""
    
    bl_idname = "aruco.select_markers_file"
    bl_label = "Выбрать файл маркеров"
    bl_description = "Выбрать JSON файл с маркерами"
    
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    
    def execute(self, context):
        context.scene.aruco_simple_props.markers_json = self.filepath
        return {'FINISHED'}

class ARUCO_OT_auto_find_files(Operator):
    """Автопоиск файлов"""
    
    bl_idname = "aruco.auto_find_files"
    bl_label = "Автопоиск"
    bl_description = "Автоматически найти data/ и results/blender_aruco_markers.json"
    
    def execute(self, context):
        props = context.scene.aruco_simple_props
        
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
                # Проверяем что там есть XMP файлы
                xmp_files = [f for f in os.listdir(data_path) if f.lower().endswith('.xmp')]
                if xmp_files:
                    props.xmp_folder = data_path
                    found_data = True
            
            # Поиск JSON файла
            json_path = os.path.join(base_path, "results", "blender_aruco_markers.json")
            if os.path.exists(json_path) and not found_json:
                props.markers_json = json_path
                found_json = True
            
            # Также ищем прямо в папке
            direct_json = os.path.join(base_path, "blender_aruco_markers.json")
            if os.path.exists(direct_json) and not found_json:
                props.markers_json = direct_json
                found_json = True
        
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

# =============================================================================
# ИНТЕРФЕЙС
# =============================================================================

class ARUCO_PT_simple_main_panel(Panel):
    """Главная панель простого аддона"""
    
    bl_label = "ArUco Simple Import"
    bl_idname = "ARUCO_PT_simple_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco Simple"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_simple_props
        
        # Заголовок
        layout.label(text="Простой импорт ArUco", icon='IMPORT')
        
        # Автопоиск
        layout.operator("aruco.auto_find_files", icon='VIEWZOOM')
        layout.separator()
        
        # Пути к данным
        box = layout.box()
        box.label(text="Данные:")
        
        # XMP папка
        row = box.row(align=True)
        row.prop(props, "xmp_folder", text="")
        if props.xmp_folder and os.path.exists(props.xmp_folder):
            # Подсчет XMP файлов
            xmp_count = len([f for f in os.listdir(props.xmp_folder) 
                           if f.lower().endswith('.xmp')])
            box.label(text=f"✅ Найдено XMP файлов: {xmp_count}")
        else:
            box.label(text="❌ Папка XMP не найдена", icon='ERROR')
        
        # JSON файл
        row = box.row(align=True)
        row.prop(props, "markers_json", text="")
        row.operator("aruco.select_markers_file", text="", icon='FILEBROWSER')
        
        if props.markers_json and os.path.exists(props.markers_json):
            # Подсчет маркеров в JSON
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
        
        # Кнопка импорта
        layout.separator()
        row = layout.row()
        row.scale_y = 2.0
        
        can_import = ((props.import_cameras and props.xmp_folder and os.path.exists(props.xmp_folder)) or 
                     (props.import_markers and props.markers_json and os.path.exists(props.markers_json)))
        
        if can_import:
            row.operator("aruco.simple_import", icon='IMPORT')
        else:
            row.enabled = False
            row.operator("aruco.simple_import", text="Укажите файлы", icon='ERROR')

class ARUCO_PT_simple_settings_panel(Panel):
    """Панель настроек"""
    
    bl_label = "Настройки отображения"
    bl_idname = "ARUCO_PT_simple_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco Simple"
    bl_parent_id = "ARUCO_PT_simple_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_simple_props
        
        # Настройки маркеров
        box = layout.box()
        box.label(text="Маркеры:")
        box.prop(props, "marker_size")
        box.prop(props, "size_by_quality")
        box.prop(props, "color_by_quality")
        
        # Очистка
        box = layout.box()
        box.label(text="Очистка:")
        box.prop(props, "clear_existing")

class ARUCO_PT_simple_info_panel(Panel):
    """Информационная панель"""
    
    bl_label = "Информация"
    bl_idname = "ARUCO_PT_simple_info_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco Simple"
    bl_parent_id = "ARUCO_PT_simple_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Статистика сцены
        cameras_count = sum(1 for obj in bpy.data.objects if obj.type == 'CAMERA')
        markers_count = sum(1 for obj in bpy.data.objects if obj.name.startswith('ArUco_Marker_'))
        
        if cameras_count > 0 or markers_count > 0:
            box = layout.box()
            box.label(text="В сцене:")
            box.label(text=f"🎥 Камер: {cameras_count}")
            box.label(text=f"🏷️ Маркеров: {markers_count}")
            
            # Качество маркеров
            if markers_count > 0:
                high_quality = sum(1 for obj in bpy.data.objects 
                                 if obj.name.startswith('ArUco_Marker_') 
                                 and obj.get('quality') == 'high')
                medium_quality = sum(1 for obj in bpy.data.objects 
                                   if obj.name.startswith('ArUco_Marker_') 
                                   and obj.get('quality') == 'medium')
                low_quality = markers_count - high_quality - medium_quality
                
                box.label(text=f"🟢 Высокого качества: {high_quality}")
                box.label(text=f"🟡 Среднего качества: {medium_quality}")
                box.label(text=f"🟠 Низкого качества: {low_quality}")
        
        # Инструкции
        box = layout.box()
        box.label(text="💡 Инструкция:")
        box.label(text="1. Запустите main.py для создания JSON")
        box.label(text="2. Нажмите 'Автопоиск' или укажите пути")
        box.label(text="3. Выберите что импортировать")
        box.label(text="4. Нажмите 'Импортировать'")
        
        box.label(text="🎨 Цвета маркеров:")
        box.label(text="🟢 Зеленый = высокое качество")
        box.label(text="🟡 Желтый = среднее качество")
        box.label(text="🟠 Оранжевый = низкое качество")

# =============================================================================
# РЕГИСТРАЦИЯ
# =============================================================================

classes = [
    ArUcoSimpleProperties,
    ARUCO_OT_simple_import,
    ARUCO_OT_select_markers_file,
    ARUCO_OT_auto_find_files,
    ARUCO_PT_simple_main_panel,
    ARUCO_PT_simple_settings_panel,
    ARUCO_PT_simple_info_panel,
]

def register():
    """Регистрация аддона"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.aruco_simple_props = bpy.props.PointerProperty(
        type=ArUcoSimpleProperties
    )
    
    print("ArUco Simple Addon зарегистрирован")

def unregister():
    """Отмена регистрации аддона"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.aruco_simple_props
    
    print("ArUco Simple Addon удален")

if __name__ == "__main__":
    register()