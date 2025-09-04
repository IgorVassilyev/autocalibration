#!/usr/bin/env python3
"""
ArUco Markers + Cameras Importer - Blender Addon
================================================

Addon для импорта ArUco маркеров и камер в Blender из файлов автокалибровки.

Установка:
1. Сохраните как aruco_importer.py
2. Blender → Edit → Preferences → Add-ons → Install
3. Выберите файл и активируйте addon
4. Панель появится в 3D Viewport → N → ArUco

Использование:
1. Запустите main.py для создания blender_aruco_markers.json
2. В Blender откройте панель ArUco (N → ArUco)
3. Выберите что импортировать и нажмите "Import Scene"
"""

bl_info = {
    "name": "ArUco Markers + Cameras Importer",
    "author": "ArUco Autocalibration Project", 
    "version": (2, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport → Sidebar → ArUco",
    "description": "Import 3D ArUco markers and cameras from autocalibration pipeline",
    "category": "Import-Export",
    "doc_url": "https://github.com/your-project/aruco-autocalibration",
}

import bpy
import json
import os
import bmesh
from mathutils import Matrix, Vector
from bpy.props import (
    StringProperty, 
    BoolProperty, 
    FloatProperty, 
    EnumProperty,
    IntProperty
)
from bpy.types import Panel, Operator, PropertyGroup
from bpy_extras.io_utils import ImportHelper


class ArUcoImporterProperties(PropertyGroup):
    """Настройки импорта ArUco маркеров и камер"""
    
    # Путь к файлу данных
    filepath: StringProperty(
        name="Файл данных",
        description="Путь к файлу blender_aruco_markers.json",
        default="",
        subtype='FILE_PATH'
    )
    
    # Что импортировать
    import_cameras: BoolProperty(
        name="Импорт камер",
        description="Импортировать камеры из RealityCapture",
        default=True
    )
    
    import_markers: BoolProperty(
        name="Импорт маркеров", 
        description="Импортировать ArUco маркеры",
        default=True
    )
    
    # === НАСТРОЙКИ КАМЕР ===
    camera_size: FloatProperty(
        name="Размер камер",
        description="Размер отображения объектов камер",
        default=0.2,
        min=0.01,
        max=2.0
    )
    
    show_frustum: BoolProperty(
        name="Показать frustum",
        description="Отображать пирамиду видимости камеры",
        default=True
    )
    
    frustum_scale: FloatProperty(
        name="Масштаб frustum",
        description="Размер пирамиды видимости",
        default=1.0,
        min=0.1,
        max=5.0
    )
    
    camera_color_by_quality: BoolProperty(
        name="Цвет камер по качеству",
        description="Раскрашивать камеры по качеству калибровки",
        default=True
    )
    
    # === НАСТРОЙКИ МАРКЕРОВ ===
    marker_size: FloatProperty(
        name="Размер маркеров",
        description="Размер Empty объектов маркеров",
        default=0.1,
        min=0.01,
        max=1.0
    )
    
    size_by_quality: BoolProperty(
        name="Размер по качеству",
        description="Изменять размер в зависимости от качества триангуляции",
        default=True
    )
    
    color_by_quality: BoolProperty(
        name="Цвет по качеству", 
        description="Раскрашивать маркеры по качеству триангуляции",
        default=True
    )
    
    # Фильтры качества маркеров
    import_low_quality: BoolProperty(
        name="Импорт низкого качества",
        description="Импортировать маркеры с низким качеством триангуляции",
        default=True
    )
    
    min_confidence: FloatProperty(
        name="Мин. уверенность",
        description="Минимальная уверенность триангуляции (0-1)",
        default=0.0,
        min=0.0,
        max=1.0
    )
    
    # Отображение объектов
    empty_type: EnumProperty(
        name="Тип Empty маркеров",
        description="Тип Empty объекта для маркеров",
        items=[
            ('PLAIN_AXES', 'Оси', 'Простые оси'),
            ('ARROWS', 'Стрелки', 'Стрелки'),
            ('SINGLE_ARROW', 'Одна стрелка', 'Одна стрелка'),
            ('CIRCLE', 'Круг', 'Круг'),
            ('CUBE', 'Куб', 'Куб'),
            ('SPHERE', 'Сфера', 'Сфера'),
            ('CONE', 'Конус', 'Конус'),
        ],
        default='PLAIN_AXES'
    )
    
    # Настройки коллекций
    cameras_collection_name: StringProperty(
        name="Коллекция камер",
        description="Имя коллекции для камер",
        default="RC_Cameras"
    )
    
    markers_collection_name: StringProperty(
        name="Коллекция маркеров",
        description="Имя коллекции для маркеров",
        default="ArUco_Markers"
    )
    
    clear_existing: BoolProperty(
        name="Очистить существующие",
        description="Удалить существующие объекты перед импортом",
        default=True
    )


# === ФУНКЦИИ ИЗ ОРИГИНАЛЬНОГО СКРИПТА ===

def to_blender_cam_matrix(R_w2cv_3x3, C_world_vec3):
    """Преобразование матрицы камеры из OpenCV в Blender формат"""
    # OpenCV: +X right, +Y down, +Z forward
    # Blender cam: +X right, +Y up, +Z backward (looks along -Z)
    R_bcam2cv = Matrix(((1,0,0),(0,-1,0),(0,0,-1)))
    R_cv2bcam = R_bcam2cv.transposed()

    # world -> blender_cam
    R_w2bcam = R_cv2bcam @ R_w2cv_3x3

    # Blender needs object matrix (local->world): invert rotation to get blender_cam->world
    R_bcam2w = R_w2bcam.transposed()  # rotation, so inverse == transpose

    # Build 4x4 safely
    M = R_bcam2w.to_4x4()
    M.translation = Vector(C_world_vec3)
    return M


def ensure_collection(name):
    """Создание или получение коллекции"""
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


class ARUCO_OT_import_scene(Operator, ImportHelper):
    """Импорт камер и ArUco маркеров из JSON файла"""
    
    bl_idname = "aruco.import_scene"
    bl_label = "Import Scene"
    bl_description = "Импорт камер и 3D позиций ArUco маркеров в Blender"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Фильтр файлов
    filename_ext = ".json"
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        """Основная функция импорта"""
        
        props = context.scene.aruco_importer
        
        print(f"\n🚀 Начинаем импорт сцены ArUco")
        print(f"   Импорт камер: {props.import_cameras}")
        print(f"   Импорт маркеров: {props.import_markers}")
        
        # Используем выбранный файл или из настроек
        data_file = self.filepath if hasattr(self, 'filepath') and self.filepath else props.filepath
        print(f"   Файл данных: {data_file}")
        
        if not data_file:
            # Автопоиск файла
            data_file = self.find_data_file()
            print(f"   Автопоиск результат: {data_file}")
        
        if not data_file or not os.path.exists(data_file):
            self.report({'ERROR'}, f"Файл данных не найден: {data_file}")
            return {'CANCELLED'}
        
        # Обновляем путь в настройках
        props.filepath = data_file
        
        try:
            # Загрузка данных
            print(f"📖 Загрузка данных из файла...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   Ключи в данных: {list(data.keys())}")
            
            # Настройка единиц измерения
            bpy.context.scene.unit_settings.system = 'METRIC'
            bpy.context.scene.unit_settings.scale_length = 1.0
            print(f"   Единицы измерения настроены: METRIC")
            
            # Результаты импорта
            result = {'cameras_imported': 0, 'markers_imported': 0, 'success': True, 'error': ''}
            
            # Импорт камер
            if props.import_cameras:
                if 'cameras' in data:
                    print(f"\n📷 ИМПОРТ КАМЕР")
                    print(f"   Найдено камер в данных: {len(data['cameras'])}")
                    camera_result = self.import_cameras_from_data(data['cameras'], props)
                    result['cameras_imported'] = camera_result['imported']
                    if not camera_result['success']:
                        result['success'] = False
                        result['error'] += f"Камеры: {camera_result['error']}; "
                else:
                    print(f"   ⚠️  Данные камер не найдены в файле")
                    result['error'] += "Камеры: данные не найдены; "
            
            # Импорт маркеров
            if props.import_markers:
                if 'markers' in data:
                    print(f"\n🎯 ИМПОРТ МАРКЕРОВ")
                    print(f"   Найдено маркеров в данных: {len(data['markers'])}")
                    marker_result = self.import_markers_from_data(data['markers'], props)
                    result['markers_imported'] = marker_result['imported']
                    if not marker_result['success']:
                        result['success'] = False
                        result['error'] += f"Маркеры: {marker_result['error']}; "
                else:
                    print(f"   ⚠️  Данные маркеров не найдены в файле")
                    result['error'] += "Маркеры: данные не найдены; "
            
            print(f"\n🎉 ИМПОРТ ЗАВЕРШЕН")
            print(f"   Камер импортировано: {result['cameras_imported']}")
            print(f"   Маркеров импортировано: {result['markers_imported']}")
            print(f"   Успех: {result['success']}")
            if result['error']:
                print(f"   Ошибки: {result['error']}")
            
            if result['success']:
                self.report({'INFO'}, 
                    f"Импорт завершен: камер {result['cameras_imported']}, "
                    f"маркеров {result['markers_imported']}"
                )
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"Ошибки импорта: {result['error']}")
                return {'CANCELLED'}
                
        except Exception as e:
            print(f"💥 Критическая ошибка импорта: {e}")
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Ошибка чтения файла: {str(e)}")
            return {'CANCELLED'}
    
    def find_data_file(self):
        """Автоматический поиск файла данных"""
        
        possible_paths = [
            # В папке с blend файлом
            os.path.join(os.path.dirname(bpy.data.filepath), "blender_aruco_markers.json") if bpy.data.filepath else None,
            os.path.join(os.path.dirname(bpy.data.filepath), "results", "blender_aruco_markers.json") if bpy.data.filepath else None,
            
            # В текущей директории
            os.path.join(os.getcwd(), "blender_aruco_markers.json"),
            os.path.join(os.getcwd(), "results", "blender_aruco_markers.json"),
            
            # На рабочем столе
            os.path.join(os.path.expanduser("~"), "Desktop", "blender_aruco_markers.json"),
            os.path.join(os.path.expanduser("~"), "Desktop", "results", "blender_aruco_markers.json"),
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def import_cameras_from_data(self, cameras_data, props):
        """Импорт камер из загруженных данных"""
        
        try:
            # Очистка существующих камер
            if props.clear_existing:
                self.clear_existing_cameras(props.cameras_collection_name)
            
            # Создание/получение коллекции
            collection = ensure_collection(props.cameras_collection_name)
            
            imported_count = 0
            
            for camera_id, camera_info in cameras_data.items():
                try:
                    # Создание камеры
                    camera_obj = self.create_camera_object(camera_id, camera_info, props)
                    
                    if camera_obj:
                        # Добавление в коллекцию
                        collection.objects.link(camera_obj)
                        
                        # Убираем из Scene Collection
                        if camera_obj.name in bpy.context.scene.collection.objects:
                            bpy.context.scene.collection.objects.unlink(camera_obj)
                        
                        imported_count += 1
                        
                except Exception as e:
                    print(f"Ошибка создания камеры {camera_id}: {e}")
                    continue
            
            return {'success': True, 'imported': imported_count, 'error': ''}
            
        except Exception as e:
            return {'success': False, 'imported': 0, 'error': str(e)}
    
    def create_camera_object(self, camera_id, camera_info, props):
        """Создание объекта камеры в Blender"""
        
        try:
            pos = camera_info["position"]
            rot = camera_info["rotation"]  # 9-элементный список (3x3 матрица)
            attrs = camera_info["attributes"]
            validation = camera_info.get("validation", {'is_valid': True, 'warnings': [], 'errors': []})
            
            if len(pos) != 3 or len(rot) != 9:
                print(f"[WARN] Пропускаем {camera_id}: неверный формат Position/Rotation")
                return None
            
            # Преобразуем rotation в 3x3 Matrix
            C_world = Vector(pos)
            R_w2cv = Matrix((rot[0:3], rot[3:6], rot[6:9]))
            
            # Создание камеры
            cam_data_block = bpy.data.cameras.new(name=camera_id+"_DATA")
            cam_obj = bpy.data.objects.new(name=camera_id, object_data=cam_data_block)
            
            # Настройка матрицы трансформации
            cam_obj.matrix_world = to_blender_cam_matrix(R_w2cv, C_world)
            
            # === ВНУТРЕННИЕ ПАРАМЕТРЫ КАМЕРЫ ===
            f35 = float(attrs.get("FocalLength35mm", 0.0) or 0.0)
            if f35 > 0:
                cam_data_block.sensor_fit = 'HORIZONTAL'
                cam_data_block.sensor_width = 36.0
                cam_data_block.lens = f35
            
            # Главная точка (principal point)
            try:
                ppu = float(attrs.get("PrincipalPointU", ""))
                ppv = float(attrs.get("PrincipalPointV", ""))
                # Если значения малые (<0.05), считаем их смещением от центра
                if abs(ppu) < 0.05 and abs(ppv) < 0.05:
                    cam_data_block.shift_x = ppu
                    cam_data_block.shift_y = -ppv
                else:
                    # предполагаем диапазон [0..1]
                    cam_data_block.shift_x = (ppu - 0.5)
                    cam_data_block.shift_y = -(ppv - 0.5)
            except Exception:
                pass
            
            # Размер отображения
            cam_obj.empty_display_size = props.camera_size
            
            # === ЦВЕТ ПО КАЧЕСТВУ ===
            if props.camera_color_by_quality:
                if not validation['is_valid']:
                    cam_obj.color = (1.0, 0.0, 0.0, 1.0)  # Красный - ошибки
                elif validation['warnings']:
                    cam_obj.color = (1.0, 0.5, 0.0, 1.0)  # Оранжевый - предупреждения
                else:
                    cam_obj.color = (0.0, 0.0, 1.0, 1.0)  # Синий - валидная
            
            # === КАСТОМНЫЕ СВОЙСТВА ===
            cam_obj["camera_id"] = camera_id
            cam_obj["focal_length_35mm"] = f35
            cam_obj["principal_point"] = [attrs.get("PrincipalPointU", 0.0), attrs.get("PrincipalPointV", 0.0)]
            cam_obj["distortion_coefficients"] = camera_info.get("distortion", [])
            cam_obj["rc_attributes"] = attrs
            cam_obj["validation_status"] = validation
            
            # === СОЗДАНИЕ FRUSTUM ===
            if props.show_frustum:
                self.create_camera_frustum(cam_obj, props.frustum_scale, attrs)
            
            return cam_obj
            
        except Exception as e:
            print(f"Ошибка создания камеры {camera_id}: {e}")
            return None
    
    def create_camera_frustum(self, camera_obj, scale, attrs):
        """Создание пирамиды видимости камеры"""
        
        try:
            # Получаем параметры камеры
            focal_length = float(attrs.get("FocalLength35mm", 35.0))
            aspect_ratio = float(attrs.get("AspectRatio", 1.0))
            
            # Создание mesh для frustum
            mesh = bpy.data.meshes.new(f"{camera_obj.name}_Frustum")
            frustum_obj = bpy.data.objects.new(f"{camera_obj.name}_Frustum", mesh)
            
            # Создание геометрии frustum
            bm = bmesh.new()
            
            # Размеры на единичном расстоянии
            sensor_width = 36.0  # мм
            w = (sensor_width / focal_length) * scale
            h = w / aspect_ratio
            d = scale
            
            # Вершины frustum (пирамида)
            verts = [
                (0, 0, 0),      # Центр камеры
                (-w, -h, -d),   # Левый нижний
                (w, -h, -d),    # Правый нижний  
                (w, h, -d),     # Правый верхний
                (-w, h, -d),    # Левый верхний
            ]
            
            # Создание вершин
            for v in verts:
                bm.verts.new(v)
            
            bm.verts.ensure_lookup_table()
            
            # Создание ребер frustum
            edges = [
                (0, 1), (0, 2), (0, 3), (0, 4),  # От центра к углам
                (1, 2), (2, 3), (3, 4), (4, 1),  # Прямоугольник
            ]
            
            for edge in edges:
                bm.edges.new([bm.verts[edge[0]], bm.verts[edge[1]]])
            
            bm.to_mesh(mesh)
            bm.free()
            
            # Привязываем frustum к камере
            frustum_obj.parent = camera_obj
            frustum_obj.parent_type = 'OBJECT'
            
            # Настройки отображения
            frustum_obj.display_type = 'WIRE'
            frustum_obj.color = camera_obj.color
            
            # Добавляем в ту же коллекцию что и камера
            for collection in camera_obj.users_collection:
                collection.objects.link(frustum_obj)
                
        except Exception as e:
            print(f"Ошибка создания frustum для {camera_obj.name}: {e}")
    
    def clear_existing_cameras(self, collection_name):
        """Очистка существующих камер"""
        
        # Удаляем объекты камер и frustums
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA' or obj.name.endswith('_Frustum'):
                obj.select_set(True)
        bpy.ops.object.delete()
        
        # Удаляем коллекцию если есть
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
            bpy.data.collections.remove(collection)
    
    def import_markers_from_data(self, markers_data, props):
        """Импорт маркеров из загруженных данных"""
        
        try:
            # Очистка существующих маркеров
            if props.clear_existing:
                self.clear_existing_markers(props.markers_collection_name)
            
            # Создание/получение коллекции
            collection = ensure_collection(props.markers_collection_name)
            
            total_markers = len(markers_data)
            imported_count = 0
            high_quality_count = 0
            
            for marker_name, marker_info in markers_data.items():
                marker_id = marker_info['id']
                position = marker_info['position']
                confidence = marker_info['confidence']
                quality = marker_info.get('quality', 'unknown')
                
                # Фильтрация по качеству
                if confidence < props.min_confidence:
                    continue
                
                if not props.import_low_quality and quality == 'low':
                    continue
                
                # Создание маркера
                marker_obj = self.create_marker_object(
                    marker_id, position, confidence, quality, props
                )
                
                # Добавление в коллекцию
                collection.objects.link(marker_obj)
                
                # Убираем из Scene Collection
                if marker_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(marker_obj)
                
                imported_count += 1
                if quality == 'high':
                    high_quality_count += 1
            
            return {
                'success': True,
                'imported': imported_count,
                'total': total_markers,
                'high_quality': high_quality_count,
                'error': ''
            }
            
        except Exception as e:
            return {
                'success': False,
                'imported': 0,
                'total': 0,
                'high_quality': 0,
                'error': str(e)
            }
    
    def clear_existing_markers(self, collection_name):
        """Очистка существующих ArUco маркеров"""
        
        # Удаляем объекты маркеров
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.name.startswith('ArUco_Marker_'):
                obj.select_set(True)
        bpy.ops.object.delete()
        
        # Удаляем коллекцию если есть
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
            bpy.data.collections.remove(collection)
    
    def create_marker_object(self, marker_id, position, confidence, quality, props):
        """Создание объекта маркера"""
        
        # Создание Empty объекта
        bpy.ops.object.empty_add(
            type=props.empty_type,
            location=tuple(position)
        )
        
        marker_obj = bpy.context.active_object
        marker_obj.name = f"ArUco_Marker_{marker_id:02d}"
        
        # Размер в зависимости от настроек
        if props.size_by_quality:
            if quality == 'high':
                size = props.marker_size
            elif quality == 'medium':
                size = props.marker_size * 0.8
            else:
                size = props.marker_size * 0.6
        else:
            size = props.marker_size
        
        marker_obj.empty_display_size = size
        
        # Цвет в зависимости от качества
        if props.color_by_quality:
            if quality == 'high':
                marker_obj.color = (0.0, 1.0, 0.0, 1.0)  # Зеленый
            elif quality == 'medium':
                marker_obj.color = (1.0, 1.0, 0.0, 1.0)  # Желтый
            else:
                marker_obj.color = (1.0, 0.5, 0.0, 1.0)  # Оранжевый
        
        # Кастомные свойства
        marker_obj["aruco_id"] = marker_id
        marker_obj["confidence"] = confidence
        marker_obj["quality"] = quality
        marker_obj["triangulated_position"] = position
        
        return marker_obj


class ARUCO_OT_auto_find_file(Operator):
    """Автоматический поиск файла данных"""
    
    bl_idname = "aruco.auto_find_file"
    bl_label = "Auto Find"
    bl_description = "Автоматически найти файл blender_aruco_markers.json"
    
    def execute(self, context):
        props = context.scene.aruco_importer
        
        # Поиск файла
        importer = ARUCO_OT_import_scene()
        found_file = importer.find_data_file()
        
        if found_file:
            props.filepath = found_file
            self.report({'INFO'}, f"Найден файл: {os.path.basename(found_file)}")
        else:
            self.report({'WARNING'}, "Файл blender_aruco_markers.json не найден")
        
        return {'FINISHED'}


class ARUCO_OT_create_preview_mesh(Operator):
    """Создание preview mesh для маркеров"""
    
    bl_idname = "aruco.create_preview_mesh"
    bl_label = "Create Preview Meshes"
    bl_description = "Создать preview mesh объекты для выбранных маркеров"
    
    def execute(self, context):
        selected_markers = [obj for obj in context.selected_objects 
                          if obj.name.startswith('ArUco_Marker_')]
        
        if not selected_markers:
            self.report({'WARNING'}, "Выберите маркеры ArUco")
            return {'CANCELLED'}
        
        for marker in selected_markers:
            self.create_marker_mesh(marker)
        
        self.report({'INFO'}, f"Создано {len(selected_markers)} preview mesh")
        return {'FINISHED'}
    
    def create_marker_mesh(self, marker_obj):
        """Создание mesh представления маркера"""
        
        # Создание mesh
        mesh = bpy.data.meshes.new(f"ArUco_Mesh_{marker_obj['aruco_id']}")
        mesh_obj = bpy.data.objects.new(f"ArUco_Mesh_{marker_obj['aruco_id']}", mesh)
        
        # Создание геометрии квадрата
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=0.1)
        bm.to_mesh(mesh)
        bm.free()
        
        # Позиционирование
        mesh_obj.location = marker_obj.location
        mesh_obj.rotation_euler = marker_obj.rotation_euler
        
        # Добавление в ту же коллекцию
        for collection in marker_obj.users_collection:
            collection.objects.link(mesh_obj)


class ARUCO_PT_main_panel(Panel):
    """Главная панель ArUco импорта"""
    
    bl_label = "ArUco Scene Import"
    bl_idname = "ARUCO_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_importer
        
        # Заголовок
        layout.label(text="Cameras + Markers Import", icon='OUTLINER_OB_CAMERA')
        
        # Выбор файла
        box = layout.box()
        box.label(text="Файл данных:")
        row = box.row(align=True)
        row.prop(props, "filepath", text="")
        row.operator("aruco.auto_find_file", text="", icon='VIEWZOOM')
        
        # Что импортировать
        box = layout.box()
        box.label(text="Импортировать:")
        row = box.row()
        row.prop(props, "import_cameras")
        row.prop(props, "import_markers")
        
        # Кнопка импорта
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        if props.filepath and os.path.exists(props.filepath):
            row.operator("aruco.import_scene", text="Import Scene", icon='IMPORT')
        else:
            op = row.operator("aruco.import_scene", text="Choose File & Import", icon='FILEBROWSER')
        
        # Информация о файле
        if props.filepath:
            if os.path.exists(props.filepath):
                layout.label(text=f"✓ {os.path.basename(props.filepath)}", icon='CHECKMARK')
                
                # Показываем информацию из файла если можем
                try:
                    with open(props.filepath, 'r') as f:
                        data = json.load(f)
                    metadata = data.get('metadata', {})
                    
                    info_box = layout.box()
                    if 'total_cameras' in metadata:
                        info_box.label(text=f"📷 Камер: {metadata['total_cameras']}")
                    if 'total_markers' in metadata:
                        info_box.label(text=f"🎯 Маркеров: {metadata['total_markers']}")
                    if 'high_confidence_markers' in metadata:
                        info_box.label(text=f"🟢 Высокого качества: {metadata['high_confidence_markers']}")
                        
                except:
                    pass
            else:
                layout.label(text="✗ Файл не найден", icon='ERROR')


class ARUCO_PT_cameras_panel(Panel):
    """Панель настроек камер"""
    
    bl_label = "Настройки камер"
    bl_idname = "ARUCO_PT_cameras_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco"
    bl_parent_id = "ARUCO_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_importer
        
        layout.enabled = props.import_cameras
        
        # Настройки отображения камер
        box = layout.box()
        box.label(text="Отображение:")
        box.prop(props, "camera_size")
        box.prop(props, "show_frustum")
        if props.show_frustum:
            box.prop(props, "frustum_scale")
        box.prop(props, "camera_color_by_quality")
        
        # Коллекция
        box = layout.box()
        box.label(text="Организация:")
        box.prop(props, "cameras_collection_name")


class ARUCO_PT_markers_panel(Panel):
    """Панель настроек маркеров"""
    
    bl_label = "Настройки маркеров"
    bl_idname = "ARUCO_PT_markers_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco"
    bl_parent_id = "ARUCO_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_importer
        
        layout.enabled = props.import_markers
        
        # Настройки отображения маркеров
        box = layout.box()
        box.label(text="Отображение:")
        box.prop(props, "empty_type")
        box.prop(props, "marker_size")
        box.prop(props, "size_by_quality")
        box.prop(props, "color_by_quality")
        
        # Фильтры качества
        box = layout.box()
        box.label(text="Фильтры качества:")
        box.prop(props, "import_low_quality")
        box.prop(props, "min_confidence")
        
        # Коллекция
        box = layout.box()
        box.label(text="Организация:")
        box.prop(props, "markers_collection_name")


class ARUCO_PT_general_panel(Panel):
    """Панель общих настроек"""
    
    bl_label = "Общие настройки"
    bl_idname = "ARUCO_PT_general_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco"
    bl_parent_id = "ARUCO_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_importer
        
        # Общие настройки
        box = layout.box()
        box.label(text="Импорт:")
        box.prop(props, "clear_existing")


class ARUCO_PT_tools_panel(Panel):
    """Панель инструментов"""
    
    bl_label = "Инструменты"
    bl_idname = "ARUCO_PT_tools_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco"
    bl_parent_id = "ARUCO_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Инструменты для работы с маркерами
        layout.operator("aruco.create_preview_mesh", icon='MESH_CUBE')
        
        # Информация о выбранных объектах
        selected_cameras = [obj for obj in context.selected_objects 
                          if obj.type == 'CAMERA']
        selected_markers = [obj for obj in context.selected_objects 
                          if obj.name.startswith('ArUco_Marker_')]
        
        if selected_cameras or selected_markers:
            box = layout.box()
            box.label(text="Выбранные объекты:")
            
            if selected_cameras:
                box.label(text=f"📷 Камер: {len(selected_cameras)}")
                for cam in selected_cameras[:3]:  # Показываем первые 3
                    box.label(text=f"   {cam.name}")
                if len(selected_cameras) > 3:
                    box.label(text=f"   ... и еще {len(selected_cameras) - 3}")
            
            if selected_markers:
                box.label(text=f"🎯 Маркеров: {len(selected_markers)}")
                for marker in selected_markers[:3]:  # Показываем первые 3
                    quality = marker.get('quality', '?')
                    box.label(text=f"   ID {marker.get('aruco_id', '?')} ({quality})")
                if len(selected_markers) > 3:
                    box.label(text=f"   ... и еще {len(selected_markers) - 3}")


# Регистрация классов
classes = [
    ArUcoImporterProperties,
    ARUCO_OT_import_scene,
    ARUCO_OT_auto_find_file,
    ARUCO_OT_create_preview_mesh,
    ARUCO_PT_main_panel,
    ARUCO_PT_cameras_panel,
    ARUCO_PT_markers_panel,
    ARUCO_PT_general_panel,
    ARUCO_PT_tools_panel,
]


def register():
    """Регистрация addon"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Регистрация свойств
    bpy.types.Scene.aruco_importer = bpy.props.PointerProperty(
        type=ArUcoImporterProperties
    )
    
    print("ArUco Cameras + Markers Importer addon зарегистрирован")


def unregister():
    """Отмена регистрации addon"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Удаление свойств
    del bpy.types.Scene.aruco_importer
    
    print("ArUco Cameras + Markers Importer addon удален")


if __name__ == "__main__":
    register()