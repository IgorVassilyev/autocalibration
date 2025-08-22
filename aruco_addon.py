#!/usr/bin/env python3
"""
ArUco Markers Importer - Blender Addon
=======================================

Addon для импорта ArUco маркеров в Blender из файлов автокалибровки.

Установка:
1. Сохраните как aruco_importer.py
2. Blender → Edit → Preferences → Add-ons → Install
3. Выберите файл и активируйте addon
4. Панель появится в 3D Viewport → N → ArUco

Использование:
1. Запустите main.py для создания blender_aruco_markers.json
2. В Blender откройте панель ArUco (N → ArUco)
3. Нажмите "Import ArUco Markers"
"""

bl_info = {
    "name": "ArUco Markers Importer",
    "author": "ArUco Autocalibration Project", 
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "3D Viewport → Sidebar → ArUco",
    "description": "Import 3D ArUco markers from autocalibration pipeline",
    "category": "Import-Export",
    "doc_url": "https://github.com/your-project/aruco-autocalibration",
}

import bpy
import json
import os
import bmesh
from mathutils import Vector
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
    """Настройки импорта ArUco маркеров"""
    
    # Путь к файлу данных
    filepath: StringProperty(
        name="Файл данных",
        description="Путь к файлу blender_aruco_markers.json",
        default="",
        subtype='FILE_PATH'
    )
    
    # Настройки отображения
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
    
    # Фильтры качества
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
        name="Тип Empty",
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
    
    # Настройки коллекции
    collection_name: StringProperty(
        name="Имя коллекции",
        description="Имя коллекции для маркеров",
        default="ArUco_Markers"
    )
    
    clear_existing: BoolProperty(
        name="Очистить существующие",
        description="Удалить существующие маркеры перед импортом",
        default=True
    )


class ARUCO_OT_import_markers(Operator, ImportHelper):
    """Импорт ArUco маркеров из JSON файла"""
    
    bl_idname = "aruco.import_markers"
    bl_label = "Import ArUco Markers"
    bl_description = "Импорт 3D позиций ArUco маркеров в Blender"
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
        
        # Используем выбранный файл или из настроек
        data_file = self.filepath if hasattr(self, 'filepath') and self.filepath else props.filepath
        
        if not data_file:
            # Автопоиск файла
            data_file = self.find_data_file()
        
        if not data_file or not os.path.exists(data_file):
            self.report({'ERROR'}, f"Файл данных не найден: {data_file}")
            return {'CANCELLED'}
        
        # Обновляем путь в настройках
        props.filepath = data_file
        
        try:
            # Загрузка данных
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Импорт маркеров
            result = self.import_markers_from_data(data, props)
            
            if result['success']:
                self.report({'INFO'}, 
                    f"Импортировано {result['imported']}/{result['total']} маркеров "
                    f"(высокого качества: {result['high_quality']})"
                )
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"Ошибка импорта: {result['error']}")
                return {'CANCELLED'}
                
        except Exception as e:
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
    
    def import_markers_from_data(self, data, props):
        """Импорт маркеров из загруженных данных"""
        
        try:
            # Очистка существующих маркеров
            if props.clear_existing:
                self.clear_existing_markers(props.collection_name)
            
            # Создание/получение коллекции
            collection = self.get_or_create_collection(props.collection_name)
            
            # Импорт маркеров
            markers_data = data.get('markers', {})
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
                'high_quality': high_quality_count
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'imported': 0,
                'total': 0,
                'high_quality': 0
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
    
    def get_or_create_collection(self, collection_name):
        """Создание или получение коллекции"""
        
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
        else:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
        
        return collection
    
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
        importer = ARUCO_OT_import_markers()
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
    
    bl_label = "ArUco Markers"
    bl_idname = "ARUCO_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_importer
        
        # Заголовок
        layout.label(text="ArUco Markers Import", icon='TRACKING')
        
        # Выбор файла
        box = layout.box()
        box.label(text="Файл данных:")
        row = box.row(align=True)
        row.prop(props, "filepath", text="")
        row.operator("aruco.auto_find_file", text="", icon='VIEWZOOM')
        
        # Кнопка импорта
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        if props.filepath and os.path.exists(props.filepath):
            row.operator("aruco.import_markers", text="Import Markers", icon='IMPORT')
        else:
            op = row.operator("aruco.import_markers", text="Choose File & Import", icon='FILEBROWSER')
        
        # Информация о файле
        if props.filepath:
            if os.path.exists(props.filepath):
                layout.label(text=f"✓ {os.path.basename(props.filepath)}", icon='CHECKMARK')
            else:
                layout.label(text="✗ Файл не найден", icon='ERROR')


class ARUCO_PT_settings_panel(Panel):
    """Панель настроек импорта"""
    
    bl_label = "Настройки импорта"
    bl_idname = "ARUCO_PT_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ArUco"
    bl_parent_id = "ARUCO_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.aruco_importer
        
        # Настройки отображения
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
        
        # Настройки коллекции
        box = layout.box()
        box.label(text="Коллекция:")
        box.prop(props, "collection_name")
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
        
        # Информация о выбранных маркерах
        selected_markers = [obj for obj in context.selected_objects 
                          if obj.name.startswith('ArUco_Marker_')]
        
        if selected_markers:
            box = layout.box()
            box.label(text=f"Выбрано маркеров: {len(selected_markers)}")
            
            for marker in selected_markers[:5]:  # Показываем первые 5
                row = box.row()
                row.label(text=f"ID {marker.get('aruco_id', '?')}")
                row.label(text=f"Q: {marker.get('quality', '?')}")
            
            if len(selected_markers) > 5:
                box.label(text=f"... и еще {len(selected_markers) - 5}")


# Регистрация классов
classes = [
    ArUcoImporterProperties,
    ARUCO_OT_import_markers,
    ARUCO_OT_auto_find_file,
    ARUCO_OT_create_preview_mesh,
    ARUCO_PT_main_panel,
    ARUCO_PT_settings_panel,
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
    
    print("ArUco Markers Importer addon зарегистрирован")


def unregister():
    """Отмена регистрации addon"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Удаление свойств
    del bpy.types.Scene.aruco_importer
    
    print("ArUco Markers Importer addon удален")


if __name__ == "__main__":
    register()