"""
Упрощенный скрипт для детекции ArUco-маркеров и отображения в Blender
Показывает маркеры как Empty объекты и камеры для анализа
"""

import bpy
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from mathutils import Vector, Matrix

# Добавляем пользовательский путь для OpenCV
user_site_packages = r"C:\Users\admin\AppData\Roaming\Python\Python311\site-packages"
if user_site_packages not in sys.path:
    sys.path.insert(0, user_site_packages)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} готов к работе")
except ImportError:
    print("❌ OpenCV не найден. Проверьте установку.")
    exit()

class SimpleXMPParser:
    """Простой парсер XMP-файлов для извлечения данных камер"""
    
    def __init__(self):
        self.cameras_data = {}
        
    def parse_xmp_file(self, xmp_path):
        """Парсит XMP-файл и извлекает данные камеры"""
        try:
            tree = ET.parse(xmp_path)
            root = tree.getroot()
            
            # Ищем элемент Description
            ns = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                  'xcr': 'http://www.capturingreality.com/ns/xcr/1.1#'}
            
            desc = root.find('.//rdf:Description', ns)
            if desc is None:
                return None
            
            # Извлекаем основные параметры
            camera_data = {
                'file_path': xmp_path,
                'filename': os.path.basename(xmp_path),
                'position': self._parse_position(desc.find('xcr:Position', ns)),
                'rotation': self._parse_rotation(desc.find('xcr:Rotation', ns)),
                'focal_length': float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}FocalLength35mm', 35.0)),
                'principal_point_u': float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}PrincipalPointU', 0.0)),
                'principal_point_v': float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}PrincipalPointV', 0.0)),
                'distortion': self._parse_distortion(desc.find('xcr:DistortionCoeficients', ns)),
                'aspect_ratio': float(desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}AspectRatio', 1.0))
            }
            
            return camera_data
            
        except Exception as e:
            print(f"❌ Ошибка парсинга {xmp_path}: {e}")
            return None
    
    def _parse_position(self, pos_element):
        if pos_element is not None and pos_element.text:
            coords = list(map(float, pos_element.text.split()))
            return np.array(coords)
        return np.array([0.0, 0.0, 0.0])
    
    def _parse_rotation(self, rot_element):
        if rot_element is not None and rot_element.text:
            values = list(map(float, rot_element.text.split()))
            return np.array(values).reshape(3, 3)
        return np.eye(3)
    
    def _parse_distortion(self, dist_element):
        if dist_element is not None and dist_element.text:
            return np.array(list(map(float, dist_element.text.split())))
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def load_all_cameras(self, directory):
        """Загружает все камеры из директории"""
        xmp_files = [f for f in os.listdir(directory) if f.endswith('.xmp')]
        
        print(f"📁 Найдено {len(xmp_files)} XMP-файлов")
        
        for xmp_file in sorted(xmp_files):
            xmp_path = os.path.join(directory, xmp_file)
            camera_data = self.parse_xmp_file(xmp_path)
            
            if camera_data:
                camera_id = os.path.splitext(xmp_file)[0]
                self.cameras_data[camera_id] = camera_data
                print(f"✓ Загружена камера: {camera_id}")
        
        print(f"🎥 Всего камер: {len(self.cameras_data)}")
        return self.cameras_data

class SimpleArUcoDetector:
    """Простой детектор ArUco-маркеров"""
    
    def __init__(self):
        # Новый API OpenCV 4.7+
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # ОСЛАБЛЯЕМ параметры для лучшей детекции
        # Уменьшаем минимальный размер маркера
        self.parameters.minMarkerPerimeterRate = 0.01  # было 0.03, делаем меньше
        self.parameters.maxMarkerPerimeterRate = 8.0   # было 4.0, делаем больше
        
        # Ослабляем требования к точности углов
        self.parameters.polygonalApproxAccuracyRate = 0.05  # было 0.03, делаем мягче
        self.parameters.minCornerDistanceRate = 0.03       # было 0.05, делаем меньше
        
        # Уменьшаем требования к расстоянию от границ
        self.parameters.minDistanceToBorder = 1  # было 3, делаем меньше
        
        # Настраиваем адаптивную пороговую обработку
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.adaptiveThreshConstant = 5  # было 7, делаем мягче
        
        # Ослабляем другие ограничения
        self.parameters.minMarkerDistanceRate = 0.03  # минимальное расстояние между маркерами
        self.parameters.markerBorderBits = 1          # количество черных битов вокруг маркера
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        print("🔧 Настроены мягкие параметры детекции ArUco")
        
    def detect_markers(self, image_path):
        """Детектирует маркеры на изображении"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Не удалось загрузить: {image_path}")
                return {}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.detector.detectMarkers(gray)
            
            markers_info = {}
            
            if ids is not None:
                print(f"🎯 Найдено {len(ids)} маркеров в {os.path.basename(image_path)}")
                
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i][0]
                    center = np.mean(marker_corners, axis=0)
                    
                    # Вычисляем размер маркера для диагностики
                    distances = []
                    for j in range(4):
                        next_j = (j + 1) % 4
                        dist = np.linalg.norm(marker_corners[j] - marker_corners[next_j])
                        distances.append(dist)
                    avg_size = np.mean(distances)
                    
                    markers_info[int(marker_id)] = {
                        'corners': marker_corners,
                        'center': center,
                        'size': avg_size,
                        'image_path': image_path
                    }
                    
                    print(f"  📍 Маркер {marker_id}: центр ({center[0]:.0f}, {center[1]:.0f}), размер {avg_size:.0f}px")
            else:
                print(f"⚠️ Маркеры не найдены в {os.path.basename(image_path)}")
                print(f"   📋 Отклоненных кандидатов: {len(rejected)}")
                if len(rejected) > 0:
                    print("   💡 Совет: попробуйте еще более мягкие параметры детекции")
            
            return markers_info
            
        except Exception as e:
            print(f"❌ Ошибка детекции в {image_path}: {e}")
            return {}

class SimpleTriangulator:
    """Простая триангуляция для вычисления 3D-координат"""
    
    def __init__(self, cameras_data):
        self.cameras_data = cameras_data
        self.sensor_width = 36.0  # мм
        
    def create_camera_matrix(self, camera_data, image_width, image_height):
        """Создает матрицу камеры"""
        focal_length_mm = camera_data['focal_length'] * self.sensor_width / 36.0
        focal_length_px = focal_length_mm * image_width / self.sensor_width
        
        cx = image_width / 2.0 + camera_data['principal_point_u'] * image_width
        cy = image_height / 2.0 + camera_data['principal_point_v'] * image_height
        
        camera_matrix = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px * camera_data['aspect_ratio'], cy],
            [0, 0, 1]
        ])
        
        return camera_matrix
    
    def triangulate_marker(self, marker_observations, image_width=1920, image_height=1080):
        """Триангулирует 3D-позицию маркера"""
        if len(marker_observations) < 2:
            return None
        
        try:
            # Берем первые две камеры для простоты
            obs1, obs2 = marker_observations[0], marker_observations[1]
            
            cam1_data = self.cameras_data[obs1['camera_id']]
            cam2_data = self.cameras_data[obs2['camera_id']]
            
            # Создаем матрицы камер
            K1 = self.create_camera_matrix(cam1_data, image_width, image_height)
            K2 = self.create_camera_matrix(cam2_data, image_width, image_height)
            
            # Создаем матрицы проекции
            R1, t1 = cam1_data['rotation'], cam1_data['position']
            R2, t2 = cam2_data['rotation'], cam2_data['position']
            
            P1 = np.dot(K1, np.hstack([R1, t1.reshape(-1, 1)]))
            P2 = np.dot(K2, np.hstack([R2, t2.reshape(-1, 1)]))
            
            # Триангулируем
            point1 = obs1['center'].reshape(-1, 1)
            point2 = obs2['center'].reshape(-1, 1)
            
            point_4d = cv2.triangulatePoints(P1, P2, point1, point2)
            point_3d = point_4d[:3] / point_4d[3]
            
            return point_3d.flatten()
            
        except Exception as e:
            print(f"❌ Ошибка триангуляции: {e}")
            return None

class BlenderVisualizer:
    """Создание визуализации в Blender"""
    
    def __init__(self):
        self.clear_scene()
        
    def clear_scene(self):
        """Очищает сцену"""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
    
    def create_marker_empty(self, marker_id, position):
        """Создает Empty объект для маркера"""
        # Конвертируем координаты для Blender (Y инвертируется)
        blender_pos = [position[0], -position[1], position[2]]
        
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=blender_pos)
        empty = bpy.context.object
        empty.name = f"ArUco_Marker_{marker_id}"
        empty.empty_display_size = 0.1
        
        # Добавляем текстовую метку
        bpy.ops.object.text_add(location=(blender_pos[0], blender_pos[1], blender_pos[2] + 0.15))
        text_obj = bpy.context.object
        text_obj.name = f"Label_{marker_id}"
        text_obj.data.body = str(marker_id)
        text_obj.data.size = 0.1
        
        print(f"✓ Создан маркер {marker_id} в позиции {blender_pos}")
        
        return empty
    
    def create_camera_object(self, camera_id, camera_data):
        """Создает объект камеры"""
        # Конвертируем позицию для Blender
        pos = camera_data['position']
        blender_pos = [pos[0], -pos[1], pos[2]]
        
        bpy.ops.object.camera_add(location=blender_pos)
        camera_obj = bpy.context.object
        camera_obj.name = f"Camera_{camera_id}"
        
        # Настраиваем параметры камеры
        camera_obj.data.lens = camera_data['focal_length']
        camera_obj.data.sensor_width = 36.0
        
        print(f"✓ Создана камера {camera_id}")
        
        return camera_obj
    
    def create_coordinate_system(self):
        """Создает систему координат"""
        # Ось X - красная
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1.0, location=(0.5, 0, 0), rotation=(0, 1.5708, 0))
        x_axis = bpy.context.object
        x_axis.name = "X_Axis"
        
        # Ось Y - зеленая  
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1.0, location=(0, 0.5, 0), rotation=(1.5708, 0, 0))
        y_axis = bpy.context.object
        y_axis.name = "Y_Axis"
        
        # Ось Z - синяя
        bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1.0, location=(0, 0, 0.5))
        z_axis = bpy.context.object
        z_axis.name = "Z_Axis"
        
        print("✓ Создана система координат")

def main():
    """Главная функция"""
    
    print("🚀 Запуск простого детектора ArUco-маркеров")
    
    # НАСТРОЙКИ - ИЗМЕНИТЕ НА ВАШИ ДАННЫЕ
    DATA_DIRECTORY = r"C:\Users\admin\PyCharmProjects\autocalibration\data"  # Ваш путь
    IMAGE_WIDTH = 2592   # Ваше разрешение из диагностики
    IMAGE_HEIGHT = 1944  # Ваше разрешение из диагностики
    
    print(f"📁 Папка с данными: {DATA_DIRECTORY}")
    
    # Проверяем папку
    if not os.path.exists(DATA_DIRECTORY):
        print(f"❌ Папка не найдена: {DATA_DIRECTORY}")
        print("💡 Измените DATA_DIRECTORY на путь к вашей папке с файлами")
        return
    
    # Шаг 1: Загружаем камеры
    print("\n🎥 Шаг 1: Загрузка данных камер")
    parser = SimpleXMPParser()
    cameras_data = parser.load_all_cameras(DATA_DIRECTORY)
    
    if not cameras_data:
        print("❌ Камеры не загружены")
        return
    
    # Шаг 2: Детекция маркеров
    print("\n🔍 Шаг 2: Поиск ArUco-маркеров")
    detector = SimpleArUcoDetector()
    all_detections = {}
    
    for camera_id in cameras_data.keys():
        jpg_path = os.path.join(DATA_DIRECTORY, f"{camera_id}.jpg")
        
        if os.path.exists(jpg_path):
            markers = detector.detect_markers(jpg_path)
            
            for marker_id, marker_data in markers.items():
                if marker_id not in all_detections:
                    all_detections[marker_id] = []
                
                all_detections[marker_id].append({
                    'camera_id': camera_id,
                    'center': marker_data['center']
                })
    
    print(f"🎯 Найдено уникальных маркеров: {len(all_detections)}")
    
    # Шаг 3: Триангуляция
    print("\n📐 Шаг 3: Вычисление 3D-координат")
    triangulator = SimpleTriangulator(cameras_data)
    marker_positions = {}
    
    for marker_id, observations in all_detections.items():
        if len(observations) >= 2:
            position = triangulator.triangulate_marker(observations, IMAGE_WIDTH, IMAGE_HEIGHT)
            if position is not None:
                marker_positions[marker_id] = position
                print(f"✓ Маркер {marker_id}: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    
    print(f"📍 Успешно триангулировано: {len(marker_positions)} маркеров")
    
    # Шаг 4: Создание визуализации в Blender
    print("\n🎨 Шаг 4: Создание визуализации в Blender")
    visualizer = BlenderVisualizer()
    
    # Создаем систему координат
    visualizer.create_coordinate_system()
    
    # Создаем маркеры как Empty объекты
    for marker_id, position in marker_positions.items():
        visualizer.create_marker_empty(marker_id, position)
    
    # Создаем камеры
    for camera_id, camera_data in cameras_data.items():
        visualizer.create_camera_object(camera_id, camera_data)
    
    # Финальная статистика
    print(f"\n✨ Готово!")
    print(f"📊 Результаты:")
    print(f"   🎥 Камер: {len(cameras_data)}")
    print(f"   🎯 Найдено маркеров: {len(all_detections)}")
    print(f"   📍 Триангулировано: {len(marker_positions)}")
    print(f"   🎨 Объектов в Blender: {len(marker_positions)} маркеров + {len(cameras_data)} камер")
    
    if marker_positions:
        print(f"\n📍 Координаты найденных маркеров:")
        for marker_id, pos in sorted(marker_positions.items()):
            print(f"   Маркер {marker_id:2d}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")

# Запуск
if __name__ == "__main__":
    main()