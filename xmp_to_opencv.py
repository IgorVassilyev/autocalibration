# xmp_to_opencv.py - Преобразование XMP → OpenCV параметры

"""
ПРЕОБРАЗОВАНИЕ #1: RealityCapture XMP → OpenCV
==============================================

Преобразует параметры камеры из формата RealityCapture XMP
в формат матрицы камеры OpenCV с учетом размеров изображения.

Основная функция:
- convert_cameras_to_opencv(xmp_cameras, image_size) → opencv_cameras
"""

import numpy as np
from typing import Dict, Tuple

class XMPToOpenCVConverter:
    """Конвертер параметров камеры: XMP → OpenCV"""
    
    def __init__(self, sensor_width_35mm: float = 36.0):
        """
        Parameters:
        -----------
        sensor_width_35mm : float
            Ширина стандартного полнокадрового сенсора в мм
        """
        self.sensor_width_35mm = sensor_width_35mm
    
# xmp_to_opencv.py - Преобразование XMP → OpenCV параметры

"""
ПРЕОБРАЗОВАНИЕ #1: RealityCapture XMP → OpenCV
==============================================

Преобразует параметры камеры из формата RealityCapture XMP
в формат матрицы камеры OpenCV с учетом размеров изображения.

Основная функция:
- convert_cameras_to_opencv(xmp_cameras, image_size) → opencv_cameras
"""

import numpy as np
from typing import Dict, Tuple

class XMPToOpenCVConverter:
    """Конвертер параметров камеры: XMP → OpenCV"""
    
    def __init__(self, sensor_width_35mm: float = 36.0):
        """
        Parameters:
        -----------
        sensor_width_35mm : float
            Ширина стандартного полнокадрового сенсора в мм
        """
        self.sensor_width_35mm = sensor_width_35mm
    
    def convert_single_camera(self, camera_id: str, xmp_data: Dict, 
                            image_size: Tuple[int, int]) -> Dict:
        """
        Преобразование одной камеры из XMP в OpenCV формат
        
        Parameters:
        -----------
        camera_id : str
            Идентификатор камеры
        xmp_data : dict
            Данные из XMP файла (от SimpleXMPParser)
        image_size : tuple
            Размеры изображения (width, height) в пикселях
            
        Returns:
        --------
        dict
            Параметры камеры в формате OpenCV
        """
        image_width, image_height = image_size
        
        # === ИЗВЛЕЧЕНИЕ XMP ДАННЫХ ===
        focal_length_35mm = xmp_data['focal_length']
        principal_point_u = xmp_data['principal_point_u']  # нормализованные [-1,+1]
        principal_point_v = xmp_data['principal_point_v']  # нормализованные [-1,+1]
        aspect_ratio = xmp_data['aspect_ratio']
        distortion_coeffs = xmp_data['distortion']
        
        # === ПРЕОБРАЗОВАНИЕ ФОКУСНОГО РАССТОЯНИЯ ===
        # Из миллиметров (35mm эквивалент) в пиксели
        fx_pixels = (focal_length_35mm / self.sensor_width_35mm) * image_width
        fy_pixels = fx_pixels * aspect_ratio
        
        # === ПРЕОБРАЗОВАНИЕ ГЛАВНОЙ ТОЧКИ ===
        # Из нормализованных координат [-1,+1] в пиксели [0, width/height]
        cx_pixels = image_width / 2 + principal_point_u * (image_width / 2)
        cy_pixels = image_height / 2 + principal_point_v * (image_height / 2)
        
        # === СОЗДАНИЕ МАТРИЦЫ КАМЕРЫ ===
        camera_matrix = np.array([
            [fx_pixels,      0.0, cx_pixels],
            [     0.0, fy_pixels, cy_pixels],
            [     0.0,      0.0,      1.0]
        ])
        
        # === СОХРАНЯЕМ ОРИГИНАЛЬНЫЕ ДАННЫЕ ===
        position = np.array(xmp_data['position'])
        rotation = np.array(xmp_data['rotation'])
        
        # === ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ ===
        validation_warnings = self._validate_opencv_params(
            fx_pixels, fy_pixels, cx_pixels, cy_pixels, 
            image_width, image_height
        )
        
        # === СОЗДАНИЕ РЕЗУЛЬТАТА ===
        return {
            # Основные OpenCV параметры
            'camera_matrix': camera_matrix,
            'distortion_coeffs': np.array(distortion_coeffs),
            'fx': fx_pixels,
            'fy': fy_pixels,
            'cx': cx_pixels,
            'cy': cy_pixels,
            'image_size': image_size,
            
            # Пространственные данные из XMP
            'position': position,
            'rotation': rotation,
            
            # Метаданные преобразования
            'original_focal_35mm': focal_length_35mm,
            'original_principal_u': principal_point_u,
            'original_principal_v': principal_point_v,
            'conversion_warnings': validation_warnings
        }
    
    def convert_all_cameras(self, xmp_cameras: Dict, 
                          image_size: Tuple[int, int]) -> Dict:
        """
        Преобразование всех камер из XMP в OpenCV формат
        
        Parameters:
        -----------
        xmp_cameras : dict
            Словарь камер от SimpleXMPParser
        image_size : tuple
            Размеры изображения (width, height) в пикселях
            
        Returns:
        --------
        dict
            Словарь {camera_id: opencv_params}
        """
        opencv_cameras = {}
        
        for camera_id, xmp_data in xmp_cameras.items():
            try:
                opencv_params = self.convert_single_camera(camera_id, xmp_data, image_size)
                opencv_cameras[camera_id] = opencv_params
                
                # Выводим предупреждения если есть
                warnings = opencv_params['conversion_warnings']
                if warnings:
                    print(f"   Предупреждения {camera_id}: {'; '.join(warnings)}")
                
            except Exception as e:
                print(f"   Ошибка преобразования {camera_id}: {e}")
                continue
        
        return opencv_cameras
    
    def _validate_opencv_params(self, fx: float, fy: float, cx: float, cy: float,
                              width: int, height: int) -> list:
        """Валидация параметров OpenCV"""
        
        warnings = []
        
        # 1. Проверка фокусного расстояния
        if not (500 <= fx <= 10000):
            warnings.append(f"Необычное фокусное расстояние fx={fx:.1f}")
        if not (500 <= fy <= 10000):
            warnings.append(f"Необычное фокусное расстояние fy={fy:.1f}")
        
        # 2. Проверка главной точки
        if not (0 <= cx <= width):
            warnings.append(f"Главная точка cx={cx:.1f} вне изображения")
        if not (0 <= cy <= height):
            warnings.append(f"Главная точка cy={cy:.1f} вне изображения")
        
        # 3. Проверка соотношения fx/fy
        ratio_diff = abs(fx - fy) / max(fx, fy)
        if ratio_diff > 0.05:  # больше 5%
            warnings.append(f"Большая разница fx/fy: {ratio_diff*100:.1f}%")
        
        # 4. Проверка смещения главной точки от центра
        center_x, center_y = width/2, height/2
        offset_x = abs(cx - center_x) / center_x
        offset_y = abs(cy - center_y) / center_y
        
        if offset_x > 0.1:  # больше 10%
            warnings.append(f"Большое смещение главной точки по X: {offset_x*100:.1f}%")
        if offset_y > 0.1:  # больше 10%
            warnings.append(f"Большое смещение главной точки по Y: {offset_y*100:.1f}%")
        
        return warnings

# Удобные функции для использования

def convert_cameras_to_opencv(xmp_cameras: Dict, image_size: Tuple[int, int]) -> Dict:
    """
    Главная функция преобразования камер XMP → OpenCV
    
    Parameters:
    -----------
    xmp_cameras : dict
        Результат от SimpleXMPParser.load_all_cameras()
    image_size : tuple
        Размеры изображения (width, height) в пикселях
        
    Returns:
    --------
    dict
        Словарь камер с OpenCV параметрами
        
    Example:
    --------
    >>> from xmp_parser import SimpleXMPParser
    >>> from xmp_to_opencv import convert_cameras_to_opencv
    >>> 
    >>> parser = SimpleXMPParser()
    >>> xmp_cameras = parser.load_all_cameras("data")
    >>> opencv_cameras = convert_cameras_to_opencv(xmp_cameras, (4032, 3024))
    """
    converter = XMPToOpenCVConverter()
    return converter.convert_all_cameras(xmp_cameras, image_size)

def print_conversion_example(camera_id: str, xmp_data: Dict, opencv_data: Dict):
    """
    Печать примера преобразования для одной камеры
    
    Полезно для отладки и понимания процесса
    """
    print(f"\nПРИМЕР ПРЕОБРАЗОВАНИЯ: {camera_id}")
    print("-" * 40)
    
    # XMP данные
    print("RealityCapture XMP:")
    print(f"   Фокусное расстояние: {xmp_data['focal_length']:.3f} mm (35mm эквивалент)")
    print(f"   Главная точка U: {xmp_data['principal_point_u']:.6f} (нормализованная)")
    print(f"   Главная точка V: {xmp_data['principal_point_v']:.6f} (нормализованная)")
    
    # OpenCV данные
    print("\nOpenCV результат:")
    print(f"   Размер изображения: {opencv_data['image_size'][0]}x{opencv_data['image_size'][1]} пикселей")
    print(f"   Фокусное расстояние fx: {opencv_data['fx']:.2f} пикселей")
    print(f"   Фокусное расстояние fy: {opencv_data['fy']:.2f} пикселей")
    print(f"   Главная точка cx: {opencv_data['cx']:.2f} пикселей")
    print(f"   Главная точка cy: {opencv_data['cy']:.2f} пикселей")
    
    # Матрица камеры
    matrix = opencv_data['camera_matrix']
    print(f"\n   Матрица камеры:")
    print(f"   [{matrix[0][0]:7.1f},     0.0, {matrix[0][2]:7.1f}]")
    print(f"   [    0.0, {matrix[1][1]:7.1f}, {matrix[1][2]:7.1f}]")
    print(f"   [    0.0,     0.0,     1.0]")
    
    # Предупреждения
    warnings = opencv_data['conversion_warnings']
    if warnings:
        print(f"\n   Предупреждения:")
        for warning in warnings:
            print(f"      • {warning}")
    else:
        print(f"\n   Все проверки пройдены")

# Функция для тестирования модуля
def test_conversion():
    """Тест модуля преобразования"""
    
    print("ТЕСТ МОДУЛЯ XMP → OpenCV")
    print("=" * 35)
    
    # Создаем тестовые XMP данные
    test_xmp_data = {
        'focal_length': 36.28,
        'principal_point_u': -0.004,
        'principal_point_v': 0.012,
        'aspect_ratio': 1.0,
        'distortion': [-0.35, 0.014, 0.35, 0, 0, 0],
        'position': [1.2, -0.8, 2.1],
        'rotation': [
            [0.971, -0.001, 0.239],
            [0.047, -0.980, -0.193],
            [0.234, 0.199, -0.952]
        ]
    }
    
    test_image_size = (4032, 3024)
    
    # Тестируем преобразование
    converter = XMPToOpenCVConverter()
    opencv_data = converter.convert_single_camera("test_camera", test_xmp_data, test_image_size)
    
    # Показываем результат
    print_conversion_example("test_camera", test_xmp_data, opencv_data)
    
    print(f"\nТест завершен успешно!")

if __name__ == "__main__":
    test_conversion()