import os
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass


@dataclass
class MarkerDetection:
    """Структура для хранения данных о детектированном маркере"""
    marker_id: int
    center: Tuple[float, float]
    corners: np.ndarray
    confidence: float
    area: float


class EnhancedArUcoDetector:
    """
    Улучшенный детектор ArUco маркеров 4x4 с повышенной точностью
    для проекта автокалибровки камер.
    """

    def __init__(self, enable_logging: bool = True, debug_mode: bool = False):
        """
        Инициализация детектора
        
        Parameters:
        -----------
        enable_logging : bool
            Включить логирование процесса детекции
        debug_mode : bool
            Сохранять промежуточные изображения для отладки
        """
        # Настройка логирования
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)
        self.debug_mode = debug_mode
        
        # Инициализация детектора ArUco
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = self._create_detection_parameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Статистика детекции
        self.detection_stats = {
            'total_images_processed': 0,
            'total_markers_detected': 0,
            'markers_by_id': {},
            'failed_detections': []
        }

    def _create_detection_parameters(self) -> cv2.aruco.DetectorParameters:
        """
        Создание оптимизированных параметров детекции для данного проекта
        """
        params = cv2.aruco.DetectorParameters()
        
        # === ОСНОВНЫЕ ПАРАМЕТРЫ ===
        # Адаптивная пороговая обработка
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        
        # === КОНТУРНЫЙ АНАЛИЗ ===
        # Минимальный размер контура (в пикселях)
        params.minMarkerPerimeterRate = 0.03  # 3% от размера изображения
        params.maxMarkerPerimeterRate = 4.0   # максимум 400%
        
        # Точность аппроксимации контура
        params.polygonalApproxAccuracyRate = 0.03
        
        # === ФИЛЬТРАЦИЯ КАНДИДАТОВ ===
        # Минимальный размер углового региона
        params.minCornerDistanceRate = 0.05
        
        # Минимальное расстояние между маркерами
        params.minDistanceToBorder = 3
        
        # === КОДИРОВКА И КОРРЕКЦИЯ ===
        # Количество бит для коррекции ошибок
        params.maxErroneousBitsInBorderRate = 0.35
        
        # Минимальная дисперсия отсэмплированных точек
        params.minOtsuStdDev = 5.0
        
        # === УГЛОВАЯ ДЕТЕКЦИЯ ===
        # Точность детекции углов
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        
        return params

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения для улучшения детекции маркеров
        
        Parameters:
        -----------
        image : np.ndarray
            Входное изображение (цветное или серое)
            
        Returns:
        --------
        np.ndarray
            Обработанное серое изображение
        """
        # Конвертация в серый если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # === УЛУЧШЕНИЕ КОНТРАСТА ===
        # Адаптивная гистограммная эквализация
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # === ШУМОПОДАВЛЕНИЕ ===
        # Небольшое размытие для уменьшения шума
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Сохранение промежуточного результата для отладки
        if self.debug_mode:
            cv2.imwrite('debug_preprocessed.jpg', denoised)
            
        return denoised

    def detect_markers(self, image_path: str) -> Dict[int, MarkerDetection]:
        """
        Основная функция детекции маркеров с расширенной информацией
        
        Parameters:
        -----------
        image_path : str
            Путь к изображению для анализа
            
        Returns:
        --------
        Dict[int, MarkerDetection]
            Словарь детектированных маркеров {marker_id: MarkerDetection}
        """
        try:
            # Загрузка изображения
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                self.logger.error(f"Не удалось загрузить изображение: {image_path}")
                self.detection_stats['failed_detections'].append(image_path)
                return {}
            
            # Предобработка
            processed_image = self.preprocess_image(image)
            
            # Детекция маркеров
            corners, ids, rejected = self.detector.detectMarkers(processed_image)
            
            # Обработка результатов
            detections = {}
            
            if ids is not None and len(ids) > 0:
                self.logger.info(f"Найдено {len(ids)} маркеров в {os.path.basename(image_path)}")
                
                for i, marker_id in enumerate(ids.flatten()):
                    marker_corners = corners[i].reshape(4, 2)
                    
                    # Вычисление центра маркера
                    center = marker_corners.mean(axis=0)
                    
                    # Вычисление площади маркера
                    area = cv2.contourArea(marker_corners)
                    
                    # Вычисление "уверенности" (простая метрика на основе площади)
                    confidence = min(1.0, area / 10000.0)  # нормализация на типичную площадь
                    
                    # Создание объекта детекции
                    detection = MarkerDetection(
                        marker_id=int(marker_id),
                        center=(float(center[0]), float(center[1])),
                        corners=marker_corners,
                        confidence=confidence,
                        area=area
                    )
                    
                    detections[int(marker_id)] = detection
                    
                    # Обновление статистики
                    if int(marker_id) not in self.detection_stats['markers_by_id']:
                        self.detection_stats['markers_by_id'][int(marker_id)] = 0
                    self.detection_stats['markers_by_id'][int(marker_id)] += 1
                
                # Сохранение отладочного изображения с маркерами
                if self.debug_mode:
                    debug_image = image.copy()
                    cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
                    debug_filename = f"debug_detected_{os.path.basename(image_path)}"
                    cv2.imwrite(debug_filename, debug_image)
                    self.logger.info(f"Сохранено отладочное изображение: {debug_filename}")
                
            else:
                self.logger.warning(f"Маркеры не найдены в {os.path.basename(image_path)}")
                if len(rejected) > 0:
                    self.logger.info(f"Отклонено {len(rejected)} кандидатов")
            
            # Обновление общей статистики
            self.detection_stats['total_images_processed'] += 1
            self.detection_stats['total_markers_detected'] += len(detections)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке {image_path}: {e}")
            self.detection_stats['failed_detections'].append(image_path)
            return {}

    def detect_from_directory(self, directory: str, 
                            expected_markers: Optional[List[int]] = None) -> Dict[str, Dict[int, MarkerDetection]]:
        """
        Детекция маркеров во всех изображениях директории
        
        Parameters:
        -----------
        directory : str
            Путь к директории с изображениями
        expected_markers : List[int], optional
            Список ожидаемых ID маркеров для валидации
            
        Returns:
        --------
        Dict[str, Dict[int, MarkerDetection]]
            Результаты детекции {camera_id: {marker_id: MarkerDetection}}
        """
        if not os.path.exists(directory):
            self.logger.error(f"Директория не существует: {directory}")
            return {}
        
        # Поиск изображений
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in os.listdir(directory) 
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if not image_files:
            self.logger.warning(f"Изображения не найдены в {directory}")
            return {}
        
        self.logger.info(f"Обработка {len(image_files)} изображений из {directory}")
        
        all_detections = {}
        
        for image_file in sorted(image_files):
            camera_id = os.path.splitext(image_file)[0]
            image_path = os.path.join(directory, image_file)
            
            detections = self.detect_markers(image_path)
            all_detections[camera_id] = detections
            
            # Вывод результата для каждого изображения
            if detections:
                marker_ids = sorted(detections.keys())
                self.logger.info(f"  {camera_id}: маркеры {marker_ids}")
            else:
                self.logger.warning(f"  {camera_id}: маркеры не найдены")
        
        # Валидация против ожидаемых маркеров
        if expected_markers:
            self._validate_expected_markers(all_detections, expected_markers)
        
        return all_detections

    def _validate_expected_markers(self, detections: Dict[str, Dict[int, MarkerDetection]], 
                                 expected_markers: List[int]) -> None:
        """Валидация найденных маркеров против ожидаемого списка"""
        
        all_found_markers = set()
        for camera_detections in detections.values():
            all_found_markers.update(camera_detections.keys())
        
        expected_set = set(expected_markers)
        missing_markers = expected_set - all_found_markers
        unexpected_markers = all_found_markers - expected_set
        
        if missing_markers:
            self.logger.warning(f"Не найдены ожидаемые маркеры: {sorted(missing_markers)}")
        
        if unexpected_markers:
            self.logger.info(f"Найдены неожиданные маркеры: {sorted(unexpected_markers)}")
        
        self.logger.info(f"Найдено {len(all_found_markers)} из {len(expected_markers)} ожидаемых маркеров")

    def get_detection_statistics(self) -> Dict:
        """Получение статистики детекции"""
        stats = self.detection_stats.copy()
        
        if stats['total_images_processed'] > 0:
            stats['average_markers_per_image'] = (
                stats['total_markers_detected'] / stats['total_images_processed']
            )
        else:
            stats['average_markers_per_image'] = 0
        
        # Сортировка маркеров по частоте обнаружения
        if stats['markers_by_id']:
            sorted_markers = sorted(
                stats['markers_by_id'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            stats['most_detected_markers'] = sorted_markers[:5]
        
        return stats

    def export_detections_to_json(self, detections: Dict[str, Dict[int, MarkerDetection]], 
                                output_path: str) -> None:
        """
        Экспорт результатов детекции в JSON для дальнейшей обработки
        
        Parameters:
        -----------
        detections : dict
            Результаты детекции от detect_from_directory()
        output_path : str
            Путь для сохранения JSON файла
        """
        import json
        
        # Подготовка данных для JSON (numpy массивы не сериализуются)
        export_data = {}
        
        for camera_id, camera_detections in detections.items():
            export_data[camera_id] = {}
            
            for marker_id, detection in camera_detections.items():
                export_data[camera_id][marker_id] = {
                    'center': detection.center,
                    'corners': detection.corners.tolist(),
                    'confidence': detection.confidence,
                    'area': detection.area
                }
        
        # Добавление метаданных
        export_data['_metadata'] = {
            'detector_version': '2.0',
            'dictionary': 'DICT_4X4_50',
            'statistics': self.get_detection_statistics(),
            'detection_timestamp': str(cv2.utils.getTickCount())
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Результаты детекции сохранены в {output_path}")

    def print_detection_summary(self, detections: Dict[str, Dict[int, MarkerDetection]]) -> None:
        """Печать сводки результатов детекции"""
        
        print(f"\n{'='*60}")
        print("📊 СВОДКА ДЕТЕКЦИИ ARUCO МАРКЕРОВ")
        print(f"{'='*60}")
        
        # Общая статистика
        total_cameras = len(detections)
        total_detections = sum(len(camera_det) for camera_det in detections.values())
        
        print(f"🎥 Обработано камер: {total_cameras}")
        print(f"🏷️  Всего детекций: {total_detections}")
        
        if total_cameras > 0:
            print(f"📈 Среднее маркеров на камеру: {total_detections/total_cameras:.1f}")
        
        # Анализ по маркерам
        all_markers = set()
        marker_frequency = {}
        
        for camera_detections in detections.values():
            for marker_id in camera_detections.keys():
                all_markers.add(marker_id)
                marker_frequency[marker_id] = marker_frequency.get(marker_id, 0) + 1
        
        print(f"\n🔍 Уникальных маркеров найдено: {len(all_markers)}")
        if all_markers:
            print(f"🏷️  ID маркеров: {sorted(all_markers)}")
        
        # Частота обнаружения маркеров
        if marker_frequency:
            print(f"\n📊 ЧАСТОТА ОБНАРУЖЕНИЯ МАРКЕРОВ:")
            for marker_id in sorted(marker_frequency.keys()):
                frequency = marker_frequency[marker_id]
                percentage = (frequency / total_cameras) * 100
                print(f"   Маркер {marker_id:2d}: {frequency:2d}/{total_cameras} камер ({percentage:5.1f}%)")
        
        # Детекция по камерам
        print(f"\n📷 ДЕТЕКЦИЯ ПО КАМЕРАМ:")
        for camera_id in sorted(detections.keys()):
            camera_detections = detections[camera_id]
            if camera_detections:
                marker_list = sorted(camera_detections.keys())
                print(f"   {camera_id}: {len(marker_list)} маркеров {marker_list}")
            else:
                print(f"   {camera_id}: маркеры не найдены ❌")
        
        # Статистика детектора
        stats = self.get_detection_statistics()
        print(f"\n🔧 СТАТИСТИКА ДЕТЕКТОРА:")
        print(f"   Обработано изображений: {stats['total_images_processed']}")
        print(f"   Общее количество детекций: {stats['total_markers_detected']}")
        if stats.get('failed_detections'):
            print(f"   Ошибки обработки: {len(stats['failed_detections'])}")


# Удобные функции для использования в проекте

def detect_markers_simple(image_path: str) -> Dict[int, Tuple[float, float]]:
    """
    Простая функция детекции маркеров (совместимость с исходным API)
    
    Returns:
    --------
    Dict[int, Tuple[float, float]]
        Словарь {marker_id: (center_x, center_y)}
    """
    detector = EnhancedArUcoDetector(enable_logging=False)
    detections = detector.detect_markers(image_path)
    
    # Преобразование к простому формату
    return {
        marker_id: detection.center 
        for marker_id, detection in detections.items()
    }


def detect_all_markers_in_directory(directory: str, 
                                  expected_count: int = 13,
                                  debug_mode: bool = False) -> Dict[str, Dict[int, MarkerDetection]]:
    """
    Основная функция для детекции всех маркеров в проекте
    
    Parameters:
    -----------
    directory : str
        Директория с изображениями
    expected_count : int
        Ожидаемое количество уникальных маркеров
    debug_mode : bool
        Режим отладки (сохранение промежуточных изображений)
        
    Returns:
    --------
    Dict[str, Dict[int, MarkerDetection]]
        Полные результаты детекции
    """
    print(f"🚀 Запуск детекции ArUco маркеров в {directory}")
    print(f"🎯 Ожидается найти ~{expected_count} уникальных маркеров")
    
    detector = EnhancedArUcoDetector(enable_logging=True, debug_mode=debug_mode)
    
    # Детекция
    detections = detector.detect_from_directory(directory)
    
    # Анализ результатов
    detector.print_detection_summary(detections)
    
    # Проверка на ожидаемое количество
    unique_markers = set()
    for camera_detections in detections.values():
        unique_markers.update(camera_detections.keys())
    
    if len(unique_markers) >= expected_count:
        print(f"\n✅ Успех! Найдено {len(unique_markers)} уникальных маркеров")
    else:
        print(f"\n⚠️  Внимание: найдено только {len(unique_markers)} из {expected_count} ожидаемых маркеров")
    
    return detections


# Функция тестирования модуля
def test_detector():
    """Тест детектора ArUco маркеров"""
    
    print("🧪 ТЕСТ ДЕТЕКТОРА ARUCO МАРКЕРОВ")
    print("=" * 40)
    
    # Создание тестового детектора
    detector = EnhancedArUcoDetector(enable_logging=True, debug_mode=True)
    
    # Тест параметров
    params = detector.parameters
    print(f"✓ Параметры детекции настроены")
    print(f"  - Словарь: DICT_4X4_50")
    print(f"  - Метод уточнения углов: {params.cornerRefinementMethod}")
    print(f"  - Мин. размер периметра: {params.minMarkerPerimeterRate}")
    
    # Тест предобработки (создаем тестовое изображение)
    test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    processed = detector.preprocess_image(test_image)
    
    print(f"✓ Предобработка изображения работает")
    print(f"  - Входной размер: {test_image.shape}")
    print(f"  - Выходной размер: {processed.shape}")
    
    print(f"\n✅ Все тесты пройдены!")


if __name__ == "__main__":
    test_detector()
