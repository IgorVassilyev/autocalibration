import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any, Optional, List, Tuple


class SimpleXMPParser:
    """Enhanced parser for extracting camera parameters from XMP files exported by RealityCapture."""

    def __init__(self, enable_logging: bool = True):
        self.cameras_data = {}
        
        # Настройка логирования
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)

    def parse_xmp_file(self, xmp_path: str) -> Optional[Dict[str, Any]]:
        """Parse a single XMP file and return camera parameters.

        Parameters
        ----------
        xmp_path : str
            Path to an XMP file.

        Returns
        -------
        dict or None
            Dictionary with camera parameters or ``None`` on failure.
        """
        try:
            tree = ET.parse(xmp_path)
            root = tree.getroot()

            ns = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'xcr': 'http://www.capturingreality.com/ns/xcr/1.1#',
            }

            desc = root.find('.//rdf:Description', ns)
            if desc is None:
                self.logger.warning(f"No rdf:Description found in {xmp_path}")
                return None

            # Извлекаем все данные
            camera_data = {
                # Метаданные файла
                'file_path': xmp_path,
                'filename': os.path.basename(xmp_path),
                
                # === ВНУТРЕННИЕ ПАРАМЕТРЫ КАМЕРЫ ===
                'focal_length': self._get_float_attribute(desc, 'FocalLength35mm', 35.0),
                'principal_point_u': self._get_float_attribute(desc, 'PrincipalPointU', 0.0),
                'principal_point_v': self._get_float_attribute(desc, 'PrincipalPointV', 0.0),
                'aspect_ratio': self._get_float_attribute(desc, 'AspectRatio', 1.0),
                'skew': self._get_float_attribute(desc, 'Skew', 0.0),  # НОВОЕ
                
                # === ВНЕШНИЕ ПАРАМЕТРЫ ===
                'position': self._parse_position(desc.find('xcr:Position', ns)),
                'rotation': self._parse_rotation(desc.find('xcr:Rotation', ns)),
                
                # === ПАРАМЕТРЫ ДИСТОРСИИ ===
                'distortion_model': self._get_string_attribute(desc, 'DistortionModel', 'unknown'),  # НОВОЕ
                'distortion': self._parse_distortion(desc.find('xcr:DistortionCoeficients', ns)),
                
                # === МЕТАДАННЫЕ КАЛИБРОВКИ ===
                'xcr_version': self._get_string_attribute(desc, 'Version', 'unknown'),  # НОВОЕ
                'realitycapture_version': self._get_string_attribute(desc, 'version', 'unknown'),  # НОВОЕ
                'pose_prior': self._get_string_attribute(desc, 'PosePrior', 'unknown'),  # НОВОЕ
                'coordinates': self._get_string_attribute(desc, 'Coordinates', 'unknown'),  # НОВОЕ
                'calibration_prior': self._get_string_attribute(desc, 'CalibrationPrior', 'unknown'),  # НОВОЕ
                'calibration_group': self._get_int_attribute(desc, 'CalibrationGroup', -1),  # НОВОЕ
                'distortion_group': self._get_int_attribute(desc, 'DistortionGroup', -1),  # НОВОЕ
                
                # === ФЛАГИ ИСПОЛЬЗОВАНИЯ ===
                'in_texturing': self._get_bool_attribute(desc, 'InTexturing', True),  # НОВОЕ
                'in_meshing': self._get_bool_attribute(desc, 'InMeshing', True),  # НОВОЕ
                
                # === ГЕОЛОКАЦИЯ ===
                'latitude': self._get_string_attribute(desc, 'latitude', None),  # НОВОЕ
                'longitude': self._get_string_attribute(desc, 'longitude', None),  # НОВОЕ
                'altitude': self._parse_altitude(self._get_string_attribute(desc, 'altitude', None)),  # НОВОЕ
            }

            # Валидация критически важных параметров
            validation_result = self._validate_camera_data(camera_data)
            camera_data['validation'] = validation_result
            
            if not validation_result['is_valid']:
                self.logger.warning(f"Validation failed for {xmp_path}: {validation_result['errors']}")

            return camera_data

        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {xmp_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing {xmp_path}: {e}")
            return None

    def _get_float_attribute(self, element: ET.Element, attr_name: str, default: float) -> float:
        """Safely extract float attribute with namespace."""
        try:
            value = element.get(f'{http://www.capturingreality.com/ns/xcr/1.1#}{attr_name}')
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse float attribute {attr_name}, using default {default}")
            return default

    def _get_string_attribute(self, element: ET.Element, attr_name: str, default: Optional[str]) -> Optional[str]:
        """Safely extract string attribute with namespace."""
        value = element.get(f'{http://www.capturingreality.com/ns/xcr/1.1#}{attr_name}')
        return value if value is not None else default

    def _get_int_attribute(self, element: ET.Element, attr_name: str, default: int) -> int:
        """Safely extract integer attribute with namespace."""
        try:
            value = element.get(f'{http://www.capturingreality.com/ns/xcr/1.1#}{attr_name}')
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            self.logger.warning(f"Could not parse int attribute {attr_name}, using default {default}")
            return default

    def _get_bool_attribute(self, element: ET.Element, attr_name: str, default: bool) -> bool:
        """Safely extract boolean attribute with namespace."""
        value = element.get(f'{http://www.capturingreality.com/ns/xcr/1.1#}{attr_name}')
        if value is not None:
            return value == '1' or value.lower() == 'true'
        return default

    def _parse_position(self, element: Optional[ET.Element]) -> List[float]:
        """Parse position element with validation."""
        if element is not None and element.text:
            try:
                values = [float(v) for v in element.text.split()]
                if len(values) != 3:
                    self.logger.warning(f"Position should have 3 values, got {len(values)}")
                    return [0.0, 0.0, 0.0]
                return values
            except ValueError as e:
                self.logger.warning(f"Could not parse position values: {e}")
        return [0.0, 0.0, 0.0]

    def _parse_rotation(self, element: Optional[ET.Element]) -> List[List[float]]:
        """Parse rotation matrix with validation."""
        if element is not None and element.text:
            try:
                values = [float(v) for v in element.text.split()]
                if len(values) != 9:
                    self.logger.warning(f"Rotation matrix should have 9 values, got {len(values)}")
                    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                
                # Преобразуем в матрицу 3x3
                matrix = [
                    [values[0], values[1], values[2]],
                    [values[3], values[4], values[5]],
                    [values[6], values[7], values[8]]
                ]
                
                # Проверяем ортогональность
                if not self._is_orthogonal_matrix(matrix):
                    self.logger.warning("Rotation matrix is not orthogonal")
                
                return matrix
            except ValueError as e:
                self.logger.warning(f"Could not parse rotation matrix: {e}")
        
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _parse_distortion(self, element: Optional[ET.Element]) -> List[float]:
        """Parse distortion coefficients with validation."""
        if element is not None and element.text:
            try:
                values = [float(v) for v in element.text.split()]
                if len(values) != 6:
                    self.logger.warning(f"Distortion should have 6 coefficients, got {len(values)}")
                    return [0.0] * 6
                return values
            except ValueError as e:
                self.logger.warning(f"Could not parse distortion coefficients: {e}")
        return [0.0] * 6

    def _parse_altitude(self, altitude_str: Optional[str]) -> Optional[float]:
        """Parse altitude from fraction format."""
        if altitude_str and '/' in altitude_str:
            try:
                numerator, denominator = altitude_str.split('/')
                return float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError) as e:
                self.logger.warning(f"Could not parse altitude {altitude_str}: {e}")
        return None

    def _is_orthogonal_matrix(self, matrix: List[List[float]], tolerance: float = 1e-3) -> bool:
        """Check if rotation matrix is orthogonal."""
        try:
            # Проверяем детерминант ≈ 1
            det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                   matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                   matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
            
            return abs(det - 1.0) < tolerance
        except:
            return False

    def _validate_realitycapture_version(self, version: str) -> bool:
        """Проверка корректности версии RealityCapture/RealityScan."""
        if version == 'unknown':
            return True
            
        known_versions = [
            '1.0', '1.1', '1.2', '1.3', '1.4', '1.5',  # RealityCapture
            '2.0',  # RealityScan 2.0
        ]
        
        # Проверяем основную версию (первые два числа)
        try:
            main_version = '.'.join(version.split('.')[:2])
            return main_version in known_versions
        except:
            return False

    def _validate_camera_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted camera data."""
        errors = []
        warnings = []
        
        # Проверка модели дисторсии
        if data['distortion_model'] not in ['brown3', 'unknown']:
            errors.append(f"Unsupported distortion model: {data['distortion_model']}")
        
        # Проверка фокусного расстояния
        if not (10.0 <= data['focal_length'] <= 200.0):
            warnings.append(f"Unusual focal length: {data['focal_length']}mm")
        
        # Проверка соотношения сторон
        if not (0.5 <= data['aspect_ratio'] <= 2.0):
            warnings.append(f"Unusual aspect ratio: {data['aspect_ratio']}")
        
        # Проверка типа координат
        if data['coordinates'] != 'absolute':
            warnings.append(f"Non-absolute coordinates: {data['coordinates']}")
        
        # Проверка типа калибровки
        if data['calibration_prior'] not in ['exact', 'unknown']:
            warnings.append(f"Unusual calibration prior: {data['calibration_prior']}")
        
        # Проверка версии RealityCapture
        if not self._validate_realitycapture_version(data['realitycapture_version']):
            warnings.append(f"Unknown RealityCapture version: {data['realitycapture_version']} (may be internal build)")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def load_all_cameras(self, directory: str) -> Dict[str, Dict[str, Any]]:
        """Load all cameras from XMP files within ``directory``.

        Parameters
        ----------
        directory : str
            Directory containing ``*.xmp`` files.

        Returns
        -------
        dict
            Mapping from camera id (filename without extension) to parameters.
        """
        if not os.path.exists(directory):
            self.logger.error(f"Directory does not exist: {directory}")
            return {}

        xmp_files = [f for f in os.listdir(directory) if f.lower().endswith('.xmp')]
        
        if not xmp_files:
            self.logger.warning(f"No XMP files found in {directory}")
            return {}

        self.logger.info(f"Found {len(xmp_files)} XMP files in {directory}")

        for xmp_file in sorted(xmp_files):
            xmp_path = os.path.join(directory, xmp_file)
            data = self.parse_xmp_file(xmp_path)
            if data is not None:
                camera_id = os.path.splitext(xmp_file)[0]
                self.cameras_data[camera_id] = data
                
                # Логируем результаты валидации
                validation = data['validation']
                if validation['warnings']:
                    self.logger.warning(f"{camera_id}: {validation['warnings']}")

        self.logger.info(f"Successfully loaded {len(self.cameras_data)} cameras")
        return self.cameras_data

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of loaded cameras."""
        if not self.cameras_data:
            return {}
        
        focal_lengths = [cam['focal_length'] for cam in self.cameras_data.values()]
        altitudes = [cam['altitude'] for cam in self.cameras_data.values() if cam['altitude'] is not None]
        
        # Фильтруем версии RealityCapture - показываем только валидные
        rc_versions = []
        for cam in self.cameras_data.values():
            version = cam['realitycapture_version']
            if self._validate_realitycapture_version(version) and version != 'unknown':
                rc_versions.append(version)
        
        # Если нет валидных версий, не показываем их вообще
        if not rc_versions:
            rc_versions = ['version not available']
        
        stats = {
            'total_cameras': len(self.cameras_data),
            'focal_length_range': (min(focal_lengths), max(focal_lengths)) if focal_lengths else (0, 0),
            'focal_length_mean': sum(focal_lengths) / len(focal_lengths) if focal_lengths else 0,
            'altitude_range': (min(altitudes), max(altitudes)) if altitudes else (None, None),
            'distortion_models': list(set(cam['distortion_model'] for cam in self.cameras_data.values())),
            'coordinate_systems': list(set(cam['coordinates'] for cam in self.cameras_data.values())),
            'realitycapture_versions': list(set(rc_versions)),
            'validation_errors': sum(1 for cam in self.cameras_data.values() if not cam['validation']['is_valid'])
        }
        
        return stats

    def export_summary_report(self, output_path: str) -> None:
        """Export summary report to text file."""
        stats = self.get_summary_stats()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== XMP PARSER SUMMARY REPORT ===\n\n")
            f.write(f"Total cameras loaded: {stats['total_cameras']}\n")
            f.write(f"Validation errors: {stats['validation_errors']}\n\n")
            
            f.write("FOCAL LENGTHS:\n")
            f.write(f"  Range: {stats['focal_length_range'][0]:.2f} - {stats['focal_length_range'][1]:.2f}mm\n")
            f.write(f"  Mean: {stats['focal_length_mean']:.2f}mm\n\n")
            
            if stats['altitude_range'][0] is not None:
                f.write("ALTITUDES:\n")
                f.write(f"  Range: {stats['altitude_range'][0]:.1f} - {stats['altitude_range'][1]:.1f}m\n\n")
            
            f.write(f"DISTORTION MODELS: {', '.join(stats['distortion_models'])}\n")
            f.write(f"COORDINATE SYSTEMS: {', '.join(stats['coordinate_systems'])}\n")
            if stats['realitycapture_versions'] != ['version not available']:
                f.write(f"REALITYCAPTURE VERSIONS: {', '.join(stats['realitycapture_versions'])}\n")
            f.write("\n")
            
            f.write("PER-CAMERA DETAILS:\n")
            for cam_id, cam_data in self.cameras_data.items():
                f.write(f"\n{cam_id}:\n")
                f.write(f"  Position: [{cam_data['position'][0]:.3f}, {cam_data['position'][1]:.3f}, {cam_data['position'][2]:.3f}]\n")
                f.write(f"  Focal length: {cam_data['focal_length']:.2f}mm\n")
                f.write(f"  Distortion model: {cam_data['distortion_model']}\n")
                f.write(f"  Validation: {'+' if cam_data['validation']['is_valid'] else '-'}\n")
                if cam_data['validation']['warnings']:
                    f.write(f"  Warnings: {'; '.join(cam_data['validation']['warnings'])}\n")
        
        self.logger.info(f"Summary report exported to {output_path}")