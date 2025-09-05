import os
import json
from aruco_detector import SimpleArUcoDetector
from xmp_parser import SimpleXMPParser


def print_camera_summary(cam_id: str, cam_data: dict) -> None:
    """Печать краткой информации о камере."""
    pos = cam_data["position"]
    validation = cam_data["validation"]
    
    print(f"\n{cam_id}:")
    print(f"  Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"  Focal length: {cam_data['focal_length']:.2f}mm")
    print(f"  Distortion model: {cam_data['distortion_model']}")
    print(f"  Skew: {cam_data['skew']}")
    print(f"  RC version: {cam_data['realitycapture_version'] if cam_data['realitycapture_version'] != 'unknown' else 'not available'}")
    
    if cam_data['altitude'] is not None:
        print(f"  Altitude: {cam_data['altitude']:.1f}m")
    
    # Геолокация
    if cam_data['latitude'] and cam_data['longitude']:
        print(f"  Location: {cam_data['latitude']}, {cam_data['longitude']}")
    
    # Валидация
    status = "+" if validation['is_valid'] else "-"
    print(f"  Validation: {status}")
    
    if validation['warnings']:
        print(f"  Warnings: {'; '.join(validation['warnings'])}")
    
    if validation['errors']:
        print(f"  Errors: {'; '.join(validation['errors'])}")


def print_camera_detailed(cam_id: str, cam_data: dict) -> None:
    """Печать подробной информации о камере."""
    print(f"\n{'='*60}")
    print(f"CAMERA: {cam_id}")
    print(f"{'='*60}")
    
    # === БАЗОВАЯ ИНФОРМАЦИЯ ===
    print(f"File: {cam_data['filename']}")
    print(f"File path: {cam_data['file_path']}")
    
    # === ВНУТРЕННИЕ ПАРАМЕТРЫ КАМЕРЫ ===
    print(f"\nINTERNAL CAMERA PARAMETERS:")
    print(f"   Focal length (35mm equiv): {cam_data['focal_length']:.4f}mm")
    print(f"   Principal point U: {cam_data['principal_point_u']:.6f}")
    print(f"   Principal point V: {cam_data['principal_point_v']:.6f}")
    print(f"   Aspect ratio: {cam_data['aspect_ratio']:.6f}")
    print(f"   Skew: {cam_data['skew']:.6f}")
    
    # === ВНЕШНИЕ ПАРАМЕТРЫ ===
    print(f"\nEXTERNAL PARAMETERS:")
    pos = cam_data['position']
    print(f"   Position (X, Y, Z): [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    
    print(f"   Rotation matrix:")
    rotation = cam_data['rotation']
    for i, row in enumerate(rotation):
        print(f"     Row {i+1}: [{row[0]:9.6f}, {row[1]:9.6f}, {row[2]:9.6f}]")
    
    # === ДИСТОРСИЯ ===
    print(f"\nDISTORTION:")
    print(f"   Model: {cam_data['distortion_model']}")
    dist = cam_data['distortion']
    print(f"   Coefficients:")
    print(f"     k1 (radial): {dist[0]:10.6f}")
    print(f"     k2 (radial): {dist[1]:10.6f}")
    print(f"     k3 (radial): {dist[2]:10.6f}")
    print(f"     p1 (tangent): {dist[3]:10.6f}")
    print(f"     p2 (tangent): {dist[4]:10.6f}")
    print(f"     k4 (radial): {dist[5]:10.6f}")
    
    # === МЕТАДАННЫЕ КАЛИБРОВКИ ===
    print(f"\nCALIBRATION METADATA:")
    print(f"   XCR version: {cam_data['xcr_version']}")
    rc_version = cam_data['realitycapture_version']
    print(f"   RealityCapture version: {rc_version if rc_version != 'unknown' else 'not available'}")
    print(f"   Pose prior: {cam_data['pose_prior']}")
    print(f"   Coordinate system: {cam_data['coordinates']}")
    print(f"   Calibration prior: {cam_data['calibration_prior']}")
    print(f"   Calibration group: {cam_data['calibration_group']}")
    print(f"   Distortion group: {cam_data['distortion_group']}")
    
    # === ФЛАГИ ИСПОЛЬЗОВАНИЯ ===
    print(f"\nPROCESSING FLAGS:")
    print(f"   Used in texturing: {'Yes' if cam_data['in_texturing'] else 'No'}")
    print(f"   Used in meshing: {'Yes' if cam_data['in_meshing'] else 'No'}")
    
    # === ГЕОЛОКАЦИЯ ===
    if cam_data['latitude'] or cam_data['longitude'] or cam_data['altitude'] is not None:
        print(f"\nGEOLOCATION:")
        if cam_data['latitude']:
            print(f"   Latitude: {cam_data['latitude']}")
        if cam_data['longitude']:
            print(f"   Longitude: {cam_data['longitude']}")
        if cam_data['altitude'] is not None:
            print(f"   Altitude: {cam_data['altitude']:.2f}m")
    
    # === ВАЛИДАЦИЯ ===
    validation = cam_data['validation']
    print(f"\nVALIDATION:")
    status = "VALID" if validation['is_valid'] else "INVALID"
    status_icon = "+" if validation['is_valid'] else "-"
    print(f"   Status: {status_icon} {status}")
    
    if validation['warnings']:
        print(f"   Warnings ({len(validation['warnings'])}):")
        for i, warning in enumerate(validation['warnings'], 1):
            print(f"     {i}. {warning}")
    
    if validation['errors']:
        print(f"   Errors ({len(validation['errors'])}):")
        for i, error in enumerate(validation['errors'], 1):
            print(f"     {i}. {error}")
    
    if not validation['warnings'] and not validation['errors']:
        print(f"   No issues found")


def print_camera_comparison_table(cameras: dict) -> None:
    """Печать сравнительной таблицы камер."""
    print(f"\n{'='*100}")
    print("CAMERA COMPARISON TABLE")
    print(f"{'='*100}")
    
    # Заголовок таблицы
    header = f"{'Camera ID':<15} {'Focal (mm)':<10} {'Position X':<10} {'Position Y':<10} {'Position Z':<10} {'Altitude (m)':<12} {'Valid':<5}"
    print(header)
    print("-" * len(header))
    
    # Данные по камерам
    for cam_id in sorted(cameras.keys()):
        cam = cameras[cam_id]
        pos = cam['position']
        alt = f"{cam['altitude']:.1f}" if cam['altitude'] is not None else "N/A"
        valid = "+" if cam['validation']['is_valid'] else "-"
        
        row = f"{cam_id:<15} {cam['focal_length']:<10.2f} {pos[0]:<10.3f} {pos[1]:<10.3f} {pos[2]:<10.3f} {alt:<12} {valid:<5}"
        print(row)


def print_detailed_stats(parser: SimpleXMPParser) -> None:
    """Печать детальной статистики."""
    stats = parser.get_summary_stats()
    
    print(f"\nTotal cameras loaded: {stats['total_cameras']}")
    print(f"Validation errors: {stats['validation_errors']}")
    
    print(f"\nFOCAL LENGTHS:")
    print(f"   Range: {stats['focal_length_range'][0]:.2f} - {stats['focal_length_range'][1]:.2f}mm")
    print(f"   Mean: {stats['focal_length_mean']:.2f}mm")
    
    if stats['altitude_range'][0] is not None:
        print(f"\nALTITUDES:")
        print(f"   Range: {stats['altitude_range'][0]:.1f} - {stats['altitude_range'][1]:.1f}m")
    
    print(f"\nTECHNICAL INFO:")
    print(f"   Distortion models: {', '.join(stats['distortion_models'])}")
    print(f"   Coordinate systems: {', '.join(stats['coordinate_systems'])}")
    if stats['realitycapture_versions'] != ['version not available']:
        print(f"   RealityCapture versions: {', '.join(stats['realitycapture_versions'])}")
    else:
        print("   RealityCapture versions: not available in XMP metadata")


def export_full_data(cameras: dict, output_path: str) -> None:
    """Экспорт всех данных в JSON для дальнейшего анализа."""
    # Подготавливаем данные для JSON (убираем непереводимые в JSON объекты)
    export_data = {}
    
    for cam_id, cam_data in cameras.items():
        export_data[cam_id] = {
            # Базовая информация
            'filename': cam_data['filename'],
            'focal_length': cam_data['focal_length'],
            'principal_point': [cam_data['principal_point_u'], cam_data['principal_point_v']],
            'aspect_ratio': cam_data['aspect_ratio'],
            'skew': cam_data['skew'],
            
            # Пространственные данные
            'position': cam_data['position'],
            'rotation_matrix': cam_data['rotation'],
            
            # Дисторсия
            'distortion_model': cam_data['distortion_model'],
            'distortion_coefficients': cam_data['distortion'],
            
            # Метаданные
            'xcr_version': cam_data['xcr_version'],
            'realitycapture_version': cam_data['realitycapture_version'],
            'pose_prior': cam_data['pose_prior'],
            'coordinates': cam_data['coordinates'],
            'calibration_prior': cam_data['calibration_prior'],
            'calibration_group': cam_data['calibration_group'],
            'distortion_group': cam_data['distortion_group'],
            
            # Флаги
            'in_texturing': cam_data['in_texturing'],
            'in_meshing': cam_data['in_meshing'],
            
            # Геолокация
            'latitude': cam_data['latitude'],
            'longitude': cam_data['longitude'],
            'altitude': cam_data['altitude'],
            
            # Валидация
            'validation': cam_data['validation']
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull camera data exported to {output_path}")


def main():
    """Основная функция демонстрации."""
    print("=" * 60)
    print("ENHANCED XMP PARSER DEMO")
    print("=" * 60)
    
    # Инициализация с логированием
    parser = SimpleXMPParser(enable_logging=True)
    
    # Загрузка камер
    cameras = parser.load_all_cameras("data")
    
    if not cameras:
        print("No cameras loaded!")
        return
    
    print(f"\nSuccessfully loaded {len(cameras)} cameras")
    
    # === КРАТКАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА ===
    print_camera_comparison_table(cameras)
    
    # === ПОДРОБНАЯ ИНФОРМАЦИЯ ПО КАЖДОЙ КАМЕРЕ ===
    print(f"\n\n{'#'*80}")
    print("DETAILED CAMERA INFORMATION")
    print(f"{'#'*80}")
    
    for cam_id in sorted(cameras.keys()):
        cam_data = cameras[cam_id]
        print_camera_detailed(cam_id, cam_data)
    
    # === ОБЩАЯ СТАТИСТИКА ===
    print(f"\n\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print_detailed_stats(parser)
    
  



if __name__ == "__main__":
    main()