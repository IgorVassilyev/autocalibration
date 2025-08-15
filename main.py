import os

from aruco_detector import SimpleArUcoDetector
from xmp_parser import SimpleXMPParser


def main():
    parser = SimpleXMPParser()
    cameras = parser.load_all_cameras("data")
    print(f"Loaded {len(cameras)} cameras")

    detector = SimpleArUcoDetector()
    detections = detector.detect_from_directory("data")

    for cam_id, cam in cameras.items():
        pos = cam["position"]
        print(f"{cam_id}: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")
        for marker_id, center in detections.get(cam_id, {}).items():
            print(f"  marker {marker_id}: {center[0]:.1f}, {center[1]:.1f}")


if __name__ == "__main__":
    main()
