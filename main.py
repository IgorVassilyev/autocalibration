import os
import json
from aruco_detector import SimpleArUcoDetector
from xmp_parser import SimpleXMPParser


def main():
    """Основная функция демонстрации."""
    print("=== Enhanced XMP Parser Demo ===")
    
    parser = SimpleXMPParser(enable_logging=False)
    cameras = parser.load_all_cameras("data")
    
    if not cameras:
        print("No cameras loaded!")
        return
    
    print(f"\n✓ Loaded {len(cameras)} cameras")

    


if __name__ == "__main__":
    main()
