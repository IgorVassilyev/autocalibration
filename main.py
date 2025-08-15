from xmp_parser import SimpleXMPParser


def main():
    parser = SimpleXMPParser()
    cameras = parser.load_all_cameras("data")
    print(f"Loaded {len(cameras)} cameras")

    for cam_id, cam in cameras.items():
        pos = cam["position"]
        print(f"{cam_id}: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}")


if __name__ == "__main__":
    main()
