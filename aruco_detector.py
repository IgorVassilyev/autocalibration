import os
from typing import Dict, Tuple

import cv2


class SimpleArUcoDetector:
    """Detect ArUco 4x4 markers on images."""

    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect_markers(self, image_path: str) -> Dict[int, Tuple[float, float]]:
        """Return detected marker centers from ``image_path``.

        Parameters
        ----------
        image_path : str
            Path to an image containing potential markers.

        Returns
        -------
        dict
            Mapping from marker id to (u, v) pixel coordinates of its center.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {}

        corners, ids, _ = self.detector.detectMarkers(image)
        results: Dict[int, Tuple[float, float]] = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                # marker_corners shape: (1, 4, 2)
                pts = marker_corners.reshape(4, 2)
                center = pts.mean(axis=0)
                results[int(marker_id)] = (float(center[0]), float(center[1]))
        return results

    def detect_from_directory(self, directory: str) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """Detect markers on all images within ``directory``.

        Parameters
        ----------
        directory : str
            Directory containing ``*.jpg`` images.

        Returns
        -------
        dict
            Mapping from camera id to per-marker centers.
        """
        images = [f for f in os.listdir(directory) if f.lower().endswith('.jpg')]
        detections: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for img in sorted(images):
            cam_id = os.path.splitext(img)[0]
            image_path = os.path.join(directory, img)
            detections[cam_id] = self.detect_markers(image_path)
        return detections
