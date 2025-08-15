import os
import xml.etree.ElementTree as ET


class SimpleXMPParser:
    """Parser for extracting camera parameters from XMP files exported by RealityCapture."""

    def __init__(self):
        self.cameras_data = {}

    def parse_xmp_file(self, xmp_path):
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
                return None

            camera_data = {
                'file_path': xmp_path,
                'filename': os.path.basename(xmp_path),
                'position': self._parse_position(desc.find('xcr:Position', ns)),
                'rotation': self._parse_rotation(desc.find('xcr:Rotation', ns)),
                'focal_length': float(
                    desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}FocalLength35mm', 35.0)
                ),
                'principal_point_u': float(
                    desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}PrincipalPointU', 0.0)
                ),
                'principal_point_v': float(
                    desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}PrincipalPointV', 0.0)
                ),
                'distortion': self._parse_distortion(
                    desc.find('xcr:DistortionCoeficients', ns)
                ),
                'aspect_ratio': float(
                    desc.get('{http://www.capturingreality.com/ns/xcr/1.1#}AspectRatio', 1.0)
                ),
            }

            return camera_data
        except Exception:
            return None

    def _parse_position(self, element):
        if element is not None and element.text:
            return [float(v) for v in element.text.split()]
        return [0.0, 0.0, 0.0]

    def _parse_rotation(self, element):
        if element is not None and element.text:
            values = [float(v) for v in element.text.split()]
            if len(values) == 9:
                return [values[0:3], values[3:6], values[6:9]]
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def _parse_distortion(self, element):
        if element is not None and element.text:
            return [float(v) for v in element.text.split()]
        return [0.0] * 6

    def load_all_cameras(self, directory):
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
        xmp_files = [f for f in os.listdir(directory) if f.lower().endswith('.xmp')]

        for xmp_file in sorted(xmp_files):
            xmp_path = os.path.join(directory, xmp_file)
            data = self.parse_xmp_file(xmp_path)
            if data is not None:
                camera_id = os.path.splitext(xmp_file)[0]
                self.cameras_data[camera_id] = data

        return self.cameras_data
