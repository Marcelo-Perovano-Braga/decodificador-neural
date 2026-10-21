import struct
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WindowsPackagingTests(unittest.TestCase):
    def test_icon_contains_expected_windows_resolutions(self) -> None:
        icon_path = PROJECT_ROOT / "assets" / "app_icon.ico"
        icon_bytes = icon_path.read_bytes()
        reserved, image_type, image_count = struct.unpack_from("<HHH", icon_bytes)

        self.assertEqual(reserved, 0)
        self.assertEqual(image_type, 1)
        self.assertGreaterEqual(image_count, 7)

        widths = set()
        for index in range(image_count):
            width = icon_bytes[6 + index * 16]
            widths.add(256 if width == 0 else width)
        self.assertTrue({16, 24, 32, 48, 64, 128, 256}.issubset(widths))

    def test_build_files_and_application_assets_exist(self) -> None:
        expected_paths = (
            "assets/app_icon.png",
            "assets/app_icon.ico",
            "build_windows.ps1",
            "IzhikevichClassifier.spec",
            "packaging/windows_version_info.txt",
            "requirements-build.txt",
        )
        for relative_path in expected_paths:
            with self.subTest(path=relative_path):
                self.assertTrue((PROJECT_ROOT / relative_path).is_file())

        png_signature = (PROJECT_ROOT / "assets" / "app_icon.png").read_bytes()[:8]
        self.assertEqual(png_signature, b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
