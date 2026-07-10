"""Runtime guards for utility behavior used by HotSpot-backed evaluation."""
import subprocess
import sys
import tempfile
import unittest

from core.util import find_hotpoint, run_command


class UtilRuntimeTests(unittest.TestCase):
    def test_run_command_reraises_failed_process(self):
        with self.assertRaises(subprocess.CalledProcessError):
            run_command([sys.executable, "-c", "import sys; sys.exit(7)"])

    def test_find_hotpoint_skips_headers_and_tracks_layers(self):
        content = "\n".join([
            "Unit steady",
            "cell0 300.0K",
            "Layer 2:",
            "cell1 315.5K",
            "Layer 3:",
            "cell2 314.0K",
        ])
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            self.assertEqual(find_hotpoint(handle.name), 315.5)


if __name__ == "__main__":
    unittest.main()
