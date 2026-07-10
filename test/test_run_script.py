"""Static guard for the HotSpot wrapper used by spawned evaluations."""
from pathlib import Path
import unittest


class RunScriptTests(unittest.TestCase):
    def test_run_script_pins_hotspot_and_passes_floorplan(self):
        script = Path(__file__).resolve().parents[1] / "tmp" / "run.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn("${HOTSPOT_PATH:-", text)
        self.assertIn('-f "$flp_file"', text)


if __name__ == "__main__":
    unittest.main()
