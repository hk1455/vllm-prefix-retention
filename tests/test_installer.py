import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "vllm"
        shutil.copytree(ROOT / "reference/vllm", self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_action(self, action, ok=True):
        r = subprocess.run([sys.executable, str(ROOT / "manage_patch.py"), action,
                            "--vllm-root", str(self.root)], capture_output=True, text=True)
        self.assertEqual(r.returncode == 0, ok, r.stdout + r.stderr)
        return r

    def test_install_idempotent_restore_and_reinstall(self):
        original = (self.root / "v1/core/block_pool.py").read_bytes()
        self.run_action("check")
        self.run_action("install")
        self.run_action("check")
        self.run_action("install")
        self.run_action("restore")
        self.assertEqual((self.root / "v1/core/block_pool.py").read_bytes(), original)
        self.assertFalse((self.root / "v1/core/prefix_retention.py").exists())
        self.run_action("restore")
        self.run_action("install")

    def test_changed_dependency_refuses_install(self):
        p = self.root / "v1/core/kv_cache_manager.py"
        p.write_bytes(p.read_bytes() + b"\n# local change\n")
        self.run_action("install", ok=False)
        self.assertFalse((self.root / "v1/core/prefix_retention.py").exists())

    def test_changed_patch_refuses_restore(self):
        self.run_action("install")
        p = self.root / "v1/core/block_pool.py"
        changed = p.read_bytes() + b"\n# user change\n"
        p.write_bytes(changed)
        self.run_action("restore", ok=False)
        self.assertEqual(p.read_bytes(), changed)

    def test_interrupted_install_can_restore(self):
        self.run_action("install")
        # Crash after helper write, before pool replacement.
        shutil.copyfile(ROOT / "reference/vllm/v1/core/block_pool.py",
                        self.root / "v1/core/block_pool.py")
        self.run_action("check", ok=False)
        self.run_action("install", ok=False)
        self.run_action("restore")
        self.run_action("check")

    def test_unknown_helper_not_overwritten(self):
        p = self.root / "v1/core/prefix_retention.py"
        p.write_text("# unrelated file\n")
        self.run_action("install", ok=False)
        self.assertEqual(p.read_text(), "# unrelated file\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
