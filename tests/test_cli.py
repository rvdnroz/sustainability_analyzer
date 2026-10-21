"""Exercise CLI startup and input validation without contacting providers."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-B", "main.py", *args], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )

    def test_help_for_all_commands(self):
        for command in ((), ("ingest",), ("ask",), ("batch",), ("run",)):
            with self.subTest(command=command):
                result = self.run_cli(*command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_rejects_conflicting_report_sources_before_processing(self):
        result = self.run_cli(
            "ingest", "--company", "Example", "--year", "2023",
            "--url", "https://example.org/report.pdf", "--pdf-path", "missing.pdf",
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("not allowed with argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
