import unittest
import uuid
from pathlib import Path

from scripts.check_secrets import scan_directory, scan_text


class SecretScanTests(unittest.TestCase):
    def test_detects_tokens_and_credentials_without_returning_values(self):
        token = "gsk_" + "a" * 32
        for text in (
            token,
            'api_key = "' + "a" * 32 + '"',
            "EXA_API_KEY=" + "b" * 32,
            "AccountKey=" + "c" * 40 + "==",
            "https://example.org/?sig=" + "d" * 32,
            "-----BEGIN " + "PRIVATE KEY-----",
        ):
            with self.subTest(text_type=text[:10]):
                findings = scan_text(text)
                self.assertTrue(findings)
                self.assertNotIn(token, repr(findings))

    def test_accepts_empty_settings_and_placeholders(self):
        self.assertEqual(scan_text('API_KEY=\napi_key = None\nAPI_KEY=your_key_here\n"secretRef": "groq-api-key"'), [])

    def test_scans_hidden_env_and_ignores_virtual_environment(self):
        root = Path(__file__).resolve().parent / (".scanner-test-" + uuid.uuid4().hex)
        root.mkdir()
        try:
            token = "gsk_" + "z" * 32
            (root / ".env").write_text(token, encoding="utf-8")
            (root / ".venv").mkdir()
            (root / ".venv" / "example.py").write_text(token, encoding="utf-8")
            findings = scan_directory(root)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0][0], Path(".env"))
        finally:
            (root / ".env").unlink(missing_ok=True)
            (root / ".venv" / "example.py").unlink(missing_ok=True)
            if (root / ".venv").exists():
                (root / ".venv").rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
