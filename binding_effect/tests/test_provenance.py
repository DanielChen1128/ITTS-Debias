import hashlib
import tempfile
import unittest
from pathlib import Path

from verify_provenance import _check_wav_manifest


class WavManifestTests(unittest.TestCase):
    def test_validates_count_and_optional_binary_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records"
            records.mkdir()
            wav = root / "results" / "sample.wav"
            wav.parent.mkdir()
            wav.write_bytes(b"wav")
            digest = hashlib.sha256(wav.read_bytes()).hexdigest()
            (records / "wav_sha256.txt").write_text(
                f"{digest}  results/sample.wav\n", encoding="utf-8"
            )
            spec = {"file": "wav_sha256.txt", "wav_count": 1}
            self.assertEqual(
                _check_wav_manifest(records, root, spec, require_binaries=True), []
            )

    def test_reports_malformed_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wav_sha256.txt").write_text("invalid\n", encoding="utf-8")
            errors = _check_wav_manifest(
                root, root, {"file": "wav_sha256.txt", "wav_count": 1}
            )
            self.assertTrue(any("invalid WAV manifest line" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
