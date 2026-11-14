"""P9 read-only C2PA validator tests (stdlib unittest; no pytest dependency).

Run:
  .venv/bin/python -m unittest discover -s tests -p 'test_c2pa_read.py' -v
"""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

import c2pa_read as cr  # noqa: E402

SCRIPT = WS / "deepfake" / "scripts" / "c2pa_read.py"
FIXTURE3 = WS / "artifacts" / "mbeta" / "fixtures" / "fixture3_c2pa_signed_ai.png"
REAL = WS / "data" / "corpora" / "eval_real" / "000000335954.jpg"
TRUNC = WS / "artifacts" / "mbeta" / "fixtures" / "fixture_truncated.jpg"
DOC = WS / "docs" / "C2PA_READONLY_VALIDATOR.md"

SYNTH_TRUSTED = {
    "active_manifest": "urn:test:trusted",
    "validation_state": "Valid",
    "validation_status": [],
    "manifests": {
        "urn:test:trusted": {
            "signature_info": {"issuer": "Trusted Test CA"},
            "assertions": [{"label": "stds.exif", "data": {"Make": "TestCam"}}],
            "success": [
                {"code": "signingCredential.trusted"},
                {"code": "assertion.dataHash.match"},
            ],
            "failure": [],
        },
    },
}

SYNTH_TRUSTED_NO_BINDING = json.loads(json.dumps(SYNTH_TRUSTED))
SYNTH_TRUSTED_NO_BINDING["manifests"]["urn:test:trusted"]["success"] = [
    {"code": "signingCredential.trusted"},
]

SYNTH_TRUSTED_WARNING = json.loads(json.dumps(SYNTH_TRUSTED))
SYNTH_TRUSTED_WARNING["validation_status"] = [{"code": "timeStamp.untrusted"}]

SYNTH_AI_TRAINED = {
    "active_manifest": "urn:test:ai-trained",
    "validation_state": "Valid",
    "validation_status": [],
    "manifests": {
        "urn:test:ai-trained": {
            "signature_info": {"issuer": "Synthetic AI CA"},
            "assertions": [{
                "label": "c2pa.actions",
                "data": {"actions": [{
                    "action": "c2pa.created",
                    "digitalSourceType": (
                        "http://cv.iptc.org/newscodes/digitalsourcetype/"
                        "trainedAlgorithmicMedia"
                    ),
                }]},
            }],
            "success": [],
            "failure": [],
        },
    },
}

SYNTH_AI_GENERATOR = {
    "active_manifest": "urn:test:ai-generator",
    "validation_state": "Valid",
    "validation_status": [],
    "manifests": {
        "urn:test:ai-generator": {
            "claim_generator_info": [{"name": "Adobe Firefly"}],
            "signature_info": {"issuer": "Synthetic AI CA"},
            "assertions": [],
            "success": [],
            "failure": [],
        },
    },
}

SYNTH_CAMERA = {
    "active_manifest": "urn:test:camera",
    "validation_state": "Valid",
    "validation_status": [],
    "manifests": {
        "urn:test:camera": {
            "signature_info": {"issuer": "Camera Test CA"},
            "assertions": [
                {"label": "stds.exif", "data": {"Make": "TestCam"}},
                {"label": "c2pa.actions", "data": {"actions": [{
                    "action": "c2pa.captured",
                    "digitalSourceType": (
                        "http://cv.iptc.org/newscodes/digitalsourcetype/"
                        "digitalCapture"
                    ),
                }]}},
            ],
            "success": [],
            "failure": [],
        },
    },
}

SYNTH_HASH_ONLY = {
    "active_manifest": "urn:test:hash-only",
    "validation_state": "Valid",
    "validation_status": [],
    "manifests": {
        "urn:test:hash-only": {
            "assertions": [{"label": "c2pa.hash.data", "data": {"alg": "sha256"}}],
            "success": [{"code": "assertion.dataHash.match"}],
            "failure": [],
        },
    },
}

SYNTH_BMFF = {
    "active_manifest": "urn:test:bmff",
    "validation_state": "Valid",
    "validation_status": [],
    "manifests": {
        "urn:test:bmff": {
            "assertions": [{"label": "c2pa.hash.bmff", "data": {"alg": "sha256"}}],
            "success": [{"code": "assertion.bmffHash.match"}],
            "failure": [],
        },
    },
}

SYNTH_BMFF_MISMATCH = json.loads(json.dumps(SYNTH_BMFF))
SYNTH_BMFF_MISMATCH["manifests"]["urn:test:bmff"]["success"] = []
SYNTH_BMFF_MISMATCH["manifests"]["urn:test:bmff"]["failure"] = [
    {"code": "assertion.bmffHash.mismatch"},
]

SYNTH_NO_ACTIVE = {
    "validation_state": "Valid",
    "validation_status": [],
    "manifests": {},
}


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        cwd=str(WS), capture_output=True, text=True,
    )


def tamper_first_idat(raw):
    data = bytearray(raw)
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        chunk_type = bytes(data[pos + 4:pos + 8])
        if chunk_type == b"IDAT":
            data[pos + 8] ^= 0xFF
            return bytes(data)
        pos += 12 + length
    raise AssertionError("no IDAT chunk found")


class TestC2paRead(unittest.TestCase):
    def test_absent_no_manifest(self):
        result = cr.read_c2pa(REAL)
        self.assertEqual(result["manifest_state"], "absent")
        for key in ("trusted", "hard_binding", "ai_claimed", "camera_claimed"):
            self.assertIs(result[key], False, key)
        self.assertIsNone(result["signer_issuer"])
        self.assertIsNone(result["validation_state"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["validation_status_codes"], [])

    def test_well_formed_valid_untrusted_fixture(self):
        result = cr.read_c2pa(FIXTURE3)
        self.assertEqual(result["manifest_state"], "well_formed_valid")
        self.assertIs(result["trusted"], False)
        self.assertIs(result["hard_binding"], True)
        self.assertIs(result["ai_claimed"], True)
        self.assertEqual(result["validation_state"], "Valid")
        self.assertIn("signingCredential.untrusted",
                      result["validation_status_codes"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["signer_issuer"], "C2PA Test Signing Cert")
        self.assertIs(result["camera_claimed"], False)

    def test_trusted_synthetic(self):
        result = cr.classify_manifest(SYNTH_TRUSTED)
        self.assertEqual(result["manifest_state"], "trusted")
        self.assertIs(result["trusted"], True)
        self.assertIs(result["hard_binding"], True)
        self.assertEqual(result["signer_issuer"], "Trusted Test CA")
        self.assertEqual(result["validation_state"], "Valid")
        self.assertEqual(result["validation_status_codes"], [])
        self.assertIsNone(result["error"])

    def test_trusted_without_hard_binding(self):
        result = cr.classify_manifest(SYNTH_TRUSTED_NO_BINDING)
        self.assertEqual(result["manifest_state"], "trusted")
        self.assertIs(result["trusted"], True)
        self.assertIs(result["hard_binding"], False)

    def test_trusted_downgraded_by_warning(self):
        result = cr.classify_manifest(SYNTH_TRUSTED_WARNING)
        self.assertEqual(result["manifest_state"], "well_formed_valid")
        self.assertIs(result["trusted"], False)
        self.assertIs(result["hard_binding"], True)
        self.assertEqual(result["validation_status_codes"],
                         ["timeStamp.untrusted"])

    def test_missing_active_manifest_is_invalid(self):
        result = cr.classify_manifest(SYNTH_NO_ACTIVE)
        self.assertEqual(result["manifest_state"], "invalid")
        self.assertIs(result["trusted"], False)
        self.assertIs(result["hard_binding"], False)
        self.assertIn("manifest.activeManifest.missing",
                      result["validation_status_codes"])

    def test_bmff_hard_binding_synthetic(self):
        matched = cr.classify_manifest(SYNTH_BMFF)
        self.assertIs(matched["hard_binding"], True)
        mismatched = cr.classify_manifest(SYNTH_BMFF_MISMATCH)
        self.assertIs(mismatched["hard_binding"], False)

    def test_bmff_mismatch_downgrades_hard_binding(self):
        result = cr.classify_manifest(SYNTH_BMFF_MISMATCH)
        self.assertIs(result["hard_binding"], False)
        self.assertEqual(result["manifest_state"], "invalid")
        self.assertIn("assertion.bmffHash.mismatch",
                      result["validation_status_codes"])

    def test_tampered_hard_binding_mismatch(self):
        original = FIXTURE3.read_bytes()
        tampered = tamper_first_idat(original)
        self.assertNotEqual(tampered, original)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fixture3_tampered.png"
            target.write_bytes(tampered)
            result = cr.read_c2pa(target)
        self.assertEqual(result["manifest_state"], "invalid")
        self.assertIs(result["trusted"], False)
        self.assertIs(result["hard_binding"], False)
        self.assertEqual(result["validation_state"], "Invalid")
        self.assertIn("assertion.dataHash.mismatch",
                      result["validation_status_codes"])
        self.assertIsNone(result["error"])
        self.assertEqual(FIXTURE3.read_bytes(), original)

    def test_truncated_parse_error(self):
        result = cr.read_c2pa(TRUNC)
        self.assertEqual(result["manifest_state"], "error")
        for key in ("trusted", "hard_binding", "ai_claimed", "camera_claimed"):
            self.assertIs(result[key], False, key)
        self.assertIsNone(result["validation_state"])
        self.assertEqual(result["validation_status_codes"], [])
        self.assertIsNotNone(result["error"])

    def test_ai_claim_detection(self):
        self.assertIs(cr.read_c2pa(FIXTURE3)["ai_claimed"], True)
        trained = cr.classify_manifest(SYNTH_AI_TRAINED)
        self.assertIs(trained["ai_claimed"], True)
        self.assertIs(trained["camera_claimed"], False)
        generator = cr.classify_manifest(SYNTH_AI_GENERATOR)
        self.assertIs(generator["ai_claimed"], True)
        self.assertIs(generator["camera_claimed"], False)
        self.assertIs(cr.read_c2pa(REAL)["ai_claimed"], False)

    def test_camera_claim_detection(self):
        camera = cr.classify_manifest(SYNTH_CAMERA)
        self.assertIs(camera["camera_claimed"], True)
        hash_only = cr.classify_manifest(SYNTH_HASH_ONLY)
        self.assertIs(hash_only["camera_claimed"], False)
        self.assertIs(hash_only["hard_binding"], True)
        self.assertIs(cr.read_c2pa(REAL)["camera_claimed"], False)

    def test_determinism(self):
        first = cr.read_c2pa(FIXTURE3)
        second = cr.read_c2pa(FIXTURE3)
        self.assertEqual(first, second)
        compact_first = json.dumps(first, sort_keys=True, separators=(",", ":"))
        compact_second = json.dumps(second, sort_keys=True, separators=(",", ":"))
        self.assertEqual(compact_first, compact_second)
        cli_first = run_cli(FIXTURE3)
        cli_second = run_cli(FIXTURE3)
        self.assertEqual(cli_first.returncode, 0)
        self.assertEqual(cli_first.stdout, cli_second.stdout)
        self.assertEqual(len(cli_first.stdout.strip().splitlines()), 1)
        self.assertTrue(cli_first.stdout.endswith("\n"))

    def test_json_schema_keys_types(self):
        result = cr.read_c2pa(FIXTURE3)
        expected = {
            "schema_version", "path", "manifest_state", "trusted",
            "hard_binding", "ai_claimed", "camera_claimed", "signer_issuer",
            "validation_state", "validation_status_codes", "error",
        }
        self.assertEqual(set(result), expected)
        self.assertEqual(len(result), 11)
        self.assertEqual(result["schema_version"], "p9.c2pa_read.v1")
        self.assertIn(result["manifest_state"], cr.MANIFEST_STATES)
        for key in ("trusted", "hard_binding", "ai_claimed", "camera_claimed"):
            self.assertIs(type(result[key]), bool, key)
        self.assertIsInstance(result["path"], str)
        self.assertTrue(Path(result["path"]).is_absolute())
        self.assertEqual(result["path"], str(FIXTURE3.resolve()))
        self.assertIsInstance(result["validation_status_codes"], list)
        for code in result["validation_status_codes"]:
            self.assertIsInstance(code, str)
        self.assertTrue(result["signer_issuer"] is None
                        or isinstance(result["signer_issuer"], str))
        self.assertTrue(result["validation_state"] is None
                        or isinstance(result["validation_state"], str))
        self.assertTrue(result["error"] is None
                        or isinstance(result["error"], str))
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_no_production_reference(self):
        allowed = {SCRIPT.resolve()}
        offenders = []
        for root in (WS / "deepfake", WS / "web"):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if (not path.is_file()
                        or path.resolve() in allowed
                        or "__pycache__" in path.parts):
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                if "c2pa_read" in text:
                    offenders.append(str(path.relative_to(WS)))
        self.assertEqual(offenders, [])

    def test_source_read_only_offline(self):
        source = SCRIPT.read_text()
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        allowed_imports = {"argparse", "io", "json", "pathlib", "sys",
                           "typing", "c2pa", "__future__"}
        self.assertLessEqual(imports, allowed_imports)

        write_methods = {"write_text", "write_bytes", "mkdir", "unlink",
                         "rmdir", "rename", "replace", "touch", "chmod",
                         "symlink_to", "hardlink_to"}
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in {
                    "open", "remove", "unlink", "rmdir", "makedirs"}:
                if func.id != "open":
                    offenders.append(func.id)
                    continue
                modes = [a.value for a in node.args
                         if isinstance(a, ast.Constant)
                         and isinstance(a.value, str)]
                modes += [kw.value.value for kw in node.keywords
                          if isinstance(kw.value, ast.Constant)
                          and isinstance(kw.value.value, str)]
                if any(any(ch in mode for ch in "wax+") for mode in modes):
                    offenders.append("open-write")
            elif isinstance(func, ast.Attribute):
                if func.attr in write_methods:
                    offenders.append(func.attr)
                elif func.attr == "open":
                    modes = [a.value for a in node.args
                             if isinstance(a, ast.Constant)
                             and isinstance(a.value, str)]
                    modes += [kw.value.value for kw in node.keywords
                              if isinstance(kw.value, ast.Constant)
                              and isinstance(kw.value.value, str)]
                    if any(any(ch in mode for ch in "wax+") for mode in modes):
                        offenders.append("Path.open-write")
        self.assertEqual(offenders, [])

        try_create = any(
            isinstance(node, ast.Attribute) and node.attr == "try_create"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "Reader"
            for node in ast.walk(tree)
        )
        self.assertTrue(try_create)

        forbidden_names = {"Settings", "Context", "ContextBuilder",
                           "ContextProvider", "Signer", "Builder", "requests",
                           "urllib", "socket", "ssl", "http", "certifi",
                           "truststore"}
        used_names = {node.id for node in ast.walk(tree)
                      if isinstance(node, ast.Name)}
        self.assertFalse(forbidden_names & used_names)

    def test_cli_missing_and_size_bound(self):
        missing = WS / "artifacts" / "p9_system" / "__missing_fixture__.png"
        missing_result = run_cli(missing)
        self.assertEqual(missing_result.returncode, 2)
        missing_json = json.loads(missing_result.stdout)
        self.assertEqual(missing_json["manifest_state"], "error")
        self.assertIn("file_not_found", missing_json["error"])
        self.assertTrue(missing_result.stderr.strip())

        large_result = run_cli(FIXTURE3, "--max-bytes", "1")
        self.assertEqual(large_result.returncode, 2)
        large_json = json.loads(large_result.stdout)
        self.assertEqual(large_json["manifest_state"], "error")
        self.assertIn("file_too_large", large_json["error"])

    def test_doc_transparency_contract(self):
        self.assertTrue(DOC.is_file(), f"missing doc: {DOC}")
        text = DOC.read_text().lower()
        phrases = [
            "c2pa is not a detection gate",
            "transparency-only",
            "signed fakes exist",
            "the in-repo test chain is untrusted and must never be reported as trusted.",
            "trust is not world truth.",
            "0 detection",
            "64.4/100",
            "hard_binding",
            "well_formed_valid",
            "absent",
            "trusted",
            "network",
            "trust store",
        ]
        for phrase in phrases:
            self.assertIn(phrase, text)
        concepts = [
            "absence is not evidence",
            "hard binding",
            "revocation",
            "timestamp",
            "trust anchor",
            "synthetic",
            "bounded",
            "cli",
            "api",
            "unbooked",
        ]
        for concept in concepts:
            self.assertIn(concept, text)

    def test_absent_classify_none(self):
        result = cr.classify_manifest(None)
        self.assertEqual(result["manifest_state"], "absent")
        self.assertEqual(result["path"], "<memory>")
        self.assertIsNone(result["error"])
        self.assertEqual(result["validation_status_codes"], [])

    def test_error_precedence_and_non_dict(self):
        errored = cr.classify_manifest({"validation_state": "Valid"},
                                       error="reader_error: boom")
        self.assertEqual(errored["manifest_state"], "error")
        self.assertEqual(errored["error"], "reader_error: boom")
        self.assertIsNone(errored["validation_state"])
        self.assertEqual(errored["validation_status_codes"], [])
        non_dict = cr.classify_manifest(["not", "a", "dict"])
        self.assertEqual(non_dict["manifest_state"], "error")
        self.assertTrue(non_dict["error"].startswith("invalid_manifest_json"))
        truncated = cr.classify_manifest({}, error="x" * 600)
        self.assertLessEqual(len(truncated["error"]), 500)

    def test_status_code_extraction_union(self):
        manifest = {
            "active_manifest": "m",
            "validation_state": "Valid",
            "validation_status": [
                "zeta.code",
                {"code": " alpha.code "},
                {"code": ""},
                {"nope": 1},
            ],
            "manifests": {
                "m": {
                    "failure": [{"code": "m.code"}, "zeta.code"],
                    "success": [],
                },
            },
        }
        result = cr.classify_manifest(manifest)
        self.assertEqual(result["validation_status_codes"],
                         ["alpha.code", "m.code", "zeta.code"])

    def test_invalid_max_bytes(self):
        result = cr.read_c2pa(FIXTURE3, max_bytes=0)
        self.assertEqual(result["manifest_state"], "error")
        self.assertIn("invalid_max_bytes", result["error"])
        cli_result = run_cli(FIXTURE3, "--max-bytes", "0")
        self.assertEqual(cli_result.returncode, 2)
        self.assertIn("invalid_max_bytes", json.loads(cli_result.stdout)["error"])

    def test_cli_pretty_output(self):
        result = run_cli(FIXTURE3, "--pretty")
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.stdout.endswith("\n"))
        self.assertIn("\n  ", result.stdout)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["manifest_state"], "well_formed_valid")

    def test_classify_label_and_signer_stripping(self):
        manifest = json.loads(json.dumps(SYNTH_TRUSTED))
        manifest["manifests"]["urn:test:trusted"]["signature_info"] = {
            "issuer": "  Trusted Test CA  ",
        }
        result = cr.classify_manifest(manifest, path="label:test")
        self.assertEqual(result["path"], "label:test")
        self.assertEqual(result["signer_issuer"], "Trusted Test CA")


if __name__ == "__main__":
    unittest.main()
