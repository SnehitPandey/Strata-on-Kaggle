import os
import hashlib
import contextlib
import io
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.strata_kaggle import (
    redact,
    safe_extract_runtime,
    _sha256,
    context_window_tokens,
    opencode_config_snippet,
    _cached_model_path,
    run_streaming,
    restore_model_cache,
    _copy_verified,
    launch_strata,
    run_plan,
    start_ngrok_tunnel,
    MODEL_CACHE_FORMAT,
    RUNTIME_CACHE_FORMAT,
)


class StrataKaggleTests(unittest.TestCase):
    def test_redacts_strata_key(self):
        text = "API key sk-strata-very-secret123\n"
        self.assertIn("sk-strata-<redacted>", redact(text))
        self.assertNotIn("very-secret123", redact(text))

    def test_redacts_ngrok_authtoken(self):
        text = "NGROK_AUTHTOKEN=2abcDEF_secretvalue\n"
        self.assertIn("NGROK_AUTHTOKEN=<redacted>", redact(text))
        self.assertNotIn("secretvalue", redact(text))

    def test_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "f"
            path.write_bytes(b"hello")
            self.assertEqual(_sha256(path), hashlib.sha256(b"hello").hexdigest())

    def test_runtime_extractor_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            archive = td_path / "bad.tar"
            with tarfile.open(archive, "w:") as tf:
                info = tarfile.TarInfo("../evil")
                info.size = 1
                import io
                tf.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(RuntimeError):
                safe_extract_runtime(archive, td_path / "out")
            self.assertFalse((td_path / "evil").exists())

    def test_runtime_extractor_rejects_cross_tree_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            archive = td_path / "cross-tree.tar"
            with tarfile.open(archive, "w:") as tf:
                info = tarfile.TarInfo("terminal/escape")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../components/engine"
                tf.addfile(info)
            with self.assertRaises(RuntimeError):
                safe_extract_runtime(archive, td_path / "out")


    @unittest.skipIf(os.name != "posix", "runtime cache symlinks are created on the Kaggle/Linux runtime")
    def test_runtime_extractor_rewrites_internal_absolute_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            runtime_root = td_path / "strata-root"
            archive = td_path / "runtime.tar"
            with tarfile.open(archive, "w:") as tf:
                for directory in (
                    "terminal/0.2.6",
                    "terminal/0.2.6/bin",
                ):
                    info = tarfile.TarInfo(directory)
                    info.type = tarfile.DIRTYPE
                    tf.addfile(info)

                binary = tarfile.TarInfo("terminal/0.2.6/bin/strata")
                payload = b"strata-binary"
                binary.size = len(payload)
                tf.addfile(binary, io.BytesIO(payload))

                link = tarfile.TarInfo("terminal/current")
                link.type = tarfile.SYMTYPE
                link.linkname = str(runtime_root / "terminal" / "0.2.6")
                tf.addfile(link)

            destination = td_path / "staging"
            with patch("scripts.strata_kaggle.STRATA_ROOT", runtime_root):
                safe_extract_runtime(archive, destination)

            current = destination / "terminal" / "current"
            self.assertTrue(current.is_symlink())
            self.assertEqual(os.readlink(current), "0.2.6")
            self.assertEqual(
                (current / "bin" / "strata").read_bytes(),
                b"strata-binary",
            )

    @unittest.skipIf(os.name != "posix", "absolute POSIX link handling is validated on the Kaggle/Linux runtime")
    def test_runtime_extractor_rejects_external_absolute_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            archive = td_path / "external-link.tar"
            with tarfile.open(archive, "w:") as tf:
                link = tarfile.TarInfo("terminal/escape")
                link.type = tarfile.SYMTYPE
                link.linkname = "/etc/passwd"
                tf.addfile(link)

            with patch("scripts.strata_kaggle.STRATA_ROOT", td_path / "strata-root"):
                with self.assertRaisesRegex(RuntimeError, "Unsafe absolute link"):
                    safe_extract_runtime(archive, td_path / "out")

    def test_cached_model_path_selects_expected_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "models" / "gyro-Gyro-S"
            model_dir.mkdir(parents=True)
            (model_dir / "model.gguf").write_bytes(b"model")
            found = _cached_model_path(root, "gyro", "gyro-s")
            self.assertEqual(found, model_dir)


    def test_run_streaming_prints_subprocess_output(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = run_streaming(
                [sys.executable, "-c", "print('progress-visible'); print('done')"],
                timeout=10,
            )
        self.assertIn("progress-visible", buffer.getvalue())
        self.assertIn("done", result.stdout)


    def test_run_streaming_reports_heartbeat(self):
        with tempfile.TemporaryDirectory() as td:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = run_streaming(
                    [sys.executable, "-c", "import time; time.sleep(0.7)"],
                    timeout=10,
                    heartbeat_interval=0.1,
                    heartbeat_label="test launch",
                    progress_paths=[Path(td)],
                )
            self.assertEqual(result.returncode, 0)
            self.assertIn("test launch still running", buffer.getvalue())
            self.assertIn("0.00 GiB", buffer.getvalue())


    def test_copy_verified_reports_byte_progress_events(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "source.bin"
            dst = root / "dest.bin"
            payload = b"progress-bytes-123"
            src.write_bytes(payload)
            events = []

            _copy_verified(
                src,
                dst,
                lambda amount, path, event: events.append((amount, path, event)),
            )

            self.assertEqual(dst.read_bytes(), payload)
            self.assertEqual(
                sum(amount for amount, _path, event in events if event == "progress"),
                len(payload),
            )
            self.assertEqual(events[0][2], "start")
            self.assertEqual(events[-1][2], "complete")

    def test_restore_model_cache_copy_mode_reports_byte_progress(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "test-owner" / "strata-model-gyro-gyro-s"
            dataset.mkdir(parents=True)
            payload = b"model-progress-data"
            flattened = dataset / "0001__model.gguf"
            flattened.write_bytes(payload)
            manifest = {
                "format": "strata.model-cache.v1",
                "model": {"family": "gyro", "variant": "gyro-s"},
                "files": [{
                    "dataset_name": flattened.name,
                    "relative_path": "models/gyro-Gyro-S/model.gguf",
                    "size": len(payload),
                }],
            }
            (dataset / "manifest.json").write_text(
                __import__("json").dumps(manifest), encoding="utf-8"
            )
            output = io.StringIO()
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                with contextlib.redirect_stdout(output):
                    restored = restore_model_cache(
                        "test-owner",
                        "strata-model-gyro-gyro-s",
                        root / "local-model",
                        family="gyro",
                        variant="gyro-s",
                        restore_mode="copy",
                    )

            self.assertTrue(restored)
            self.assertIn("Model restore:", output.getvalue())
            self.assertIn("100.0%", output.getvalue())
            self.assertIn("MiB/s", output.getvalue())

    @unittest.skipIf(os.name != "posix", "symlink restore is validated on the Kaggle/Linux runtime")
    def test_restore_model_cache_symlink_mode_avoids_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input" / "datasets"
            dataset = input_root / "test-owner" / "strata-model-gyro-gyro-s"
            dataset.mkdir(parents=True)
            payload = b"model-bytes"
            flattened = dataset / "0001__model.gguf"
            flattened.write_bytes(payload)
            manifest = {
                "format": "strata.model-cache.v1",
                "model": {"family": "gyro", "variant": "gyro-s"},
                "files": [{
                    "dataset_name": flattened.name,
                    "relative_path": "models/gyro-Gyro-S/model.gguf",
                    "size": len(payload),
                }],
            }
            (dataset / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
            target_root = root / "local-model"
            with patch("scripts.strata_kaggle.INPUT_ROOT", input_root):
                restored = restore_model_cache(
                    "test-owner", "strata-model-gyro-gyro-s", target_root,
                    family="gyro", variant="gyro-s", restore_mode="symlink",
                )
            linked = target_root / "models" / "gyro-Gyro-S" / "model.gguf"
            self.assertTrue(restored)
            self.assertTrue(linked.is_symlink())
            self.assertEqual(linked.read_bytes(), payload)
            self.assertEqual(linked.resolve(), flattened.resolve())

    def test_context_window_tokens(self):
        self.assertEqual(context_window_tokens("32k"), 32 * 1024)
        self.assertEqual(context_window_tokens("200k"), 200 * 1024)
        self.assertEqual(context_window_tokens("4096"), 4096)
        self.assertEqual(context_window_tokens("1M"), 1024 * 1024)
        with self.assertRaises(ValueError):
            context_window_tokens("large")

    def test_open_code_snippet_has_provider_wrapper_and_no_key(self):
        value = opencode_config_snippet("https://sample.ngrok-free.app", "qwen3.8-flash-next-gyro-gyro-s")
        import json
        parsed = json.loads(value)
        self.assertIn("strata", parsed)
        self.assertEqual(parsed["strata"]["settings"]["apiKey"], "{file:~/.config/opencode/strata-key.txt}")
        self.assertEqual(
            parsed["strata"]["models"]["qwen3.8-flash-next-gyro-gyro-s"]["reasoningEffort"],
            "none",
        )
        self.assertEqual(
            parsed["strata"]["models"]["qwen3.8-flash-next-gyro-gyro-s"]["limit"]["context"],
            200 * 1024,
        )
        self.assertNotIn("sk-strata-", value)

    def test_open_code_snippet_accepts_context_override(self):
        import json
        parsed = json.loads(
            opencode_config_snippet(
                "https://sample.ngrok-free.app",
                "model-id",
                context_tokens=context_window_tokens("32k"),
                output_tokens=4096,
            )
        )
        limit = parsed["strata"]["models"]["model-id"]["limit"]
        self.assertEqual(limit, {"context": 32 * 1024, "output": 4096})

    def test_launch_strata_uses_configured_access_mode(self):
        captured = {}

        class Result:
            returncode = 0
            stdout = ""

        def fake_streaming(command, **_kwargs):
            captured["command"] = command
            return Result()

        with patch("scripts.strata_kaggle._model_source_args", return_value=[]):
            with patch("scripts.strata_kaggle.run_streaming", side_effect=fake_streaming):
                launch_strata(
                    "gyro", "gyro-s", "200k", "off", "0",
                    Path("/tmp/model"), access_mode="lan",
                )

        command = captured["command"]
        self.assertEqual(command[command.index("--access") + 1], "lan")
        self.assertIn("--no-tui", command)

    def test_run_plan_uses_configured_access_mode(self):
        captured = {}

        class Result:
            returncode = 0
            stdout = "plan output"

        def fake_streaming(command, **_kwargs):
            captured["command"] = command
            return Result()

        with patch("scripts.strata_kaggle._model_source_args", return_value=[]):
            with patch("scripts.strata_kaggle.run_streaming", side_effect=fake_streaming):
                output = run_plan(
                    "gyro", "gyro-s", "200k", "off", "0",
                    Path("/tmp/model"), access_mode="lan",
                )

        command = captured["command"]
        self.assertIn("--plan", command)
        self.assertEqual(command[command.index("--access") + 1], "lan")
        self.assertEqual(output, "plan output")

    def test_cache_manifest_formats(self):
        self.assertEqual(MODEL_CACHE_FORMAT, "strata.model-cache.v1")
        self.assertEqual(RUNTIME_CACHE_FORMAT, "strata.runtime-cache.v1")

    def test_start_ngrok_tunnel_reuses_existing_tunnel(self):
        class Tunnel:
            config = {"addr": "http://127.0.0.1:8080"}
            public_url = "http://sample.ngrok-free.app"

        class FakeNgrok:
            def __init__(self):
                self.connect_calls = 0
                self.token = None

            def set_auth_token(self, token):
                self.token = token

            def get_tunnels(self):
                return [Tunnel()]

            def connect(self, *_args, **_kwargs):
                self.connect_calls += 1
                return Tunnel()

        fake = FakeNgrok()
        with patch("scripts.strata_kaggle.install_ngrok_if_missing"):
            with patch("scripts.strata_kaggle._load_ngrok", return_value=fake):
                url = start_ngrok_tunnel(auth_token="test-token")

        self.assertEqual(url, "https://sample.ngrok-free.app")
        self.assertEqual(fake.token, "test-token")
        self.assertEqual(fake.connect_calls, 0)

    def test_start_ngrok_tunnel_passes_optional_domain(self):
        class Tunnel:
            config = {"addr": "http://127.0.0.1:8080"}
            public_url = "http://sample.ngrok-free.app"

        class FakeNgrok:
            def __init__(self):
                self.options = None

            def set_auth_token(self, _token):
                pass

            def get_tunnels(self):
                return []

            def connect(self, _port, _proto, **options):
                self.options = options
                return Tunnel()

        fake = FakeNgrok()
        with patch("scripts.strata_kaggle.install_ngrok_if_missing"):
            with patch("scripts.strata_kaggle._load_ngrok", return_value=fake):
                url = start_ngrok_tunnel(auth_token="test-token", domain="my-name.ngrok-free.app")

        self.assertEqual(url, "https://sample.ngrok-free.app")
        self.assertEqual(fake.options, {"domain": "my-name.ngrok-free.app"})


if __name__ == "__main__":
    unittest.main()
