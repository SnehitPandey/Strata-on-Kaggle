"""Guided Kaggle helpers for running Strata-supported models safely."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import platform
import posixpath
import re
import shutil
import signal
import queue
import threading
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

WORKING = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input/datasets")
STRATA_ROOT = Path("/root/.local/share/strata")
STRATA_BIN = Path("/root/.local/bin/strata")
STRATA_PORT = 8080
DEFAULT_ACCESS_MODE = "lan"
DEFAULT_NGROK_SECRET = "NGROK_AUTHTOKEN"
MODEL_CACHE_FORMAT = "strata.model-cache.v1"
RUNTIME_CACHE_FORMAT = "strata.runtime-cache.v1"
DISK_SAFETY_MARGIN_GB = 8.0


def log(message: str) -> None:
    print("[Strata] " + message, flush=True)


def disk_free_gb(path: Path) -> float | None:
    """Free space on the filesystem holding `path`, in decimal GB, or None if it cannot be read."""
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


def check_download_fits(data_dir: Path, download_gb: float | None,
                        margin_gb: float = DISK_SAFETY_MARGIN_GB) -> None:
    """Stop before a download that cannot fit on the filesystem holding `data_dir`."""
    if download_gb is None or download_gb <= 0:
        return
    free_gb = disk_free_gb(data_dir)
    if free_gb is None:
        return
    needed_gb = download_gb + margin_gb
    if free_gb >= needed_gb:
        return
    raise RuntimeError(
        f"Not enough free disk: {free_gb:.1f} GB free, about {needed_gb:.1f} GB needed "
        f"({download_gb:.1f} GB download + {margin_gb:.0f} GB scratch/reserve). "
        "This check uses the actual free space for MODEL_DATA_DIR; Kaggle's displayed disk "
        "maximum can be lower than the real /root overlay. If space is genuinely low, attach "
        "your private model and runtime datasets in Add Input and set "
        "MODEL_RESTORE_MODE='symlink' so the model is read from /kaggle/input instead of "
        "downloaded or copied. Nothing was downloaded."
    )


def redact(text: str) -> str:
    text = re.sub(r"sk-strata-[A-Za-z0-9._~-]+", "sk-strata-<redacted>", text)
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)\S+", r"\1<redacted>", text)
    text = re.sub(r"(?i)(KAGGLE_API_TOKEN\s*=\s*)\S+", r"\1<redacted>", text)
    text = re.sub(r"(?i)(NGROK_AUTHTOKEN\s*=\s*)\S+", r"\1<redacted>", text)
    return text


def run_command(command: list[str], *, check: bool = True, timeout: int | None = None,
                env: dict[str, str] | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None, env=env,
        timeout=timeout, check=False,
    )
    output = (result.stdout or "") if capture else ""
    safe_output = redact(output)
    if safe_output.strip():
        print(safe_output, end="" if safe_output.endswith("\n") else "\n", flush=True)
    if check and result.returncode != 0:
        raise RuntimeError("Command failed (" + str(result.returncode) + "): " +
                           " ".join(command) + "\n" + safe_output[-5000:])
    return result


def kaggle_env(username: str, secret_name: str = "KAGGLE_API_TOKEN") -> dict[str, str]:
    if not username or username.lower().startswith("your-"):
        raise ValueError("Set KAGGLE_USERNAME in the notebook configuration cell.")
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(secret_name).strip()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle Secret " + secret_name + " was not available. Add it in Add-ons > Secrets, "
            "enable notebook access, and rerun. Do not paste the token into notebook code."
        ) from exc
    if not token:
        raise RuntimeError("Kaggle Secret " + secret_name + " is empty.")
    # Keep credentials outside the saved notebook workspace and never print the token.
    home = Path("/tmp/strata-kaggle-home")
    config = home / ".kaggle"
    config.mkdir(parents=True, exist_ok=True)
    token_path = config / "access_token"
    token_path.write_text(token + "\n", encoding="utf-8")
    token_path.chmod(0o600)
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "TMPDIR": "/kaggle/working/tmp",
        "XDG_CACHE_HOME": str(home / ".cache"),
        "KAGGLE_CONFIG_DIR": str(config),
        "KAGGLE_USERNAME": username,
    })
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = str(config)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["TMPDIR"] = env["TMPDIR"]
    os.environ["XDG_CACHE_HOME"] = env["XDG_CACHE_HOME"]
    log("Kaggle Secret loaded into a temporary credential directory (value hidden).")
    return env


def install_kaggle_cli(env: dict[str, str] | None = None) -> None:
    run_command([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "kaggle"],
                timeout=900, env=env)
    run_command(["kaggle", "--version"], timeout=60, env=env)


def _directory_snapshot(path: Path) -> tuple[int, int]:
    """Return file count and total bytes without following symlinks."""
    count = 0
    size = 0
    if not path.exists():
        return count, size
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
        for name in files:
            file_path = Path(root) / name
            try:
                if file_path.is_symlink():
                    continue
                stat = file_path.stat()
                count += 1
                size += stat.st_size
            except OSError:
                continue
    return count, size


def _terminate_process(process: subprocess.Popen, force: bool = False) -> None:
    if process.poll() is not None:
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, sig)
            return
        except OSError:
            pass
    if force:
        process.kill()
    else:
        process.terminate()


def run_streaming(command: list[str], *, check: bool = True, timeout: int | None = None,
                  env: dict[str, str] | None = None,
                  heartbeat_interval: float | None = None,
                  heartbeat_label: str = "Command",
                  progress_paths: list[Path] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command while streaming output and reporting real directory-size heartbeats."""
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env,
        bufsize=8192, start_new_session=True,
    )
    if process.stdout is None:
        raise RuntimeError("Could not capture process output.")
    output_queue: queue.Queue[bytes | None] = queue.Queue()
    captured: list[str] = []

    def pump() -> None:
        pending = bytearray()
        try:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                pending.extend(chunk)
                while True:
                    split_at = next((i for i, byte in enumerate(pending) if byte in (10, 13)), None)
                    if split_at is None:
                        break
                    separator = pending[split_at]
                    piece = bytes(pending[:split_at + 1])
                    del pending[:split_at + 1]
                    # Combine CRLF into one newline event.
                    if separator == 13 and pending[:1] == b"\n":
                        piece += b"\n"
                        del pending[:1]
                    output_queue.put(piece)
            if pending:
                output_queue.put(bytes(pending))
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout if timeout is not None else None
    last_heartbeat = time.monotonic()
    previous_snapshots: dict[str, tuple[int, int]] = {}
    reached_eof = False
    try:
        while not reached_eof:
            if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                _terminate_process(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _terminate_process(process, force=True)
                    process.wait()
                raise TimeoutError(
                    f"Command timed out after {timeout}s: {' '.join(command)}. "
                    f"Recent output: {redact(''.join(captured)[-3000:])}"
                )
            try:
                item = output_queue.get(timeout=0.5)
            except queue.Empty:
                now = time.monotonic()
                if heartbeat_interval and now - last_heartbeat >= heartbeat_interval and process.poll() is None:
                    details = []
                    for path in progress_paths or []:
                        count, size = _directory_snapshot(path)
                        label = str(path)
                        current = (count, size)
                        previous = previous_snapshots.get(label)
                        change = ""
                        if previous is not None:
                            delta = size - previous[1]
                            if delta > 0:
                                change = f", +{delta / (1024**2):.1f} MiB since last check"
                        details.append(f"{path}: {count} files, {size / (1024**3):.2f} GiB{change}")
                        previous_snapshots[label] = current
                    print(f"[Strata] {heartbeat_label} still running ({int(now - (deadline - timeout)) if deadline and timeout else int(now - last_heartbeat)}s elapsed)."
                          + (("\n  " + "\n  ".join(details)) if details else ""), flush=True)
                    last_heartbeat = now
                continue
            if item is None:
                reached_eof = True
                continue
            text_piece = item.decode("utf-8", errors="replace")
            safe_piece = redact(text_piece)
            captured.append(text_piece)
            if text_piece.endswith("\r") and not text_piece.endswith("\r\n"):
                print("\r" + safe_piece.rstrip("\r"), end="", flush=True)
            else:
                print(safe_piece, end="" if safe_piece.endswith("\n") else "\n", flush=True)
    finally:
        if process.poll() is None:
            _terminate_process(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_process(process, force=True)
                process.wait()
        reader.join(timeout=2)
        process.stdout.close()

    output = redact("".join(captured))
    completed = subprocess.CompletedProcess(command, process.returncode, output, None)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            + output[-5000:]
        )
    return completed


def _gpu_names() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=20, check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def hardware_summary() -> dict[str, Any]:
    return {"platform": platform.platform(), "machine": platform.machine(), "gpu_names": _gpu_names()}


def default_gpu_list() -> str:
    names = _gpu_names()
    return ",".join(str(i) for i in range(len(names))) if names else "0"


def find_attached_dataset(owner: str, slug: str, marker: str | None = None) -> Path | None:
    exact = INPUT_ROOT / owner / slug
    if exact.is_dir() and (marker is None or (exact / marker).is_file()):
        return exact
    if not INPUT_ROOT.exists():
        return None
    marker_name = marker or "dataset-metadata.json"
    for candidate in INPUT_ROOT.rglob(marker_name):
        parent = candidate.parent
        if parent.name == slug and (parent.parent.name == owner or owner in parent.parts):
            return parent
    return None


def safe_extract_runtime(archive: Path, destination: Path) -> None:
    """Safely extract runtime paths, normalizing internal absolute symlinks.

    Runtime archives created from STRATA_ROOT can contain absolute symlinks
    such as terminal/current -> /root/.local/share/strata/terminal/<version>.
    Rewrite those to archive-relative links, but only when the target remains
    inside STRATA_ROOT and within the same allowed top-level runtime tree.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    runtime_root = Path(STRATA_ROOT).absolute()

    with tarfile.open(archive, "r:") as tf:
        members = tf.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError("Unsafe path in runtime archive: " + member.name)
            if not name.parts or name.parts[0] not in {"terminal", "components", "tools"}:
                raise RuntimeError("Unexpected runtime archive path: " + member.name)

            if not (member.issym() or member.islnk()):
                continue

            link_text = member.linkname
            if os.path.isabs(link_text):
                # Absolute links are only accepted when their target is inside
                # the known Strata runtime root. Convert that target back to
                # a path relative to the archive's top-level tree.
                absolute_target = Path(os.path.normpath(link_text))
                try:
                    archive_target = absolute_target.relative_to(runtime_root)
                except ValueError as exc:
                    raise RuntimeError(
                        "Unsafe absolute link in runtime archive: " + member.name
                    ) from exc

                resolved = PurePosixPath(archive_target.as_posix())
                if (
                    not resolved.parts
                    or resolved.parts[0] != name.parts[0]
                    or ".." in resolved.parts
                ):
                    raise RuntimeError(
                        "Unsafe absolute/cross-tree link in runtime archive: "
                        + member.name
                    )

                if member.issym():
                    # Symlinks are relative to the link's containing directory.
                    member.linkname = posixpath.relpath(
                        resolved.as_posix(),
                        start=name.parent.as_posix(),
                    )
                else:
                    # Tar hard-link targets are relative to archive root.
                    member.linkname = resolved.as_posix()
                continue

            link = PurePosixPath(link_text)
            if member.issym():
                resolved = PurePosixPath(
                    posixpath.normpath(str(name.parent / link))
                )
            else:
                resolved = PurePosixPath(posixpath.normpath(str(link)))
            if (
                not resolved.parts
                or resolved.parts[0] != name.parts[0]
                or ".." in resolved.parts
            ):
                raise RuntimeError(
                    "Unsafe cross-tree link in runtime archive: " + member.name
                )

        tf.extractall(destination, members=members, filter="data")



def _remove_existing_path(path: Path) -> None:
    """Remove a directory/file/symlink safely before atomically installing staged runtime files."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)

def restore_runtime_cache(owner: str, slug: str) -> bool:
    ds = find_attached_dataset(owner, slug, "strata-runtime-cache.bin")
    if not ds:
        log("Runtime cache not attached; Strata will be installed if needed.")
        return False
    archive = ds / "strata-runtime-cache.bin"
    if not archive.is_file():
        log("Runtime archive missing; skipping runtime restore.")
        return False
    checksum = ds / "strata-runtime-cache.bin.sha256"
    if checksum.is_file():
        expected = checksum.read_text(encoding="utf-8").strip().split()[0]
        if _sha256(archive) != expected:
            raise RuntimeError("Runtime cache checksum mismatch; refusing extraction.")

    manifest = ds / "runtime-manifest.json"
    meta: dict[str, Any] = {}
    if manifest.is_file():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("machine") and meta["machine"] != platform.machine():
            log("Runtime machine differs from cached build; skipping cache restore.")
            return False
        saved_gpus = meta.get("gpu_names") or []
        current_gpus = _gpu_names()
        if saved_gpus and current_gpus and saved_gpus != current_gpus:
            log("GPU differs from cache; restoring terminal but Strata may need a compatible engine.")

    # Avoid overwriting an executable currently being used by a Strata service/CLI.
    # If the installed runtime matches this cache, reuse it without extracting over it.
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    existing_version = None
    if shutil.which("strata"):
        probe = subprocess.run(["strata", "--version"], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=60, check=False)
        existing_version = (probe.stdout or "").strip().splitlines()[0] if probe.stdout else None
    cached_version = str(meta.get("strata_version", "")).strip()
    engine_dirs = [str(x) for x in meta.get("engine_dirs", [])]
    engine_cache_ok = bool(engine_dirs) and all(
        (STRATA_ROOT / "components" / "engine" / name).is_dir() for name in engine_dirs
    )
    if (existing_version and cached_version and existing_version == cached_version
            and STRATA_BIN.exists() and engine_cache_ok):
        log("A matching Strata terminal and engine are already installed; reusing them without extraction to avoid overwriting a busy Python executable.")
        return True

    if shutil.which("strata"):
        log("Stopping any existing Strata server before replacing its runtime files…")
        stopped = subprocess.run(["strata", "launch", "--stop"], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=90, check=False)
        if stopped.stdout and stopped.stdout.strip():
            print(redact(stopped.stdout), flush=True)
        # The CLI can report that nothing was running; that is not a failure.
        time.sleep(1)

    log("Verifying and extracting Strata runtime cache into a staging directory…")
    staging_root = Path("/root/.local/share/strata-restore-staging")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    safe_extract_runtime(archive, staging_root)

    staged_terminal = staging_root / "terminal"
    staged_components = staging_root / "components"
    if not staged_terminal.is_dir() or not staged_components.is_dir():
        shutil.rmtree(staging_root, ignore_errors=True)
        raise RuntimeError("Runtime archive must contain terminal/ and components/ directories.")
    staged_bin = staged_terminal / "current" / "bin" / "strata"
    if not staged_bin.exists():
        matches = sorted(staged_terminal.glob("*/bin/strata"))
        if not matches:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise RuntimeError("No strata executable found in runtime archive.")

    STRATA_ROOT.mkdir(parents=True, exist_ok=True)
    # Replace only after staging has extracted and validated successfully.
    for part in ("terminal", "components"):
        old_path = STRATA_ROOT / part
        _remove_existing_path(old_path)
        os.replace(staging_root / part, old_path)
    staged_tools = staging_root / "tools"
    if staged_tools.is_dir():
        tools_root = STRATA_ROOT / "tools"
        tools_root.mkdir(parents=True, exist_ok=True)
        for tool in staged_tools.iterdir():
            target = tools_root / tool.name
            if target.exists() or target.is_symlink():
                target.unlink()
            os.replace(tool, target)
    shutil.rmtree(staging_root, ignore_errors=True)

    STRATA_BIN.parent.mkdir(parents=True, exist_ok=True)
    target = STRATA_ROOT / "terminal" / "current" / "bin" / "strata"
    if not target.exists():
        matches = sorted((STRATA_ROOT / "terminal").glob("*/bin/strata"))
        if not matches:
            raise RuntimeError("No strata executable found in restored runtime cache.")
        target = matches[-1]
    STRATA_BIN.unlink(missing_ok=True)
    STRATA_BIN.symlink_to(target)
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    run_command([str(STRATA_BIN), "--version"], timeout=60)
    log("Runtime cache restore completed.")
    return True


def install_strata_if_missing() -> None:
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    if shutil.which("strata"):
        run_command(["strata", "--version"], timeout=60)
        return
    installer = Path("/tmp/strata-install.sh")
    run_command(["curl", "-fsSL",
                 "https://github.com/Niko1221/Strata/releases/latest/download/install.sh",
                 "-o", str(installer)], timeout=180)
    run_command(["sh", str(installer)], timeout=900)
    os.environ["PATH"] = "/root/.local/bin:" + os.environ.get("PATH", "")
    if not shutil.which("strata"):
        raise RuntimeError("Strata installer completed but the command is not on PATH.")
    run_command(["strata", "--version"], timeout=60)


def _copy_verified(
    src: Path,
    dst: Path,
    progress_callback: Callable[[int, Path, str], None] | None = None,
) -> tuple[int, bool]:
    """Copy one verified file and optionally report byte-level progress events."""
    expected = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(0, dst, "start")

    # Copy mode must materialize symlinked input files into local storage.
    if dst.is_symlink():
        dst.unlink()
    elif dst.is_file() and dst.stat().st_size == expected:
        if progress_callback:
            progress_callback(0, dst, "complete")
        return expected, False

    temp = dst.with_name(dst.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with src.open("rb") as inp, temp.open("wb") as out:
            while True:
                chunk = inp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                if progress_callback:
                    progress_callback(len(chunk), dst, "progress")
            out.flush()
            os.fsync(out.fileno())
        if temp.stat().st_size != expected:
            raise RuntimeError("Size verification failed: " + src.name)
        temp.replace(dst)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    if progress_callback:
        progress_callback(0, dst, "complete")
    return expected, True


def restore_model_cache(owner: str, slug: str, data_dir: Path, workers: int = 8,
                        family: str | None = None, variant: str | None = None,
                        restore_mode: str = "copy") -> bool:
    """Restore a private cache by copying files locally or symlinking to the Kaggle input mount.

    'copy' duplicates bytes into local storage and reports live byte-level progress.
    'symlink' avoids the extra local copy but may be slower because the engine reads the dataset mount.
    Kaggle still has to make attached input dataset files available to the runtime.
    """
    if restore_mode not in {"copy", "symlink"}:
        raise ValueError("restore_mode must be 'copy' or 'symlink'.")
    ds = find_attached_dataset(owner, slug, "manifest.json")
    if not ds:
        log("Model cache not attached; Strata will download the selected model on first launch.")
        return False
    try:
        manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("Cannot read model-cache manifest.") from exc
    if manifest.get("format") != MODEL_CACHE_FORMAT:
        log("Unrecognized model cache manifest; not restoring it.")
        return False
    saved = manifest.get("model", {})
    if family and saved.get("family") != family:
        raise RuntimeError("Attached model cache family does not match the selected family.")
    if variant and saved.get("variant") != variant:
        raise RuntimeError("Attached model cache variant does not match the selected variant.")
    entries = manifest.get("files", [])
    if not entries:
        raise RuntimeError("Model cache manifest is empty.")
    total = sum(int(e["size"]) for e in entries)
    data_dir.mkdir(parents=True, exist_ok=True)
    if restore_mode == "copy":
        free_gb = disk_free_gb(data_dir)
        needed_gb = total / 1e9 + 2.0
        if free_gb is not None and free_gb < needed_gb:
            log(f"Copy restore needs about {needed_gb:.1f} GB but only {free_gb:.1f} GB is free; "
                "using symlink mode instead so the model is not duplicated onto this disk.")
            restore_mode = "symlink"
    log("Restoring " + str(len(entries)) + " model files (" + f"{total / (1024**3):.2f} GiB" + ") using " + restore_mode + " mode.")

    validated: list[tuple[Path, Path, int]] = []
    for item in entries:
        rel = PurePosixPath(item["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("Unsafe path in model manifest: " + str(rel))
        src = ds / item["dataset_name"]
        dst = data_dir.joinpath(*rel.parts)
        size = int(item["size"])
        if not src.is_file() or src.stat().st_size != size:
            raise RuntimeError("Model cache file missing or size mismatch: " + item["dataset_name"])
        validated.append((src, dst, size))

    if restore_mode == "symlink":
        created = 0
        already_linked = 0
        for src, dst, size in validated:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                try:
                    if dst.resolve(strict=True) == src.resolve(strict=True) and dst.stat().st_size == size:
                        already_linked += 1
                        continue
                except OSError:
                    pass
                # Replace a stale/dangling symlink from a previous mount/session.
                dst.unlink()
            elif dst.exists():
                if dst.is_file() and dst.stat().st_size == size:
                    # Keep a valid existing local copy rather than replacing it with a link.
                    continue
                if dst.is_file():
                    # A failed/partial download can leave a truncated file here; the dataset
                    # copy was just size-checked, so replace the bad file and reclaim the space.
                    log("Replacing an incomplete local model file with a verified dataset link: " + str(dst))
                    dst.unlink()
                else:
                    raise RuntimeError("Restore target already exists with unexpected content: " + str(dst))
            dst.symlink_to(src.resolve())
            created += 1
        partials = [p for p in data_dir.rglob("*.part") if p.is_file()]
        if partials:
            reclaimed = sum(p.stat().st_size for p in partials)
            removed = 0
            for part in partials:
                try:
                    part.unlink()
                    removed += 1
                except OSError:
                    pass
            if removed:
                log(f"Removed {removed} stale partial download(s), reclaiming "
                    f"{reclaimed / (1024**3):.2f} GiB.")
        log(f"Model cache linked without copying model bytes: {created} links created, {already_linked} already linked.")
        log("Note: Kaggle still stages the attached input dataset; symlink mode avoids only the extra copy to local storage.")
        return True

    # Files already present as regular local files count as restored without re-copying.
    existing = {
        str(dst): (dst, size)
        for _src, dst, size in validated
        if not dst.is_symlink() and dst.is_file() and dst.stat().st_size == size
    }
    state: dict[str, Any] = {
        "progress_bytes": sum(size for _dst, size in existing.values()),
        "copied_bytes": 0,
        "completed": set(existing),
        "active": set(),
        "last_report_time": 0.0,
        "last_report_bytes": 0,
        "printed": False,
    }
    state_lock = threading.Lock()
    started_at = time.monotonic()
    report_interval = 1.5
    report_byte_interval = 512 * 1024 * 1024
    total_files = len(validated)

    def progress_callback(delta: int, dst: Path, event: str) -> None:
        now = time.monotonic()
        with state_lock:
            key = str(dst)
            if event == "start":
                state["active"].add(key)
            elif event == "progress":
                state["progress_bytes"] += delta
                state["copied_bytes"] += delta
            elif event == "complete":
                state["active"].discard(key)
                state["completed"].add(key)

            is_finished = (
                state["progress_bytes"] >= total
                and len(state["completed"]) >= total_files
                and not state["active"]
            )
            should_report = (
                not state["printed"]
                or now - state["last_report_time"] >= report_interval
                or state["progress_bytes"] - state["last_report_bytes"] >= report_byte_interval
                or is_finished
            )
            if not should_report:
                return

            elapsed_seconds = max(now - started_at, 0.001)
            elapsed = int(elapsed_seconds)
            elapsed_text = f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
            percent = 100.0 if total == 0 else min(100.0, 100 * state["progress_bytes"] / total)
            speed_mib_s = state["copied_bytes"] / (1024 ** 2) / elapsed_seconds
            progress_text = (
                f"[Strata] Model restore: "
                f"{state['progress_bytes'] / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GiB "
                f"({percent:.1f}%) | {len(state['completed'])}/{total_files} files complete "
                f"| {len(state['active'])} copying | {speed_mib_s:.1f} MiB/s "
                f"| {elapsed_text} elapsed"
            )
            print("\r" + progress_text, end="", flush=True)
            state["last_report_time"] = now
            state["last_report_bytes"] = state["progress_bytes"]
            state["printed"] = True

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_copy_verified, src, dst, progress_callback)
                for src, dst, _size in validated
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        if state["printed"]:
            print(flush=True)

    log("Model cache copied into local storage and size-verified.")
    return True



def _cached_model_path(data_dir: Path, family: str, variant: str) -> Path | None:
    """Find a compatible-looking local GGUF to prevent Strata redownloading restored models."""
    models_dir = data_dir / "models"
    if not models_dir.is_dir():
        return None
    files = [p for p in models_dir.rglob("*.gguf") if p.is_file() and not p.name.endswith(".part")]
    if not files:
        return None
    expected_folder = (family + "-" + variant).lower().replace("_", "-")
    matching = [p for p in files if expected_folder in p.parent.as_posix().lower().replace("_", "-")]
    if matching:
        return max(matching, key=lambda p: p.stat().st_size).parent
    parent_dirs = {p.parent for p in files}
    if len(parent_dirs) == 1:
        return next(iter(parent_dirs))
    raise RuntimeError(
        "Multiple cached GGUF model folders were found under " + str(models_dir) +
        ". Keep one selected model in this cache or choose a dedicated MODEL_DATA_DIR."
    )


def _model_source_args(data_dir: Path, family: str, variant: str) -> list[str]:
    cached = _cached_model_path(data_dir, family, variant)
    if cached:
        log("Using local model files at " + str(cached) + " (avoiding a new model download).")
        return ["--model-path", str(cached)]
    return ["--data-dir", str(data_dir)]


def plan_download_gb(family: str, variant: str, context: str, vision: str,
                     gpu_list: str, data_dir: Path,
                     access_mode: str = DEFAULT_ACCESS_MODE) -> float | None:
    """Ask Strata's machine-readable plan how much it would download, or None if unavailable."""
    cmd = ["strata", "launch", "--plan", "--family", family, "--model", variant,
           *_model_source_args(data_dir, family, variant), "--context", context, "--vision", vision,
           "--mtp", "off", "--tune", "skip", "--access", access_mode,
           "--port", str(STRATA_PORT), "--gpu", gpu_list, "--output", "json"]
    try:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=180, check=False)
        if result.returncode != 0 or not (result.stdout or "").strip():
            return None
        payload = json.loads(result.stdout)
        download = (payload.get("plan") or {}).get("download") or {}
        value = download.get("gb")
        return float(value) if value is not None else None
    except (OSError, ValueError, TypeError, AttributeError, subprocess.SubprocessError):
        return None


def run_plan(family: str, variant: str, context: str, vision: str,
             gpu_list: str, data_dir: Path, access_mode: str = DEFAULT_ACCESS_MODE) -> str:
    cmd = ["strata", "launch", "--plan", "--family", family, "--model", variant,
           *_model_source_args(data_dir, family, variant), "--context", context, "--vision", vision,
           "--mtp", "off", "--tune", "skip", "--access", access_mode,
           "--port", str(STRATA_PORT), "--gpu", gpu_list]
    result = run_streaming(cmd, timeout=300, check=False,
                           heartbeat_interval=15, heartbeat_label="Strata launch plan",
                           progress_paths=[data_dir, STRATA_ROOT / "components"])
    output = result.stdout or ""
    if result.returncode:
        raise RuntimeError("Strata plan failed. Choose a supported family/model variant.")
    return output


def launch_strata(family: str, variant: str, context: str, vision: str,
                  gpu_list: str, data_dir: Path, access_mode: str = DEFAULT_ACCESS_MODE) -> str:
    cmd = ["strata", "launch", "--family", family, "--model", variant,
           *_model_source_args(data_dir, family, variant), "--context", context, "--vision", vision,
           "--mtp", "off", "--tune", "skip", "--access", access_mode,
           "--port", str(STRATA_PORT), "--gpu", gpu_list, "--yes", "--no-tui"]
    log("Launching Strata. First-time model/engine downloads may take a while.")
    result = run_streaming(cmd, timeout=3600, check=False,
                           heartbeat_interval=20, heartbeat_label="Strata model/engine launch",
                           progress_paths=[data_dir, STRATA_ROOT / "components"])
    output = result.stdout or ""
    if result.returncode:
        raise RuntimeError("Strata launch failed. Inspect the redacted output above.")
    return output


def install_ngrok_if_missing(env: dict[str, str] | None = None) -> None:
    """Install pyngrok once; the tunnel itself is created by start_ngrok_tunnel."""
    try:
        import pyngrok  # noqa: F401
        return
    except ImportError:
        pass
    run_command([sys.executable, "-m", "pip", "install", "--quiet", "pyngrok"],
                timeout=600, env=env)


def _load_ngrok():
    try:
        from pyngrok import ngrok
    except ImportError as exc:
        raise RuntimeError(
            "pyngrok is required for the ngrok tunnel. Run install_ngrok_if_missing() "
            "or `pip install pyngrok` first."
        ) from exc
    return ngrok


def _resolve_ngrok_token(secret_name: str = DEFAULT_NGROK_SECRET,
                         auth_token: str | None = None,
                         env: dict[str, str] | None = None) -> str:
    if auth_token:
        return auth_token.strip()
    if env and env.get("NGROK_AUTHTOKEN"):
        return str(env["NGROK_AUTHTOKEN"]).strip()
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(secret_name).strip()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle Secret " + secret_name + " was not available. Add it in Add-ons > Secrets, "
            "enable notebook access, and rerun. Do not paste the ngrok token into notebook code."
        ) from exc
    if not token:
        raise RuntimeError("Kaggle Secret " + secret_name + " is empty.")
    return token


def _read_key() -> str:
    result = subprocess.run(["strata", "launch", "--show-key"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=60, check=True)
    match = re.search(r"sk-strata-[A-Za-z0-9._~-]+", result.stdout or "")
    if not match:
        raise RuntimeError("Could not safely identify the Strata API key.")
    return match.group(0)


def wait_health(timeout_seconds: int = 900) -> dict[str, Any]:
    import requests
    deadline, last = time.time() + timeout_seconds, "no response"
    while time.time() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{STRATA_PORT}/health", timeout=5)
            if response.ok:
                body = response.json()
                if body.get("loaded") is True:
                    if not body.get("api_key"):
                        raise RuntimeError("Strata has no API-key protection; refusing a public tunnel.")
                    log("Strata health is OK; model=" + str(body.get("model")))
                    return body
                last = "model not loaded yet"
            else:
                last = "HTTP " + str(response.status_code)
        except RuntimeError:
            raise
        except Exception as exc:
            last = str(exc)
        time.sleep(5)
    raise TimeoutError("Strata did not become healthy; last result: " + last)


def start_ngrok_tunnel(existing_url: str | None = None, *,
                       port: int = STRATA_PORT,
                       secret_name: str = DEFAULT_NGROK_SECRET,
                       domain: str = "",
                       auth_token: str | None = None,
                       env: dict[str, str] | None = None) -> str:
    """Open (or reuse) an ngrok HTTPS tunnel to the local Strata server."""
    if existing_url:
        log("Using HTTPS URL printed by Strata.")
        return existing_url
    install_ngrok_if_missing(env)
    ngrok = _load_ngrok()
    ngrok.set_auth_token(_resolve_ngrok_token(secret_name, auth_token, env))

    for tunnel in ngrok.get_tunnels():
        try:
            addr = str(tunnel.config.get("addr", ""))
        except Exception:
            addr = ""
        if str(port) in addr:
            url = tunnel.public_url.replace("http://", "https://")
            log("Reusing existing ngrok tunnel: " + url)
            return url

    options = {"domain": domain.strip()} if domain and domain.strip() else {}
    tunnel = ngrok.connect(port, "http", **options)
    url = tunnel.public_url.replace("http://", "https://")
    log("ngrok tunnel started. URL: " + url)
    return url


# Backward-compatible alias for notebooks written before the ngrok migration.
start_quick_tunnel = start_ngrok_tunnel


def verify_public_endpoint(public_url: str, expected_model: str | None = None) -> dict[str, Any]:
    import requests
    base, key = public_url.rstrip("/"), _read_key()
    unauth = requests.get(base + "/v1/models", timeout=20)
    if unauth.status_code not in (401, 403):
        raise RuntimeError("Unauthenticated /v1/models returned HTTP " + str(unauth.status_code) +
                           "; expected 401/403. Stop the tunnel and secure the server.")
    headers = {"Authorization": "Bearer " + key}
    response = requests.get(base + "/v1/models", headers=headers, timeout=30)
    if not response.ok:
        raise RuntimeError("Authenticated /v1/models failed: HTTP " + str(response.status_code) +
                           ": " + redact(response.text[:1000]))
    rows = response.json().get("data", [])
    available = [x.get("id") for x in rows if isinstance(x, dict) and x.get("id")]
    model_id = expected_model if expected_model in available else (available[0] if available else expected_model)
    if not model_id:
        raise RuntimeError("Strata did not advertise a model ID from /v1/models.")
    log("Unauthenticated request rejected with HTTP " + str(unauth.status_code) + ".")
    log("Sending short authenticated generation test…")
    chat = requests.post(
        base + "/v1/chat/completions", headers={**headers, "Content-Type": "application/json"},
        json={"model": model_id, "messages": [{"role": "user", "content": "Reply with exactly: Strata connection working."}],
              "max_tokens": 32, "reasoning_effort": "none", "stream": False},
        timeout=240,
    )
    if not chat.ok:
        raise RuntimeError("Authenticated chat test failed: HTTP " + str(chat.status_code) + ": " + redact(chat.text[:1500]))
    body = chat.json()
    choices = body.get("choices", [])
    answer = choices[0].get("message", {}).get("content", "") if choices else ""
    if isinstance(answer, list):
        answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    print("Authenticated test response: " + str(answer))
    print("OpenCode base URL: " + base + "/v1")
    print("API key was not printed. Retrieve it privately using: strata launch --show-key")
    return {"model_id": model_id, "url": base, "unauthenticated_status": unauth.status_code,
            "authenticated_status": chat.status_code}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_dataset_metadata(stage: Path, username: str, slug: str, title: str, description: str) -> None:
    metadata = {"title": title, "id": username + "/" + slug,
                "licenses": [{"name": "other"}],
                "description": description + " Components/model files retain upstream terms. Private by default."}
    (stage / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _upload_create_or_version(stage: Path, username: str, slug: str,
                              env: dict[str, str], message: str) -> None:
    cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "skip"]
    log("Uploading dataset files to Kaggle. Progress output will stream below; large uploads may take several minutes.")
    result = run_streaming(
        cmd, timeout=7200, env=env, check=False,
        heartbeat_interval=30, heartbeat_label="Kaggle dataset upload",
    )
    output = redact(result.stdout or "")
    lower = output.lower()

    # Some Kaggle CLI versions can print a slug-conflict error but still exit zero.
    slug_conflict = (
        "already in use by a dataset" in lower
        or ("requested slug" in lower and "already" in lower)
        or "duplicate slug" in lower
        or "slug is already taken" in lower
        or "dataset already exists" in lower
    )
    if result.returncode == 0 and not slug_conflict:
        log("Upload submitted for " + username + "/" + slug + ". Check Kaggle's processing status.")
        return

    if not slug_conflict:
        raise RuntimeError(
            "Kaggle dataset creation failed (exit " + str(result.returncode) + "): "
            + output[-4000:]
        )

    log("Dataset slug already exists; uploading a new private dataset version instead.")
    version = run_streaming(
        ["kaggle", "datasets", "version", "-p", str(stage), "-m", message, "--dir-mode", "skip"],
        timeout=7200, env=env, check=False,
        heartbeat_interval=30, heartbeat_label="Kaggle dataset version upload",
    )
    version_output = redact(version.stdout or "")
    if version.returncode:
        raise RuntimeError("Kaggle dataset version failed: " + version_output[-4000:])
    version_lower = version_output.lower()
    if "error" in version_lower and not any(
        success in version_lower for success in ("upload successful", "version created", "version updated")
    ):
        raise RuntimeError("Kaggle reported an error while creating the dataset version: " + version_output[-4000:])
    log("New version upload submitted for " + username + "/" + slug + ". Wait for Kaggle processing to finish.")

def _clean_stage(stage: Path) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)


def upload_model_cache(data_dir: Path, username: str, slug: str, family: str, variant: str,
                       env: dict[str, str], license_confirmed: bool, force: bool = False) -> bool:
    if not license_confirmed:
        log("Skipped model upload. Confirm the model terms and enable the explicit license flag first.")
        return False
    attached = find_attached_dataset(username, slug, "manifest.json")
    if attached and not force:
        try:
            old = json.loads((attached / "manifest.json").read_text(encoding="utf-8"))
            if (old.get("format") == MODEL_CACHE_FORMAT
                    and old.get("model") == {"family": family, "variant": variant}):
                old_entries = old.get("files", [])
                cache_matches = bool(old_entries)
                for item in old_entries:
                    rel = PurePosixPath(item["relative_path"])
                    if rel.is_absolute() or ".." in rel.parts:
                        cache_matches = False
                        break
                    local_file = data_dir.joinpath(*rel.parts)
                    if not local_file.is_file() or local_file.stat().st_size != int(item["size"]):
                        cache_matches = False
                        break
                if cache_matches:
                    log("Attached private model cache already matches the selected model; skipping duplicate upload.")
                    return False
        except Exception:
            pass

    allowed_roots = {"models", "packs", "mtp"}
    files = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and not p.name.endswith(".part")
        and p.relative_to(data_dir).parts and p.relative_to(data_dir).parts[0] in allowed_roots
    )
    if not files:
        raise RuntimeError("No model/packs/mtp files found under " + str(data_dir) + ".")
    sizes = {p.relative_to(data_dir).as_posix(): p.stat().st_size for p in files}
    stage = WORKING / "strata-model-dataset-stage"
    _clean_stage(stage)
    entries = []
    for number, path in enumerate(files, 1):
        relative = path.relative_to(data_dir).as_posix()
        name = f"{number:04d}__" + re.sub(r"[^A-Za-z0-9._-]+", "__", relative.replace("/", "__"))
        # Resolve input-mount symlinks so the Kaggle CLI sees the underlying file.
        (stage / name).symlink_to(path.resolve())
        entries.append({"dataset_name": name, "relative_path": relative, "size": sizes[relative]})
    manifest = {"format": MODEL_CACHE_FORMAT,
                "model": {"family": family, "variant": variant}, "files": entries}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_dataset_metadata(stage, username, slug, "Strata Model Cache",
                            f"Personal private cache for Strata-supported {family}/{variant} model data.")
    total = sum(item["size"] for item in entries)
    log(f"Preparing {len(entries)} model files ({total / (1024**3):.2f} GiB) for private upload.")
    _upload_create_or_version(stage, username, slug, env, f"Update Strata model cache: {family}/{variant}")
    return True



def upload_runtime_cache(username: str, slug: str, env: dict[str, str], force: bool = False) -> bool:
    if not (STRATA_ROOT / "terminal").exists() or not (STRATA_ROOT / "components").exists():
        log("Runtime upload skipped: terminal/components directory missing.")
        return False
    attached = find_attached_dataset(username, slug, "strata-runtime-cache.bin")
    version = subprocess.run(["strata", "--version"], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=60, check=False).stdout.strip()
    hw = hardware_summary()
    engine_dirs = sorted(p.name for p in (STRATA_ROOT / "components" / "engine").glob("*") if p.is_dir())
    if attached and not force:
        try:
            old = json.loads((attached / "runtime-manifest.json").read_text(encoding="utf-8"))
            if (old.get("strata_version") == version and old.get("machine") == hw["machine"]
                    and old.get("gpu_names") == hw["gpu_names"] and old.get("engine_dirs", []) == engine_dirs):
                log("Runtime dataset matches installed terminal/engine; skipping upload.")
                return False
        except Exception:
            pass
    stage = WORKING / "strata-runtime-dataset-stage"
    _clean_stage(stage)
    archive = stage / "strata-runtime-cache.bin"
    log("Packaging Strata terminal and engine cache (model data remains separate)…")
    with tarfile.open(archive, "w:") as tf:
        tf.add(STRATA_ROOT / "terminal", arcname="terminal", recursive=True)
        tf.add(STRATA_ROOT / "components", arcname="components", recursive=True)
    (stage / "strata-runtime-cache.bin.sha256").write_text(
        _sha256(archive) + "  strata-runtime-cache.bin\n", encoding="utf-8")
    meta = {"format": RUNTIME_CACHE_FORMAT, "strata_version": version,
            **hw, "engine_dirs": engine_dirs,
            "contents": ["terminal", "components"]}
    (stage / "runtime-manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _write_dataset_metadata(stage, username, slug, "Strata Runtime Cache",
                            "Private runtime cache for Strata terminal and engine components.")
    log(f"Runtime archive size: {archive.stat().st_size / (1024**3):.2f} GiB.")
    _upload_create_or_version(stage, username, slug, env, "Update Strata terminal and engine cache")
    return True


def context_window_tokens(context: str) -> int:
    """Convert a Strata context label such as '200k' into an OpenCode token limit."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", str(context))
    if not match:
        raise ValueError("Unsupported context value: " + str(context))
    amount = float(match.group(1))
    unit = match.group(2).lower()
    multiplier = {"": 1, "k": 1024, "m": 1024 ** 2}[unit]
    return int(amount * multiplier)


def opencode_config_snippet(public_url: str, model_id: str,
                            provider_package: str = "@ai-sdk/openai-compatible",
                            context_tokens: int = 200 * 1024,
                            output_tokens: int = 8192) -> str:
    provider = {
        "npm": provider_package,
        "name": "Strata Kaggle",
        "options": {
            "baseURL": public_url.rstrip("/") + "/v1",
            "apiKey": "{file:~/.config/opencode/strata-key.txt}",
            "headers": {"ngrok-skip-browser-warning": "1"},
        },
        "models": {
            model_id: {
                "name": model_id,
                "tool_call": True,
                "limit": {"context": int(context_tokens), "output": int(output_tokens)},
            }
        },
    }
    return json.dumps({"provider": {"strata": provider}}, indent=2)
