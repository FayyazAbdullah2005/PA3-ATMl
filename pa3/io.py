"""Atomic, checksummed artifacts and resumable units. Never accept partial output."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if type(value).__module__=="torch" and type(value).__name__ in ("dtype","device"):
        return str(value)
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def canonical(value) -> bytes:
    return json.dumps(clean_json(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash(*names: str) -> str:
    base = Path(__file__).parent
    return digest({n: sha256(base / n) for n in names})


@contextlib.contextmanager
def atomic_path(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        yield tmp
        with tmp.open("r+b") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Directory fsync is unavailable on Windows and some mounted filesystems.
        if os.name != "nt":
            try:
                dfd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, value):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                                  allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_parquet(path: Path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if isinstance(rows, pa.Table):
        table = rows
    else:
        cleaned = clean_json(list(rows))
        try:
            table = pa.Table.from_pylist(cleaned)
        except (pa.ArrowTypeError, pa.ArrowInvalid):
            if cleaned and isinstance(cleaned[0], dict):
                for k in {k for r in cleaned for k in r}:
                    types = {type(r[k]) for r in cleaned if r.get(k) is not None}
                    if str in types and len(types) > 1:
                        for r in cleaned:
                            if r.get(k) is not None:
                                r[k] = str(r[k])
            table = pa.Table.from_pylist(cleaned)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")


def read_parquet(path: Path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def append_events(path: Path, values):
    rows = list(values)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0,os.SEEK_END)
        length=f.tell()
        if length:
            f.seek(-1,os.SEEK_END)
            if f.read(1)!=b"\n":
                # Repair only an incomplete trailing record before appending.
                position=length
                boundary=0
                while position:
                    size=min(65536,position)
                    position-=size
                    f.seek(position)
                    chunk=f.read(size)
                    newline=chunk.rfind(b"\n")
                    if newline>=0:
                        boundary=position+newline+1
                        break
                f.truncate(boundary)
        f.seek(0,os.SEEK_END)
        f.write(b"".join(canonical(v) + b"\n" for v in rows))
        f.flush()
        os.fsync(f.fileno())


def append_event(path: Path, value):
    append_events(path, [value])


def read_events(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("rb") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A killed process can leave one incomplete trailing line.
                continue
    return rows


def environment():
    packages = ("torch", "transformers", "datasets", "lm-eval", "numpy", "pyarrow",
                "math-verify", "modal", "scipy", "safetensors")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    frozen=sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,"installed_packages":frozen}


class Store:
    def __init__(self, root: str | Path, commit=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.commit = commit or (lambda: None)

    def done(self, directory: Path, signature: str | None = None) -> bool:
        marker = Path(directory) / "DONE.json"
        if not marker.exists():
            return False
        data = read_json(marker)
        if signature is not None and data["signature"] != signature:
            raise RuntimeError(f"Configuration/code changed for {directory}. Use a new run ID; existing output was preserved.")
        for rel, expected in data["files"].items():
            p = Path(directory) / rel
            if not p.resolve().is_relative_to(Path(directory).resolve()):
                raise RuntimeError(f"Artifact manifest escapes its unit directory: {rel}")
            if not p.is_file() or sha256(p) != expected:
                raise RuntimeError(f"Artifact failed integrity check: {p}. Existing files were preserved.")
        return True

    def finish(self, directory: Path, signature: str, files: list[Path], **metadata):
        directory = Path(directory)
        hashes = {str(Path(f).relative_to(directory)): sha256(f) for f in files}
        write_json(directory / "DONE.json", {"signature": signature, "files": hashes,
                   "finished_unix": time.time(), **metadata})
        self.commit()

    def status(self):
        return [{"unit": str(p.parent.relative_to(self.root)),
                 "finished": read_json(p).get("finished_unix")} for p in sorted(self.root.rglob("DONE.json"))]


def download(url: str, path: Path, expected_sha: str | None = None):
    """HTTP Range resume; an interrupted download leaves a reusable .download file."""
    import requests
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and receipt.exists():
        info = read_json(receipt)
        if info["url"] != url or sha256(path) != info["sha256"]:
            raise RuntimeError(f"Download source/integrity changed: {path}")
        return path
    partial = path.with_suffix(path.suffix + ".download")
    offset = partial.stat().st_size if partial.exists() else 0
    headers={"Accept-Encoding":"identity"}
    if offset:
        headers["Range"]=f"bytes={offset}-"
    with requests.get(url, stream=True, headers=headers, timeout=(30, 120)) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError(f"Unexpected Content-Range for {url}")
        expected_bytes = int(response.headers.get("Content-Length", "0"))
        written = 0
        with partial.open("ab" if append else "wb") as f:
            for block in response.iter_content(8 * 1024 * 1024):
                f.write(block)
                written += len(block)
            f.flush()
            os.fsync(f.fileno())
        if expected_bytes and written != expected_bytes and not response.headers.get("Content-Encoding"):
            raise RuntimeError(f"Incomplete download: {path}")
    actual = sha256(partial)
    if expected_sha and actual != expected_sha:
        raise RuntimeError(f"Checksum mismatch: {path}")
    os.replace(partial, path)
    write_json(receipt, {"url": url, "sha256": actual, "bytes": path.stat().st_size})
    return path


def safe_extract_zip(archive: Path, destination: Path, members=None):
    import zipfile
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = set(members) if members is not None else None
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if selected is not None and item.filename not in selected:
                continue
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == item.file_size:
                continue
            with atomic_path(target) as tmp, z.open(item) as source, tmp.open("wb") as out:
                import shutil
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)

