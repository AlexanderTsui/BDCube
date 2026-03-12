from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import trimesh

MESH_SUFFIXES = {".obj", ".ply", ".glb", ".gltf", ".stl", ".off"}
TEXT_KEYS = ("text", "caption", "prompt", "description", "name", "title")
MESH_KEYS = ("mesh_path", "path", "file_path", "mesh", "object_path")
BOUNDING_BOX_MAX_SIZE = 1.925
MESH_SCALE = 0.96


def normalize_bbox(bounding_box_xyz: tuple[float, float, float]) -> list[float]:
    max_length = max(bounding_box_xyz)
    if max_length <= 0:
        return [0.0, 0.0, 0.0]
    return [BOUNDING_BOX_MAX_SIZE * elem / max_length for elem in bounding_box_xyz]


def rescale(vertices: np.ndarray, mesh_scale: float = MESH_SCALE) -> np.ndarray:
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    return (vertices - center) * scale


def load_scaled_mesh(file_path: str) -> "trimesh.Trimesh":
    import trimesh

    mesh: trimesh.Trimesh = trimesh.load(file_path, force="mesh")
    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces after cleaning")
    mesh.vertices = rescale(mesh.vertices)
    return mesh


def _first_value(record: dict, keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return None


def _load_json_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return list(data["samples"])
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_manifest(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported manifest format: {path}")


def _resolve_manifest_entries(root: Path, manifest_path: Path) -> list[dict]:
    records = _load_manifest(manifest_path)
    entries = []
    for record in records:
        mesh_value = _first_value(record, MESH_KEYS)
        if mesh_value is None:
            continue
        mesh_path = Path(mesh_value)
        if not mesh_path.is_absolute():
            mesh_path = (root / mesh_path).resolve()
        if not mesh_path.exists():
            continue
        text = _first_value(record, TEXT_KEYS) or mesh_path.stem.replace("_", " ")
        entries.append({"mesh_path": mesh_path, "text": text})
    return entries


def _read_sidecar_text(mesh_path: Path) -> Optional[str]:
    txt_path = mesh_path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text().strip()
        if text:
            return text

    json_path = mesh_path.with_suffix(".json")
    if json_path.exists():
        try:
            record = json.loads(json_path.read_text())
            if isinstance(record, dict):
                return _first_value(record, TEXT_KEYS)
        except json.JSONDecodeError:
            pass
    return None


def _scan_root(root: Path) -> list[dict]:
    entries = []
    for mesh_path in sorted(root.rglob("*")):
        if not mesh_path.is_file() or mesh_path.suffix.lower() not in MESH_SUFFIXES:
            continue
        text = _read_sidecar_text(mesh_path) or mesh_path.stem.replace("_", " ")
        entries.append({"mesh_path": mesh_path.resolve(), "text": text})
    return entries


def _sample_point_cloud(mesh: "trimesh.Trimesh", n_samples: int) -> torch.Tensor:
    import trimesh

    positions, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_indices]
    point_cloud = np.concatenate([positions, normals], axis=1).astype(np.float32)
    return torch.from_numpy(point_cloud)


def _compute_bbox(mesh: "trimesh.Trimesh") -> torch.Tensor:
    extent = mesh.vertices.max(0) - mesh.vertices.min(0)
    bbox = normalize_bbox(tuple(float(v) for v in extent))
    return torch.tensor(bbox, dtype=torch.float32)


class ObjaverseDataset(Dataset):
    def __init__(
        self,
        root: str,
        manifest_path: Optional[str] = None,
        point_samples: int = 8192,
        cache_dir: Optional[str] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve() if manifest_path else None
        )
        self.point_samples = point_samples
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path is not None:
            self.entries = _resolve_manifest_entries(self.root, self.manifest_path)
        else:
            self.entries = _scan_root(self.root)

        if not self.entries:
            raise ValueError(
                f"No valid Objaverse entries found under {self.root} "
                f"(manifest={self.manifest_path})"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def _cache_path(self, mesh_path: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        path_digest = hashlib.sha1(str(mesh_path).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{mesh_path.stem}_{path_digest}.pt"

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        mesh_path = entry["mesh_path"]
        cache_path = self._cache_path(mesh_path)
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            cached["prompt_text"] = entry["text"]
            cached["mesh_path"] = str(mesh_path)
            return cached

        mesh = load_scaled_mesh(str(mesh_path))
        sample = {
            "prompt_text": entry["text"],
            "point_cloud": _sample_point_cloud(mesh, self.point_samples),
            "bbox_xyz": _compute_bbox(mesh),
            "mesh_path": str(mesh_path),
        }
        if cache_path is not None:
            torch.save(
                {
                    "point_cloud": sample["point_cloud"],
                    "bbox_xyz": sample["bbox_xyz"],
                },
                cache_path,
            )
        return sample


def collate_objaverse_batch(samples: list[dict]) -> dict:
    point_cloud = torch.stack([sample["point_cloud"] for sample in samples], dim=0)
    bbox_xyz = torch.stack([sample["bbox_xyz"] for sample in samples], dim=0)
    return {
        "prompt_text": [sample["prompt_text"] for sample in samples],
        "point_cloud": point_cloud,
        "bbox_xyz": bbox_xyz,
        "mesh_path": [sample["mesh_path"] for sample in samples],
    }
