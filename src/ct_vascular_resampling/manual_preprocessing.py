"""从手工 Slicer 标签图生成器官网格并引用既有重建血管。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import SimpleITK as sitk
import yaml

from .contract import BLACK_RATIO_LIMIT
from .ct_resampling import read_ct_image
from .preprocessing import _case_config, build_binary_masks, mask_to_mesh, validate_geometry


MANUAL_ORGAN_LABEL_VALUES = {
    "spleen": (1,),
    "kidney_right": (2,),
    "kidney_left": (3,),
    "gallbladder": (4,),
    "esophagus": (5,),
    "liver": (6,),
    "stomach": (7,),
    "aorta": (8,),
    "inferior_vena_cava": (9,),
    "pancreas": (11,),
    "adrenal_gland_right": (12,),
    "adrenal_gland_left": (13,),
    "duodenum": (14,),
    "portal_vein": (23, 26, 33, 34, 35, 36, 37),
}
MANUAL_ORGAN_MODEL_IDS = {
    **{identifier: identifier for identifier in MANUAL_ORGAN_LABEL_VALUES if identifier != "portal_vein"},
    "portal_vein": "portal_vein_and_splenic_vein",
}
EUS_VESSEL_LABEL_VALUES = {
    "aorta": (8,),
    "inferior_vena_cava": (9,),
    "portal_vein": (26, 33, 34, 35, 36, 37),
}
EUS_VESSEL_COLORS = {
    "aorta": (255, 0, 0),
    "inferior_vena_cava": (0, 0, 255),
    "portal_vein": (170, 85, 255),
}


def _required_file(path: str | Path, description: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{description}不存在: {source}")
    return source


def _required_ct(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file() and not source.is_dir():
        raise FileNotFoundError(f"CT 输入不存在: {source}")
    return source


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        digest.update(b"directory\0")
        for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = candidate.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with candidate.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    raise FileNotFoundError(f"输入不存在: {path}")


def _geometry_record(image: sitk.Image) -> dict[str, list[int] | list[float]]:
    return {
        "size": [int(value) for value in image.GetSize()],
        "spacing": [float(value) for value in image.GetSpacing()],
        "origin": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
    }


def _manual_case_config(
    *,
    case_id: str,
    ct_path: Path,
    dicom_series_uid: str | None,
    segmentation_relative_path: str,
    artery_model_path: Path,
    vein_model_path: Path,
    output_root: str | Path,
) -> dict[str, Any]:
    config = _case_config(
        case_id,
        output_root=str(Path(output_root).expanduser().resolve()),
    )
    config["ct_path"] = str(ct_path)
    if dicom_series_uid is not None:
        config["dicom_series_uid"] = dicom_series_uid
    config["organ_models"] = {
        model_id: f"models/{model_id}.ply"
        for model_id in MANUAL_ORGAN_MODEL_IDS.values()
    }
    config["vessel_models"] = [
        {
            "id": "artery_tree",
            "path": str(artery_model_path),
            "label": "artery",
            "color": [255, 82, 0],
        },
        {
            "id": "vein_tree",
            "path": str(vein_model_path),
            "label": "vein",
            "color": [0, 188, 212],
        },
    ]
    config["filtering"]["black_ratio_limit"] = BLACK_RATIO_LIMIT
    config["manual_segmentation"] = {
        "path": segmentation_relative_path,
        "organ_label_values": {
            identifier: list(values)
            for identifier, values in MANUAL_ORGAN_LABEL_VALUES.items()
        },
        "eus_vessel_label_values": {
            identifier: list(values)
            for identifier, values in EUS_VESSEL_LABEL_VALUES.items()
        },
        "eus_vessel_colors": {
            identifier: list(color)
            for identifier, color in EUS_VESSEL_COLORS.items()
        },
    }
    return config


def write_manual_segmentation_case(
    *,
    ct_path: str | Path,
    dicom_series_uid: str | None = None,
    segmentation_path: str | Path,
    artery_model_path: str | Path,
    vein_model_path: str | Path,
    output_directory: str | Path,
    output_root: str | Path,
    case_id: str,
) -> dict[str, object]:
    """事务式写出 14 类手工器官，不复制或重建外部动静脉网格。"""

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id 必须是非空字符串")
    ct_source = _required_ct(ct_path)
    segmentation_source = _required_file(segmentation_path, "手工分割文件")
    artery_source = _required_file(artery_model_path, "外部 artery_tree 模型")
    vein_source = _required_file(vein_model_path, "外部 vein_tree 模型")

    destination = Path(output_directory).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"输出目录已有内容: {destination}")

    ct = read_ct_image(ct_source, dicom_series_uid=dicom_series_uid)
    segmentation = sitk.ReadImage(str(segmentation_source))
    validate_geometry(ct, segmentation)
    labels = sitk.GetArrayViewFromImage(segmentation)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("手工分割必须是三维整数标签图")
    missing = [
        identifier
        for identifier, values in MANUAL_ORGAN_LABEL_VALUES.items()
        if not np.any(np.isin(labels, values))
    ]
    if missing:
        raise ValueError(f"手工分割缺少必要器官标签: {', '.join(missing)}")
    required_source_values = {
        value
        for values in MANUAL_ORGAN_LABEL_VALUES.values()
        for value in values
    }
    present_source_values = {int(value) for value in np.unique(labels)}
    missing_source_values = sorted(required_source_values - present_source_values)
    if missing_source_values:
        raise ValueError(f"手工分割缺少必要源标签值: {missing_source_values}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        mask_directory = temporary / "masks"
        model_directory = temporary / "models"
        segmentation_directory = temporary / "segmentation"
        for directory in (mask_directory, model_directory, segmentation_directory):
            directory.mkdir(parents=True, exist_ok=True)

        copied_segmentation = segmentation_directory / "EUS-main-organ.seg.nrrd"
        shutil.copyfile(segmentation_source, copied_segmentation)
        segmentation_sha256 = _sha256_path(segmentation_source)
        if _sha256_path(copied_segmentation) != segmentation_sha256:
            raise OSError("手工分割逐字节复制校验失败")

        organ_records: dict[str, dict[str, object]] = {}
        for canonical_id, values in MANUAL_ORGAN_LABEL_VALUES.items():
            model_id = MANUAL_ORGAN_MODEL_IDS[canonical_id]
            mask = build_binary_masks(segmentation, {model_id: values})[model_id]
            mask_path = mask_directory / f"{model_id}.nrrd"
            model_path = model_directory / f"{model_id}.ply"
            sitk.WriteImage(mask, str(mask_path), useCompression=True)
            mesh = mask_to_mesh(mask)
            mesh.export(str(model_path), file_type="ply")
            organ_records[model_id] = {
                "canonical_organ_label": canonical_id,
                "source_label_values": list(values),
                "voxel_count": int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))),
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
                "is_watertight": bool(mesh.is_watertight),
                "mask": str(mask_path.relative_to(temporary)),
                "mesh": str(model_path.relative_to(temporary)),
            }

        segmentation_relative_path = str(copied_segmentation.relative_to(temporary))
        config = _manual_case_config(
            case_id=case_id.strip(),
            ct_path=ct_source,
            dicom_series_uid=dicom_series_uid,
            segmentation_relative_path=segmentation_relative_path,
            artery_model_path=artery_source,
            vein_model_path=vein_source,
            output_root=output_root,
        )
        manifest = {
            "case_id": case_id.strip(),
            "ct": {
                "path": str(ct_source),
                "kind": "directory" if ct_source.is_dir() else "file",
                "dicom_series_uid": dicom_series_uid,
                "sha256": _sha256_path(ct_source),
                "geometry": _geometry_record(ct),
            },
            "segmentation": {
                "source_path": str(segmentation_source),
                "path": segmentation_relative_path,
                "sha256": segmentation_sha256,
                "geometry": _geometry_record(segmentation),
            },
            "organ_label_values": {
                identifier: list(values)
                for identifier, values in MANUAL_ORGAN_LABEL_VALUES.items()
            },
            "organ_models": organ_records,
            "external_vessel_models": {
                "artery_tree": {
                    "path": str(artery_source),
                    "sha256": _sha256_path(artery_source),
                },
                "vein_tree": {
                    "path": str(vein_source),
                    "sha256": _sha256_path(vein_source),
                },
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "case_manual_segmentation.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output_directory": str(destination),
        "segmentation_path": str(destination / "segmentation" / "EUS-main-organ.seg.nrrd"),
        "manifest_path": str(destination / "manifest.json"),
        "case_config_path": str(destination / "case_manual_segmentation.yaml"),
        "organ_model_count": len(MANUAL_ORGAN_MODEL_IDS),
    }
