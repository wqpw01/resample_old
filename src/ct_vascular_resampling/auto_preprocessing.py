"""通过 TotalSegmentator 自动生成重采样所需器官网格。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk
import yaml

from .preprocessing import (
    ORGAN_LABEL_VALUES,
    build_binary_masks,
    validate_geometry,
    write_preprocessed_masks_case,
)


AUTO_ORGAN_IDS = (*ORGAN_LABEL_VALUES, "portal_vein_and_splenic_vein")
_REQUIRED_VESSEL_LABELS = frozenset({"artery", "vein"})
_LEGACY_VESSEL_LABEL = "portal"
_CT_SUFFIXES = (".nii", ".nii.gz", ".nrrd")


@dataclass(frozen=True)
class AutoCaseConfig:
    case_id: str
    ct_path: Path
    vascular_segmentation_path: Path
    vessel_label_values: dict[str, tuple[int, ...]]
    registration_module_path: Path
    output_root: Path
    dicom_series_uid: str | None = None
    totalsegmentator_executable: str = "TotalSegmentator"
    totalsegmentator_device: str = "gpu:0"
    totalsegmentator_cache_directory: Path | None = None


def _resolve_path(value: Any, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是非空路径字符串")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _label_values(raw: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(raw, dict):
        raise ValueError("vessel_label_values 必须是 YAML 映射")
    keys = set(raw)
    allowed_keys = _REQUIRED_VESSEL_LABELS | {_LEGACY_VESSEL_LABEL}
    if not _REQUIRED_VESSEL_LABELS.issubset(keys) or not keys.issubset(allowed_keys):
        raise ValueError("vessel_label_values 必须包含 artery、vein，且只能额外包含兼容键 portal")
    result: dict[str, tuple[int, ...]] = {}
    all_values: set[int] = set()
    for name in ("artery", "vein", "portal"):
        if name not in raw:
            continue
        values = raw[name]
        if not isinstance(values, list) or not values or any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"vessel_label_values.{name} 必须是非空非负整数列表")
        labels = tuple(values)
        overlap = all_values.intersection(labels)
        if overlap:
            raise ValueError(f"血管标签不能归属多个类别: {sorted(overlap)}")
        all_values.update(labels)
        result[name] = labels
    result["vein"] += result.pop("portal", ())
    return {"artery": result["artery"], "vein": result["vein"]}


def load_auto_case_config(path: str | Path) -> AutoCaseConfig:
    """读取 CT、混合标签体及其动静脉标签映射的自动病例配置。"""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("自动病例配置必须是 YAML 映射")
    case_id = raw.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id 必须是非空字符串")
    total = raw.get("totalsegmentator", {})
    if total is None:
        total = {}
    if not isinstance(total, dict):
        raise ValueError("totalsegmentator 必须是 YAML 映射")
    executable = total.get("executable", "TotalSegmentator")
    device = total.get("device", "gpu:0")
    cache_value = total.get("cache_directory")
    if not isinstance(executable, str) or not executable:
        raise ValueError("totalsegmentator.executable 必须是非空字符串")
    if not isinstance(device, str) or not device:
        raise ValueError("totalsegmentator.device 必须是非空字符串")
    cache_directory = (
        None
        if cache_value is None
        else _resolve_path(cache_value, source.parent, "totalsegmentator.cache_directory")
    )
    series_uid = raw.get("dicom_series_uid")
    if series_uid is not None and (not isinstance(series_uid, str) or not series_uid):
        raise ValueError("dicom_series_uid 必须是非空字符串或 null")
    return AutoCaseConfig(
        case_id=case_id,
        ct_path=_resolve_path(raw.get("ct_path"), source.parent, "ct_path"),
        vascular_segmentation_path=_resolve_path(
            raw.get("vascular_segmentation_path"), source.parent, "vascular_segmentation_path"
        ),
        vessel_label_values=_label_values(raw.get("vessel_label_values")),
        registration_module_path=_resolve_path(
            raw.get("registration_module_path"), source.parent, "registration_module_path"
        ),
        output_root=_resolve_path(raw.get("output_root"), source.parent, "output_root"),
        dicom_series_uid=series_uid,
        totalsegmentator_executable=executable,
        totalsegmentator_device=device,
        totalsegmentator_cache_directory=cache_directory,
    )


def build_totalsegmentator_command(
    executable: str,
    ct_path: str | Path,
    output_directory: str | Path,
    device: str,
) -> list[str]:
    """构造限定器官集合的 TotalSegmentator 全分辨率 GPU 调用。"""

    return [
        executable,
        "-i",
        str(ct_path),
        "-o",
        str(output_directory),
        "--task",
        "total",
        "--roi_subset",
        *AUTO_ORGAN_IDS,
        "--device",
        device,
    ]


def _read_dicom_series(directory: Path, series_uid: str | None) -> tuple[sitk.Image, str]:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or []
    if not series_ids:
        raise ValueError(f"DICOM 目录中没有可读取的序列: {directory}")
    if series_uid is None:
        if len(series_ids) != 1:
            raise ValueError("DICOM 目录包含多个序列，请设置 dicom_series_uid")
        selected = series_ids[0]
    elif series_uid in series_ids:
        selected = series_uid
    else:
        raise ValueError(f"指定的 dicom_series_uid 不在目录中: {series_uid}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), selected))
    return reader.Execute(), selected


def read_ct_input(path: str | Path, dicom_series_uid: str | None = None) -> tuple[sitk.Image, str | None]:
    """读取 NIfTI/NRRD 文件或单一 DICOM 序列，保持原始物理空间。"""

    source = Path(path)
    if source.is_dir():
        return _read_dicom_series(source, dicom_series_uid)
    if not source.is_file():
        raise FileNotFoundError(f"CT 不存在: {source}")
    if not source.name.lower().endswith(_CT_SUFFIXES):
        raise ValueError(f"仅支持 NIfTI、NRRD 或 DICOM 目录 CT: {source}")
    return sitk.ReadImage(str(source)), None


def _mask_file(directory: Path, structure: str) -> Path:
    candidates = (directory / f"{structure}.nii.gz", directory / f"{structure}.nii", directory / f"{structure}.nrrd")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"TotalSegmentator 未输出必需器官 {structure}: {directory}")


def load_totalsegmentator_masks(directory: str | Path, ct: sitk.Image) -> dict[str, sitk.Image]:
    """读取并严格校验 TotalSegmentator 的必需器官掩膜。"""

    root = Path(directory)
    masks: dict[str, sitk.Image] = {}
    for structure in AUTO_ORGAN_IDS:
        mask = sitk.ReadImage(str(_mask_file(root, structure)))
        validate_geometry(ct, mask)
        binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        values = sitk.GetArrayViewFromImage(binary_mask)
        if not np.any(values):
            raise ValueError(f"TotalSegmentator 输出器官为空: {structure}")
        masks[structure] = binary_mask
    return masks


def write_auto_preprocessed_case(
    ct: sitk.Image,
    organ_masks: Mapping[str, sitk.Image],
    vascular_segmentation: sitk.Image,
    vessel_label_values: Mapping[str, tuple[int, ...]],
    output_directory: str | Path,
    registration_module_path: str | Path,
    case_id: str,
    total_segmentator_metadata: Mapping[str, object],
    output_root: str = "resampling_output",
) -> dict[str, object]:
    """合并门静脉、生成全部模型并写入可直接运行的内部病例 YAML。"""

    if set(organ_masks) != set(AUTO_ORGAN_IDS):
        raise ValueError("organ_masks 必须包含全部 TotalSegmentator 必需器官")
    validate_geometry(ct, vascular_segmentation)
    organs: dict[str, sitk.Image] = {}
    for name, mask in organ_masks.items():
        validate_geometry(ct, mask)
        organs[name] = sitk.Cast(mask > 0, sitk.sitkUInt8)
    artery_values = tuple(vessel_label_values["artery"])
    vein_values = tuple(vessel_label_values["vein"]) + tuple(vessel_label_values.get("portal", ()))
    vascular_masks = build_binary_masks(
        vascular_segmentation,
        {
            "artery_tree": artery_values,
            "vein_tree": vein_values,
        },
    )
    for mask in vascular_masks.values():
        mask.CopyInformation(ct)
    source_values = {
        **{name: tuple() for name in AUTO_ORGAN_IDS},
        "artery_tree": artery_values,
        "vein_tree": vein_values,
    }
    return write_preprocessed_masks_case(
        ct=ct,
        masks={**organs, **vascular_masks},
        source_label_values=source_values,
        output_directory=output_directory,
        registration_module_path=registration_module_path,
        case_id=case_id,
        output_root=output_root,
        provenance={"organ_source": "TotalSegmentator", **dict(total_segmentator_metadata)},
    )


def prepare_auto_case(config: AutoCaseConfig) -> Path:
    """运行自动器官分割并返回供既有 run_case 使用的内部 YAML 路径。"""

    if not config.registration_module_path.is_file():
        raise FileNotFoundError(f"2021.py 不存在: {config.registration_module_path}")
    ct, series_uid = read_ct_input(config.ct_path, config.dicom_series_uid)
    vascular = sitk.ReadImage(str(config.vascular_segmentation_path))
    validate_geometry(ct, vascular)
    case_directory = config.output_root / config.case_id
    preprocessing_directory = case_directory / "preprocessing"
    total_input = preprocessing_directory / "totalsegmentator_input.nii.gz"
    total_output = config.totalsegmentator_cache_directory or preprocessing_directory / "totalsegmentator"
    command = build_totalsegmentator_command(
        config.totalsegmentator_executable,
        total_input,
        total_output,
        config.totalsegmentator_device,
    )
    try:
        organ_masks = load_totalsegmentator_masks(total_output, ct)
        cache_reused = True
    except (FileNotFoundError, RuntimeError, ValueError):
        cache_reused = False
        total_input.parent.mkdir(parents=True, exist_ok=True)
        total_output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ct, str(total_input), useCompression=True)
        subprocess.run(command, check=True)
        organ_masks = load_totalsegmentator_masks(total_output, ct)
    result = write_auto_preprocessed_case(
        ct=ct,
        organ_masks=organ_masks,
        vascular_segmentation=vascular,
        vessel_label_values=config.vessel_label_values,
        output_directory=preprocessing_directory,
        registration_module_path=config.registration_module_path,
        case_id=config.case_id,
        output_root=str(config.output_root),
        total_segmentator_metadata={
            "task": "total",
            "device": config.totalsegmentator_device,
            "structures": list(AUTO_ORGAN_IDS),
            "dicom_series_uid": series_uid,
            "cache_reused": cache_reused,
            "cache_directory": str(total_output),
            "command": command,
            "command_executed": not cache_reused,
        },
    )
    return Path(result["case_config_path"])


def describe_auto_case(config: AutoCaseConfig) -> str:
    """为 CLI dry-run 提供不写文件的输入校验摘要。"""

    ct, series_uid = read_ct_input(config.ct_path, config.dicom_series_uid)
    vascular = sitk.ReadImage(str(config.vascular_segmentation_path))
    validate_geometry(ct, vascular)
    return json.dumps(
        {
            "case_id": config.case_id,
            "ct_size": list(ct.GetSize()),
            "ct_spacing": list(ct.GetSpacing()),
            "dicom_series_uid": series_uid,
            "total_segmentator_structures": list(AUTO_ORGAN_IDS),
        },
        ensure_ascii=False,
    )
