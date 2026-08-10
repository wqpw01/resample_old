from __future__ import annotations

import json

import pytest

from ct_vascular_resampling.eus_organs import (
    EUS_CATALOG_SCHEMA_VERSION,
    EUS_ORGAN_METADATA_SCHEMA_VERSION,
    load_eus_organ_catalog,
    parse_eus_organ_catalog,
)


EXPECTED_SHA256 = "54b8bf06fc48d1733e98b32a01dc10e056f5db3b4cddb34e18905dd8d97bf63d"
EXPECTED_LABELS = (
    "adrenal_gland_left",
    "adrenal_gland_right",
    "aorta",
    "duodenum",
    "inferior_vena_cava",
    "kidney_left",
    "kidney_right",
    "liver",
    "pancreas",
    "portal_vein",
    "spleen",
)


def _entry(
    organ_label: str,
    label_ids: list[int],
    *,
    role: str,
    vessel_type: str | None,
    canonical_vessel_label_id: int | None = None,
) -> dict[str, object]:
    return {
        "organ_label": organ_label,
        "eus_label_ids": label_ids,
        "eus_label_names": [f"label_{label_id}" for label_id in label_ids],
        "role": role,
        "vessel_type": vessel_type,
        "canonical_vessel_label_id": canonical_vessel_label_id,
    }


def _payload(*extra_entries: dict[str, object]) -> bytes:
    organs = [
        _entry("aorta", [3], role="organ_and_vessel", vessel_type="artery"),
        *extra_entries,
        _entry("inferior_vena_cava", [30], role="organ_and_vessel", vessel_type="vein"),
        _entry(
            "portal_vein",
            [26, 27],
            role="organ_and_vessel",
            vessel_type="vein",
            canonical_vessel_label_id=26,
        ),
    ]
    raw = {"schema_version": "eus-possible-organs/v1", "organs": organs}
    return (json.dumps(raw, ensure_ascii=False) + "\n").encode("utf-8")


def test_packaged_eus_catalog_matches_approved_source() -> None:
    catalog = load_eus_organ_catalog()

    assert EUS_CATALOG_SCHEMA_VERSION == "eus-possible-organs/v1"
    assert EUS_ORGAN_METADATA_SCHEMA_VERSION == "eus-organ-metadata/v1"
    assert catalog.sha256 == EXPECTED_SHA256
    assert tuple(item.organ_label for item in catalog.organs) == EXPECTED_LABELS
    assert catalog.candidate_labels(["stomach", "liver", "aorta", "liver"]) == ["aorta", "liver"]
    roles = {item.organ_label: (item.role, item.vessel_type) for item in catalog.organs}
    assert roles["aorta"] == ("organ_and_vessel", "artery")
    assert roles["inferior_vena_cava"] == ("organ_and_vessel", "vein")
    assert roles["portal_vein"] == ("organ_and_vessel", "vein")
    assert catalog.to_record()["geometry_sources"] == {
        "portal_vein": "portal_vein_and_splenic_vein"
    }


@pytest.mark.parametrize("forbidden", ["bile_duct", "common_bile_duct"])
def test_catalog_rejects_bile_duct_labels(forbidden: str) -> None:
    forbidden_entry = _entry(forbidden, [99], role="organ", vessel_type=None)

    with pytest.raises(ValueError, match="胆管"):
        parse_eus_organ_catalog(_payload(forbidden_entry))


def test_catalog_rejects_eus_label_id_shared_by_organs() -> None:
    duplicate = _entry("duodenum", [3], role="organ", vessel_type=None)

    with pytest.raises(ValueError, match="不能跨器官重复"):
        parse_eus_organ_catalog(_payload(duplicate))


def test_catalog_rejects_incorrect_dual_role_vessel_type() -> None:
    raw = json.loads(_payload())
    raw["organs"][0]["vessel_type"] = "vein"

    with pytest.raises(ValueError, match="双身份"):
        parse_eus_organ_catalog((json.dumps(raw, ensure_ascii=False) + "\n").encode("utf-8"))
