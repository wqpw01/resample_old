from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import trimesh

from ct_vascular_resampling import zero_plane_visualization as visualization
from ct_vascular_resampling.zero_plane_visualization import (
    load_visualization_organ_meshes,
    project_orthographic,
    render_interactive_html,
    render_static_views,
    select_zero_planes,
    write_structured_exports,
)


def _record(index: int = 0, *, organ: str = "stomach") -> dict:
    return {
        "slice_id": f"{organ}-{index:06d}-rp000-pp000-yp000",
        "organ": organ,
        "probe_point_world": [0.0, 0.0, 0.0],
        "angles_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "square_vertices_world": [
            [0.0, -50.0, 0.0],
            [0.0, 50.0, 0.0],
            [100.0, 50.0, 0.0],
            [100.0, -50.0, 0.0],
        ],
        "local_axes_world": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        },
        "input_normal_world": [1.0, 0.0, 0.0],
        "coordinate_system": "RAS",
    }


def test_select_zero_planes_filters_nonzero_angles_and_preserves_recorded_geometry():
    nonzero = _record(0)
    nonzero["angles_degrees"]["roll"] = 5.0
    selected = select_zero_planes([nonzero, _record(0)], {"stomach": 1})

    assert len(selected) == 1
    plane = selected[0]
    assert plane.slice_id == "stomach-000000-rp000-pp000-yp000"
    assert plane.organ == "stomach"
    assert plane.point_index == 0
    assert np.array_equal(plane.probe, np.zeros(3))
    assert np.array_equal(
        plane.vertices,
        np.asarray(
            [
                [0.0, -50.0, 0.0],
                [0.0, 50.0, 0.0],
                [100.0, 50.0, 0.0],
                [100.0, -50.0, 0.0],
            ]
        ),
    )


def test_select_zero_planes_sorts_by_expected_organ_order_and_point_index():
    selected = select_zero_planes(
        [_record(1), _record(0), _record(0, organ="liver")],
        {"stomach": 2, "liver": 1},
    )

    assert [(item.organ, item.point_index) for item in selected] == [
        ("stomach", 0),
        ("stomach", 1),
        ("liver", 0),
    ]


def test_select_zero_planes_rejects_duplicate_sample():
    with pytest.raises(ValueError, match="重复"):
        select_zero_planes([_record(), _record()], {"stomach": 1})


def test_select_zero_planes_rejects_missing_expected_sample():
    with pytest.raises(ValueError, match="计数"):
        select_zero_planes([_record()], {"stomach": 2})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record.update(coordinate_system="LPS"),
            "RAS",
        ),
        (
            lambda record: record["square_vertices_world"].__setitem__(
                2, [99.0, 50.0, 0.0]
            ),
            "100 mm 正方形",
        ),
        (
            lambda record: record.__setitem__(
                "probe_point_world", [0.0, 1.0, 0.0]
            ),
            "底边中点",
        ),
        (
            lambda record: record["local_axes_world"].__setitem__(
                "z", [0.0, 0.0, -1.0]
            ),
            "右手",
        ),
        (
            lambda record: record["local_axes_world"].__setitem__(
                "y", [1.0, 1.0, 0.0]
            ),
            "正交",
        ),
    ],
)
def test_select_zero_planes_rejects_invalid_recorded_geometry(mutate, message):
    record = deepcopy(_record())
    mutate(record)

    with pytest.raises(ValueError, match=message):
        select_zero_planes([record], {"stomach": 1})


def test_write_structured_exports_preserves_geometry_and_provenance(tmp_path):
    records = select_zero_planes([_record()], {"stomach": 1})
    provenance = {
        "source_manifest_sha256": "a" * 64,
        "core_design_sha256": "b" * 64,
        "build_git_commit": "c" * 40,
    }

    write_structured_exports(records, tmp_path, provenance)

    point_text = (tmp_path / "sampling_points.ply").read_text(encoding="utf-8")
    edge_text = (tmp_path / "zero_planes_edges.ply").read_text(encoding="utf-8")
    face_text = (tmp_path / "zero_planes_faces.ply").read_text(encoding="utf-8")
    assert "element vertex 1" in point_text
    assert "property float nx" in point_text
    assert "property uchar red" in point_text
    assert "element vertex 4" in edge_text
    assert "element edge 4" in edge_text
    assert "element face 1" in face_text

    payload = json.loads(
        (tmp_path / "sampling_points_zero_planes.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "zero-plane-visualization/v1"
    assert payload["coordinate_system"] == "RAS"
    assert payload["unit"] == "mm"
    assert payload["record_count"] == 1
    assert payload["organ_counts"] == {"stomach": 1}
    assert payload["provenance"] == provenance
    assert payload["records"][0]["vertices_world"][2] == [100.0, 50.0, 0.0]

    with (tmp_path / "sampling_points_zero_planes.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["slice_id"] == "stomach-000000-rp000-pp000-yp000"
    assert float(rows[0]["v2_x_mm"]) == 100.0


def test_render_outputs_are_offline_and_nonblank(tmp_path):
    records = select_zero_planes([_record()], {"stomach": 1})
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [-5.0, -5.0, -5.0],
                [5.0, -5.0, -5.0],
                [0.0, 5.0, -5.0],
                [0.0, 0.0, 5.0],
            ]
        ),
        faces=np.asarray([[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
        process=False,
    )
    meshes = {"stomach": mesh}

    render_interactive_html(
        records,
        meshes,
        tmp_path / "sampling_points_zero_planes_interactive.html",
    )
    render_static_views(records, meshes, tmp_path)

    html = (tmp_path / "sampling_points_zero_planes_interactive.html").read_text(
        encoding="utf-8"
    )
    assert 'src="https://cdn.plot.ly' not in html
    assert "Plotly.newPlot" in html
    assert "stomach" in html
    assert visualization.INTERACTIVE_ORGAN_MESH_OPACITY == pytest.approx(0.70)
    assert visualization.STATIC_ORGAN_MESH_ALPHA == pytest.approx(0.14)
    assert visualization.ORGAN_OPACITY_MIN == pytest.approx(0.10)
    assert visualization.ORGAN_OPACITY_MAX == pytest.approx(1.00)
    assert visualization.ORGAN_OPACITY_STEP == pytest.approx(0.05)
    assert '"opacity":0.7' in html
    assert 'id="zero-plane-visualization"' in html
    assert 'id="zero-plane-visibility-toggle"' in html
    assert 'id="organ-mesh-opacity-slider"' in html
    assert 'id="organ-mesh-opacity-value"' in html
    assert 'min="0.1" max="1.0" step="0.05" value="0.7"' in html
    assert 'class="zero-plane-toolbar"' in html
    assert "position: fixed" not in html
    assert "显示 0° 基准面" in html
    assert "器官网格不透明度" in html
    assert "70%" in html
    assert "const organMeshTraceIndices = [0];" in html
    assert "Plotly.restyle(graph, {opacity}, organMeshTraceIndices)" in html
    assert "const planeTraceIndices = [2,3];" in html
    assert (
        '"visible":[true,true,false,false,false,false,false]}],'
        '"label":"Points only"' in html
    )
    assert "plotly_buttonclicked" in html
    assert "indeterminate" in html
    for name in ("isometric", "axial", "coronal", "sagittal"):
        path = tmp_path / f"sampling_points_zero_planes_{name}.png"
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape[0] >= 600
        assert pixels.shape[1] >= 800
        assert float(pixels.std()) > 0.0


@pytest.mark.parametrize(
    ("view", "expected", "labels"),
    [
        ("axial", [[1.0, 2.0], [4.0, 5.0]], ("R (+) / L (-) [mm]", "A (+) / P (-) [mm]")),
        ("coronal", [[1.0, 3.0], [4.0, 6.0]], ("R (+) / L (-) [mm]", "S (+) / I (-) [mm]")),
        ("sagittal", [[2.0, 3.0], [5.0, 6.0]], ("A (+) / P (-) [mm]", "S (+) / I (-) [mm]")),
    ],
)
def test_project_orthographic_uses_two_patient_axes(view, expected, labels):
    projected, x_label, y_label = project_orthographic(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), view
    )

    assert np.array_equal(projected, np.asarray(expected))
    assert (x_label, y_label) == labels


def test_load_visualization_meshes_converts_lps_to_ras_and_duplicates_esophagus(tmp_path):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    for organ in TARGET_COUNTS:
        if organ == "esophagus":
            vertices = np.asarray(
                [
                    [x, y, z]
                    for z in (0.0, 5.0, 10.0)
                    for x, y in ((8.0, 17.0), (12.0, 17.0), (12.0, 23.0), (8.0, 23.0))
                ]
            )
            faces = []
            for lower in (0, 4):
                upper = lower + 4
                for index in range(4):
                    following = (index + 1) % 4
                    faces.extend(
                        [
                            [lower + index, lower + following, upper + following],
                            [lower + index, upper + following, upper + index],
                        ]
                    )
            mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
        else:
            mesh = trimesh.creation.box(extents=(4.0, 6.0, 10.0))
            mesh.apply_translation((10.0, 20.0, 5.0))
        if organ == "liver":
            mesh.apply_translation((0.0, 0.0, -5.0))
        mesh.export(meshes / f"{organ}.ply")

    loaded, esophagus_span = load_visualization_organ_meshes(
        meshes, input_coordinate_system="LPS"
    )

    stomach_vertices = np.asarray(loaded["stomach"].vertices)
    assert np.allclose(stomach_vertices.min(axis=0), [-12.0, -23.0, 0.0])
    assert np.allclose(stomach_vertices.max(axis=0), [-8.0, -17.0, 10.0])
    assert esophagus_span == pytest.approx(5.0)
    esophagus_z = np.asarray(loaded["esophagus"].vertices)[:, 2]
    assert float(esophagus_z.min()) == pytest.approx(-5.0)
    assert float(esophagus_z.max()) == pytest.approx(5.0)


TARGET_COUNTS = {
    "stomach": 118,
    "liver": 162,
    "pancreas": 37,
    "duodenum": 53,
    "esophagus": 30,
}

LEGACY_NAMES = {
    "stomach": "Stomach",
    "liver": "Liver",
    "pancreas": "Pancreas",
    "duodenum": "Duodenum",
    "esophagus": "Esophagus",
}


def _shifted_record(organ: str, index: int, z_offset: float) -> dict:
    record = _record(index, organ=organ)
    shift = np.asarray([0.0, 0.0, z_offset])
    record["probe_point_world"] = shift.tolist()
    record["square_vertices_world"] = (
        np.asarray(record["square_vertices_world"]) + shift
    ).tolist()
    return record


def _write_surface_ply(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(records)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "end_header",
    ]
    for record in records:
        values = record["probe_point_world"] + record["input_normal_world"]
        lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cli_exports_complete_case2_bundle(tmp_path):
    inputs = tmp_path / "input"
    samples = inputs / "ResampledpointPLY"
    meshes = inputs / "target_organ_meshes"
    all_records: list[dict] = []
    global_index = 0
    for organ, count in TARGET_COUNTS.items():
        organ_records = []
        for point_index in range(count):
            record = _shifted_record(organ, point_index, float(global_index) * 0.25)
            organ_records.append(record)
            all_records.append(record)
            global_index += 1
        _write_surface_ply(samples / f"FPS-{LEGACY_NAMES[organ]}.ply", organ_records)
        mesh = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
        meshes.mkdir(parents=True, exist_ok=True)
        mesh.export(meshes / f"{organ}.ply")

    zero_records = inputs / "zero_records.jsonl"
    zero_records.write_text(
        "".join(json.dumps(record) + "\n" for record in all_records),
        encoding="utf-8",
    )
    run_metadata = inputs / "run_metadata.json"
    run_metadata.write_text(
        json.dumps(
            {
                "run_state": "complete",
                "total_squares": 1431118,
                "input_coordinate_system": "RAS",
                "core_design_sha256": "b" * 64,
                "build_git_commit": "c" * 40,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "delivery"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_zero_plane_visualization.py",
            "--zero-records-jsonl",
            str(zero_records),
            "--sample-ply-dir",
            str(samples),
            "--organ-mesh-dir",
            str(meshes),
            "--run-metadata",
            str(run_metadata),
            "--source-manifest-sha256",
            "a" * 64,
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["record_count"] == 400
    expected_files = {
        "sampling_points.ply",
        "zero_planes_edges.ply",
        "zero_planes_faces.ply",
        "sampling_points_zero_planes.csv",
        "sampling_points_zero_planes.json",
        "sampling_points_zero_planes_interactive.html",
        "sampling_points_zero_planes_isometric.png",
        "sampling_points_zero_planes_axial.png",
        "sampling_points_zero_planes_coronal.png",
        "sampling_points_zero_planes_sagittal.png",
        "README_中文.txt",
        "SHA256SUMS.txt",
    }
    assert expected_files <= {path.name for path in output.iterdir() if path.is_file()}
    assert len(list((output / "target_organ_meshes").glob("*.ply"))) == 5
    assert not list(tmp_path.glob(".delivery.tmp*"))
