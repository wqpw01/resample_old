"""20260813 基准设计及 20260819 正式修订的不可变算法常量。"""

BASE_CORE_DESIGN_FILENAME = "基于目标器官的采样方法-20260813.docx"
BASE_CORE_DESIGN_SHA256 = "de56e7a1b984f925e97631b076d6b729e77575eb6513b4d57f3028818b7e71ca"
CORE_DESIGN_FILENAME = "core-design-amendment-20260819.md"
CORE_DESIGN_SHA256 = "001e033727172593b79577cb1ee53a738189fff5efab2bdbffefc8c197571d9a"

BLACK_RATIO_LIMIT = 0.60
MINIMUM_POINT_SPACING_MM = 10.0
RAY_LENGTH_MM = 100.0
RAY_BATCH_SIZE = 2048
SAMPLING_SEED = 0
CENTERLINE_VOXEL_PITCH_MM = 1.0
CENTERLINE_TANGENT_WINDOW_MM = 10.0
CENTERLINE_MAX_TERMINAL_SPUR_MM = 5.0
SQUARE_SIDE_LENGTH_MM = 100.0
OUTPUT_RESOLUTION = 300
WINDOW_LEVEL_HU = 40.0
WINDOW_WIDTH_HU = 400.0
FILL_HU_VALUE = -1000.0
BLACK_THRESHOLD = 50
LINE_MIN_DIAGONAL_FRACTION = 0.70
BLACK_SIDE_MIN_RATIO = 0.90
VALID_SIDE_MAX_BLACK_RATIO = 0.10
ESOPHAGUS_EXTENSION_TARGET_FILTER = "original_and_translated_segments_independently"
POSE_CONVENTION = {
    "coordinate_frame": "local_right_handed",
    "matrix_order": "B @ Rz(yaw) @ Ry(pitch) @ Rx(roll)",
    "positive_yaw": "counterclockwise",
    "yaw_observer": "local_positive_z_looking_toward_probe",
    "rotation_center": "probe_at_square_bottom_edge_midpoint",
}
FOV_POLICY = {
    "vertex_rule": "any_square_vertex_outside_ct",
    "outside_status": "excluded_fov",
    "saved_artifacts": ["ct_png"],
    "out_of_bounds_png_value": 0,
}

ROLL_ANGLES_DEGREES = tuple(float(value) for value in range(-45, 46, 5))
PITCH_ANGLES_DEGREES = tuple(float(value) for value in range(-30, 31, 5))
STANDARD_YAW_ANGLES_DEGREES = tuple(float(value) for value in range(-30, 31, 5))
SPECIAL_YAW_ANGLES_DEGREES = tuple(float(value) for value in range(-120, 31, 5))
LIVER_REGION_TWO_YAW_ANGLES_DEGREES = tuple(float(value) for value in range(-60, 61, 5))
