"""20260813 论文重采样合同的不可变算法常量。"""

CORE_DESIGN_FILENAME = "基于目标器官的采样方法-20260813.docx"
CORE_DESIGN_SHA256 = "de56e7a1b984f925e97631b076d6b729e77575eb6513b4d57f3028818b7e71ca"

BLACK_RATIO_LIMIT = 0.60

ROLL_ANGLES_DEGREES = tuple(float(value) for value in range(-45, 46, 5))
PITCH_ANGLES_DEGREES = tuple(float(value) for value in range(-30, 31, 5))
STANDARD_YAW_ANGLES_DEGREES = tuple(float(value) for value in range(-30, 31, 5))
SPECIAL_YAW_ANGLES_DEGREES = tuple(float(value) for value in range(-120, 31, 5))
