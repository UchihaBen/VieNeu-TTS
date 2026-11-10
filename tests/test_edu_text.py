import pytest

from apps.edu_text import normalize_edu


@pytest.mark.parametrize("src, want", [
    ("12 : 4 = 3", "12 chia 4 = 3"),
    ("12:4 = 3", "12 chia 4 = 3"),
    ("15 ÷ 3 = 5", "15 chia 3 = 5"),
    ("3 × 4 = 12", "3 nhân 4 = 12"),
    ("Tô màu 1/3 hình tròn.", "Tô màu 1 phần 3 hình tròn."),
    ("3,14", "3 phẩy 14"),
    ("2,5 kg", "2 phẩy 5 kg"),
    ("1,05", "1,05"),
    ("các số 1,2,3", "các số 1, 2, 3"),
    ("Bài IV: Phép chia", "Bài 4: Phép chia"),
    ("Thế kỷ XXI", "Thế kỷ 21"),
    # left to the stock normalizer
    ("8:30", "8:30"),
    ("Bài 3: Tính nhẩm", "Bài 3: Tính nhẩm"),
    ("12/11/2026", "12/11/2026"),
    ("1.000.000", "1.000.000"),
    ("I love", "I love"),
])
def test_normalize_edu(src, want):
    assert normalize_edu(src) == want
