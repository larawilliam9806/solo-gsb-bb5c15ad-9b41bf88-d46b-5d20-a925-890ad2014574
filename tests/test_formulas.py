"""分子式反演（``POST /api/v1/formulas/infer``）的核心算法与 HTTP API 测试。

核心（区间收紧、O 闭区间筛选、总偏差键、堆前 20 截取、计数上限与
总数大于零、闭区间端点包含）直接测试 ``app.formulas``；
HTTP 层测试成功路径、回显、排序/截断标志与 422 输入校验。
"""

import time
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.formulas import (
    ELEMENTS,
    MAX_ATOMS_PER_ELEMENT,
    RESULT_LIMIT,
    ElementMeasurement,
    infer_formulas,
    render_formula,
)
from app.main import app

client = TestClient(app)

PATH = "/api/v1/formulas/infer"

WEIGHT = {"C": 12, "H": 1, "N": 14, "O": 16}


# ------------------------------------------------------------ 辅助
def make_measurements(mapping, tol="0"):
    """``mapping`` 为 element -> (measurement, tolerance) 或 measurement。"""
    out = []
    for element in ELEMENTS:
        value = mapping[element]
        if isinstance(value, tuple):
            measurement, tolerance = value
        else:
            measurement, tolerance = value, tol
        out.append(
            ElementMeasurement(
                element, Fraction(measurement), Fraction(tolerance)
            )
        )
    return out


def ethanol(tol="0.5"):
    return make_measurements(
        {
            "C": ("52.14", tol),
            "H": ("13.13", tol),
            "N": ("0", "0"),
            "O": ("34.73", tol),
        }
    )


# ------------------------------------------------------------ 分子式渲染
def test_render_formula_hill_order():
    assert render_formula({"C": 2, "H": 6, "N": 0, "O": 1}) == "C2H6O"
    assert render_formula({"C": 1, "H": 4, "N": 0, "O": 0}) == "CH4"
    assert render_formula({"C": 0, "H": 2, "N": 0, "O": 1}) == "H2O"
    assert render_formula({"C": 6, "H": 0, "N": 0, "O": 0}) == "C6"
    assert render_formula({"C": 0, "H": 0, "N": 2, "O": 1}) == "N2O"


# ------------------------------------------------------------ 核心：基础命中
def test_infer_ethanol_family_sorted_by_mass():
    # 以精确计算百分比为实测、1e-12 容差（区间仍含精确值）：
    # 只保留最简式 C2H6O 的整数倍，且受每种元素 ≤30 约束
    matches, total = infer_formulas(
        make_measurements(
            {
                "C": ("52.17391304347826086956521739", "1e-12"),
                "H": ("13.04347826086956521739130435", "1e-12"),
                "N": ("0", "0"),
                "O": ("34.78260869565217391304347826", "1e-12"),
            }
        ),
        1,
        1290,
    )
    assert total == len(matches) == 5
    assert [m.formula for m in matches] == [
        "C2H6O", "C4H12O2", "C6H18O3", "C8H24O4", "C10H30O5",
    ]
    first = matches[0]
    assert first.molar_mass == 46
    assert first.counts == {"C": 2, "H": 6, "N": 0, "O": 1}
    assert first.percentages["C"] == Fraction(2400, 46)
    assert first.percentages["N"] == 0
    assert sum(first.percentages.values()) == 100


def test_infer_loose_tolerance_admits_near_neighbors():
    # 容差 0.5 时除 5 个精确倍数外还纳入 C8H25O4、C10H29O5 等邻近式
    matches, total = infer_formulas(ethanol("0.5"), 1, 1290)
    assert total == 7
    assert "C8H25O4" in [m.formula for m in matches]
    # 同偏差倍数在前、同质量偏差由 (C,H,N,O) 次序决定，整体有序
    keys = [(m.total_deviation, m.molar_mass,
             tuple(m.counts[e] for e in ELEMENTS)) for m in matches]
    assert keys == sorted(keys)


def test_infer_mass_interval_restricts_multiple():
    matches, total = infer_formulas(ethanol(), 40, 50)
    assert total == 1
    assert matches[0].formula == "C2H6O"


def test_infer_mass_closed_endpoints():
    # M=46 恰为闭区间端点时仍应命中
    matches, total = infer_formulas(ethanol("0.5"), 46, 46)
    assert total == 1
    assert matches[0].molar_mass == 46


def test_infer_no_solution_empty_list():
    # C 几乎 100% 但又要求少量其它元素，无解
    matches, total = infer_formulas(
        make_measurements(
            {
                "C": ("99", "0.1"),
                "H": ("5", "0.1"),
                "N": ("0", "0"),
                "O": ("0", "0"),
            }
        ),
        1,
        1290,
    )
    assert matches == []
    assert total == 0


def test_infer_exact_zero_tolerance_inclusive_boundary():
    # CH4O 的百分比恰为有限小数 37.5 / 12.5 / 0 / 50：
    # 容差为 0（闭区间退化为等式）时最简式倍数必须全部精确命中
    matches, total = infer_formulas(
        make_measurements(
            {"C": "37.5", "H": "12.5", "N": "0", "O": "50"}
        ),
        1,
        1290,
    )
    assert [m.formula for m in matches] == [
        "CH4O", "C2H8O2", "C3H12O3", "C4H16O4", "C5H20O5",
        "C6H24O6", "C7H28O7",
    ]
    assert matches[0].total_deviation == 0
    # 等式边界不含任何近似：H2O（11.11…%）不得混入
    assert all(m.counts["N"] == 0 for m in matches)


def test_infer_measurements_need_not_sum_to_100():
    # 实测合计 100.2%（实验误差，不强制归一）：逐元素落区间即命中
    matches, total = infer_formulas(
        make_measurements(
            {
                "C": ("52.20", "0.1"),
                "H": ("13.20", "0.2"),
                "N": ("0", "0"),
                "O": ("34.80", "0.1"),
            }
        ),
        46,
        46,
    )
    assert total == 1
    assert matches[0].formula == "C2H6O"


def test_infer_zero_count_element_positive_lower_bound_excluded():
    # N 下界为正：任何 n=0 的计数都应被排除
    matches, total = infer_formulas(
        make_measurements(
            {
                "C": ("40", "40"),
                "H": ("0", "100"),
                "N": ("0.1", "50"),
                "O": ("0", "100"),
            }
        ),
        1,
        1290,
    )
    assert total > 0
    assert all(m.counts["N"] >= 1 for m in matches)


def test_infer_upper_bound_zero_forces_count_zero():
    # H 上界恰为 0：h 必须为 0
    matches, total = infer_formulas(
        make_measurements(
            {
                "C": ("0", "100"),
                "H": ("0", "0"),
                "N": ("0", "100"),
                "O": ("0", "100"),
            }
        ),
        1,
        1290,
    )
    assert total > 0
    assert all(m.counts["H"] == 0 for m in matches)


def test_infer_total_atoms_positive():
    # 全部区间接受，但 M ≥ 1 保证全零计数 (0,0,0,0) 不出现
    matches, total = infer_formulas(
        make_measurements({e: ("0", "100") for e in ELEMENTS}), 1, 1290
    )
    assert all(
        (m.counts["C"] + m.counts["H"] + m.counts["N"] + m.counts["O"]) > 0
        for m in matches
    )
    # 最轻候选是 H（M=1），排序首位
    assert matches[0].formula == "H"
    assert matches[0].molar_mass == 1


def test_infer_per_element_count_capped_at_30():
    # C 区间逼出 31 个 C 也必须被上限截断：C%=100 只允许 C1..C30
    matches, total = infer_formulas(
        make_measurements(
            {"C": "100", "H": "0", "N": "0", "O": "0"}
        ),
        1,
        100000,
    )
    # C%=100 只允许 C1..C30；共 30 个候选，截取前 20 个（C1..C20）
    assert total == MAX_ATOMS_PER_ELEMENT
    assert len(matches) == RESULT_LIMIT
    assert [m.formula for m in matches] == ["C" if i == 1 else f"C{i}"
                                            for i in range(1, 21)]
    assert all(m.counts["C"] <= MAX_ATOMS_PER_ELEMENT for m in matches)


# ------------------------------------------------------------ 核心：排序与截取
def test_infer_truncation_returns_top_20_and_total():
    matches, total = infer_formulas(
        make_measurements({e: ("0", "100") for e in ELEMENTS}), 1, 60
    )
    assert total > RESULT_LIMIT
    assert len(matches) == RESULT_LIMIT == 20
    # 返回页按 (偏差, M, C, H, N, O) 升序
    keys = [
        (m.total_deviation, m.molar_mass, m.counts["C"], m.counts["H"],
         m.counts["N"], m.counts["O"])
        for m in matches
    ]
    assert keys == sorted(keys)


def test_infer_ordering_deviation_before_mass():
    # 构造两个同质量区间内的候选，验证偏差小者在前（即使质量更大也靠前）
    # H2O：H≈11.11 O≈88.89；给 H 实测 12（容差 2）、O 实测 87（容差 3）
    matches, _ = infer_formulas(
        make_measurements(
            {
                "C": ("0", "0"),
                "H": ("12", "2"),
                "N": ("0", "0"),
                "O": ("87", "3"),
            }
        ),
        1,
        1290,
    )
    deviations = [m.total_deviation for m in matches]
    assert deviations == sorted(deviations)


def test_infer_all_hydrogen_isotope_tiebreak_by_counts():
    # 同偏差、同质量在 CHNO 名义质量下极少出现；验证同偏差同质量时
    # 按 C/H/N/O 计数元组升序（直接检查排序键单调）
    matches, _ = infer_formulas(
        make_measurements({e: ("0", "100") for e in ELEMENTS}), 1, 1290
    )
    keys = [(m.total_deviation, m.molar_mass,
             tuple(m.counts[e] for e in ELEMENTS)) for m in matches]
    assert keys == sorted(keys)


# ------------------------------------------------------------ 核心：精确有理数
def test_infer_nonterminating_decimal_exact():
    # 0.1 是二进制浮点无法精确表示的十进制；算法须按精确 Fraction 判定。
    # H2O 的 H%=100*2/18=11.11…：测量 11.1、容差 0.02 必含；
    # 测量 11.09、容差 0（与 11.11… 不等）必不含
    inclusive, total_in = infer_formulas(
        make_measurements(
            {"C": "0", "H": ("11.1", "0.02"), "N": "0", "O": ("88.89", "0.02")}
        ),
        18,
        18,
    )
    assert total_in == 1
    assert inclusive[0].formula == "H2O"

    exclusive, total_ex = infer_formulas(
        make_measurements(
            {"C": "0", "H": "11.09", "N": "0", "O": "88.91"}
        ),
        18,
        18,
    )
    assert total_ex == 0
    assert exclusive == []


def test_infer_worst_case_performance():
    # 全部区间接受、质量区间拉满：92 万余候选也应在合理时间内完成
    start = time.perf_counter()
    matches, total = infer_formulas(
        make_measurements({e: ("0", "100") for e in ELEMENTS}), 1, 100000
    )
    elapsed = time.perf_counter() - start
    assert len(matches) == RESULT_LIMIT
    assert total == 31**4 - 1  # 除全零计数外全部命中
    assert elapsed < 10.0


# ------------------------------------------------------------ HTTP：成功路径
def test_api_success_shape_and_echo():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "O", "measurement": "34.73", "tolerance": "0.5"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "C", "measurement": "52.14", "tolerance": "0.5"},
                {"element": "H", "measurement": "13.13", "tolerance": "0.5"},
            ],
            "molar_mass_min": 40,
            "molar_mass_max": 50,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # 回显固定按 C/H/N/O 排序，与提交顺序无关
    assert [item["element"] for item in data["elements"]] == list(ELEMENTS)
    assert data["molar_mass_min"] == 40
    assert data["molar_mass_max"] == 50
    assert data["total_matches"] == data["count"] == 1
    assert data["truncated"] is False
    candidate = data["formulas"][0]
    assert candidate["formula"] == "C2H6O"
    assert candidate["counts"] == {"C": 2, "H": 6, "N": 0, "O": 1}
    assert candidate["molar_mass"] == 46
    assert candidate["percentages"]["N"] == "0"
    assert set(candidate) == {
        "formula", "counts", "molar_mass", "percentages", "total_deviation",
    }


def test_api_empty_result_returns_empty_list():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "C", "measurement": "99", "tolerance": "0.1"},
                {"element": "H", "measurement": "0", "tolerance": "0"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "O", "measurement": "0", "tolerance": "0"},
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 1290,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["formulas"] == []
    assert data["count"] == 0
    assert data["total_matches"] == 0
    assert data["truncated"] is False


def test_api_truncation_flags_and_top_20():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": e, "measurement": "0", "tolerance": "100"}
                for e in ELEMENTS
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 60,
        },
    )
    data = resp.json()
    assert resp.status_code == 200
    assert len(data["formulas"]) == 20
    assert data["count"] == 20
    assert data["total_matches"] > 20
    assert data["truncated"] is True


def test_api_percentages_are_decimal_strings_summing_100():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "C", "measurement": "37.5", "tolerance": "0"},
                {"element": "H", "measurement": "12.5", "tolerance": "0"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "O", "measurement": "50", "tolerance": "0"},
            ],
            "molar_mass_min": 32,
            "molar_mass_max": 32,
        },
    )
    candidate = resp.json()["formulas"][0]
    assert candidate["formula"] == "CH4O"
    for value in candidate["percentages"].values():
        assert isinstance(value, str)
    assert sum(Fraction(v) for v in candidate["percentages"].values()) == 100


# ------------------------------------------------------------ HTTP：422 校验
def test_api_missing_element():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": e, "measurement": "0", "tolerance": "0"}
                for e in ("C", "H", "N")
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 10,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_api_duplicate_and_missing_elements():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "C", "measurement": "52", "tolerance": "1"},
                {"element": "H", "measurement": "13", "tolerance": "1"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 1290,
        },
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["duplicate_elements"] == ["N"]
    assert details["missing_elements"] == ["O"]


@pytest.mark.parametrize("bad_element", ["S", "Cl", "c", "carbon", ""])
def test_api_unexpected_element_rejected(bad_element):
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": e, "measurement": "0", "tolerance": "0"}
                for e in ("C", "H", "N")
            ]
            + [{"element": bad_element, "measurement": "0", "tolerance": "0"}],
            "molar_mass_min": 1,
            "molar_mass_max": 10,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize("bad_value", ["-0.1", "100.1", "NaN", "Infinity", "abc"])
def test_api_invalid_measurement(bad_value):
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "C", "measurement": bad_value, "tolerance": "0"},
                {"element": "H", "measurement": "0", "tolerance": "0"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "O", "measurement": "0", "tolerance": "0"},
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 10,
        },
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["invalid_measurements"][0]["element"] == "C"


@pytest.mark.parametrize("bad_value", ["-0.1", "100.1", "NaN", "Infinity", "abc"])
def test_api_invalid_tolerance(bad_value):
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "C", "measurement": "50", "tolerance": bad_value},
                {"element": "H", "measurement": "0", "tolerance": "0"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "O", "measurement": "0", "tolerance": "0"},
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 10,
        },
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["invalid_tolerances"][0]["element"] == "C"
    assert details["invalid_tolerances"][0]["tolerance"] == bad_value


def test_api_json_numbers_rejected():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "C", "measurement": 52.14, "tolerance": 0.5},
                {"element": "H", "measurement": "13.13", "tolerance": 0.5},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "O", "measurement": "34.73", "tolerance": 0.5},
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 1290,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize(
    "payload",
    [
        {"molar_mass_min": 0, "molar_mass_max": 10},
        {"molar_mass_min": -5, "molar_mass_max": 10},
        {"molar_mass_min": 1.5, "molar_mass_max": 10},
        {"molar_mass_min": 100, "molar_mass_max": 50},
    ],
)
def test_api_invalid_mass_bounds(payload):
    body = {
        "elements": [
            {"element": e, "measurement": "0", "tolerance": "100"}
            for e in ELEMENTS
        ],
        **payload,
    }
    resp = client.post(PATH, json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_api_inverted_mass_range_details():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": e, "measurement": "0", "tolerance": "100"}
                for e in ELEMENTS
            ],
            "molar_mass_min": 100,
            "molar_mass_max": 50,
        },
    )
    details = resp.json()["error"]["details"]
    assert details["molar_mass_min"] == 100
    assert details["molar_mass_max"] == 50


def test_api_blank_string_rejected():
    resp = client.post(
        PATH,
        json={
            "elements": [
                {"element": "C", "measurement": "  ", "tolerance": "0"},
                {"element": "H", "measurement": "0", "tolerance": "0"},
                {"element": "N", "measurement": "0", "tolerance": "0"},
                {"element": "O", "measurement": "0", "tolerance": "0"},
            ],
            "molar_mass_min": 1,
            "molar_mass_max": 10,
        },
    )
    assert resp.status_code == 422
