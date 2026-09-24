"""由元素质量百分比实测值反推 CHNO 分子式（元素分析式反演）。

给定 C、H、N、O 四种元素的实测质量百分比与绝对容差（闭区间）以及
摩尔质量的正整数闭区间，枚举原子计数 ``(c, h, n, o)``（每种元素
**0～30**，总数大于零），按名义原子质量 C=12、H=1、N=14、O=16
计算分子式摩尔质量

    M = 12c + h + 14n + 16o

与各元素计算质量百分比

    p_X = 100·w_X·k_X / M

只保留同时满足

    measurement_X − tolerance_X ≤ p_X ≤ measurement_X + tolerance_X
    （对四种元素逐一起作用，实测值不要求恰好合计 100）
    molar_mass_min ≤ M ≤ molar_mass_max

的计数。百分比与容差先把十进制字符串**无损**转为
:class:`fractions.Fraction`，热循环中的比较全部改写为整数交叉相乘
（``a/b ≤ c/d`` ⇔ ``a·d ≤ c·b``），不经过任何浮点运算。

枚举结构：外层只遍历 (c, h, n) 三元组（至多 31³ ≈ 2.98 万组），由
三种元素的百分比闭区间与质量闭区间把 M 夹成一个整数区间，再在该
区间内扫描 o（O 原子量最大、区间内至多 31 个）。即使全部 92 万余个
非零计数都命中（如全部元素容差为 100），热循环也不构造任何
:class:`Fraction`，只用整数交叉相乘维护一个至多 20 项的堆（总偏差
为有理数，比较按交叉相乘进行），因此最坏情形也在约 3 秒内完成；
仅对最终返回的至多 20 个候选构造精确百分比与偏差。

结果按 **总百分比绝对偏差** Σ|p_X − measurement_X|（精确有理数）、
**摩尔质量**、**(C,H,N,O) 计数** 依次升序排序；分子式文本按 Hill
顺序（C、H 在前，其余按字母序——本服务固定为 C、H、N、O）渲染，
原子数为 0 的元素省略、为 1 时省略下标。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

#: 固定支持的元素（也是结果排序与 Hill 渲染顺序）
ELEMENTS: tuple[str, ...] = ("C", "H", "N", "O")

#: 名义原子质量（C=12、H=1、N=14、O=16）
ATOMIC_WEIGHT: dict[str, int] = {"C": 12, "H": 1, "N": 14, "O": 16}

#: 每种元素的原子数上限（含端点）
MAX_ATOMS_PER_ELEMENT = 30

#: 最多返回的分子式条数
RESULT_LIMIT = 20


@dataclass(frozen=True)
class ElementMeasurement:
    """一种元素的实测质量百分比（百分数，0～100）与绝对容差（≥0）。"""

    element: str
    measurement: Fraction
    tolerance: Fraction

    @property
    def lower(self) -> Fraction:
        """百分比接受区间下界 = 实测 − 容差（可能为负，筛选时按闭区间）。"""
        return self.measurement - self.tolerance

    @property
    def upper(self) -> Fraction:
        """百分比接受区间上界 = 实测 + 容差。"""
        return self.measurement + self.tolerance


@dataclass(frozen=True)
class FormulaMatch:
    """一个落入全部容差与摩尔质量区间的候选分子式。"""

    formula: str
    counts: dict[str, int]
    molar_mass: int
    percentages: dict[str, Fraction]
    total_deviation: Fraction


def _ceil_div(a: int, b: int) -> int:
    """整数 ceil(a / b)（b > 0；a 可负）。"""
    return -((-a) // b)


def render_formula(counts: dict[str, int]) -> str:
    """按 Hill 顺序渲染 CHNO 计数；原子数为 0 省略、为 1 省略下标。"""
    parts: list[str] = []
    for element in ELEMENTS:
        count = counts.get(element, 0)
        if count:
            parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


@dataclass(frozen=True)
class _Bounds:
    """某元素实测百分比接受区间与实测值的整数分子/分母表示。"""

    lower_num: int
    lower_den: int
    upper_num: int
    upper_den: int
    meas_num: int
    meas_den: int


class _HeapEntry:
    """堆中的一项：以交叉相乘比较有理数偏差，再依次比较整数次序键。

    总偏差为 ``dev_num / (M·Q)``（Q 为四元素实测分母之积，对全部
    候选相同），交叉相乘比较时 Q 可约去，故只需存 ``dev_num`` 与 M；
    堆为 **max 序**（:meth:`__lt__` 表示“在堆序中更小”，即候选
    更差），因此堆顶恒为当前保留集合中**最差**的候选，便于在发现
    更优候选时以 :func:`heapq.heapreplace` 替换。

    使用 ``__slots__`` 与方法局部化以压低 92 万级候选最坏情形下
    逐候选的构造/比较开销；仅在真正入堆时才构造实例。
    """

    __slots__ = ("dev_num", "molar_mass", "c", "h", "n", "o")

    def __init__(
        self, dev_num: int, molar_mass: int, c: int, h: int, n: int, o: int
    ) -> None:
        self.dev_num = dev_num
        self.molar_mass = molar_mass
        self.c = c
        self.h = h
        self.n = n
        self.o = o

    def __lt__(self, other: "_HeapEntry") -> bool:
        # 堆为 max-heap：展示排序键（偏差, M, C, H, N, O）**更大**者
        # 才是堆序中的“更小”（堆顶）；调用 heapq（最小堆）后堆顶
        # 即当前保留集合中展示排序最大的候选（最差候选）
        return _sort_greater(self, other)


def _sort_greater(left: _HeapEntry, right: _HeapEntry) -> bool:
    """展示排序：left 的排序键严格大于 right。"""
    cross_l = left.dev_num * right.molar_mass
    cross_r = right.dev_num * left.molar_mass
    if cross_l != cross_r:
        return cross_l > cross_r
    if left.molar_mass != right.molar_mass:
        return left.molar_mass > right.molar_mass
    return (left.c, left.h, left.n, left.o) > (
        right.c,
        right.h,
        right.n,
        right.o,
    )


def infer_formulas(
    measurements: Sequence[ElementMeasurement],
    mass_min: int,
    mass_max: int,
) -> tuple[list[FormulaMatch], int]:
    """枚举落入全部百分比容差与摩尔质量闭区间的全部 CHNO 分子式。

    ``measurements`` 必须恰好按 :data:`ELEMENTS` 顺序给出 C、H、N、O
    四条（由业务层保证）；``mass_min ≤ mass_max`` 同样由业务层保证。
    返回 ``(matches, total)``：``matches`` 为按（总偏差, 摩尔质量,
    C/H/N/O 计数）升序排列的至多 :data:`RESULT_LIMIT` 条候选，
    ``total`` 为截取前的全部候选数（无候选时两者分别为空列表与 0）。
    """
    by_element = {item.element: item for item in measurements}
    bounds: dict[str, _Bounds] = {}
    denominator_product = 1
    for element in ELEMENTS:
        lower = by_element[element].lower
        upper = by_element[element].upper
        meas = by_element[element].measurement
        bounds[element] = _Bounds(
            *lower.as_integer_ratio(),
            *upper.as_integer_ratio(),
            *meas.as_integer_ratio(),
        )
        denominator_product *= meas.denominator

    lim = MAX_ATOMS_PER_ELEMENT
    o_weight = ATOMIC_WEIGHT["O"]
    heap: list[_HeapEntry] = []
    total_matches = 0
    heappush = heapq.heappush
    heapreplace = heapq.heapreplace

    # 热循环只做整数运算：预先取出各元素实测 (num, den) 与 100·w
    bc, bh, bn, bo = (bounds[e] for e in ELEMENTS)
    qprod = denominator_product
    c100, h100, n100, o100 = (100 * ATOMIC_WEIGHT[e] for e in ELEMENTS)
    qc, qh, qn, qo = (qprod // b.meas_den for b in (bc, bh, bn, bo))

    def consider(
        molar_mass: int, c: int, h: int, n: int, o: int
    ) -> None:
        # 总偏差 Σ|100·w·k/M − meas_num/meas_den| 通分为
        # dev_num / (M·Q)：各元素项
        # |100·w·k·q − meas_num·M|·(Q/q) / (M·Q)，分子相加即可
        dev_num = (
            abs(c100 * c * bc.meas_den - bc.meas_num * molar_mass) * qc
            + abs(h100 * h * bh.meas_den - bh.meas_num * molar_mass) * qh
            + abs(n100 * n * bn.meas_den - bn.meas_num * molar_mass) * qn
            + abs(o100 * o * bo.meas_den - bo.meas_num * molar_mass) * qo
        )
        if len(heap) < RESULT_LIMIT:
            heappush(
                heap, _HeapEntry(dev_num, molar_mass, c, h, n, o)
            )
            return
        worst = heap[0]
        # 新候选是否严格优于堆顶（最差）候选；Q 在交叉相乘中约去
        cross_new = dev_num * worst.molar_mass
        cross_worst = worst.dev_num * molar_mass
        if (
            cross_new < cross_worst
            or (
                cross_new == cross_worst
                and (
                    molar_mass < worst.molar_mass
                    or (
                        molar_mass == worst.molar_mass
                        and (c, h, n, o) < (worst.c, worst.h, worst.n, worst.o)
                    )
                )
            )
        ):
            heapreplace(
                heap, _HeapEntry(dev_num, molar_mass, c, h, n, o)
            )

    for c in range(lim + 1):
        for h in range(lim + 1):
            for n in range(lim + 1):
                base_mass = (
                    ATOMIC_WEIGHT["C"] * c
                    + ATOMIC_WEIGHT["H"] * h
                    + ATOMIC_WEIGHT["N"] * n
                )
                # M 的可行闭区间 [mass_low, mass_high]：
                # 总体质量区间 ∩ [base_mass, base_mass + 16·30]（M ≥ base
                # 因 o ≥ 0；M=0 的空分子在下面由 o 区间自动排除）
                mass_low = max(mass_min, base_mass)
                mass_high = min(mass_max, base_mass + o_weight * lim)
                if mass_low > mass_high:
                    continue

                infeasible = False
                # C/H/N 的百分比闭区间进一步收紧 M：
                #   100·w·k/M ∈ [lower, upper]
                # k=0 时只需 lower ≤ 0；k>0 时
                #   M ≥ 100·w·k / upper（upper=0 则不可能）
                #   M ≤ 100·w·k / lower（仅 lower>0 时提供上界）
                for count, b, x_factor in (
                    (c, bc, c100),
                    (h, bh, h100),
                    (n, bn, n100),
                ):
                    if count == 0:
                        if b.lower_num > 0:
                            infeasible = True
                            break
                        continue
                    if b.upper_num == 0:
                        infeasible = True
                        break
                    x = x_factor * count
                    mass_low = max(
                        mass_low, _ceil_div(x * b.upper_den, b.upper_num)
                    )
                    if b.lower_num > 0:
                        mass_high = min(
                            mass_high, (x * b.lower_den) // b.lower_num
                        )
                    if mass_low > mass_high:
                        infeasible = True
                        break
                if infeasible:
                    continue

                # 由 M ∈ [mass_low, mass_high] 反推 o 的整数区间
                o_low = max(0, _ceil_div(mass_low - base_mass, o_weight))
                o_high = min(lim, (mass_high - base_mass) // o_weight)
                for o in range(o_low, o_high + 1):
                    molar_mass = base_mass + o_weight * o
                    x_o = o100 * o
                    # O 的百分比闭区间（整数交叉相乘，闭区间含端点）
                    if x_o * bo.lower_den < bo.lower_num * molar_mass:
                        continue
                    if x_o * bo.upper_den > bo.upper_num * molar_mass:
                        continue
                    total_matches += 1
                    consider(molar_mass, c, h, n, o)

    # 堆顶为最差候选，依次弹出为“最差 → 最优”，反转后即展示顺序
    entries = [heapq.heappop(heap) for _ in range(len(heap))]
    entries.reverse()

    result: list[FormulaMatch] = []
    for entry in entries:
        counts = {"C": entry.c, "H": entry.h, "N": entry.n, "O": entry.o}
        molar_mass = entry.molar_mass
        percentages = {
            element: Fraction(
                100 * ATOMIC_WEIGHT[element] * counts[element], molar_mass
            )
            for element in ELEMENTS
        }
        total_deviation = sum(
            (
                abs(percentages[element] - by_element[element].measurement)
                for element in ELEMENTS
            ),
            Fraction(0),
        )
        result.append(
            FormulaMatch(
                formula=render_formula(counts),
                counts=counts,
                molar_mass=molar_mass,
                percentages=percentages,
                total_deviation=total_deviation,
            )
        )
    return result, total_matches
