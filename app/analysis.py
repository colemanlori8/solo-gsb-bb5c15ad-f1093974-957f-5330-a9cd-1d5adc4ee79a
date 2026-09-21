"""三阶互调产物分析：枚举 2*fa - fb，聚合来源并评估最近命中对象。"""
from __future__ import annotations

from .schemas import GuardBand, HitObject, ProductOut, RiskGroup, SourcePair


def enumerate_third_order_products(
    carriers_hz: list[int], scan_start_hz: int, scan_end_hz: int
) -> dict[int, set[tuple[int, int]]]:
    """枚举不同载频的三阶产物 2*fa - fb（fa != fb）。

    仅保留落在扫描范围 [scan_start_hz, scan_end_hz] 内的产物；
    同频产物聚合全部来源组合 (fa, fb)。
    """
    products: dict[int, set[tuple[int, int]]] = {}
    for fa in carriers_hz:
        for fb in carriers_hz:
            if fa == fb:
                continue
            freq = 2 * fa - fb
            if scan_start_hz <= freq <= scan_end_hz:
                products.setdefault(freq, set()).add((fa, fb))
    return products


def _distance_to_guard_band(freq: int, band: GuardBand) -> int:
    """到保护频段的距离：落在带内为 0，否则为到最近边界的距离。"""
    if band.start_hz <= freq <= band.end_hz:
        return 0
    return min(abs(freq - band.start_hz), abs(freq - band.end_hz))


def nearest_hit(
    freq: int, carriers_hz: list[int], guard_bands: list[GuardBand]
) -> tuple[int, HitObject]:
    """返回 freq 到载频或保护频段的最短距离及对应命中对象。

    距离相同时按确定性规则决胜：载频优先于保护频段，其次按参考频率
    （载频值或频段起点）升序，保证结果稳定可复现。
    """
    best: tuple[int, int, int, HitObject] | None = None
    for carrier in carriers_hz:
        candidate = (
            abs(freq - carrier),
            0,
            carrier,
            HitObject(type="carrier", carrier_hz=carrier),
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    for band in guard_bands:
        candidate = (
            _distance_to_guard_band(freq, band),
            1,
            band.start_hz,
            HitObject(type="guard_band", guard_band=band),
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None  # carriers_hz 至少含一个载频
    return best[0], best[3]


def analyze(
    carriers_hz: list[int],
    scan_start_hz: int,
    scan_end_hz: int,
    guard_bands: list[GuardBand],
    guard_interval_hz: int,
) -> list[ProductOut]:
    """完整分析：枚举产物 -> 过滤范围外 -> 同频聚合 -> 计算最近命中对象。

    返回按产物频率升序排列的结果（同频已聚合，频率唯一，排序稳定确定）。
    """
    grouped = enumerate_third_order_products(carriers_hz, scan_start_hz, scan_end_hz)
    results: list[ProductOut] = []
    for freq in sorted(grouped):
        distance, hit = nearest_hit(freq, carriers_hz, guard_bands)
        sources = [SourcePair(fa_hz=fa, fb_hz=fb) for fa, fb in sorted(grouped[freq])]
        results.append(
            ProductOut(
                freq_hz=freq,
                distance_hz=distance,
                hit=hit,
                violates_guard_interval=distance < guard_interval_hz,
                sources=sources,
            )
        )
    return results


def build_risk_groups(products: list[ProductOut]) -> list[RiskGroup]:
    """由已持久化的分析结果生成风险组（二部图连通分量）。

    仅 violates_guard_interval 的违规产物参与：每个违规产物是一个风险节点，
    其全部来源组合中的载频 fa_hz/fb_hz 与该节点连边，构成二部图；每个连通
    分量即一个风险组。同频产物的多来源已聚合在同一节点，共享载频的产物沿
    链式边传递归组；非违规产物不参与，不会桥接任何载频。
    组按 (最小载频, 最小产物频率) 升序排序，group_id 从 1 连续编号。
    """
    violating = [p for p in products if p.violates_guard_interval]
    if not violating:
        return []

    # 并查集：节点为 ("product", freq_hz) 或 ("carrier", carrier_hz)，
    # 两类节点分开编号，避免产物频率与载频数值相同而混淆。
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]  # 路径压缩
            node = parent[node]
        return node

    def link(a: tuple[str, int], b: tuple[str, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for product in violating:
        pnode = ("product", product.freq_hz)
        parent.setdefault(pnode, pnode)
        for source in product.sources:
            for carrier_hz in (source.fa_hz, source.fb_hz):
                cnode = ("carrier", carrier_hz)
                parent.setdefault(cnode, cnode)
                link(pnode, cnode)

    components: dict[tuple[str, int], dict[str, set[int]]] = {}
    for node in list(parent):
        kind, value = node
        bucket = components.setdefault(find(node), {"product": set(), "carrier": set()})
        bucket[kind].add(value)

    by_freq = {p.freq_hz: p for p in violating}
    groups: list[RiskGroup] = []
    for bucket in components.values():
        members = [by_freq[f] for f in bucket["product"]]
        groups.append(
            RiskGroup(
                group_id=0,  # 占位，排序后统一编号
                carriers_hz=sorted(bucket["carrier"]),
                product_freqs_hz=sorted(bucket["product"]),
                source_pair_count=sum(len(p.sources) for p in members),
                distance_hz=min(p.distance_hz for p in members),
            )
        )
    # 各组载频集合互不相交，排序键唯一，编号稳定确定
    groups.sort(key=lambda g: (g.carriers_hz[0], g.product_freqs_hz[0]))
    for group_id, group in enumerate(groups, start=1):
        group.group_id = group_id
    return groups
