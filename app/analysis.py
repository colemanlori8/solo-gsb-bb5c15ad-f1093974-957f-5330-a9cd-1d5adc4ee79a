"""三阶互调产物分析：枚举 2*fa - fb，聚合来源并评估最近命中对象。"""
from __future__ import annotations

from .schemas import GuardBand, HitObject, MitigationPlan, ProductOut, RiskGroup, SourcePair


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


def build_mitigation_plan(
    products: list[ProductOut], analysis_id: int, max_disabled: int
) -> MitigationPlan:
    """基于已持久化分析结果，求消除全部违规产物的精确最小停用载频集合。

    消除语义：对一个违规产物，其每个来源组合 (fa, fb) 都必须至少有一个载频被
    停用才算消除；即停用集合必须"击中"该产物的全部来源边。对全部违规产物而言，
    等价于在"全部违规产物来源边（去重、无向）的并图"上求最小顶点覆盖。

    由于 max_disabled <= 8，采用边分支有界搜索：任取一条未被覆盖的边，必须停用
    其两个端点之一，递归深度即停用数，叶子最多 2**max_disabled 个。搜索中先以
    较小端点分支，并始终保留 (集合尺寸, 升序列表) 字典序最弱的最优解，保证
    多个最小解时的决胜规则稳定确定。

    限制内不存在覆盖时 feasible=false，停用集合为空，remaining 列出全部违规
    产物频率（即不采取任何停用时仍未消除的产物）。
    """
    violating = [p for p in products if p.violates_guard_interval]
    remaining = sorted(p.freq_hz for p in violating)

    # 来源边：同频产物的多来源及不同产物的重复来源只保留一条（无向去重）
    edges: set[frozenset[int]] = set()
    for product in violating:
        for source in product.sources:
            edges.add(frozenset((source.fa_hz, source.fb_hz)))

    best: tuple[int, ...] | None = None  # 当前最优解（升序元组），None 表示尚未找到

    def search(remaining_edges: set[frozenset[int]], used: tuple[int, ...]) -> None:
        nonlocal best
        # 不劣于已有最优解即剪枝：只接受严格更小、或同尺寸字典序更弱的解
        if best is not None and (len(used), used) >= (len(best), best):
            return
        if len(used) > max_disabled:
            return
        if not remaining_edges:
            best = used
            return
        # 已用满预算但仍有未覆盖边；分支只会增加停用数
        if len(used) == max_disabled:
            return

        edge = min(remaining_edges, key=lambda e: (min(e), max(e)))
        u, v = sorted(edge)  # 先尝试较小端点，使字典序弱的解更早出现
        for endpoint in (u, v):
            # 未覆盖边的两个端点都不在 used 中，故每个分支恰好新增一个停用载频
            search(
                {e for e in remaining_edges if endpoint not in e},
                tuple(sorted(used + (endpoint,))),
            )

    # 边集为空（无违规产物）时空集即尺寸 0 的覆盖：search 立即以 () 命中
    search(edges, ())

    feasible = best is not None
    return MitigationPlan(
        analysis_id=analysis_id,
        max_disabled=max_disabled,
        feasible=feasible,
        disabled_carriers_hz=list(best) if feasible else [],
        remaining_product_freqs_hz=[] if feasible else remaining,
    )
