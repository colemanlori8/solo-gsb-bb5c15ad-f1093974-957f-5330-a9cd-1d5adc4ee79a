"""FastAPI 入口：射频三阶互调（2fa - fb）分析 API。"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status

from . import database
from .analysis import analyze, build_mitigation_plan, build_risk_groups
from .schemas import (
    AnalysisMeta,
    AnalysisResult,
    MitigationPlan,
    RiskGroup,
    SiteCreate,
    SiteOut,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()  # 启动时初始化数据库（幂等）
    yield


app = FastAPI(
    title="RF Intermodulation Analysis API",
    summary="射频三阶互调产物（2fa - fb）分析：站点管理、产物枚举与命中评估",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/sites",
    response_model=SiteOut,
    status_code=status.HTTP_201_CREATED,
    tags=["sites"],
    summary="创建站点（频率方案）",
    description="任一载频重复或频段非法时整批拒绝（422），不写入任何数据；名称重复返回 409。",
)
def create_site(payload: SiteCreate) -> SiteOut:
    try:
        site_id = database.insert_site(payload)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"site name already exists: {payload.name}"
        ) from exc
    site = database.fetch_site(site_id)
    assert site is not None
    return site


@app.get("/sites", response_model=list[SiteOut], tags=["sites"], summary="列出全部已保存方案")
def list_sites() -> list[SiteOut]:
    return database.list_sites()


@app.get("/sites/{site_id}", response_model=SiteOut, tags=["sites"], summary="查询单个已保存方案")
def get_site(site_id: int) -> SiteOut:
    site = database.fetch_site(site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"site {site_id} not found")
    return site


@app.post(
    "/sites/{site_id}/analysis",
    response_model=AnalysisResult,
    status_code=status.HTTP_201_CREATED,
    tags=["analysis"],
    summary="对方案执行三阶互调分析并保存结果",
)
def run_analysis(site_id: int) -> AnalysisResult:
    site = database.fetch_site(site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"site {site_id} not found")
    products = analyze(
        carriers_hz=site.carriers_hz,
        scan_start_hz=site.scan_start_hz,
        scan_end_hz=site.scan_end_hz,
        guard_bands=site.guard_bands,
        guard_interval_hz=site.guard_interval_hz,
    )
    analysis_id, created_at = database.insert_analysis(site_id, products)
    return AnalysisResult(
        analysis_id=analysis_id,
        site_id=site_id,
        created_at=created_at,
        product_count=len(products),
        products=products,
    )


@app.get(
    "/sites/{site_id}/analysis",
    response_model=AnalysisResult,
    tags=["analysis"],
    summary="查询该方案最近一次分析结果",
)
def get_latest_analysis(site_id: int) -> AnalysisResult:
    if database.fetch_site(site_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"site {site_id} not found")
    result = database.fetch_latest_analysis(site_id)
    if result is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no analysis for site {site_id} yet; POST /sites/{site_id}/analysis first",
        )
    return result


@app.get(
    "/sites/{site_id}/analyses",
    response_model=list[AnalysisMeta],
    tags=["analysis"],
    summary="列出该方案的全部分析记录",
)
def list_site_analyses(site_id: int) -> list[AnalysisMeta]:
    if database.fetch_site(site_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"site {site_id} not found")
    return database.list_analyses(site_id)


@app.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResult,
    tags=["analysis"],
    summary="按 id 查询已保存的分析结果",
)
def get_analysis(analysis_id: int) -> AnalysisResult:
    result = database.fetch_analysis(analysis_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"analysis {analysis_id} not found")
    return result


@app.get(
    "/analyses/{analysis_id}/risk-groups",
    response_model=list[RiskGroup],
    tags=["analysis"],
    summary="按已保存分析结果生成风险分组（违规产物与来源载频的二部图连通分量）",
    description=(
        "仅使用已持久化的分析结果：每个违规产物为一个风险节点，其全部来源组合中的 "
        "fa_hz/fb_hz 载频与该节点连边，按连通分量归组；非违规产物不参与。"
        "无违规产物时返回空列表。"
    ),
)
def get_risk_groups(analysis_id: int) -> list[RiskGroup]:
    result = database.fetch_analysis(analysis_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"analysis {analysis_id} not found")
    return build_risk_groups(result.products)


@app.get(
    "/analyses/{analysis_id}/mitigation-plan",
    response_model=MitigationPlan,
    tags=["analysis"],
    summary="求消除全部违规产物的精确最小停用载频集合（最小顶点覆盖）",
    description=(
        "仅使用已持久化的分析结果，不重新扫描：把每个违规产物的每个来源组合 "
        "(fa_hz, fb_hz) 视为无向边，停用集合必须覆盖全部来源边才算消除该产物，"
        "即在来源边并图上求最小顶点覆盖。存在多个最小解时按载频升序列表的字典序"
        "择优；max_disabled 限制内无解时 feasible=false，并给出仍未消除的产物频率。"
    ),
)
def get_mitigation_plan(
    analysis_id: int,
    max_disabled: int = Query(
        ..., ge=1, le=8, description="允许停用的载频数上限，取 1～8 的整数"
    ),
) -> MitigationPlan:
    result = database.fetch_analysis(analysis_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"analysis {analysis_id} not found")
    return build_mitigation_plan(result.products, analysis_id, max_disabled)
