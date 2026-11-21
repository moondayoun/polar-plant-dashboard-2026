import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


@st.cache_data
def load_csv(file):
    """UTF-8-SIG로 읽고 실패 시 CP949로 다시 읽기"""
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    """학교마다 다른 열 이름을 표준 이름으로 통일"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df


def safe_mean(series):
    """NaN 제외 평균이 유효한지 확인"""
    s = series.dropna()
    if len(s) < 2:
        return None
    return s.mean()


def safe_corr(x, y):
    """상관계수 계산 (NaN 제거, 2개 미만이면 None)"""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return None, None
    r = df["x"].corr(df["y"])
    return r, df


def add_trendline(fig, x, y, color="black"):
    """np.polyfit으로 추세선 추가"""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    x_line = np.array([df["x"].min(), df["x"].max()])
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color=color, dash="dash"), name="추세선"))


def scatter_with_trend(x, y, xlabel, ylabel, color="#2E86AB"):
    """산점도 + 추세선 Figure 반환"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=color, size=8), name="데이터"))
    add_trendline(fig, x, y)
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def bar_chart_single(col, df, title, color="#2E86AB"):
    """개체별 막대 그래프 (개체번호 유지)"""
    df2 = df[[col, "개체번호"]].dropna().sort_values(col, ascending=False)
    if len(df2) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df2["개체번호"].astype(str), y=df2[col], marker_color=color, name=title))
    fig.update_layout(
        xaxis_title="개체번호",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
        showlegend=False,
    )
    return fig


def box_plot(col, df):
    """생중량 분포 박스플롯"""
    s = df[col].dropna()
    if len(s) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Box(y=s, name=col, fillcolor="#2E86AB"))
    fig.update_layout(
        yaxis_title=col,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
        showlegend=False,
    )
    return fig


def time_series(df, col, title):
    """날짜별 일별 평균 꺾은선 그래프"""
    if col not in df.columns or "time" not in df.columns:
        return None
    d = df[["time", col]].dropna()
    if len(d) == 0:
        return None
    daily = d.set_index("time").resample("D").mean().dropna()
    if len(daily) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", line=dict(color="#2E86AB"), name=title))
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def csv_download_button(df, label, filename):
    """한글 깨짐 방지 CSV 다운로드 버튼"""
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def extract_school_name(filename):
    """파일명에서 학교명 추출 (예: OO고_환경데이터.csv → OO고)"""
    if not filename:
        return "우리 학교"
    base = filename.split("/")[-1].split("\\")[-1]
    m = re.match(r"^(.+?)\s*[_-]", base)
    if m:
        return m.group(1)
    return base.split("_")[0].split("-")[0]


# ──────────────────────────────────────────────
# Streamlit 앱 시작
# ──────────────────────────────────────────────

st.set_page_config(page_title="극지식물 생육 분석 대시보드", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# 사이드바: 파일 업로드
st.sidebar.header("📂 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.header("📏 생육 측정 항목")
growth_metrics = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_metrics = []
for m in growth_metrics:
    if st.sidebar.checkbox(m, value=True, key=f"chk_{m}"):
        selected_metrics.append(m)

# 데이터 로딩
with st.spinner("데이터를 읽어오는 중..."):
    env_df = load_csv(env_file)
    growth_df = load_csv(growth_file)

env_df = std_cols(env_df) if not env_df.empty else pd.DataFrame()
growth_df = std_cols(growth_df) if not growth_df.empty else pd.DataFrame()

school_name = extract_school_name(env_file.name if env_file else growth_file.name if growth_file else "")

# ──────────────────────────────────────────────
# 탭 1: 실험 개요
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header(f"📖 {school_name} 극지식물 실험 개요")

        col_bg, col_purpose = st.columns(2)
        with col_bg:
            st.subheader("연구 배경")
            st.markdown(
                """
- 극지식물은 저온·건조 등 척박한 환경에서 자라는 식물로, 환경 조건에 따라 생육 차이가 크게 나타납니다.
- 본 실험은 우리 학교 환경에서 극지식물 개체별로 생육 차이가 얼마나 나는지 관찰하고, 환경 요인과의 관계를 알아보기 위해 진행되었습니다.
"""
            )
        with col_purpose:
            st.subheader("연구 목적")
            st.markdown(
                """
1. 개체별 생중량 분포를 확인하고 개체 간 차이를 정량적으로 비교합니다.
2. 잎 수, 지상부 길이 등 측정 항목과 생중량의 관계를 상관분석으로 살펴봅니다.
3. 생중량 상위 25%와 하위 25% 개체를 비교해 생육 차이의 특징을 분석합니다.
"""
            )

        # 데이터 요약
        st.subheader("📋 데이터 요약")

        env_rows = []
        growth_rows = []

        if not env_df.empty:
            time_col = "time" if "time" in env_df.columns else None
            if time_col is not None:
                env_rows.append(["측정 기간 (환경)", f"{env_df[time_col].min()} ~ {env_df[time_col].max()}"])
            env_row_count = len(env_df)
            env_rows.append(["환경 데이터 행 수", f"{env_row_count}행"])
            numeric_env = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            env_rows.append(["환경 측정 항목", f"{len(numeric_env)}개 ({', '.join(numeric_env)})"])

        if not growth_df.empty:
            growth_rows.append(["총 개체 수", f"{growth_df['개체번호'].nunique()}개체"])
            growth_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"] if c in growth_df.columns]
            growth_rows.append(["생육 측정 항목", f"{len(growth_cols)}개 ({', '.join(growth_cols)})"])

        if env_rows or growth_rows:
            summary_tbl = []
            if env_rows:
                for r in env_rows:
                    summary_tbl.append(r)
            if growth_rows:
                for r in growth_rows:
                    summary_tbl.append(r)
            st.dataframe(pd.DataFrame(summary_tbl, columns=["구분", "내용"]), width="stretch")
        else:
            st.info("📁 CSV 파일을 업로드하면 데이터 요약이 표시됩니다.")

        # 주요 지표 카드 4개
        st.subheader("📊 주요 지표")

        c1, c2, c3, c4 = st.columns(4)

        total_individuals = growth_df["개체번호"].nunique() if not growth_df.empty and "개체번호" in growth_df.columns else 0
        c1.metric("총 개체 수", f"{total_individuals}개체")

        avg_temp = safe_mean(env_df["temperature"]) if not env_df.empty and "temperature" in env_df.columns else None
        c2.metric("평균 온도", f"{avg_temp:.1f}°C" if avg_temp is not None else "데이터 없음")

        avg_hum = safe_mean(env_df["humidity"]) if not env_df.empty and "humidity" in env_df.columns else None
        c3.metric("평균 습도", f"{avg_hum:.1f}%" if avg_hum is not None else "데이터 없음")

        avg_biomass = safe_mean(growth_df["생중량"]) if not growth_df.empty and "생중량" in growth_df.columns else None
        c4.metric("평균 생중량", f"{avg_biomass:.2f}g" if avg_biomass is not None else "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ──────────────────────────────────────────────
# 탭 2: 환경 데이터
# ──────────────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 데이터 CSV를 업로드하면 환경 데이터 탭이 표시됩니다.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터 분석")

            # 환경 요약 표
            st.subheader("📈 환경 요약 (평균·최소·최대)")
            env_stats_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_stats_cols:
                stats_data = []
                for col in env_stats_cols:
                    s = env_df[col].dropna()
                    if len(s) > 0:
                        stats_data.append([col, f"{s.mean():.2f}", f"{s.min():.2f}", f"{s.max():.2f}"])
                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data, columns=["항목", "평균", "최소", "최대"]), width="stretch")
                else:
                    st.info("환경 데이터의 숫자 값을 계산할 수 없습니다.")
            else:
                st.info("환경 데이터에 temperature, humidity, ph, ec 열이 없어 환경 요약을 표시할 수 없습니다.")

            # 날짜별 시계열
            st.subheader("📅 날짜별 환경 변화 (일별 평균)")

            ts_cols = st.columns(2)
            with ts_cols[0]:
                temp_fig = time_series(env_df, "temperature", "온도 변화")
                if temp_fig:
                    st.plotly_chart(temp_fig, width="stretch")
                else:
                    st.info("온도 데이터가 없어 온도 변화 그래프를 표시할 수 없습니다.")

            with ts_cols[1]:
                hum_fig = time_series(env_df, "humidity", "습도 변화")
                if hum_fig:
                    st.plotly_chart(hum_fig, width="stretch")
                else:
                    st.info("습도 데이터가 없어 습도 변화 그래프를 표시할 수 없습니다.")

            ph_ec_cols = st.columns(2)
            with ph_ec_cols[0]:
                if "ph" in env_df.columns:
                    ph_fig = time_series(env_df, "ph", "pH 변화")
                    if ph_fig:
                        st.plotly_chart(ph_fig, width="stretch")
                    else:
                        st.info("pH 데이터로 일별 그래프를 만들 수 없습니다.")
                else:
                    st.info("pH 열이 없어 pH 변화 그래프를 표시할 수 없습니다.")

            with ph_ec_cols[1]:
                if "ec" in env_df.columns:
                    ec_fig = time_series(env_df, "ec", "EC 변화")
                    if ec_fig:
                        st.plotly_chart(ec_fig, width="stretch")
                    else:
                        st.info("EC 데이터로 일별 그래프를 만들 수 없습니다.")
                else:
                    st.info("EC 열이 없어 EC 변화 그래프를 표시할 수 없습니다.")

            # 원본 테이블 + 다운로드
            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch")
                    csv_download_button(env_df, "환경 데이터 CSV 다운로드", f"{school_name}_환경데이터.csv")
                else:
                    st.info(" 표시할 원본 데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ──────────────────────────────────────────────
# 탭 3: 생육 결과
# ──────────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("📊 생육 결과 데이터 CSV를 업로드하면 생육 결과 탭이 표시됩니다.")
        else:
            st.header(f"📊 {school_name} 생육 결과 분석")

            # 핵심 결과 카드
            st.subheader("🥇 핵심 결과")

            k1, k2, k3 = st.columns(3)
            avg_biomass = safe_mean(growth_df["생중량"]) if "생중량" in growth_df.columns else None
            k1.metric("평균 생중량", f"{avg_biomass:.2f}g" if avg_biomass is not None else "데이터 없음")

            if "생중량" in growth_df.columns and growth_df["생중량"].notna().sum() > 0:
                max_idx = growth_df["생중량"].idxmax()
                min_idx = growth_df["생중량"].idxmin()
                max_val = growth_df.loc[max_idx, "생중량"]
                min_val = growth_df.loc[min_idx, "생중량"]
                max_id = growth_df.loc[max_idx, "개체번호"]
                min_id = growth_df.loc[min_idx, "개체번호"]
                k2.metric("최대 생중량", f"{max_val:.2f}g ({max_id})")
                k3.metric("최소 생중량", f"{min_val:.2f}g ({min_id})")
            else:
                k2.metric("최대 생중량", "데이터 없음")
                k3.metric("최소 생중량", "데이터 없음")

            # 개체별 생육 비교 (막대 그래프)
            st.subheader("📊 개체별 생육 비교")

            if "생중량" in growth_df.columns:
                bio_bar = bar_chart_single("생중량", growth_df, "평균 생중량", "#2E86AB")
                if bio_bar:
                    st.plotly_chart(bio_bar, width="stretch")
                else:
                    st.info("생중량 막대 그래프를 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 생중량 막대 그래프를 표시할 수 없습니다.")

            metric_cols = st.columns(3)
            other_metrics = [m for m in ["잎 수", "지상부 길이", "지하부 길이"] if m in growth_df.columns]
            if other_metrics:
                idx = 0
                for mc in metric_cols:
                    if idx < len(other_metrics):
                        m = other_metrics[idx]
                        bar_fig = bar_chart_single(m, growth_df, f"평균 {m}", "#A23B72")
                        if bar_fig:
                            mc.plotly_chart(bar_fig, width="stretch")
                        else:
                            mc.info(f"{m} 그래프를 표시할 수 없습니다.")
                        idx += 1
            else:
                st.info("잎 수, 지상부 길이, 지하부 길이 열이 없어 해당 그래프를 표시할 수 없습니다.")

            # 생중량 분포
            st.subheader("📦 생중량 분포")
            if "생중량" in growth_df.columns:
                box_fig = box_plot("생중량", growth_df)
                if box_fig:
                    st.plotly_chart(box_fig, width="stretch")
                else:
                    st.info("생중량 분포 그래프를 표시할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 분포 그래프를 표시할 수 없습니다.")

            # 생중량 통계: 평균·표준편차·CV
            if "생중량" in growth_df.columns:
                s = growth_df["생중량"].dropna()
                if len(s) >= 2:
                    mean_val = s.mean()
                    std_val = s.std()
                    cv_val = std_val / mean_val if mean_val != 0 else None
                    stat_cols = st.columns(3)
                    stat_cols[0].metric("평균 생중량", f"{mean_val:.2f}g")
                    stat_cols[1].metric("표준편차", f"{std_val:.2f}g")
                    stat_cols[2].metric("변동계수(CV)", f"{cv_val:.3f}" if cv_val is not None else "N/A")
                    st.caption("※ 표준편차: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 값 / CV(변동계수): 표준편차를 평균으로 나눈 값, 단위가 다른 집단 간 비교에 사용")

            # 상관관계 분석
            st.subheader("🔗 상관관계 분석")

            if "잎 수" in growth_df.columns and "생중량" in growth_df.columns:
                leaf_biomass = growth_df[["잎 수", "생중량"]].dropna()
                if len(leaf_biomass) >= 2:
                    r1, df1 = safe_corr(leaf_biomass["잎 수"], leaf_biomass["생중량"])
                    c_corr1, fig_corr1 = st.columns([1, 3])
                    with c_corr1:
                        st.metric("잎 수 vs 생중량 상관계수(r)", f"{r1:.3f}" if r1 is not None else "N/A")
                        st.caption("※ 상관계수(r): -1~1 사이 값, 1에 가까울수록 강한 양의 상관 / 0에 가까우면 상관 약함")
                    with fig_corr1:
                        fig_leaf = scatter_with_trend(df1["잎 수"], df1["생중량"], "잎 수", "생중량(g)", "#2E86AB")
                        st.plotly_chart(fig_leaf, width="stretch")
                else:
                    st.info("잎 수와 생중량 데이터가 충분히 없어 상관분석을 할 수 없습니다.")

            if "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
                len_biomass = growth_df[["지상부 길이", "생중량"]].dropna()
                if len(len_biomass) >= 2:
                    r2, df2 = safe_corr(len_biomass["지상부 길이"], len_biomass["생중량"])
                    c_corr2, fig_corr2 = st.columns([1, 3])
                    with c_corr2:
                        st.metric("지상부 길이 vs 생중량 상관계수(r)", f"{r2:.3f}" if r2 is not None else "N/A")
                    with fig_corr2:
                        fig_len = scatter_with_trend(df2["지상부 길이"], df2["생중량"], "지상부 길이(mm)", "생중량(g)", "#A23B72")
                        st.plotly_chart(fig_len, width="stretch")
                else:
                    st.info("지상부 길이와 생중량 데이터가 충분히 없어 상관분석을 할 수 없습니다.")

            if "잎 수" not in growth_df.columns and "지상부 길이" not in growth_df.columns:
                st.info("잎 수, 지상부 길이 열이 없어 상관관계 분석을 표시할 수 없습니다.")

            # 생중량 상위 25% vs 하위 25% 비교
            if "생중량" in growth_df.columns:
                s = growth_df["생중량"].dropna()
                if len(s) >= 4:
                    q75 = s.quantile(0.75)
                    q25 = s.quantile(0.25)
                    top = growth_df[growth_df["생중량"] >= q75]
                    bottom = growth_df[growth_df["생중량"] <= q25]

                    st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")

                    comp_cols = st.columns(2)
                    with comp_cols[0]:
                        st.markdown("#### 🔺 상위 25% (생중량 ≥ 75퍼센타일)")
                        st.metric("개체 수", f"{len(top)}개체")
                        if "잎 수" in top.columns:
                            st.metric("평균 잎 수", f"{top['잎 수'].mean():.1f}장")
                        if "지상부 길이" in top.columns:
                            st.metric("평균 지상부 길이", f"{top['지상부 길이'].mean():.1f}mm")
                        if "지하부 길이" in top.columns:
                            st.metric("평균 지하부 길이", f"{top['지하부 길이'].mean():.1f}mm")

                    with comp_cols[1]:
                        st.markdown("#### 🔻 하위 25% (생중량 ≤ 25퍼센타일)")
                        st.metric("개체 수", f"{len(bottom)}개체")
                        if "잎 수" in bottom.columns:
                            st.metric("평균 잎 수", f"{bottom['잎 수'].mean():.1f}장")
                        if "지상부 길이" in bottom.columns:
                            st.metric("평균 지상부 길이", f"{bottom['지상부 길이'].mean():.1f}mm")
                        if "지하부 길이" in bottom.columns:
                            st.metric("평균 지하부 길이", f"{bottom['지하부 길이'].mean():.1f}mm")

            # 원본 테이블 + 다운로드
            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch")
                    csv_download_button(growth_df, "생육 결과 CSV 다운로드", f"{school_name}_생육결과데이터.csv")
                else:
                    st.info("표시할 원본 데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
