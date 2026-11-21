import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ──────────────────────────────────────────────
# 데이터 로딩 & 표준화 함수
# ──────────────────────────────────────────────

@st.cache_data
def load_csv(file):
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
    s = series.dropna()
    if len(s) < 2:
        return None
    return s.mean()


def safe_corr(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return None, None
    r = df["x"].corr(df["y"])
    return r, df


def add_trendline(fig, x, y, color="black"):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    x_line = np.array([df["x"].min(), df["x"].max()])
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                             line=dict(color=color, dash="dash"), name="추세선"))


def scatter_with_trend(x, y, xlabel, ylabel, color="#2E86AB"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(color=color, size=8), name="데이터"))
    add_trendline(fig, x, y)
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def bar_chart_single(col, df, title, color="#2E86AB"):
    df2 = df[[col, "개체번호"]].dropna().sort_values(col, ascending=False)
    if len(df2) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df2["개체번호"].astype(str), y=df2[col],
                         marker_color=color, name=title))
    fig.update_layout(
        xaxis_title="개체번호",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
        showlegend=False,
    )
    return fig


def box_plot(col, df):
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
    if col not in df.columns or "time" not in df.columns:
        return None
    d = df[["time", col]].dropna()
    if len(d) == 0:
        return None
    daily = d.set_index("time").resample("D").mean().dropna()
    if len(daily) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines",
                             line=dict(color="#2E86AB"), name=title))
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def csv_download_button(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def extract_school_name(filename):
    if not filename:
        return "우리 학교"
    base = filename.split("/")[-1].split("\\")[-1]
    m = re.match(r"^(.+?)\s*[_-]", base)
    if m:
        return m.group(1)
    return base.split("_")[0].split("-")[0]


def group_summary_card(col1, col2, label1, label2, metric_label, fmt=".2f", unit=""):
    """두 그룹 비교 지표 카드"""
    c1, c2 = st.columns(2)
    v1 = safe_mean(col1)
    v2 = safe_mean(col2)
    c1.metric(f"{label1} {metric_label}", f"{v1:{fmt}}{unit}" if v1 is not None else "데이터 없음")
    c2.metric(f"{label2} {metric_label}", f"{v2:{fmt}}{unit}" if v2 is not None else "데이터 없음")


# ──────────────────────────────────────────────
# Streamlit 앱
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


# ── 사이드바: CSV 업로드 4개 ──────────────────
st.sidebar.header("📂 데이터 업로드")

st.sidebar.subheader("환경 데이터")
env_treat_file = st.sidebar.file_uploader("🌿 환경 CSV - 처리구", type="csv")
env_ctrl_file = st.sidebar.file_uploader("🌿 환경 CSV - 미처리구", type="csv")

st.sidebar.subheader("생육 결과 데이터")
growth_treat_file = st.sidebar.file_uploader("📊 생육 CSV - 처리구", type="csv")
growth_ctrl_file = st.sidebar.file_uploader("📊 생육 CSV - 미처리구", type="csv")

st.sidebar.markdown("---")
st.sidebar.header("📏 생육 측정 항목")
growth_metrics = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_metrics = []
for m in growth_metrics:
    if st.sidebar.checkbox(m, value=True, key=f"chk_{m}"):
        selected_metrics.append(m)


# ── 데이터 로딩 ─────────────────────────────
with st.spinner("데이터를 읽어오는 중..."):
    env_treat_df = load_csv(env_treat_file)
    env_ctrl_df = load_csv(env_ctrl_file)
    growth_treat_df = load_csv(growth_treat_file)
    growth_ctrl_df = load_csv(growth_ctrl_file)

env_treat_df = std_cols(env_treat_df) if not env_treat_df.empty else pd.DataFrame()
env_ctrl_df = std_cols(env_ctrl_df) if not env_ctrl_df.empty else pd.DataFrame()
growth_treat_df = std_cols(growth_treat_df) if not growth_treat_df.empty else pd.DataFrame()
growth_ctrl_df = std_cols(growth_ctrl_df) if not growth_ctrl_df.empty else pd.DataFrame()

school_name = "우리 학교"
for f in [env_treat_file, env_ctrl_file, growth_treat_file, growth_ctrl_file]:
    if f:
        school_name = extract_school_name(f.name)
        break


# ──────────────────────────────────────────────
# 탭 1: 실험 개요
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header(f"📖 {school_name} 극지식물 실험 개요 - 처리구 vs 미처리구")

        col_bg, col_purpose = st.columns(2)
        with col_bg:
            st.subheader("연구 배경")
            st.markdown(
                """
- 극지식물은 저온·건조 등 척박한 환경에서 자라는 식물입니다.
- 본 실험은 **미생물 처리 여부에 따라** 극지식물 개체별 생육 차이가 어떻게 나타나는지 비교합니다.
- 처리구(미생물 처리)와 미처리구(처리 안 함)의 환경 조건과 생육 결과를 나란히 분석합니다.
"""
            )
        with col_purpose:
            st.subheader("연구 목적")
            st.markdown(
                """
1. 처리구와 미처리구의 생중량 분포·평균을 비교하여 미생물 처리 효과를 확인합니다.
2. 잎 수, 지상부 길이 등 측정 항목과 생중량의 관계를 각 그룹별로 상관분석합니다.
3. 생중량 상위 25% / 하위 25% 개체를 각 그룹에서 비교해 생육 차이의 특징을 살펴봅니다.
"""
            )

        # 데이터 요약 표
        st.subheader("📋 데이터 요약")

        summary_rows = []

        # 처리구 환경
        if not env_treat_df.empty:
            tcol = "time" if "time" in env_treat_df.columns else None
            if tcol is not None:
                summary_rows.append(["처리구 환경 - 측정 기간",
                                     f"{env_treat_df[tcol].min()} ~ {env_treat_df[tcol].max()}"])
            summary_rows.append(["처리구 환경 - 데이터 행 수", f"{len(env_treat_df)}행"])
            ncols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_treat_df.columns]
            summary_rows.append(["처리구 환경 - 측정 항목", f"{len(ncols)}개 ({', '.join(ncols)})"])
        else:
            summary_rows.append(["처리구 환경", "데이터 없음"])

        # 미처리구 환경
        if not env_ctrl_df.empty:
            tcol = "time" if "time" in env_ctrl_df.columns else None
            if tcol is not None:
                summary_rows.append(["미처리구 환경 - 측정 기간",
                                     f"{env_ctrl_df[tcol].min()} ~ {env_ctrl_df[tcol].max()}"])
            summary_rows.append(["미처리구 환경 - 데이터 행 수", f"{len(env_ctrl_df)}행"])
            ncols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_ctrl_df.columns]
            summary_rows.append(["미처리구 환경 - 측정 항목", f"{len(ncols)}개 ({', '.join(ncols)})"])
        else:
            summary_rows.append(["미처리구 환경", "데이터 없음"])

        # 처리구 생육
        if not growth_treat_df.empty:
            summary_rows.append(["처리구 - 총 개체 수",
                                 f"{growth_treat_df['개체번호'].nunique()}개체"])
            gcols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                                  "지상부 질량", "지하부 질량"] if c in growth_treat_df.columns]
            summary_rows.append(["처리구 - 생육 측정 항목",
                                 f"{len(gcols)}개 ({', '.join(gcols)})"])
        else:
            summary_rows.append(["처리구 생육", "데이터 없음"])

        # 미처리구 생육
        if not growth_ctrl_df.empty:
            summary_rows.append(["미처리구 - 총 개체 수",
                                 f"{growth_ctrl_df['개체번호'].nunique()}개체"])
            gcols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량",
                                  "지상부 질량", "지하부 질량"] if c in growth_ctrl_df.columns]
            summary_rows.append(["미처리구 - 생육 측정 항목",
                                 f"{len(gcols)}개 ({', '.join(gcols)})"])
        else:
            summary_rows.append(["미처리구 생육", "데이터 없음"])

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows, columns=["구분", "내용"]), width="stretch")
        else:
            st.info("📁 CSV 파일을 업로드하면 데이터 요약이 표시됩니다.")

        # 주요 지표 카드 (두 그룹 비교)
        st.subheader("📊 주요 지표 비교 (처리구 vs 미처리구)")

        # 총 개체 수
        t_ind = growth_treat_df["개체번호"].nunique() if not growth_treat_df.empty and "개체번호" in growth_treat_df.columns else 0
        c_ind = growth_ctrl_df["개체번호"].nunique() if not growth_ctrl_df.empty and "개체번호" in growth_ctrl_df.columns else 0
        c1, c2 = st.columns(2)
        c1.metric("처리구 총 개체 수", f"{t_ind}개체")
        c2.metric("미처리구 총 개체 수", f"{c_ind}개체")

        # 평균 온도
        t_temp = safe_mean(env_treat_df["temperature"]) if not env_treat_df.empty and "temperature" in env_treat_df.columns else None
        c_temp = safe_mean(env_ctrl_df["temperature"]) if not env_ctrl_df.empty and "temperature" in env_ctrl_df.columns else None
        c1, c2 = st.columns(2)
        c1.metric("처리구 평균 온도", f"{t_temp:.1f}°C" if t_temp is not None else "데이터 없음")
        c2.metric("미처리구 평균 온도", f"{c_temp:.1f}°C" if c_temp is not None else "데이터 없음")

        # 평균 습도
        t_hum = safe_mean(env_treat_df["humidity"]) if not env_treat_df.empty and "humidity" in env_treat_df.columns else None
        c_hum = safe_mean(env_ctrl_df["humidity"]) if not env_ctrl_df.empty and "humidity" in env_ctrl_df.columns else None
        c1, c2 = st.columns(2)
        c1.metric("처리구 평균 습도", f"{t_hum:.1f}%" if t_hum is not None else "데이터 없음")
        c2.metric("미처리구 평균 습도", f"{c_hum:.1f}%" if c_hum is not None else "데이터 없음")

        # 평균 생중량
        t_bio = safe_mean(growth_treat_df["생중량"]) if not growth_treat_df.empty and "생중량" in growth_treat_df.columns else None
        c_bio = safe_mean(growth_ctrl_df["생중량"]) if not growth_ctrl_df.empty and "생중량" in growth_ctrl_df.columns else None
        c1, c2 = st.columns(2)
        c1.metric("처리구 평균 생중량", f"{t_bio:.2f}g" if t_bio is not None else "데이터 없음")
        c2.metric("미처리구 평균 생중량", f"{c_bio:.2f}g" if c_bio is not None else "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ──────────────────────────────────────────────
# 탭 2: 환경 데이터
# ──────────────────────────────────────────────
with tab2:
    try:
        has_env = not env_treat_df.empty or not env_ctrl_df.empty
        if not has_env:
            st.info("🌡️ 환경 데이터 CSV를 업로드하면 환경 데이터 탭이 표시됩니다.")
        else:
            st.header(f"🌡️ {school_name} 환경 데이터 - 처리구 vs 미처리구")

            # ── 처리구 환경 요약 ──
            st.subheader("🌿 처리구 환경 요약 (평균·최소·최대)")
            env_cols_common = [c for c in ["temperature", "humidity", "ph", "ec"]
                               if c in env_treat_df.columns]
            if env_cols_common:
                rows_t = []
                for col in env_cols_common:
                    s = env_treat_df[col].dropna()
                    if len(s) > 0:
                        rows_t.append([col, f"{s.mean():.2f}", f"{s.min():.2f}", f"{s.max():.2f}"])
                if rows_t:
                    st.dataframe(pd.DataFrame(rows_t, columns=["항목", "평균", "최소", "최대"]), width="stretch")
                else:
                    st.info("처리구 환경 데이터의 숫자 값을 계산할 수 없습니다.")
            else:
                st.info("처리구 환경 데이터에 measurement 열이 없습니다.")

            # ── 미처리구 환경 요약 ──
            st.subheader("🌿 미처리구 환경 요약 (평균·최소·최대)")
            env_cols_ctrl = [c for c in ["temperature", "humidity", "ph", "ec"]
                             if c in env_ctrl_df.columns]
            if env_cols_ctrl:
                rows_c = []
                for col in env_cols_ctrl:
                    s = env_ctrl_df[col].dropna()
                    if len(s) > 0:
                        rows_c.append([col, f"{s.mean():.2f}", f"{s.min():.2f}", f"{s.max():.2f}"])
                if rows_c:
                    st.dataframe(pd.DataFrame(rows_c, columns=["항목", "평균", "최소", "최대"]), width="stretch")
                else:
                    st.info("미처리구 환경 데이터의 숫자 값을 계산할 수 없습니다.")
            else:
                st.info("미처리구 환경 데이터에 measurement 열이 없습니다.")

            # ── 온도 시계열 비교 ──
            st.subheader("📅 날짜별 온도 변화 비교 (일별 평균)")
            temp_t_fig = time_series(env_treat_df, "temperature", "처리구 온도") if "temperature" in env_treat_df.columns else None
            temp_c_fig = time_series(env_ctrl_df, "temperature", "미처리구 온도") if "temperature" in env_ctrl_df.columns else None
            if temp_t_fig or temp_c_fig:
                comp_fig = go.Figure()
                if temp_t_fig:
                    temp_t_data = env_treat_df[["time", "temperature"]].dropna()
                    daily_t = temp_t_data.set_index("time").resample("D").mean().dropna()
                    if len(daily_t) > 0:
                        comp_fig.add_trace(go.Scatter(x=daily_t.index, y=daily_t["temperature"],
                                                      mode="lines", name="처리구 온도",
                                                      line=dict(color="#2E86AB")))
                if temp_c_fig:
                    temp_c_data = env_ctrl_df[["time", "temperature"]].dropna()
                    daily_c = temp_c_data.set_index("time").resample("D").mean().dropna()
                    if len(daily_c) > 0:
                        comp_fig.add_trace(go.Scatter(x=daily_c.index, y=daily_c["temperature"],
                                                      mode="lines", name="미처리구 온도",
                                                      line=dict(color="#E94F37")))
                comp_fig.update_layout(
                    xaxis_title="날짜",
                    yaxis_title="온도 (°C)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(comp_fig, width="stretch")
            else:
                st.info("온도 데이터가 없어 온도 비교 그래프를 표시할 수 없습니다.")

            # ── 습도 시계열 비교 ──
            st.subheader("📅 날짜별 습도 변화 비교 (일별 평균)")
            hum_t_fig = time_series(env_treat_df, "humidity", "처리구 습도") if "humidity" in env_treat_df.columns else None
            hum_c_fig = time_series(env_ctrl_df, "humidity", "미처리구 습도") if "humidity" in env_ctrl_df.columns else None
            if hum_t_fig or hum_c_fig:
                comp_fig2 = go.Figure()
                if hum_t_fig:
                    hum_t_data = env_treat_df[["time", "humidity"]].dropna()
                    daily_ht = hum_t_data.set_index("time").resample("D").mean().dropna()
                    if len(daily_ht) > 0:
                        comp_fig2.add_trace(go.Scatter(x=daily_ht.index, y=daily_ht["humidity"],
                                                       mode="lines", name="처리구 습도",
                                                       line=dict(color="#2E86AB")))
                if hum_c_fig:
                    hum_c_data = env_ctrl_df[["time", "humidity"]].dropna()
                    daily_hc = hum_c_data.set_index("time").resample("D").mean().dropna()
                    if len(daily_hc) > 0:
                        comp_fig2.add_trace(go.Scatter(x=daily_hc.index, y=daily_hc["humidity"],
                                                       mode="lines", name="미처리구 습도",
                                                       line=dict(color="#E94F37")))
                comp_fig2.update_layout(
                    xaxis_title="날짜",
                    yaxis_title="습도 (%)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(comp_fig2, width="stretch")
            else:
                st.info("습도 데이터가 없어 습도 비교 그래프를 표시할 수 없습니다.")

            # ── pH·EC 시계열 (있는 경우만) ──
            ph_ec_available = ["ph", "ec"]
            ph_ec_cols = st.columns(2)
            for idx, col in enumerate(ph_ec_available):
                with ph_ec_cols[idx]:
                    t_has = col in env_treat_df.columns
                    c_has = col in env_ctrl_df.columns
                    if t_has or c_has:
                        comp_fig3 = go.Figure()
                        if t_has:
                            d = env_treat_df[["time", col]].dropna()
                            daily = d.set_index("time").resample("D").mean().dropna()
                            if len(daily) > 0:
                                comp_fig3.add_trace(go.Scatter(x=daily.index, y=daily[col],
                                                             mode="lines", name=f"처리구 {col.upper()}",
                                                             line=dict(color="#2E86AB")))
                        if c_has:
                            d = env_ctrl_df[["time", col]].dropna()
                            daily = d.set_index("time").resample("D").mean().dropna()
                            if len(daily) > 0:
                                comp_fig3.add_trace(go.Scatter(x=daily.index, y=daily[col],
                                                             mode="lines", name=f"미처리구 {col.upper()}",
                                                             line=dict(color="#E94F37")))
                        ylabel = "pH" if col == "ph" else "EC (mS/cm)"
                        comp_fig3.update_layout(
                            xaxis_title="날짜",
                            yaxis_title=ylabel,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=40, r=20, t=30, b=40),
                        )
                        st.plotly_chart(comp_fig3, width="stretch")
                    else:
                        st.info(f"{col.upper()} 열이 없어 그래프를 표시할 수 없습니다.")

            # ── 원본 테이블 + 다운로드 ──
            with st.expander("📄 환경 데이터 원본 + CSV 다운로드"):
                env_download_cols = st.columns(2)
                with env_download_cols[0]:
                    if not env_treat_df.empty:
                        st.markdown("#### 🌿 처리구 환경 데이터")
                        st.dataframe(env_treat_df, width="stretch")
                        csv_download_button(env_treat_df, "처리구 환경 CSV 다운로드",
                                            f"{school_name}_처리구_환경데이터.csv")
                    else:
                        st.info("처리구 환경 데이터가 없습니다.")
                with env_download_cols[1]:
                    if not env_ctrl_df.empty:
                        st.markdown("#### 🌿 미처리구 환경 데이터")
                        st.dataframe(env_ctrl_df, width="stretch")
                        csv_download_button(env_ctrl_df, "미처리구 환경 CSV 다운로드",
                                            f"{school_name}_미처리구_환경데이터.csv")
                    else:
                        st.info("미처리구 환경 데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ──────────────────────────────────────────────
# 탭 3: 생육 결과
# ──────────────────────────────────────────────
with tab3:
    try:
        has_growth = not growth_treat_df.empty or not growth_ctrl_df.empty
        if not has_growth:
            st.info("📊 생육 결과 데이터 CSV를 업로드하면 생육 결과 탭이 표시됩니다.")
        else:
            st.header(f"📊 {school_name} 생육 결과 - 처리구 vs 미처리구")

            treat_has_bio = "생중량" in growth_treat_df.columns if not growth_treat_df.empty else False
            ctrl_has_bio = "생중량" in growth_ctrl_df.columns if not growth_ctrl_df.empty else False

            # ── 🥇 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과: 처리구 vs 미처리구")

            if treat_has_bio:
                t_avg = safe_mean(growth_treat_df["생중량"])
                t_max_idx = growth_treat_df["생중량"].idxmax() if growth_treat_df["생중량"].notna().sum() > 0 else None
                t_min_idx = growth_treat_df["생중량"].idxmin() if growth_treat_df["생중량"].notna().sum() > 0 else None
            else:
                t_avg = None
            if ctrl_has_bio:
                c_avg = safe_mean(growth_ctrl_df["생중량"])
                c_max_idx = growth_ctrl_df["생중량"].idxmax() if growth_ctrl_df["생중량"].notna().sum() > 0 else None
                c_min_idx = growth_ctrl_df["생중량"].idxmin() if growth_ctrl_df["생중량"].notna().sum() > 0 else None
            else:
                c_avg = None

            k1, k2, k3 = st.columns(3)
            k1.metric("처리구 평균 생중량",
                      f"{t_avg:.2f}g" if t_avg is not None else "데이터 없음")
            k2.metric("처리구 최대 생중량",
                      f"{growth_treat_df.loc[t_max_idx, '생중량']:.2f}g ({growth_treat_df.loc[t_max_idx, '개체번호']})"
                      if t_max_idx is not None and treat_has_bio else "데이터 없음")
            k3.metric("처리구 최소 생중량",
                      f"{growth_treat_df.loc[t_min_idx, '생중량']:.2f}g ({growth_treat_df.loc[t_min_idx, '개체번호']})"
                      if t_min_idx is not None and treat_has_bio else "데이터 없음")

            k1, k2, k3 = st.columns(3)
            k1.metric("미처리구 평균 생중량",
                      f"{c_avg:.2f}g" if c_avg is not None else "데이터 없음")
            k2.metric("미처리구 최대 생중량",
                      f"{growth_ctrl_df.loc[c_max_idx, '생중량']:.2f}g ({growth_ctrl_df.loc[c_max_idx, '개체번호']})"
                      if c_max_idx is not None and ctrl_has_bio else "데이터 없음")
            k3.metric("미처리구 최소 생중량",
                      f"{growth_ctrl_df.loc[c_min_idx, '생중량']:.2f}g ({growth_ctrl_df.loc[c_min_idx, '개체번호']})"
                      if c_min_idx is not None and ctrl_has_bio else "데이터 없음")

            # ── 평균 생중량 직접 비교 ──
            if t_avg is not None and c_avg is not None:
                st.subheader("📊 평균 생중량 비교")
                diff = t_avg - c_avg
                pct = (diff / c_avg * 100) if c_avg != 0 else None
                comp_cols = st.columns(2)
                comp_cols[0].metric("처리구 평균", f"{t_avg:.2f}g")
                comp_cols[1].metric("미처리구 평균", f"{c_avg:.2f}g")
                st.metric("차이 (처리구 − 미처리구)", f"{diff:+.2f}g",
                          delta=f"{pct:+.1f}%" if pct is not None else None)

            # ── 개체별 생육 비교 (막대 그래프) ──
            st.subheader("📊 개체별 생육 비교")

            bio_bar_cols = st.columns(2)
            with bio_bar_cols[0]:
                if treat_has_bio:
                    fig = bar_chart_single("생중량", growth_treat_df, "처리구 생중량", "#2E86AB")
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("처리구 생중량 막대 그래프를 표시할 수 없습니다.")
                else:
                    st.info("처리구 생육 데이터에 생중량 열이 없습니다.")
            with bio_bar_cols[1]:
                if ctrl_has_bio:
                    fig = bar_chart_single("생중량", growth_ctrl_df, "미처리구 생중량", "#E94F37")
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("미처리구 생중량 막대 그래프를 표시할 수 없습니다.")
                else:
                    st.info("미처리구 생육 데이터에 생중량 열이 없습니다.")

            # 잎 수, 지상부 길이, 지하부 길이 비교 막대
            other_metrics = [m for m in ["잎 수", "지상부 길이", "지하부 길이"]
                            if m in growth_treat_df.columns or m in growth_ctrl_df.columns]
            if other_metrics:
                st.subheader("📊 측정 항목별 개체 비교")
                metric_bar_cols = st.columns(3)
                for idx, m in enumerate(other_metrics):
                    with metric_bar_cols[idx % 3]:
                        t_has = m in growth_treat_df.columns if not growth_treat_df.empty else False
                        c_has = m in growth_ctrl_df.columns if not growth_ctrl_df.empty else False
                        if t_has and c_has:
                            fig = go.Figure()
                            t_data = growth_treat_df[[m, "개체번호"]].dropna().sort_values(m, ascending=False)
                            c_data = growth_ctrl_df[[m, "개체번호"]].dropna().sort_values(m, ascending=False)
                            if len(t_data) > 0:
                                fig.add_trace(go.Bar(x=t_data["개체번호"].astype(str), y=t_data[m],
                                                     marker_color="#2E86AB", name="처리구"))
                            if len(c_data) > 0:
                                fig.add_trace(go.Bar(x=c_data["개체번호"].astype(str), y=c_data[m],
                                                     marker_color="#E94F37", name="미처리구"))
                            fig.update_layout(
                                xaxis_title="개체번호",
                                yaxis_title=m,
                                barmode="group",
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                margin=dict(l=40, r=20, t=30, b=40),
                            )
                            st.plotly_chart(fig, width="stretch")
                        elif t_has:
                            fig = bar_chart_single(m, growth_treat_df, f"처리구 {m}", "#2E86AB")
                            if fig:
                                st.plotly_chart(fig, width="stretch")
                            else:
                                st.info(f"처리구 {m} 그래프를 표시할 수 없습니다.")
                        elif c_has:
                            fig = bar_chart_single(m, growth_ctrl_df, f"미처리구 {m}", "#E94F37")
                            if fig:
                                st.plotly_chart(fig, width="stretch")
                            else:
                                st.info(f"미처리구 {m} 그래프를 표시할 수 없습니다.")
                        else:
                            st.info(f"{m} 데이터가 없습니다.")

            # ── 생중량 분포 (박스플롯 비교) ──
            st.subheader("📦 생중량 분포 비교 (처리구 vs 미처리구)")
            if treat_has_bio and ctrl_has_bio:
                fig_box = go.Figure()
                t_s = growth_treat_df["생중량"].dropna()
                c_s = growth_ctrl_df["생중량"].dropna()
                if len(t_s) > 0:
                    fig_box.add_trace(go.Box(y=t_s, name="처리구", fillcolor="#2E86AB"))
                if len(c_s) > 0:
                    fig_box.add_trace(go.Box(y=c_s, name="미처리구", fillcolor="#E94F37"))
                fig_box.update_layout(
                    yaxis_title="생중량 (g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig_box, width="stretch")
            elif treat_has_bio:
                fig = box_plot("생중량", growth_treat_df)
                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("처리구 생중량 분포 그래프를 표시할 수 없습니다.")
            elif ctrl_has_bio:
                fig = box_plot("생중량", growth_ctrl_df)
                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("미처리구 생중량 분포 그래프를 표시할 수 없습니다.")
            else:
                st.info("생중량 데이터가 없어 분포 그래프를 표시할 수 없습니다.")

            # ── 생중량 통계 (평균·표준편차·CV) ──
            if treat_has_bio:
                s = growth_treat_df["생중량"].dropna()
                if len(s) >= 2:
                    m, std, cv = s.mean(), s.std(), s.std() / s.mean() if s.mean() != 0 else None
                    sc = st.columns(3)
                    sc[0].metric("처리구 평균", f"{m:.2f}g")
                    sc[1].metric("처리구 표준편차", f"{std:.2f}g")
                    sc[2].metric("처리구 CV",
                                 f"{cv:.3f}" if cv is not None else "N/A")
            if ctrl_has_bio:
                s = growth_ctrl_df["생중량"].dropna()
                if len(s) >= 2:
                    m, std, cv = s.mean(), s.std(), s.std() / s.mean() if s.mean() != 0 else None
                    sc = st.columns(3)
                    sc[0].metric("미처리구 평균", f"{m:.2f}g")
                    sc[1].metric("미처리구 표준편차", f"{std:.2f}g")
                    sc[2].metric("미처리구 CV",
                                 f"{cv:.3f}" if cv is not None else "N/A")
            st.caption("※ 표준편차: 값들이 평균에서 얼마나 흩어져 있는지 / CV(변동계수): 표준편차÷평균, 단위가 다른 집단 비교에 사용")

            # ── 상관관계 분석 (각 그룹별) ──
            st.subheader("🔗 상관관계 분석 (그룹별)")

            corr_pairs = [
                ("잎 수", "생중량", "#2E86AB", "처리구"),
                ("잎 수", "생중량", "#E94F37", "미처리구"),
                ("지상부 길이", "생중량", "#2E86AB", "처리구"),
                ("지상부 길이", "생중량", "#E94F37", "미처리구"),
            ]

            shown_any_corr = False
            for x_col, y_col, color, grp_label in corr_pairs:
                if grp_label == "처리구":
                    src_df = growth_treat_df
                else:
                    src_df = growth_ctrl_df
                if src_df.empty:
                    continue
                if x_col not in src_df.columns or y_col not in src_df.columns:
                    continue
                subset = src_df[[x_col, y_col]].dropna()
                if len(subset) < 2:
                    continue
                r, df_corr = safe_corr(subset[x_col], subset[y_col])
                if r is None:
                    continue
                shown_any_corr = True
                c_label, c_fig = st.columns([1, 3])
                c_label.metric(f"{grp_label} {x_col} vs {y_col} (r)", f"{r:.3f}")
                with c_fig:
                    fig_scatter = scatter_with_trend(
                        df_corr[x_col], df_corr[y_col],
                        x_col, y_col, color
                    )
                    st.plotly_chart(fig_scatter, width="stretch")

            if not shown_any_corr:
                st.info("상관관계 분석에 필요한 데이터(잎 수·지상부 길이·생중량)가 충분하지 않습니다.")

            # ── 생중량 상위 25% vs 하위 25% (그룹별) ──
            st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")

            group_stats_cols = st.columns(2)

            for grp_idx, (grp_label, grp_df, has_bio) in enumerate([
                ("처리구", growth_treat_df, treat_has_bio),
                ("미처리구", growth_ctrl_df, ctrl_has_bio),
            ]):
                with group_stats_cols[grp_idx]:
                    st.markdown(f"#### {grp_label}")
                    if not has_bio or grp_df.empty:
                        st.info(f"{grp_label} 생중량 데이터가 없습니다.")
                        continue
                    s = grp_df["생중량"].dropna()
                    if len(s) < 4:
                        st.info(f"{grp_label} 개체가 4개 미만이라 상·하위 25%를 나눌 수 없습니다.")
                        continue
                    q75 = s.quantile(0.75)
                    q25 = s.quantile(0.25)
                    top = grp_df[grp_df["생중량"] >= q75]
                    bottom = grp_df[grp_df["생중량"] <= q25]

                    st.metric("개체 수 (전체)", f"{grp_df['개체번호'].nunique()}개체")
                    st.metric("상위 25% 개체 수", f"{len(top)}개체")
                    st.metric("하위 25% 개체 수", f"{len(bottom)}개체")

                    sub_cols = st.columns(3)
                    for i, col in enumerate(["잎 수", "지상부 길이", "지하부 길이"]):
                        if col in top.columns and len(top) > 0:
                            sub_cols[i].metric(f"상위 평균 {col}",
                                                f"{top[col].mean():.1f}")
                        if col in bottom.columns and len(bottom) > 0:
                            pass  # 아래는 별도 메트릭
                    st.markdown("**상위 25% 평균**")
                    if "잎 수" in top.columns:
                        st.metric("잎 수", f"{top['잎 수'].mean():.1f}장")
                    if "지상부 길이" in top.columns:
                        st.metric("지상부 길이", f"{top['지상부 길이'].mean():.1f}mm")
                    if "지하부 길이" in top.columns:
                        st.metric("지하부 길이", f"{top['지하부 길이'].mean():.1f}mm")

                    st.markdown("**하위 25% 평균**")
                    if "잎 수" in bottom.columns:
                        st.metric("잎 수", f"{bottom['잎 수'].mean():.1f}장")
                    if "지상부 길이" in bottom.columns:
                        st.metric("지상부 길이", f"{bottom['지상부 길이'].mean():.1f}mm")
                    if "지하부 길이" in bottom.columns:
                        st.metric("지하부 길이", f"{bottom['지하부 길이'].mean():.1f}mm")

            # ── 원본 테이블 + 다운로드 ──
            with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                gd_cols = st.columns(2)
                with gd_cols[0]:
                    if not growth_treat_df.empty:
                        st.markdown("#### 📊 처리구 생육 데이터")
                        st.dataframe(growth_treat_df, width="stretch")
                        csv_download_button(growth_treat_df, "처리구 생육 CSV 다운로드",
                                            f"{school_name}_처리구_생육결과데이터.csv")
                    else:
                        st.info("처리구 생육 데이터가 없습니다.")
                with gd_cols[1]:
                    if not growth_ctrl_df.empty:
                        st.markdown("#### 📊 미처리구 생육 데이터")
                        st.dataframe(growth_ctrl_df, width="stretch")
                        csv_download_button(growth_ctrl_df, "미처리구 생육 CSV 다운로드",
                                            f"{school_name}_미처리구_생육결과데이터.csv")
                    else:
                        st.info("미처리구 생육 데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
