import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 데이터 로딩 ──────────────────────────────────────────────────────────────
@st.cache_data
def read_csv(file):
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
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ── 한글 폰트 CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

school_name = ""
env_file = st.sidebar.file_uploader("🌡️ 환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("📊 생육 결과 CSV", type="csv")

if env_file is not None:
    school_name = os.path.splitext(os.path.basename(env_file.name))[0]
    school_name = school_name.replace("_환경데이터", "").replace("_환경", "")
if growth_file is not None and not school_name:
    school_name = os.path.splitext(os.path.basename(growth_file.name))[0]
    school_name = school_name.replace("_생육결과데이터", "").replace("_생육", "")

with st.sidebar:
    st.markdown("---")
    st.info("📁 CSV 파일을 업로드하면 분석이 시작됩니다.")
    st.markdown("**용어 설명**")
    st.markdown("- **평균**: 모든 값을 더해 개수로 나눈 값")
    st.markdown("- **표준편차**: 값들이 평균에서 얼마나 퍼져있는지")
    st.markdown("- **CV(변동계수)**: 표준편차를 평균으로 나눈 비율(%)")
    st.markdown("- **상관계수(r)**: 두 변수의 관계가 얼마나 강한지 (-1~1)")

env_df = read_csv(env_file)
growth_df = read_csv(growth_file)

if not env_df.empty:
    env_df = std_cols(env_df)
if not growth_df.empty:
    growth_df = std_cols(growth_df)

# ── 원본 CSV 다운로드 헬퍼 ───────────────────────────────────────────────────
def make_download_btn(df, label, filename):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, csv_bytes, file_name=filename, mime="text/csv")

# ── 그래프 헬퍼 ──────────────────────────────────────────────────────────────
def bar_with_highlight(labels, values, title, ylabel, q1, q3):
    colors = []
    for v in values:
        if v >= q3:
            colors.append("#d32f2f")
        elif v <= q1:
            colors.append("#1976d2")
        else:
            colors.append("#90caf9")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(labels), y=list(values), marker_color=colors,
                         marker_line_color="white", marker_line_width=0.5))
    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"},
                      xaxis_title="개체", yaxis_title=ylabel,
                      font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                      height=380, margin=dict(l=50, r=20, t=50, b=50))
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.15)", griddash="dot")
    return fig

def boxplot_single(values, title, ylabel):
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    fig = go.Figure()
    fig.add_trace(go.Box(y=list(values), name="생중량",
                         marker_color="#bbdefb",
                         boxmean=True,
                         line=dict(color="#1976d2"),
                         fillcolor="#bbdefb"))
    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"},
                      yaxis_title=ylabel,
                      font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                      height=350, margin=dict(l=50, r=20, t=50, b=50))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.15)", griddash="dot")
    return fig

def scatter_with_trend(x, y, xlabel, ylabel, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="markers",
                             marker=dict(color="#1976d2", size=8, opacity=0.75,
                                        line=dict(color="white", width=0.5)),
                             name="데이터"))
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 100)
    fig.add_trace(go.Scatter(x=list(xs), y=list(slope*xs + intercept),
                             mode="lines", line=dict(color="#d32f2f", width=2),
                             name=f"추세선 (기울기 {slope:.3f})"))
    fig.update_layout(title={"text": title, "x": 0.5, "xanchor": "center"},
                      xaxis_title=xlabel, yaxis_title=ylabel,
                      font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                      height=380, margin=dict(l=50, r=20, t=50, b=50))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.15)", griddash="dot")
    return fig

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ====================== 탭 1: 실험 개요 =====================═
with tab1:
    try:
        if env_df.empty and growth_df.empty:
            st.info("📁 사이드바에서 CSV 파일을 업로드해주세요.")
        else:
            # 데이터 요약
            if not env_df.empty:
                n_env = len(env_df)
                time_col = "time" if "time" in env_df.columns else None
                period_msg = ""
                if time_col and not env_df[time_col].isna().all():
                    tmin = env_df[time_col].min()
                    tmax = env_df[time_col].max()
                    period_msg = f"{tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}"
            else:
                n_env = 0
                period_msg = "데이터 없음"

            if not growth_df.empty:
                n_growth = len(growth_df)
                growth_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이",
                                           "지상부 질량", "지하부 질량"] if c in growth_df.columns]
            else:
                n_growth = 0
                growth_cols = []

            st.header("연구 배경 및 목적")
            st.markdown(f"""**{school_name} 극지식물 생육 분석**
- 극지식물(폴라 플랜트)이 서로 다른 환경에서 어떻게 자라는지 알아보기 위한 실험입니다.
- 개체별 생중량(식물 무게) 차이를 비교하고, 잎 수·길이와의 관계를 살펴봅니다.
            """)

            st.subheader("데이터 요약")
            col_summary = st.columns(4)
            with col_summary[0]:
                st.metric("총 개체 수", f"{n_growth}개체" if n_growth else "데이터 없음", delta=None)
            with col_summary[1]:
                if not env_df.empty and "temperature" in env_df.columns:
                    avg_t = env_df["temperature"].dropna().mean()
                    st.metric("평균 온도", f"{avg_t:.1f}℃")
                else:
                    st.metric("평균 온도", "데이터 없음")
            with col_summary[2]:
                if not env_df.empty and "humidity" in env_df.columns:
                    avg_h = env_df["humidity"].dropna().mean()
                    st.metric("평균 습도", f"{avg_h:.1f}%")
                else:
                    st.metric("평균 습도", "데이터 없음")
            with col_summary[3]:
                if not growth_df.empty and "생중량" in growth_df.columns:
                    w = growth_df["생중량"].dropna()
                    if len(w) > 0:
                        st.metric("평균 생중량", f"{w.mean():.2f}g")
                    else:
                        st.metric("평균 생중량", "데이터 없음")
                else:
                    st.metric("평균 생중량", "데이터 없음")

            st.info(f"📅 측정 기간: {period_msg}" if period_msg else "📅 측정 기간 정보가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ====================== 탭 2: 환경 데이터 =================═
with tab2:
    try:
        if env_df.empty:
            st.info("📁 사이드바에서 환경 CSV 파일을 업로드해주세요.")
        else:
            st.header("🌡️ 환경 데이터 요약")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"]
                        if c in env_df.columns]
            if env_cols:
                st.subheader("통계 요약")
                env_stats = env_df[env_cols].dropna().agg(["mean", "min", "max"]).T
                env_stats.columns = ["평균", "최소", "최대"]
                st.dataframe(env_stats, width="stretch")
            else:
                st.warning("표시할 숫자 환경 열이 없습니다.")

            # 시계열
            st.subheader("날짜별 변화 (일별 평균)")
            if "time" in env_df.columns and not env_df["time"].isna().all():
                daily = env_df.set_index("time").resample("D").mean(numeric_only=True)
                env_cols_daily = [c for c in ["temperature", "humidity", "ph", "ec"]
                                   if c in daily.columns]
                if env_cols_daily:
                    sub_tabs = st.tabs([
                        "📈 온도", "💧 습도", "🧪 pH·EC"
                    ])
                    with sub_tabs[0]:
                        if "temperature" in daily.columns:
                            fig_t = go.Figure()
                            fig_t.add_trace(go.Scatter(x=daily.index, y=daily["temperature"],
                                                        mode="lines+markers",
                                                        line=dict(color="#d32f2f", width=2),
                                                        marker=dict(size=5)))
                            fig_t.update_layout(title={"text": "일별 평균 온도", "x": 0.5},
                                                xaxis_title="날짜", yaxis_title="온도 (℃)",
                                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                                height=300)
                            st.plotly_chart(fig_t, width="stretch")
                        else:
                            st.info("온도 열이 없어 표시하지 못했습니다.")
                    with sub_tabs[1]:
                        if "humidity" in daily.columns:
                            fig_h = go.Figure()
                            fig_h.add_trace(go.Scatter(x=daily.index, y=daily["humidity"],
                                                       mode="lines+markers",
                                                       line=dict(color="#1976d2", width=2),
                                                       marker=dict(size=5)))
                            fig_h.update_layout(title={"text": "일별 평균 습도", "x": 0.5},
                                                xaxis_title="날짜", yaxis_title="습도 (%)",
                                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                                height=300)
                            st.plotly_chart(fig_h, width="stretch")
                        else:
                            st.info("습도 열이 없어 표시하지 못했습니다.")
                    with sub_tabs[2]:
                        ph_ec_cols = [c for c in ["ph", "ec"] if c in daily.columns]
                        if ph_ec_cols:
                            fig_ph = make_subplots(rows=1, cols=len(ph_ec_cols),
                                                   subplot_titles=["pH", "EC"] if "ph" in ph_ec_cols and "ec" in ph_ec_cols
                                                   else (["pH"] if "ph" in ph_ec_cols else ["EC"]))
                            for i, col in enumerate(ph_ec_cols):
                                fig_ph.add_trace(go.Scatter(x=daily.index, y=daily[col],
                                                            mode="lines", line=dict(width=2),
                                                            name=col),
                                                 row=1, col=i+1)
                            fig_ph.update_layout(title={"text": "일별 pH·EC 변화", "x": 0.5},
                                                 font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                                 height=300)
                            st.plotly_chart(fig_ph, width="stretch")
                        else:
                            st.info("pH·EC 열이 없어 표시하지 못했습니다.")
                else:
                    st.info("표시할 환경 숫자 열이 없습니다.")
            else:
                st.info("time(시간) 열이 없거나 유효한 날짜가 없어 시계열을 그릴 수 없습니다.")

            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch")
                if not env_df.empty:
                    make_download_btn(env_df, "📥 환경 데이터 CSV 다운로드", f"{school_name}_환경데이터.csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ====================== 탭 3: 생육 결과 =================═
with tab3:
    try:
        if growth_df.empty:
            st.info("📁 사이드바에서 생육 결과 CSV 파일을 업로드해주세요.")
        else:
            # 핵심 결과 카드
            st.header("🥇 핵심 결과")
            if "생중량" in growth_df.columns:
                w = growth_df["생중량"].dropna()
                if len(w) >= 2:
                    avg_w = w.mean()
                    max_idx = w.idxmax()
                    min_idx = w.idxmin()
                    max_id = growth_df.loc[max_idx, "개체번호"]
                    min_id = growth_df.loc[min_idx, "개체번호"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("평균 생중량", f"{avg_w:.2f}g")
                    c2.metric("최대 생중량", f"{w.max():.2f}g", delta=f"개체 {max_id}")
                    c3.metric("최소 생중량", f"{w.min():.2f}g", delta=f"개체 {min_id}")
                else:
                    st.metric("평균 생중량", "데이터 부족")
            else:
                st.info("생중량 열이 없어 핵심 결과를 표시할 수 없습니다.")

            # 개체별 생육 비교
            st.subheader("개체별 생육 비교 (막대 그래프)")
            growth_num_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                                if c in growth_df.columns]
            if growth_num_cols:
                sub_tabs_g = st.tabs([
                    "⭐ 생중량", "🍃 잎 수", "📏 지상부 길이", "📏 지하부 길이"
                ])
                for idx, col in enumerate(growth_num_cols):
                    vals = growth_df[col].dropna()
                    ids = growth_df.loc[vals.index, "개체번호"].tolist()
                    if len(vals) >= 2:
                        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                        with sub_tabs_g[idx]:
                            fig_bar = bar_with_highlight(ids, vals.values,
                                                         f"개체별 {col}",
                                                         col, q1, q3)
                            st.plotly_chart(fig_bar, width="stretch")
                    else:
                        with sub_tabs_g[idx]:
                            st.info(f"{col} 데이터가 충분하지 않아 그래프를 표시할 수 없습니다.")
            else:
                st.info("표시할 생육 숫자 열이 없습니다.")

            # 생중량 분포 (박스플롯)
            st.subheader("생중량 분포")
            if "생중량" in growth_df.columns:
                w_all = growth_df["생중량"].dropna()
                if len(w_all) >= 2:
                    fig_box = boxplot_single(w_all.values, "개체별 생중량 분포 (박스플롯)",
                                             "생중량 (g)")
                    # 사분위 정보 텍스트 추가
                    q1_b, q3_b = np.percentile(w_all.values, 25), np.percentile(w_all.values, 75)
                    st.plotly_chart(fig_box, width="stretch")
                    st.info(f"📊 Q1={q1_b:.2f}g, Q3={q3_b:.2f}g, IQR={q3_b-q1_b:.2f}g")
                else:
                    st.info("생중량 데이터가 충분하지 않습니다.")
            else:
                st.info("생중량 열이 없어 박스플롯을 표시할 수 없습니다.")

            # 상관관계 분석
            st.subheader("상관관계 분석 (산점도 + 추세선)")
            if "생중량" in growth_df.columns:
                w_col = "생중량"
                w_vals = growth_df[w_col].dropna()
                pair_cols = [c for c in ["잎 수", "지상부 길이"] if c in growth_df.columns]
                if pair_cols:
                    corr_tabs = st.tabs([f"🍃 잎 수 vs 생중량", f"📏 지상부 길이 vs 생중량"])
                    for idx, col in enumerate(pair_cols):
                        leaf_vals = growth_df[col].dropna()
                        common_idx = w_vals.index.intersection(leaf_vals.index)
                        x = leaf_vals.loc[common_idx].values
                        y = w_vals.loc[common_idx].values
                        with corr_tabs[idx]:
                            if len(x) >= 2:
                                r = np.corrcoef(x, y)[0, 1]
                                fig_sc = scatter_with_trend(x, y, f"{col}", "생중량 (g)",
                                                            f"{col} vs 생중량  (r = {r:.2f})")
                                st.plotly_chart(fig_sc, width="stretch")
                                st.info(f"📐 상관계수 r = {r:.2f} | 기울기 = {np.polyfit(x, y, 1)[0]:.3f}")
                            else:
                                st.info(f"{col}과 생중량 공통 데이터가 충분하지 않습니다.")
                else:
                    st.info("잎 수·지상부 길이 열이 없어 상관관계 분석을 할 수 없습니다.")
            else:
                st.info("생중량 열이 없어 상관관계 분석을 할 수 없습니다.")

            with st.expander("📄 생육 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch")
                if not growth_df.empty:
                    make_download_btn(growth_df, "📥 생육 데이터 CSV 다운로드",
                                      f"{school_name}_생육결과데이터.csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
