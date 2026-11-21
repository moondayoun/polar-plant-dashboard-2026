import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ------------------------------------------------------------------
# CSV 읽기: file.getvalue()로 바이트를 꺼내 인코딩 두 번 시도
# pd.read_csv(file)로 두 번 읽으면 빈 파일이 되는 문제를 방지
# ------------------------------------------------------------------
@st.cache_data
def read_csv_cached(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ------------------------------------------------------------------
# 열 이름 표준화 (KeyError 방지)
# ------------------------------------------------------------------
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
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + \
                pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df

# ------------------------------------------------------------------
# 그래프 헬퍼: 평균 막대 그래프
# ------------------------------------------------------------------
def draw_bar_chart(df, col, title, color):
    valid = df[[col]].dropna()
    if len(valid) < 1:
        return None
    fig = px.bar(
        x=valid.index.tolist(),
        y=valid[col].tolist(),
        labels={"x": "개체", "y": title},
        color=valid[col].tolist(),
        color_continuous_scale=color,
    )
    fig.update_traces(hovertemplate=col + ": %{y:.2f}" + "<extra></extra>")
    fig.update_layout(
        title=title,
        xaxis_title="개체",
        yaxis_title=title,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        coloraxis_showscale=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

# ------------------------------------------------------------------
# 그래프 헬퍼: 산점도 + np.polyfit 추세선
# ------------------------------------------------------------------
def draw_scatter_plot(df, weight_col, metric_col, label):
    tmp = df[[weight_col, metric_col]].dropna()
    if len(tmp) < 2:
        return None
    x = tmp[metric_col].values
    y = tmp[weight_col].values
    r, _ = stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x.tolist(),
        y=y.tolist(),
        mode="markers",
        marker=dict(size=8, color="#1976d2", opacity=0.75),
        name=label,
        hovertemplate=label + ": %{x:.2f}<br>생중량: %{y:.2f} g<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=line_x.tolist(),
        y=line_y.tolist(),
        mode="lines",
        line=dict(color="#d32f2f", width=2),
        name="추세선 (r=" + f"{r:.2f}" + ")",
    ))
    fig.update_layout(
        title=label + " vs 생중량 (상관계수 r = " + f"{r:.2f}" + ")",
        xaxis_title=label,
        yaxis_title="생중량 (g)",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

# ------------------------------------------------------------------
# 그래프 헬퍼: 생중량 분포 박스플롯
# ------------------------------------------------------------------
def draw_boxplot(df, weight_col):
    valid = df[[weight_col]].dropna()
    if len(valid) < 2:
        return None
    fig = px.box(
        y=valid[weight_col].tolist(),
        labels={"y": "생중량 (g)"},
        color_discrete_sequence=["#1976d2"],
    )
    fig.update_traces(fillcolor="#bbdefb", boxmean=True)
    fig.update_layout(
        title="개체별 생중량 분포 (박스플롯)",
        yaxis_title="생중량 (g)",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

# ------------------------------------------------------------------
# 안전표 출력 헬퍼
# ------------------------------------------------------------------
def safe_dataframe(df):
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.info("표를 표시할 수 없습니다.")

# ============================================================
# 메인 앱
# ============================================================
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")

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

st.title("극지식물 개체별 생육 차이 분석")

# ---- 사이드바: CSV 업로드 ----
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

with st.sidebar:
    st.markdown("---")
    st.markdown("### 생육 측정 항목 선택")
    st.caption("비교할 항목을 고르세요. (평균 생중량은 항상 표시됩니다)")

# CSV 읽는 중 spinner 표시
with st.spinner("CSV를 읽는 중이에요..."):
    env_df = read_csv_cached(env_file)
    growth_df = read_csv_cached(growth_file)

# 열 표준화 적용
env_df = std_cols(env_df)
growth_df = std_cols(growth_df)

# 학교명: 업로드한 파일명 앞부분에서 추출
school_name = "우리 학교"
if env_file is not None:
    school_name = env_file.name.split("_")[0].strip() or school_name
elif growth_file is not None:
    school_name = growth_file.name.split("_")[0].strip() or school_name

# 생육 측정 항목 선택
growth_cols = ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"]
available_growth_cols = [c for c in growth_cols if c in growth_df.columns]
selected_metrics = st.sidebar.multiselect(
    "비교 항목",
    options=available_growth_cols,
    default=available_growth_cols,
    help="여러 항목을 고를 수 있어요.",
)

# 다운로드용 원본 데이터 복구 헬퍼
def orig_env():
    if env_file is None or env_df.empty:
        return pd.DataFrame()
    cols = [c for c in env_df.columns
            if c in ("time", "temperature", "humidity", "ph", "ec",
                     "water_temp", "co2", "categories")]
    return env_df[[c for c in cols if c in env_df.columns]].copy()

def orig_growth():
    if growth_file is None or growth_df.empty:
        return pd.DataFrame()
    cols = [c for c in growth_df.columns
            if c in ("개체번호", "잎 수", "지상부 길이", "지하부 길이",
                     "생중량", "지상부 질량", "지하부 질량")]
    return growth_df[[c for c in cols if c in growth_df.columns]].copy()

# ============================================================
# 탭 1: 실험 개요
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["실험 개요", "환경 데이터", "생육 결과"]
)

with tab1:
    try:
        st.header("실험 개요")
        st.subheader("연구 배경 및 목적")
        st.info(
            "극지식물은 추운 환경에서도 살아남는 강한 식물이에요. "
            "우리 학교 친구들이 키운 극지식물 개체마다 생육 상태가 "
            "어떻게 다른지 살펴보고, 온도·습도 같은 환경 조건과 "
            "어떤 관계가 있는지 알아봅니다."
        )
        st.markdown("**통계 용어 간단 설명**")
        st.markdown(
            "- **평균(mean)**: 여러 값을 더해 개수로 나눈 '대표값'이에요. "
            "- **표준편차(standard deviation)**: 값이 평균에서 얼마나 흩어져 있는지 나타내요. "
            "- **상관계수(correlation coefficient, r)**: 두 변수가 함께 움직이는 정도를 "
            "-1 ~ 1로 보여줘요. 1에 가까울수록 함께 증가해요."
        )

        if env_df is None or env_df.empty:
            st.info("환경 CSV를 올려주세요. 이 탭의 환경 정보가 표시됩니다.")
        if growth_df is None or growth_df.empty:
            st.info("생육 CSV를 올려주세요. 이 탭의 생육 정보가 표시됩니다.")

        if (env_df is not None and not env_df.empty) or \
           (growth_df is not None and not growth_df.empty):
            st.subheader("데이터 요약")
            rows = []
            if growth_df is not None and not growth_df.empty:
                rows.append(["생육 결과 데이터", str(len(growth_df)) + "개체",
                             "개체번호, 잎 수, 지상부 길이, 지하부 길이, 생중량 등"])
            if env_df is not None and not env_df.empty:
                time_col = env_df["time"] if "time" in env_df.columns else None
                if time_col is not None and not time_col.isna().all():
                    start = time_col.min().strftime("%Y-%m-%d")
                    end = time_col.max().strftime("%Y-%m-%d")
                    period = start + " ~ " + end
                else:
                    period = "기간 정보 없음"
                rows.append(["환경 데이터", str(len(env_df)) + "개 측정",
                             "측정 기간: " + period])
            if rows:
                summary_df = pd.DataFrame(rows, columns=["데이터", "크기", "주요 항목"])
                safe_dataframe(summary_df)

        col1, col2, col3, col4 = st.columns(4)

        if growth_df is not None and not growth_df.empty and \
                "개체번호" in growth_df.columns:
            total = len(growth_df["개체번호"].dropna())
            col1.metric("총 개체수", str(total) + "개체")
        else:
            col1.metric("총 개체수", "-")

        if env_df is not None and not env_df.empty and \
                "temperature" in env_df.columns:
            t_avg = env_df["temperature"].dropna()
            if len(t_avg) >= 1:
                col2.metric("평균 온도", f"{t_avg.mean():.1f} C")
            else:
                col2.metric("평균 온도", "-")
        else:
            col2.metric("평균 온도", "-")

        if env_df is not None and not env_df.empty and \
                "humidity" in env_df.columns:
            h_avg = env_df["humidity"].dropna()
            if len(h_avg) >= 1:
                col3.metric("평균 습도", f"{h_avg.mean():.1f}%")
            else:
                col3.metric("평균 습도", "-")
        else:
            col3.metric("평균 습도", "-")

        if growth_df is not None and not growth_df.empty and \
                "생중량" in growth_df.columns:
            w_avg = growth_df["생중량"].dropna()
            if len(w_avg) >= 1:
                col4.metric("평균 생중량", f"{w_avg.mean():.2f}g")
            else:
                col4.metric("평균 생중량", "-")
        else:
            col4.metric("평균 생중량", "-")

    except Exception as e:
        st.warning("경고: 이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))

# ============================================================
# 탭 2: 환경 데이터
# ============================================================
with tab2:
    try:
        st.header("환경 데이터")

        if env_df is None or env_df.empty:
            st.info("환경 CSV를 올려주세요. 환경 데이터의 그래프와 표가 표시돼요.")
        else:
            st.subheader("환경 요약 (평균, 최소, 최대)")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                        if c in env_df.columns]
            if env_cols:
                rows = []
                for c in env_cols:
                    s = env_df[c].dropna()
                    if len(s) >= 1:
                        label = c.capitalize()
                        if c == "temperature":
                            label = "온도 (C)"
                        elif c == "humidity":
                            label = "습도 (%)"
                        elif c == "ph":
                            label = "pH"
                        elif c == "ec":
                            label = "EC (dS/m)"
                        rows.append([
                            label,
                            f"{s.mean():.2f}",
                            f"{s.min():.2f}",
                            f"{s.max():.2f}",
                        ])
                if rows:
                    env_summary = pd.DataFrame(rows, columns=["항목", "평균", "최소", "최대"])
                    safe_dataframe(env_summary)
                else:
                    st.info("환경 수치를 계산할 수 없어요.")
            else:
                st.info("온도, 습도, pH, EC 열이 없어서 환경 요약을 표시할 수 없어요.")

            st.subheader("날짜별 환경 변화 (일별 평균)")
            if "time" in env_df.columns:
                env_df["day"] = env_df["time"].dt.date
                daily_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                              if c in env_df.columns]
                for col in daily_cols:
                    daily = env_df.groupby("day")[col].mean().dropna()
                    if len(daily) < 1:
                        st.info(col + "의 일별 평균을 계산할 수 없어요.")
                        continue
                    fig = px.line(
                        x=daily.index.astype(str),
                        y=daily.values,
                        labels={"x": "날짜", "y": col},
                        markers=True,
                    )
                    y_label = col
                    if col == "temperature":
                        y_label = "온도 (C)"
                    elif col == "humidity":
                        y_label = "습도 (%)"
                    elif col == "ph":
                        y_label = "pH"
                    elif col == "ec":
                        y_label = "EC (dS/m)"
                    fig.update_traces(hovertemplate="%{x}<br>" + y_label +
                                      ": %{y:.2f}<extra></extra>")
                    fig.update_layout(
                        title=y_label + " 일별 평균",
                        xaxis_title="날짜",
                        yaxis_title=y_label,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("시간(time) 열이 없어서 날짜별 그래프를 그릴 수 없어요.")

            with st.expander("환경 데이터 원본 테이블 + 다운로드"):
                orig_e = orig_env()
                if orig_e.empty:
                    st.info("표시할 원본 데이터가 없어요.")
                else:
                    safe_dataframe(orig_e.head(200))
                    csv_bytes = orig_e.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="환경 CSV 다운로드 (UTF-8)",
                        data=csv_bytes,
                        file_name="환경데이터_원본.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning("경고: 이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))

# ============================================================
# 탭 3: 생육 결과
# ============================================================
with tab3:
    try:
        st.header("생육 결과")

        if growth_df is None or growth_df.empty:
            st.info("생육 CSV를 올려주세요. 생육 분석 그래프가 표시돼요.")
        else:
            st.subheader("핵심 결과")
            if "생중량" in growth_df.columns:
                w = growth_df["생중량"].dropna()
                if len(w) >= 1:
                    mean_w = w.mean()
                    max_idx = w.idxmax()
                    min_idx = w.idxmin()
                    max_id = growth_df.loc[max_idx, "개체번호"]
                    min_id = growth_df.loc[min_idx, "개체번호"]

                    def fmt_id(v):
                        try:
                            fv = float(v)
                            if fv == int(fv):
                                return str(int(fv))
                        except Exception:
                            pass
                        return str(v)

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("평균 생중량", f"{mean_w:.2f} g")
                    col_b.metric("최대 생중량", f"{w.max():.2f} g",
                                 "개체 " + fmt_id(max_id))
                    col_c.metric("최소 생중량", f"{w.min():.2f} g",
                                 "개체 " + fmt_id(min_id))
                else:
                    st.info("생중량 값이 하나도 없어서 핵심 결과를 표시할 수 없어요.")
            else:
                st.info("생중량 열이 없어서 핵심 결과를 표시할 수 없어요.")

            st.caption(
                "통계 용어 - 표준편차: 값이 평균에서 얼마나 흩어져 있는지, "
                "CV(변동계수): 표준편차를 평균으로 나눈 값으로, 서로 다른 규모의 "
                "그룹을 비교할 때 써요."
            )

            st.subheader("개체별 생육 비교 (막대 그래프)")
            bar_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                        if c in growth_df.columns]
            if bar_cols:
                for bc in bar_cols:
                    y_label = bc
                    if bc == "생중량":
                        y_label = "생중량 (g) ☆"
                    elif bc == "잎 수":
                        y_label = "잎 수 (장)"
                    elif bc == "지상부 길이":
                        y_label = "지상부 길이 (mm)"
                    elif bc == "지하부 길이":
                        y_label = "지하부 길이 (mm)"
                    fig = draw_bar_chart(growth_df, bc, y_label,
                                         "Blues" if bc == "생중량" else "Greens")
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(y_label + " 그래프를 표시할 수 없어요.")
            else:
                st.info("비교할 생육 열이 없어요.")

            if "생중량" in growth_df.columns:
                bp = draw_boxplot(growth_df, "생중량")
                if bp is not None:
                    st.subheader("생중량 분포 (박스플롯)")
                    st.plotly_chart(bp, width="stretch")
                else:
                    st.info("생중량 분포 그래프를 표시할 수 없어요.")

            if "생중량" in growth_df.columns:
                w2 = growth_df["생중량"].dropna()
                if len(w2) >= 2:
                    mean_w2 = w2.mean()
                    std_w2 = w2.std()
                    cv_w2 = (std_w2 / mean_w2 * 100) if mean_w2 != 0 else 0
                    st.info(
                        f"생중량: 평균 {mean_w2:.2f}g, "
                        f"표준편차 {std_w2:.2f}g, "
                        f"CV {cv_w2:.1f}%"
                    )

            if "생중량" in growth_df.columns:
                w3 = growth_df["생중량"].dropna()
                if len(w3) >= 4:
                    q1 = w3.quantile(0.25)
                    q3 = w3.quantile(0.75)
                    top_df = growth_df[growth_df["생중량"] >= q3].copy()
                    bot_df = growth_df[growth_df["생중량"] <= q1].copy()
                    if not top_df.empty and not bot_df.empty:
                        st.subheader("생중량 상위 25% vs 하위 25% 비교")
                        rows = []
                        rows.append(["그룹", "개체 수", "평균 생중량 (g)",
                                     "평균 잎 수", "평균 지상부 길이 (mm)"])
                        if "잎 수" in top_df.columns:
                            top_leaf = top_df["잎 수"].dropna().mean()
                            bot_leaf = bot_df["잎 수"].dropna().mean()
                        else:
                            top_leaf = float("nan")
                            bot_leaf = float("nan")
                        if "지상부 길이" in top_df.columns:
                            top_shoot = top_df["지상부 길이"].dropna().mean()
                            bot_shoot = bot_df["지상부 길이"].dropna().mean()
                        else:
                            top_shoot = float("nan")
                            bot_shoot = float("nan")
                        rows.append([
                            "상위 25% (75분위 이상)",
                            str(len(top_df)),
                            f"{top_df['생중량'].mean():.2f}",
                            f"{top_leaf:.1f}" if not np.isnan(top_leaf) else "-",
                            f"{top_shoot:.1f}" if not np.isnan(top_shoot) else "-",
                        ])
                        rows.append([
                            "하위 25% (25분위 이하)",
                            str(len(bot_df)),
                            f"{bot_df['생중량'].mean():.2f}",
                            f"{bot_leaf:.1f}" if not np.isnan(bot_leaf) else "-",
                            f"{bot_shoot:.1f}" if not np.isnan(bot_shoot) else "-",
                        ])
                        comp_df = pd.DataFrame(rows[1:], columns=rows[0])
                        safe_dataframe(comp_df)

            st.subheader("상관관계 분석 (산점도 + 추세선)")
            st.caption(
                "각 점은 하나의 개체를 나타내요. 추세선은 직선 모양으로 두 변수의 "
                "전반적인 방향을 보여줘요. 상관계수 r이 1에 가까울수록 강한 양의 관계예요."
            )
            if "생중량" in growth_df.columns:
                metrics_to_plot = [m for m in selected_metrics
                                   if m in growth_df.columns]
                for m in metrics_to_plot:
                    sp = draw_scatter_plot(growth_df, "생중량", m, m)
                    if sp is not None:
                        st.plotly_chart(sp, width="stretch")
                    else:
                        st.info(m + " vs 생중량 산점도를 표시할 수 없어요 "
                                + "(값이 너무 적어요).")
            else:
                st.info("생중량 열이 없어서 상관관계 분석을 할 수 없어요.")

            with st.expander("생육 데이터 원본 테이블 + 다운로드"):
                orig_g = orig_growth()
                if orig_g.empty:
                    st.info("표시할 원본 데이터가 없어요.")
                else:
                    safe_dataframe(orig_g)
                    csv_bytes = orig_g.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="생육 CSV 다운로드 (UTF-8)",
                        data=csv_bytes,
                        file_name="생육결과_원본.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning("경고: 이 부분은 데이터 형식 때문에 표시하지 못했어요: " + str(e))

# ============================================================
# 사이드바 하단: 학교명 표시
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("학교명: " + school_name)
