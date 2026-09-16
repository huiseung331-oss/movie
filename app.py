import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 기본 페이지 설정 (브라우저 탭 제목 + 아이콘)
# ---------------------------------------------------------
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬")

st.title("🎬 영화 유형 나누기")

# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df_raw = load_data()
total_count = len(df_raw)

# ---------------------------------------------------------
# 파생 변수 만들기
#   - log_scrn: 스크린 수 상용로그
#   - log_audi: 누적 관객 상용로그
#   - days_in_top10: 그대로 사용
#   - longrun_index: 누적 관객 / 첫 주 관객, 최대 20으로 자르기
# ---------------------------------------------------------
df = df_raw.copy()

# 첫 주 관객이 0이거나 결측이면 나눗셈이 불가능하므로 미리 제외 대상 표시
needed_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]

# 결측 제거
df = df.dropna(subset=needed_cols)
# 첫 주 관객이 0인 영화 제외
df = df[df["first_week_audi"] != 0]

# 로그 및 롱런 지수 계산 (스크린 수, 누적 관객이 0 이하인 경우도 로그 불가하므로 제외)
df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0)]

df["log_scrn"] = np.log10(df["first_scrn"])
df["log_audi"] = np.log10(df["total_audi"])
df["longrun_index"] = df["total_audi"] / df["first_week_audi"]
df["longrun_index"] = df["longrun_index"].clip(upper=20)

used_count = len(df)

st.write(f"전체 편수: {total_count}편 / 분석에 사용한 편수: {used_count}편")

# ---------------------------------------------------------
# 속성 선택 (체크박스 방식, 두 개 이상 선택)
# ---------------------------------------------------------
st.subheader("묶음에 사용할 속성 고르기")

# 화면에 보여줄 이름 <-> 실제 데이터 열 이름 매핑
feature_options = {
    "스크린 수(로그)": "log_scrn",
    "누적 관객(로그)": "log_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}

cols = st.columns(4)
selected_features = []
default_selected = list(feature_options.keys())  # 기본값: 네 개 모두

for i, (label, col_name) in enumerate(feature_options.items()):
    with cols[i]:
        checked = st.checkbox(label, value=True)
        if checked:
            selected_features.append(col_name)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해야 분석을 진행할 수 있어요. 체크박스를 다시 확인해 주세요.")
    st.stop()

# ---------------------------------------------------------
# 표준화 + K-평균 군집화 (난수 고정)
# ---------------------------------------------------------
X = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)
df["cluster_raw"] = raw_labels

# ---------------------------------------------------------
# 묶음 번호를 누적 관객 평균이 큰 순서로 ㉮, ㉯, ㉰ 로 다시 매기기
# ---------------------------------------------------------
cluster_order = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

label_map = {}
symbols = ["㉮", "㉯", "㉰"]
for rank, raw_c in enumerate(cluster_order):
    label_map[raw_c] = symbols[rank]

df["cluster"] = df["cluster_raw"].map(label_map)

# ---------------------------------------------------------
# 2차원 산점도
# ---------------------------------------------------------
st.subheader("2차원 산점도")

axis_options = list(feature_options.keys())
# selected_features에 해당하는 라벨만 추려서 축 후보로 사용
available_labels = [label for label, col in feature_options.items() if col in selected_features]

col_x, col_y = st.columns(2)
with col_x:
    x_label_2d = st.selectbox("가로축 속성", available_labels, index=0, key="x_2d")
with col_y:
    default_y_index = 1 if len(available_labels) > 1 else 0
    y_label_2d = st.selectbox("세로축 속성", available_labels, index=default_y_index, key="y_2d")

x_col_2d = feature_options[x_label_2d]
y_col_2d = feature_options[y_label_2d]

fig_2d = px.scatter(
    df,
    x=x_col_2d,
    y=y_col_2d,
    color="cluster",
    hover_name="movieNm",
    labels={x_col_2d: x_label_2d, y_col_2d: y_label_2d, "cluster": "유형"},
    category_orders={"cluster": symbols},
)
st.plotly_chart(fig_2d, use_container_width=True)

# ---------------------------------------------------------
# 3차원 산점도
# ---------------------------------------------------------
st.subheader("3차원 산점도")

if len(available_labels) < 3:
    st.info("3차원 산점도를 그리려면 속성을 세 개 이상 선택해야 해요.")
else:
    col_x3, col_y3, col_z3 = st.columns(3)
    with col_x3:
        x_label_3d = st.selectbox("X축 속성", available_labels, index=0, key="x_3d")
    with col_y3:
        idx_y = 1 if len(available_labels) > 1 else 0
        y_label_3d = st.selectbox("Y축 속성", available_labels, index=idx_y, key="y_3d")
    with col_z3:
        idx_z = 2 if len(available_labels) > 2 else 0
        z_label_3d = st.selectbox("Z축 속성", available_labels, index=idx_z, key="z_3d")

    x_col_3d = feature_options[x_label_3d]
    y_col_3d = feature_options[y_label_3d]
    z_col_3d = feature_options[z_label_3d]

    fig_3d = px.scatter_3d(
        df,
        x=x_col_3d,
        y=y_col_3d,
        z=z_col_3d,
        color="cluster",
        hover_name="movieNm",
        labels={
            x_col_3d: x_label_3d,
            y_col_3d: y_label_3d,
            z_col_3d: z_label_3d,
            "cluster": "유형",
        },
        category_orders={"cluster": symbols},
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# 묶음별 편수와 네 속성 평균 (원래 단위)
# ---------------------------------------------------------
st.subheader("묶음별 편수와 속성 평균 (원래 단위)")

# 원래 단위로 보여주기 위해 로그를 취하기 전 값 사용
df["scrn_original"] = df["first_scrn"]
df["audi_original"] = df["total_audi"]

summary = df.groupby("cluster").agg(
    편수=("movieNm", "count"),
    평균_스크린수=("scrn_original", "mean"),
    평균_누적관객=("audi_original", "mean"),
    평균_10위권일수=("days_in_top10", "mean"),
    평균_롱런지수=("longrun_index", "mean"),
)

# ㉮, ㉯, ㉰ 순서로 정렬
summary = summary.reindex(symbols)
st.dataframe(summary)

# ---------------------------------------------------------
# 묶음별 누적 관객 상위 5편 제목
# ---------------------------------------------------------
st.subheader("묶음별 누적 관객 상위 5편")

for symbol in symbols:
    st.markdown(f"**{symbol}**")
    top5 = (
        df[df["cluster"] == symbol]
        .sort_values("total_audi", ascending=False)
        .head(5)["movieNm"]
        .tolist()
    )
    for i, name in enumerate(top5, start=1):
        st.write(f"{i}. {name}")
