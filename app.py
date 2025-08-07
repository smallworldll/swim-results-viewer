
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    # 加载数据文件（你之后可以替换为GitHub上的原始文件链接）
    pools = pd.read_csv("Pools-data.csv").iloc[1:, :4]
    pools.columns = ["泳池ID", "名称", "地点", "长度（米）"]
    pools["泳池ID"] = pools["泳池ID"].astype(str).str.zfill(2)

    results = pd.read_csv("swim-data.csv").iloc[1:, :7]
    results.columns = ["姓名", "比赛日期", "泳池ID", "项目", "成绩", "排名", "备注"]
    results["比赛日期"] = pd.to_datetime(results["比赛日期"], errors="coerce")
    results["泳池ID"] = results["泳池ID"].astype(str).str.zfill(2)

    return results, pools

results_df, pools_df = load_data()

# 合并泳池信息
merged = pd.merge(results_df, pools_df, on="泳池ID", how="left")
merged = merged.sort_values("比赛日期", ascending=False)

st.title("🏊‍♀️ 游泳成绩查询系统")
st.markdown("本系统支持按选手、项目、时间、泳池进行筛选与对比。")

# 侧边栏筛选器
with st.sidebar:
    st.header("筛选条件")
    selected_swimmer = st.selectbox("选择选手", options=sorted(merged["姓名"].unique()))
    project_options = merged[merged["姓名"] == selected_swimmer]["项目"].unique()
    selected_project = st.selectbox("选择项目", options=project_options)
    date_range = st.date_input("选择日期范围", [])

# 应用筛选
filtered = merged[merged["姓名"] == selected_swimmer]
filtered = filtered[filtered["项目"] == selected_project]
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range)
    filtered = filtered[(filtered["比赛日期"] >= start_date) & (filtered["比赛日期"] <= end_date)]

# 显示表格
st.subheader(f"{selected_swimmer} 的 {selected_project} 成绩")
st.dataframe(filtered[["比赛日期", "名称", "地点", "长度（米）", "成绩", "排名", "备注"]], use_container_width=True)

# 成绩折线图
st.subheader("📈 成绩趋势图（单位：秒，越低越好）")
def time_to_seconds(t):
    try:
        parts = t.strip().split(":")
        if len(parts) == 3:
            m, s, cs = int(parts[0]), int(parts[1]), int(parts[2])
            return m * 60 + s + cs / 100
        elif len(parts) == 2:
            m, rest = int(parts[0]), float(parts[1])
            return m * 60 + rest
        else:
            return float(t)
    except:
        return None

filtered["成绩（秒）"] = filtered["成绩"].apply(time_to_seconds)
plot_data = filtered.dropna(subset=["成绩（秒）"])

if not plot_data.empty:
    st.line_chart(plot_data.set_index("比赛日期")["成绩（秒）"])
else:
    st.info("暂无有效成绩数据用于绘图。")
