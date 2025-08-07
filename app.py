
import streamlit as st
import pandas as pd
import plotly.express as px

# 加载数据
@st.cache_data
def load_data():
    swim_data = pd.read_csv("swim-data.csv")
    pool_data = pd.read_csv("Pools-data.csv")
    return swim_data, pool_data

swim_data, pool_data = load_data()

# 合并泳池信息
swim_data = swim_data.merge(pool_data, left_on="泳池ID", right_on="泳池ID", how="left")

# 页面标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.write("本系统支持按选手、项目、时间、泳池进行筛选与对比。")

# 筛选条件
with st.sidebar:
    st.header("筛选条件")
    selected_swimmer = st.selectbox("选择选手", sorted(swim_data["姓名"].unique()))
    selected_event = st.selectbox("选择项目", sorted(swim_data["项目"].unique()))
    date_range = st.date_input("选择日期范围", [])

# 过滤数据
df = swim_data[(swim_data["姓名"] == selected_swimmer) & (swim_data["项目"] == selected_event)]
if len(date_range) == 2:
    df = df[(df["比赛日期"] >= str(date_range[0])) & (df["比赛日期"] <= str(date_range[1]))]

# 成绩转换为秒数
def time_to_seconds(t):
    try:
        parts = list(map(int, t.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        else:
            return float(t)
    except:
        return None

df["成绩_秒"] = df["成绩"].apply(time_to_seconds)

# 展示表格
st.subheader(f"{selected_swimmer} 的 {selected_event} 成绩")
st.dataframe(df[["比赛日期", "名称", "地点", "长度（米）", "成绩", "排名", "备注"]])

# 趋势图
fig = px.line(df.sort_values("比赛日期"), x="比赛日期", y="成绩_秒", markers=True,
              title="📉 成绩趋势图（单位：秒，越低越好）")
st.plotly_chart(fig)
