import streamlit as st
import pandas as pd
import plotly.express as px

# 读取数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pool_data, on="PoolID", how="left")

# 页面标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.markdown("## 🔍 请选择筛选条件")

# 筛选器
name_options = ["All"] + sorted(data["Name"].dropna().unique().tolist())
pool_options = ["All"] + sorted(data["PoolName"].dropna().unique().tolist())
event_options = ["All"] + sorted(data["Event"].dropna().unique().tolist())
city_options = ["All"] + sorted(data["City"].dropna().unique().tolist())
date_options = ["All"] + sorted(data["Date"].dropna().unique().tolist())
length_options = ["All"] + sorted(data["Length"].dropna().unique().tolist())

name_filter = st.selectbox("Name", name_options)
pool_filter = st.selectbox("Pool Name", pool_options)
event_filter = st.selectbox("Event", event_options)
city_filter = st.selectbox("City", city_options)
date_filter = st.selectbox("Date", date_options)
length_filter = st.selectbox("Pool Length", length_options)

# 应用筛选条件
if name_filter != "All":
    data = data[data["Name"] == name_filter]
if pool_filter != "All":
    data = data[data["PoolName"] == pool_filter]
if event_filter != "All":
    data = data[data["Event"] == event_filter]
if city_filter != "All":
    data = data[data["City"] == city_filter]
if date_filter != "All":
    data = data[data["Date"] == date_filter]
if length_filter != "All":
    data = data[data["Length"] == length_filter]

# 显示结果表格
st.markdown("### 比赛记录")
st.dataframe(data)

# 绘图：比赛成绩折线图（如果格式正确）
try:
    time_data = data.copy()
    time_data["Seconds"] = time_data["Result"].str.extract(r"(\d+):(\d+)\.?(\d*)")
    time_data[["Min", "Sec", "Ms"]] = data["Result"].str.extract(r"(\d+):(\d+):?(\d*)").fillna(0).astype(int)
    time_data["TotalSeconds"] = time_data["Min"] * 60 + time_data["Sec"] + time_data["Ms"] / 100
    fig = px.bar(time_data, x="Date", y="TotalSeconds", color="Event", barmode="group", title="比赛成绩（按时间）")
    st.plotly_chart(fig)
except Exception as e:
    st.warning(f"图表生成失败：{e}")
