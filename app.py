import streamlit as st
import pandas as pd
import plotly.express as px

# 加载数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pool_data, on="PoolID", how="left")

# 标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.markdown("## 🔍 请选择筛选条件")

# 多条件筛选（顺序：Name → Event → LengthMeters → PoolName → City → Date）
name_options = ["All"] + sorted(data["Name"].dropna().unique().tolist())
event_options = ["All"] + sorted(data["Event"].dropna().unique().tolist())
length_options = ["All"] + sorted(data["LengthMeters"].dropna().unique().tolist())
pool_options = ["All"] + sorted(data["PoolName"].dropna().unique().tolist())
city_options = ["All"] + sorted(data["City"].dropna().unique().tolist())
date_options = ["All"] + sorted(data["Date"].dropna().unique().tolist())

name_filter = st.selectbox("Name", name_options)
event_filter = st.selectbox("Event", event_options)
length_filter = st.selectbox("Length (Meters)", length_options)
pool_filter = st.selectbox("Pool Name", pool_options)
city_filter = st.selectbox("City", city_options)
date_filter = st.selectbox("Date", date_options)

# 应用筛选条件
filtered_data = data.copy()

if name_filter != "All":
    filtered_data = filtered_data[filtered_data["Name"] == name_filter]
if event_filter != "All":
    filtered_data = filtered_data[filtered_data["Event"] == event_filter]
if length_filter != "All":
    filtered_data = filtered_data[filtered_data["LengthMeters"] == length_filter]
if pool_filter != "All":
    filtered_data = filtered_data[filtered_data["PoolName"] == pool_filter]
if city_filter != "All":
    filtered_data = filtered_data[filtered_data["City"] == city_filter]
if date_filter != "All":
    filtered_data = filtered_data[filtered_data["Date"] == date_filter]

# 显示表格
st.markdown("## 比赛记录")
st.dataframe(filtered_data)

# 绘图（如果 Result 是时间格式）
try:
    filtered_data["Result_sec"] = pd.to_timedelta("00:" + filtered_data["Result"]).dt.total_seconds()
    fig = px.bar(
        filtered_data,
        x="Date",
        y="Result_sec",
        color="Event",
        hover_data=["Name", "PoolName", "City"],
        title="成绩趋势图（单位：秒）"
    )
    st.plotly_chart(fig)
except Exception as e:
    st.warning(f"图表生成失败：{e}")
