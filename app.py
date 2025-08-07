
import streamlit as st
import pandas as pd
import plotly.express as px

# 加载数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pool_data, on="PoolID", how="left")

# 设置页面标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.markdown("## 🔍 请选择筛选条件")

# 下拉筛选框顺序：name，event，泳池长度，泳池名称，城市，日期
name_options = ["All"] + sorted(data["Name"].dropna().unique().tolist())
name_filter = st.selectbox("Name", name_options, index=name_options.index("Anna") if "Anna" in name_options else 0)

event_options = ["All"] + sorted(data["Event"].dropna().unique().tolist())
event_filter = st.selectbox("Event", event_options)

length_options = ["All"] + sorted(data["LengthMeters"].dropna().astype(int).astype(str).unique().tolist())
length_filter = st.selectbox("Length (Meters)", length_options)

pool_options = ["All"] + sorted(data["PoolName"].dropna().unique().tolist())
pool_filter = st.selectbox("Pool Name", pool_options)

city_options = ["All"] + sorted(data["City"].dropna().unique().tolist())
city_filter = st.selectbox("City", city_options)

date_options = ["All"] + sorted(data["Date"].dropna().unique().tolist())
date_filter = st.selectbox("Date", date_options)

# 多条件筛选
filtered_data = data.copy()
if name_filter != "All":
    filtered_data = filtered_data[filtered_data["Name"] == name_filter]
if event_filter != "All":
    filtered_data = filtered_data[filtered_data["Event"] == event_filter]
if length_filter != "All":
    filtered_data = filtered_data[filtered_data["LengthMeters"].astype(int).astype(str) == length_filter]
if pool_filter != "All":
    filtered_data = filtered_data[filtered_data["PoolName"] == pool_filter]
if city_filter != "All":
    filtered_data = filtered_data[filtered_data["City"] == city_filter]
if date_filter != "All":
    filtered_data = filtered_data[filtered_data["Date"] == date_filter]

# 显示表格
st.markdown("## 比赛记录")
st.dataframe(filtered_data)

# 绘制图表
try:
    filtered_data["Seconds"] = filtered_data["Result"].apply(
        lambda x: int(x.split(":")[0]) * 60 + float(x.split(":")[1])
    )
    fig = px.bar(filtered_data, x="Date", y="Seconds", color="Event", title="比赛成绩图表")
    st.plotly_chart(fig)
except Exception as e:
    st.warning("图表生成失败：请确保成绩格式为 mm:ss 或 m:ss.fff")
