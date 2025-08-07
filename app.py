import streamlit as st
import pandas as pd
import plotly.express as px

# 加载数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pool_data, on="PoolID", how="left")

st.title("🏊‍♀️ 游泳成绩查询系统")
st.markdown("### 🔍 请选择筛选条件")

# 筛选控件
name_filter = st.selectbox("Name", ["All"] + sorted(data["Name"].dropna().unique().tolist()))
pool_filter = st.selectbox("Pool Name", ["All"] + sorted(data["PoolName"].dropna().unique().tolist()))
event_filter = st.selectbox("Event", ["All"] + sorted(data["Event"].dropna().unique().tolist()))
city_filter = st.selectbox("City", ["All"] + sorted(data["City"].dropna().unique().tolist()))
date_filter = st.selectbox("Date", ["All"] + sorted(data["Date"].dropna().unique().tolist()))

# 应用筛选
filtered_data = data.copy()
if name_filter != "All":
    filtered_data = filtered_data[filtered_data["Name"] == name_filter]
if pool_filter != "All":
    filtered_data = filtered_data[filtered_data["PoolName"] == pool_filter]
if event_filter != "All":
    filtered_data = filtered_data[filtered_data["Event"] == event_filter]
if city_filter != "All":
    filtered_data = filtered_data[filtered_data["City"] == city_filter]
if date_filter != "All":
    filtered_data = filtered_data[filtered_data["Date"] == date_filter]

# 显示数据表
st.subheader("比赛记录")
st.dataframe(filtered_data)

# 绘制图表（如果成绩列存在）
if "Result" in filtered_data.columns and not filtered_data["Result"].isnull().all():
    try:
        filtered_data["Result_seconds"] = pd.to_timedelta("00:" + filtered_data["Result"]).dt.total_seconds()
        fig = px.bar(filtered_data, x="Date", y="Result_seconds", color="Event", barmode="group",
                     labels={"Result_seconds": "成绩（秒）"})
        st.subheader("比赛成绩图表（秒）")
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"图表生成失败：{e}")
