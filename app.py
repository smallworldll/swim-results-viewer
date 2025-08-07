
import streamlit as st
import pandas as pd
import plotly.express as px

# 读取数据，并将“泳池ID”列设置为字符串，避免浮点错误
swim_data = pd.read_csv("swim-data.csv", dtype={"泳池ID": str})
pool_data = pd.read_csv("Pools-data.csv", dtype={"泳池ID": str})

# 合并数据
data = pd.merge(swim_data, pool_data, on="泳池ID", how="left")

# 页面标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.markdown("### 🔍 请选择筛选条件")

# 多项筛选
col1, col2, col3 = st.columns(3)
with col1:
    name_filter = st.selectbox("姓名", ["全部"] + sorted(data["姓名"].dropna().unique().tolist()))
with col2:
    pool_filter = st.selectbox("泳池名称", ["全部"] + sorted(data["名称"].dropna().unique().tolist()))
with col3:
    event_filter = st.selectbox("项目", ["全部"] + sorted(data["项目"].dropna().unique().tolist()))

col4, col5 = st.columns(2)
with col4:
    location_filter = st.selectbox("地点", ["全部"] + sorted(data["地点"].dropna().unique().tolist()))
with col5:
    date_filter = st.selectbox("比赛日期", ["全部"] + sorted(data["比赛日期"].dropna().unique().tolist()))

# 应用筛选条件
filtered_data = data.copy()
if name_filter != "全部":
    filtered_data = filtered_data[filtered_data["姓名"] == name_filter]
if pool_filter != "全部":
    filtered_data = filtered_data[filtered_data["名称"] == pool_filter]
if event_filter != "全部":
    filtered_data = filtered_data[filtered_data["项目"] == event_filter]
if location_filter != "全部":
    filtered_data = filtered_data[filtered_data["地点"] == location_filter]
if date_filter != "全部":
    filtered_data = filtered_data[filtered_data["比赛日期"] == date_filter]

# 显示结果表格
st.markdown("### 📋 比赛记录")
st.dataframe(filtered_data)

# 绘制成绩图表
if not filtered_data.empty:
    st.markdown("### 📈 比赛成绩（按时间）")
    fig = px.bar(filtered_data, x="项目", y="成绩", color="项目", text="成绩")
    st.plotly_chart(fig)
else:
    st.warning("未找到符合条件的比赛记录。")
