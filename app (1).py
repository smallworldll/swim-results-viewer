
import streamlit as st
import pandas as pd
import plotly.express as px

# 加载数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pool_data, on="泳池ID", how="left")

# 筛选条件
st.title("🏊‍♀️ 游泳成绩查询系统")
st.markdown("### 请选择筛选条件")

col1, col2, col3 = st.columns(3)
with col1:
    selected_name = st.selectbox("姓名", options=["全部"] + sorted(data["姓名"].dropna().unique().tolist()))
with col2:
    selected_pool = st.selectbox("泳池名称", options=["全部"] + sorted(data["名称"].dropna().unique().tolist()))
with col3:
    selected_location = st.selectbox("地点", options=["全部"] + sorted(data["地点"].dropna().unique().tolist()))

col4, col5 = st.columns(2)
with col4:
    selected_event = st.selectbox("项目", options=["全部"] + sorted(data["项目"].dropna().unique().tolist()))
with col5:
    selected_date = st.selectbox("比赛日期", options=["全部"] + sorted(data["比赛日期"].dropna().unique().tolist()))

# 应用筛选条件
filtered_data = data.copy()
if selected_name != "全部":
    filtered_data = filtered_data[filtered_data["姓名"] == selected_name]
if selected_pool != "全部":
    filtered_data = filtered_data[filtered_data["名称"] == selected_pool]
if selected_location != "全部":
    filtered_data = filtered_data[filtered_data["地点"] == selected_location]
if selected_event != "全部":
    filtered_data = filtered_data[filtered_data["项目"] == selected_event]
if selected_date != "全部":
    filtered_data = filtered_data[filtered_data["比赛日期"] == selected_date]

# 显示筛选后的数据
st.markdown("### 比赛记录")
st.dataframe(filtered_data)

# 成绩图表
if not filtered_data.empty:
    st.markdown("### 比赛成绩（按时间）")
    try:
        filtered_data["成绩（秒）"] = pd.to_timedelta(filtered_data["成绩"]).dt.total_seconds()
        fig = px.bar(filtered_data, x="比赛日期", y="成绩（秒）", color="项目", barmode="group")
        st.plotly_chart(fig)
    except:
        st.warning("成绩格式有误，无法生成图表。")
else:
    st.info("没有符合条件的记录。")
