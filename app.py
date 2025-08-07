
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="游泳成绩查询系统", layout="wide")

# 加载数据
swim_data = pd.read_csv("swim-data.csv")
pools_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pools_data, on="PoolID", how="left")

# 筛选选项顺序：Name -> Event -> Length -> Pool Name -> City -> Date
st.title("🏊‍♀️ 游泳成绩查询系统")
st.header("🔍 请选择筛选条件")

# 名字筛选，默认 Anna
name_options = ["All"] + sorted(data["Name"].dropna().unique().tolist())
selected_name = st.selectbox("Name", name_options, index=name_options.index("Anna") if "Anna" in name_options else 0)

# 项目筛选
event_options = ["All"] + sorted(data["Event"].dropna().unique().tolist())
selected_event = st.selectbox("Event", event_options)

# 泳池长度筛选（整数格式）
length_options = ["All"] + sorted(data["LengthMeters"].dropna().astype(int).unique().tolist())
selected_length = st.selectbox("Length (Meters)", length_options)

# 泳池名称筛选
poolname_options = ["All"] + sorted(data["PoolName"].dropna().unique().tolist())
selected_poolname = st.selectbox("Pool Name", poolname_options)

# 城市筛选
city_options = ["All"] + sorted(data["City"].dropna().unique().tolist())
selected_city = st.selectbox("City", city_options)

# 日期筛选
date_options = ["All"] + sorted(data["Date"].dropna().unique().tolist())
selected_date = st.selectbox("Date", date_options)

# 多条件过滤
filtered_data = data.copy()
if selected_name != "All":
    filtered_data = filtered_data[filtered_data["Name"] == selected_name]
if selected_event != "All":
    filtered_data = filtered_data[filtered_data["Event"] == selected_event]
if selected_length != "All":
    filtered_data = filtered_data[filtered_data["LengthMeters"].astype(int) == selected_length]
if selected_poolname != "All":
    filtered_data = filtered_data[filtered_data["PoolName"] == selected_poolname]
if selected_city != "All":
    filtered_data = filtered_data[filtered_data["City"] == selected_city]
if selected_date != "All":
    filtered_data = filtered_data[filtered_data["Date"] == selected_date]

st.subheader("🏅 比赛记录")
st.dataframe(filtered_data)

# 画图（如果数据格式允许）
try:
    def parse_time(t):
        parts = t.strip().split(":")
        if len(parts) == 3:
            return int(parts[0])*60 + int(parts[1]) + int(parts[2])/100
        elif len(parts) == 2:
            return int(parts[0])*60 + float(parts[1])
        else:
            return float(t)
    filtered_data["Seconds"] = filtered_data["Result"].apply(parse_time)
    fig, ax = plt.subplots()
    ax.plot(filtered_data["Date"], filtered_data["Seconds"], marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Time (Seconds)")
    ax.set_title("Performance Over Time")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"图表生成失败：{e}")
