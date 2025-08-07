import streamlit as st
import pandas as pd

# 页面标题
st.set_page_config(page_title="🏊‍♀️ 游泳成绩查询系统")

# 标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.subheader("🔍 请选择筛选条件")

# 读取数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并泳池信息
data = pd.merge(swim_data, pool_data, how="left", on="PoolID")

# 筛选字段顺序
filter_order = ["Name", "Event", "LengthMeters", "PoolName", "City", "Date"]
data = data[filter_order]

# 默认值
default_name = "Anna"

# 下拉菜单选项
name_options = ["All"] + sorted(data["Name"].dropna().unique().tolist())
event_options = ["All"] + sorted(data["Event"].dropna().unique().tolist())
length_options = ["All"] + sorted(data["LengthMeters"].dropna().astype(int).astype(str).unique().tolist())
poolname_options = ["All"] + sorted(data["PoolName"].dropna().unique().tolist())
city_options = ["All"] + sorted(data["City"].dropna().unique().tolist())

# 筛选组件
name_filter = st.selectbox("Name", name_options, index=name_options.index(default_name) if default_name in name_options else 0)
event_filter = st.selectbox("Event", event_options)
length_filter = st.selectbox("Length (Meters)", length_options)
poolname_filter = st.selectbox("Pool Name", poolname_options)
city_filter = st.selectbox("City", city_options)

# 应用筛选
filtered_data = data.copy()
if name_filter != "All":
    filtered_data = filtered_data[filtered_data["Name"] == name_filter]
if event_filter != "All":
    filtered_data = filtered_data[filtered_data["Event"] == event_filter]
if length_filter != "All":
    filtered_data = filtered_data[filtered_data["LengthMeters"].astype(str) == length_filter]
if poolname_filter != "All":
    filtered_data = filtered_data[filtered_data["PoolName"] == poolname_filter]
if city_filter != "All":
    filtered_data = filtered_data[filtered_data["City"] == city_filter]

# 显示筛选结果
st.markdown("### 查询结果")
st.dataframe(filtered_data)

# 提供下载
st.download_button(
    label="📥 下载筛选结果",
    data=filtered_data.to_csv(index=False).encode("utf-8-sig"),
    file_name="filtered_results.csv",
    mime="text/csv"
)
