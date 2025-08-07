import streamlit as st
import pandas as pd
import plotly.express as px

# 设置页面标题
st.set_page_config(page_title="游泳成绩查询", layout="wide")

# 读取 CSV 文件，并清理列名空格
try:
    pool_data = pd.read_csv("Pools-data.csv")
    pool_data.columns = pool_data.columns.str.strip()
except Exception as e:
    st.error(f"加载 Pools-data.csv 出错: {e}")
    st.stop()

try:
    swim_data = pd.read_csv("swim-data.csv")
    swim_data.columns = swim_data.columns.str.strip()
except Exception as e:
    st.error(f"加载 swim-data.csv 出错: {e}")
    st.stop()

# 检查关键列是否存在
if "泳池ID" not in pool_data.columns or "泳池ID" not in swim_data.columns:
    st.error("两个文件中必须都包含 '泳池ID' 列，请检查列名是否正确。")
    st.stop()

# 合并数据
try:
    swim_data = swim_data.merge(pool_data, on="泳池ID", how="left")
except Exception as e:
    st.error(f"合并数据时出错: {e}")
    st.stop()

# 页面标题
st.title("🏊 游泳成绩查询系统")

# 下拉选项选择姓名
names = swim_data["姓名"].dropna().unique()
selected_name = st.selectbox("请选择选手姓名", names)

# 筛选数据
filtered = swim_data[swim_data["姓名"] == selected_name]

# 展示表格
st.subheader("比赛记录")
st.dataframe(filtered)

# 绘图（如果有数据）
if not filtered.empty:
    fig = px.bar(
        filtered,
        x="比赛日期",
        y="成绩",
        color="项目",
        barmode="group",
        title="比赛成绩（按时间）"
    )
    st.plotly_chart(fig)
else:
    st.info("没有找到该选手的比赛数据。")
