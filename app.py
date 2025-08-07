import streamlit as st
import pandas as pd

st.set_page_config(page_title="🏊‍♀️ 游泳成绩查询系统", layout="wide")

# 读取数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pool_data, on="PoolID", how="left")

# 填充 PoolName：如果缺失，则使用 City 作为 PoolName
data["PoolName"] = data["PoolName"].fillna(data["City"])

# 页面标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.subheader("🔍 请选择筛选条件")

# 筛选顺序
# Name -> Event -> LengthMeters -> PoolName -> City -> Date

# Name，默认 Anna
name_opts = ["All"] + sorted(data["Name"].dropna().unique().tolist())
name_sel = st.selectbox("Name", name_opts, index=name_opts.index("Anna") if "Anna" in name_opts else 0)

# Event
event_opts = ["All"] + sorted(data["Event"].dropna().unique().tolist())
event_sel = st.selectbox("Event", event_opts)

# LengthMeters
length_opts = ["All"] + sorted(data["LengthMeters"].dropna().astype(int).unique().tolist())
length_sel = st.selectbox("Length (Meters)", length_opts)

# PoolName
pool_opts = ["All"] + sorted(data["PoolName"].dropna().unique().tolist())
pool_sel = st.selectbox("Pool Name", pool_opts)

# City
city_opts = ["All"] + sorted(data["City"].dropna().unique().tolist())
city_sel = st.selectbox("City", city_opts)

# Date
date_opts = ["All"] + sorted(data["Date"].dropna().unique().tolist())
date_sel = st.selectbox("Date", date_opts)

# 多条件过滤
df = data.copy()
if name_sel != "All":
    df = df[df["Name"] == name_sel]
if event_sel != "All":
    df = df[df["Event"] == event_sel]
if length_sel != "All":
    df = df[df["LengthMeters"].astype(int) == length_sel]
if pool_sel != "All":
    df = df[df["PoolName"] == pool_sel]
if city_sel != "All":
    df = df[df["City"] == city_sel]
if date_sel != "All":
    df = df[df["Date"] == date_sel]

# 显示结果
st.markdown("### 🏅 比赛记录")
st.dataframe(df)

# 提供下载
st.download_button("📥 下载结果", df.to_csv(index=False).encode("utf-8-sig"), file_name="filtered_results.csv")
