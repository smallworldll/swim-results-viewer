
import streamlit as st
import pandas as pd

st.set_page_config(page_title="🏊‍♀️ 游泳成绩查询系统", layout="wide")

# 读取数据
swim_data = pd.read_csv("swim-data.csv")
pool_data = pd.read_csv("Pools-data.csv")

# 合并数据
data = pd.merge(swim_data, pool_data, on="PoolID", how="left")

# 填充 PoolName：如果缺失，则使用 City
data["PoolName"] = data["PoolName"].fillna(data["City"])

# 页面标题
st.title("🏊‍♀️ 游泳成绩查询系统")
st.subheader("🔍 请选择筛选条件")

# 筛选顺序：Name -> Event -> LengthMeters -> PoolName -> City -> Date

# Name 多选
all_names = sorted(data["Name"].dropna().unique().tolist())
selected_names = st.multiselect("Name (可多选)", all_names, default=["Anna"])

# Event
event_opts = ["All"] + sorted(data["Event"].dropna().unique().tolist())
selected_event = st.selectbox("Event", event_opts)

# LengthMeters
length_opts = ["All"] + sorted(data["LengthMeters"].dropna().astype(int).unique().tolist())
selected_length = st.selectbox("Length (Meters)", length_opts)

# PoolName
pool_opts = ["All"] + sorted(data["PoolName"].dropna().unique().tolist())
selected_pool = st.selectbox("Pool Name", pool_opts)

# City
city_opts = ["All"] + sorted(data["City"].dropna().unique().tolist())
selected_city = st.selectbox("City", city_opts)

# Date
date_opts = ["All"] + sorted(data["Date"].dropna().unique().tolist())
selected_date = st.selectbox("Date", date_opts)

# 多条件过滤
df = data.copy()
if selected_names:
    df = df[df["Name"].isin(selected_names)]
if selected_event != "All":
    df = df[df["Event"] == selected_event]
if selected_length != "All":
    df = df[df["LengthMeters"].astype(str) == str(selected_length)]
if selected_pool != "All":
    df = df[df["PoolName"] == selected_pool]
if selected_city != "All":
    df = df[df["City"] == selected_city]
if selected_date != "All":
    df = df[df["Date"] == selected_date]

# 显示结果
st.markdown("### 🏅 比赛记录")
st.dataframe(df)

# 提供下载
st.download_button("📥 下载结果", df.to_csv(index=False).encode("utf-8-sig"), file_name="filtered_results.csv")

# 成绩折线图
st.subheader("📈 成绩折线图（单位：秒）")
if not df.empty and "Result" in df.columns:
    def parse_result(t):
        parts = t.split(":")
        if len(parts) == 3:
            return int(parts[0])*60 + int(parts[1]) + int(parts[2]) / 100
        elif len(parts) == 2:
            return int(parts[0])*60 + float(parts[1])
        else:
            return float(t)
    df["Seconds"] = df["Result"].apply(parse_result)
    chart_df = df.pivot(index="Date", columns="Name", values="Seconds")
    st.line_chart(chart_df)
else:
    st.info("暂无可绘制图表的数据或成绩列丢失")
