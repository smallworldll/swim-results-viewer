
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
import json
import requests
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="游泳成绩查询系统", page_icon="🏊‍♀️", layout="wide")

# -----------------------------
# Constants
# -----------------------------
MEETS_DIR = Path("meets")
TIME_REGEX = re.compile(r"^\d{1,2}:\d{2}\.\d{2}$")  # m:ss.xx  (e.g., 1:02.45)

EVENT_CHOICES = [
    "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle",
    "25m Backstroke","50m Backstroke","100m Backstroke",
    "25m Breaststroke","50m Breaststroke","100m Breaststroke",
    "25m Butterfly","50m Butterfly","100m Butterfly",
    "100m Individual Medley","200m Individual Medley"
]


# -----------------------------
# Utility
# -----------------------------
def parse_time_str_to_seconds(t: str) -> float:
    """Only accept m:ss.xx format, return seconds as float. Raise ValueError if invalid."""
    if not isinstance(t, str):
        raise ValueError("time must be string m:ss.xx")
    t = t.strip()
    if not TIME_REGEX.match(t):
        raise ValueError("format must be m:ss.xx (e.g., 1:02.45)")
    # parse
    m, rest = t.split(":")
    s, cs = rest.split(".")
    total = int(m) * 60 + int(s) + int(cs) / 100.0
    return total


def seconds_to_time_str(sec: float) -> str:
    if pd.isna(sec):
        return ""
    m = int(sec // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{m}:{s:02d}.{cs:02d}"


def read_all_meets() -> pd.DataFrame:
    """Read meets directory structure only. Return flat dataframe."""
    rows = []
    if not MEETS_DIR.exists():
        return pd.DataFrame(columns=[
            "Date","City","MeetName","PoolName","LengthMeters",
            "Name","EventName","Result","Rank","Note","Seconds","MeetFolder"
        ])
    for folder in sorted(MEETS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        meta = folder / "meta.csv"
        results = folder / "results.csv"
        if not meta.exists() or not results.exists():
            continue
        try:
            meta_df = pd.read_csv(meta, dtype={"LengthMeters":"Int64"})
            # meta expected: Date, City, MeetName, PoolName, LengthMeters
            if len(meta_df) == 0:
                continue
            mrow = meta_df.iloc[0].to_dict()
            res_df = pd.read_csv(results)
            if res_df.empty:
                continue
            res_df["Date"] = mrow.get("Date","")
            res_df["City"] = mrow.get("City","")
            res_df["MeetName"] = mrow.get("MeetName","")
            res_df["PoolName"] = mrow.get("PoolName","")
            res_df["LengthMeters"] = mrow.get("LengthMeters",pd.NA)
            res_df["MeetFolder"] = folder.name

            # strict parse: m:ss.xx -> Seconds
            def _to_sec(x):
                try:
                    return parse_time_str_to_seconds(str(x))
                except Exception:
                    return np.nan

            res_df["Seconds"] = res_df["Result"].map(_to_sec)
            rows.append(res_df)
        except Exception as e:
            st.warning(f"读取 {folder.name} 出错：{e}")
            continue
    if not rows:
        return pd.DataFrame(columns=[
            "Date","City","MeetName","PoolName","LengthMeters",
            "Name","EventName","Result","Rank","Note","Seconds","MeetFolder"
        ])
    df = pd.concat(rows, ignore_index=True)
    # reorder columns
    cols = ["Name","Date","City","MeetName","PoolName","LengthMeters","EventName","Result","Rank","Note","Seconds","MeetFolder"]
    df = df.loc[:, cols]
    return df


def seed_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for seed (best) results within Name+EventName+LengthMeters groups."""
    mask = pd.Series(False, index=df.index)
    if df.empty:
        return mask
    # smaller Seconds is better
    grouped = df.groupby(["Name","EventName","LengthMeters"], dropna=False)["Seconds"]
    mins = grouped.transform("min")
    mask = df["Seconds"] == mins
    return mask


# --------------- GitHub helpers ---------------
def _gh_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

def github_put_file(token: str, repo: str, path: str, content_bytes: bytes, commit_message: str):
    """Create or update a file via GitHub API."""
    import base64, requests
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    # Check if file exists to get sha
    r = requests.get(url, headers=_gh_headers(token))
    sha = None
    if r.status_code == 200:
        sha = r.json().get("sha")
    payload = {
        "message": commit_message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    r2 = requests.put(url, headers=_gh_headers(token), json=payload, timeout=30)
    if r2.status_code not in (200,201):
        raise RuntimeError(f"GitHub API error {r2.status_code}: {r2.text}")


# -----------------------------
# Pages
# -----------------------------
def page_browse():
    st.header("📊 查询与对比")

    data = read_all_meets()

    if data.empty:
        st.info("meets/ 目录为空。先到『新增比赛』页录入吧。")
        return

    # Filters
    with st.expander("筛选条件", expanded=True):
        names = sorted(data["Name"].dropna().unique().tolist())
        selected_names = st.multiselect("Name（可多选）", names, default=names[:1])

        events = sorted(data["EventName"].dropna().unique().tolist())
        event = st.selectbox("Event", ["All"] + events)

        lengths = sorted([int(x) for x in data["LengthMeters"].dropna().unique().tolist()])
        length = st.selectbox("Length (Meters)", ["All"] + lengths)

        pools = sorted(data["PoolName"].dropna().unique().tolist())
        pool = st.selectbox("Pool Name", ["All"] + pools)

        cities = sorted(data["City"].dropna().unique().tolist())
        city = st.selectbox("City", ["All"] + cities)

        dates = sorted(data["Date"].dropna().unique().tolist())
        date = st.selectbox("Date", ["All"] + dates)

    df = data.copy()
    if selected_names:
        df = df[df["Name"].isin(selected_names)]
    if event != "All":
        df = df[df["EventName"] == event]
    if length != "All":
        df = df[df["LengthMeters"] == int(length)]
    if pool != "All":
        df = df[df["PoolName"] == pool]
    if city != "All":
        df = df[df["City"] == city]
    if date != "All":
        df = df[df["Date"] == date]

    df = df.sort_values(["Name","EventName","LengthMeters","Date"]).reset_index(drop=True)

    # seed highlight
    seeds = seed_mask(df)

    # Prepare styled table
    def style_fn(row):
        c = ""
        if row["__seed__"]:
            # 红色字体标注种子成绩
            c = "color: #d90429; font-weight: 700;"
        return [c] * len(row)

    if df.empty:
        st.warning("没有匹配的记录。")
        return

    disp = df.drop(columns=["Seconds"])
    disp["__seed__"] = seeds.values
    styled = disp.style.apply(style_fn, axis=1).hide(axis="index")
    st.subheader("🏅 比赛记录")
    st.dataframe(styled, use_container_width=True)

    # Line chart (seconds, lower is better)
    st.subheader("📈 成绩趋势（单位：秒，越低越好）")
    # for line chart, need x=datetime; create a sortable timestamp
    # If Date is string, attempt to parse to timestamp else use index order
    try:
        ts = pd.to_datetime(df["Date"])
    except Exception:
        ts = pd.to_datetime(df["Date"], errors='coerce')
    chart_df = pd.DataFrame({
        "Date": ts,
        "Seconds": df["Seconds"].astype(float),
        "Name": df["Name"].astype(str),
        "EventName": df["EventName"].astype(str),
        "Length": df["LengthMeters"].astype(str)
    }).dropna(subset=["Seconds","Date"])

    if chart_df.empty:
        st.info("无有效时间数据可绘图。")
    else:
        # Pivot for multiple series by Name (and optionally Event if user picks All events)
        # We'll group by Name when single event; if All, group by (Name, EventName)
        if event != "All":
            chart_df["Series"] = chart_df["Name"]
        else:
            chart_df["Series"] = chart_df["Name"] + " · " + chart_df["EventName"]
        # Draw lines
        import altair as alt
        line = alt.Chart(chart_df).mark_line(point=True).encode(
            x='Date:T', y='Seconds:Q',
            color='Series:N',
            tooltip=['Series','Date:T','Seconds:Q','Length:N']
        ).interactive()
        st.altair_chart(line, use_container_width=True)

    # CSV download
    st.download_button(
        "📥 下载筛选结果 CSV",
        df.drop(columns=["Seconds"]).to_csv(index=False).encode("utf-8"),
        file_name="filtered_results.csv",
        mime="text/csv"
    )


def page_new_meet():
    st.header("📝 新增比赛")
    with st.form("meta_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            date = st.date_input("Date", value=None, format="YYYY-MM-DD")
        with c2:
            city = st.text_input("City")
        with c3:
            length = st.selectbox("LengthMeters", [25, 50], index=0)
        meet_name = st.text_input("MeetName")
        pool_name = st.text_input("PoolName")

        st.markdown("---")
        st.markdown("#### 成绩记录（可添加多条）")
        if "entry_rows" not in st.session_state:
            st.session_state.entry_rows = 1

        add_col, rm_col = st.columns([1,1])
        with add_col:
            if st.form_submit_button("➕ 增加一条记录", use_container_width=True):
                st.session_state.entry_rows += 1
                st.stop()
        with rm_col:
            if st.form_submit_button("➖ 删除最后一条", use_container_width=True):
                st.session_state.entry_rows = max(1, st.session_state.entry_rows - 1)
                st.stop()

        entries = []
        for i in range(st.session_state.entry_rows):
            st.markdown(f"**记录 {i+1}**")
            cc1, cc2, cc3, cc4, cc5 = st.columns([1.2,1.5,1,0.7,1.2])
            with cc1:
                name = st.text_input(f"Name_{i}", key=f"name_{i}")
            with cc2:
                ev = st.selectbox(f"EventName_{i}", EVENT_CHOICES, key=f"event_{i}")
            with cc3:
                res = st.text_input(f"Result_{i}（m:ss.xx）", key=f"result_{i}", placeholder="1:02.45")
            with cc4:
                rank = st.number_input(f"Rank_{i}", key=f"rank_{i}", min_value=0, step=1, value=0)
            with cc5:
                note = st.text_input(f"Note_{i}", key=f"note_{i}", placeholder="可留空")
            entries.append({"Name": name, "EventName": ev, "Result": res, "Rank": int(rank), "Note": note})

        st.markdown("---")
        push_github = st.checkbox("提交到 GitHub（免下载上传）", value=True)
        also_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=False)

        submitted = st.form_submit_button("生成/提交", use_container_width=True)

    if not submitted:
        return

    # Validate meta
    if not date or not city or not meet_name or not pool_name:
        st.error("请完整填写『Date / City / MeetName / PoolName』。")
        return

    # Validate results & time format
    valid_entries = []
    errs = []
    for idx, rec in enumerate(entries, start=1):
        if not rec["Name"] or not rec["EventName"] or not rec["Result"]:
            errs.append(f"第 {idx} 条记录缺少 Name / EventName / Result")
            continue
        try:
            _ = parse_time_str_to_seconds(rec["Result"])
        except Exception as e:
            errs.append(f"第 {idx} 条记录时间格式错误：{rec['Result']}（需要 m:ss.xx）")
            continue
        valid_entries.append(rec)
    if errs:
        st.error("；".join(errs))
        return
    if not valid_entries:
        st.error("没有有效记录。")
        return

    # Folder path
    folder = f"{date.isoformat()}_{city}"
    folder_path = MEETS_DIR / folder
    meta_csv = folder_path / "meta.csv"
    results_csv = folder_path / "results.csv"

    # Create CSV contents
    meta_df = pd.DataFrame([{
        "Date": date.isoformat(),
        "City": city,
        "MeetName": meet_name,
        "PoolName": pool_name,
        "LengthMeters": int(length)
    }])

    results_df = pd.DataFrame(valid_entries, columns=["Name","EventName","Result","Rank","Note"])

    # Save locally if requested
    if also_local:
        folder_path.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(meta_csv, index=False)
        results_df.to_csv(results_csv, index=False)
        st.success(f"已写入本地： {meta_csv} , {results_csv}")

    # Push to GitHub if requested and secrets available
    if push_github:
        token = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("REPO")
        if not token or not repo:
            st.error("缺少 Secrets：['GITHUB_TOKEN','REPO']。请在 Streamlit Cloud - App - Settings - Secrets 配置。")
            return
        # upload meta.csv and results.csv
        try:
            path_meta = f"{MEETS_DIR}/{folder}/meta.csv"
            path_results = f"{MEETS_DIR}/{folder}/results.csv"
            github_put_file(token, repo, path_meta, meta_df.to_csv(index=False).encode("utf-8"),
                            f"[streamlit] add/update meta for {folder}")
            github_put_file(token, repo, path_results, results_df.to_csv(index=False).encode("utf-8"),
                            f"[streamlit] add/update results for {folder}")
            st.success(f"已提交到 GitHub：{path_meta}, {path_results}")
        except Exception as e:
            st.error(f"推送 GitHub 失败：{e}")
            return

    st.balloons()
    st.info("提交完成！可切换到『查询与对比』页面查看。")


# -----------------------------
# Main
# -----------------------------
PAGES = {
    "查询与对比": page_browse,
    "新增比赛": page_new_meet
}

def main():
    st.title("🏊‍♀️ 游泳成绩查询系统")
    page = st.sidebar.radio("功能", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    main()
