
# -*- coding: utf-8 -*-
import os
import re
import io
import json
import base64
import glob
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None

# ------------------------
# Constants & helpers
# ------------------------
DATA_ROOT = "meets"

COMMON_EVENTS = [
    "25m Freestyle", "25m Backstroke", "25m Breaststroke", "25m Butterfly",
    "50m Freestyle", "50m Backstroke", "50m Breaststroke", "50m Butterfly",
    "100m Freestyle", "100m Backstroke", "100m Breaststroke", "100m Butterfly",
]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize(s: str) -> str:
    if s is None:
        return ""
    # replace slashes and extra spaces
    s = str(s).strip().replace("/", "-").replace("\\", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def meet_folder(date_str: str, city: str, pool: str) -> str:
    part = f"{date_str}_{sanitize(city)}_{sanitize(pool)}"
    return os.path.join(DATA_ROOT, part)

def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def write_csv_safe(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def parse_time_to_seconds(text: str) -> float:
    """Accept 'm:ss.xx' or 'ss.xx' or 'ss' and return seconds(float)."""
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s.lower() == "none":
        return None
    # m:ss.xx
    m = re.match(r"^\s*(\d+):(\d{1,2})(?:\.(\d{1,2}))?\s*$", s)
    if m:
        mm = int(m.group(1))
        ss = int(m.group(2))
        cs = m.group(3)
        frac = 0.0 if cs is None else float(f"0.{cs}")
        return mm*60 + ss + frac
    # ss.xx or ss
    m = re.match(r"^\s*(\d+)(?:\.(\d{1,2}))?\s*$", s)
    if m:
        ss = int(m.group(1))
        cs = m.group(2)
        frac = 0.0 if cs is None else float(f"0.{cs}")
        return ss + frac
    return None

def seconds_to_text(sec: float) -> str:
    if sec is None or pd.isna(sec):
        return ""
    sec = float(sec)
    m = int(sec // 60)
    s = sec - m*60
    return f"{m}:{s:05.2f}"

def load_all_results() -> pd.DataFrame:
    rows = []
    if not os.path.exists(DATA_ROOT):
        return pd.DataFrame()
    for d in sorted(os.listdir(DATA_ROOT)):
        p = os.path.join(DATA_ROOT, d)
        if not os.path.isdir(p):
            continue
        meta_path = os.path.join(p, "meta.csv")
        res_path = os.path.join(p, "results.csv")
        meta = read_csv_safe(meta_path)
        res = read_csv_safe(res_path)
        if meta.empty and res.empty:
            continue
        # Get meta row (1st or NaN)
        meta_row = meta.iloc[0] if not meta.empty else pd.Series()
        if res.empty:
            continue
        for _, r in res.iterrows():
            row = dict(r)
            # backfill from meta
            for k in ["Date", "City", "MeetName", "PoolName", "LengthMeters"]:
                if k not in row or pd.isna(row[k]) or row[k] in ("", None):
                    row[k] = meta_row.get(k, None)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        # normalize columns
        for col in ["Name","EventName","Result","Rank","Note","Date","City","MeetName","PoolName","LengthMeters"]:
            if col not in df.columns:
                df[col] = None
        # parse result seconds
        df["Seconds"] = df["Result"].apply(parse_time_to_seconds)
        # normalize Date to string for display
        df["Date"] = df["Date"].astype(str)
        # PB mark
        df["PB"] = False
        if not df["Seconds"].isna().all():
            gb = df.groupby(["Name","EventName","LengthMeters"])["Seconds"]
            best = gb.transform("min")
            df["PB"] = (df["Seconds"] == best)
    return df.sort_values(["Name","EventName","Date"])

def push_to_github_if_checked(file_path: str, rel_path: str):
    """If user checked push and secrets are available, push file via GitHub API."""
    if not st.session_state.get("push_github", False):
        return
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo = st.secrets.get("REPO", None)
    if not token or not repo:
        st.warning("未配置 GITHUB_TOKEN/REPO，已跳过推送。")
        return
    try:
        import requests
        with open(file_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")
        headers = {"Authorization": f"token {token}","Accept":"application/vnd.github+json"}
        url = f"https://api.github.com/repos/{repo}/contents/{rel_path}"
        # check if exists to get sha
        r = requests.get(url, headers=headers)
        sha = r.json().get("sha") if r.status_code == 200 else None
        data = {
            "message": f"Update {rel_path}",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            data["sha"] = sha
        r = requests.put(url, headers=headers, data=json.dumps(data))
        if r.status_code in (200,201):
            st.success(f"已推送到 GitHub: {rel_path}")
        else:
            st.warning(f"GitHub 推送失败（{r.status_code}）：{r.text[:300]}")
    except Exception as e:
        st.warning(f"推送异常：{e}")

# ------------------------
# UI Pages
# ------------------------
def page_query():
    st.header("🏊 游泳成绩查询 / 对比")
    df = load_all_results()
    if df.empty:
        st.info("当前没有成绩数据。请先在“赛事管理/成绩录入”中添加。")
        return

    # Filters
    names = sorted([x for x in df["Name"].dropna().unique().tolist() if str(x).strip()])
    default_names = ["Anna"] if "Anna" in names else None
    picked_names = st.multiselect("Name（可多选）", names, default=default_names)
    work = df.copy()
    if picked_names:
        work = work[work["Name"].isin(picked_names)]

    events = ["全部"] + sorted([x for x in work["EventName"].dropna().unique().tolist()])
    ev = st.selectbox("Event", events, index=0)
    if ev != "全部":
        work = work[work["EventName"] == ev]

    lengths = ["全部"] + [str(x) for x in sorted(work["LengthMeters"].dropna().unique().tolist())]
    ln = st.selectbox("Length (Meters)", lengths, index=0)
    if ln != "全部":
        work = work[work["LengthMeters"].astype(str) == ln]

    # Show table with PB mark
    show = work.copy()
    show["Seed"] = show["PB"].map(lambda x: "⭐" if x else "")
    show = show[["Name","Date","EventName","Result","Rank","Seed","City","PoolName","LengthMeters"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Line chart by date (Seconds)
    if alt is not None and not work.empty and work["Seconds"].notna().any():
        chart_df = work.copy()
        # parse Date to datetime if possible
        def parse_date(s):
            try:
                return pd.to_datetime(s)
            except Exception:
                return pd.NaT
        chart_df["DateDT"] = chart_df["Date"].apply(parse_date)
        chart_df = chart_df[chart_df["DateDT"].notna()]
        if not chart_df.empty:
            title = "成绩折线图（越低越好）"
            c = alt.Chart(chart_df).mark_line(point=True).encode(
                x="DateDT:T",
                y=alt.Y("Seconds:Q", title="Seconds (s)"),
                color="Name:N",
                tooltip=["Name","EventName","LengthMeters","Date","Result"]
            ).properties(height=300, title=title)
            st.altair_chart(c, use_container_width=True)

def page_manage():
    st.header("📁 赛事管理 / 成绩录入")

    ensure_dir(DATA_ROOT)

    with st.expander("① 新建/选择赛事（meta）", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            date_val = st.text_input("Date", value=dt.date.today().isoformat())
        with col2:
            city_val = st.text_input("City", value="Chiang Mai")
        with col3:
            meet_name = st.text_input("MeetName", value="Local Meet")

        pool_val = st.text_input("PoolName", value="National Sports University Chiang Mai Campus")
        length_val = st.number_input("LengthMeters", min_value=10, max_value=100, value=25, step=1)

        st.checkbox("保存时推送到 GitHub", key="push_github", value=False)

        if st.button("保存赛事信息（写入/推送 meta.csv）", type="primary"):
            folder = meet_folder(date_val, city_val, pool_val)
            ensure_dir(folder)
            meta = pd.DataFrame([{
                "Date": date_val,
                "City": city_val,
                "MeetName": meet_name,
                "PoolName": pool_val,
                "LengthMeters": length_val
            }])
            meta_path = os.path.join(folder, "meta.csv")
            write_csv_safe(meta, meta_path)
            st.success(f"已保存： {meta_path}")
            rel = os.path.relpath(meta_path, ".")
            push_to_github_if_checked(meta_path, rel)

    with st.expander("② 新增成绩（results）", expanded=True):
        # list meets
        meet_folders = [d for d in sorted(os.listdir(DATA_ROOT)) if os.path.isdir(os.path.join(DATA_ROOT, d))]
        if not meet_folders:
            st.info("还没有赛事，请先创建并保存 meta。")
            return
        meet_choice = st.selectbox("选择赛事文件夹", meet_folders, index=len(meet_folders)-1)
        folder = os.path.join(DATA_ROOT, meet_choice)
        meta = read_csv_safe(os.path.join(folder, "meta.csv"))
        meta_row = meta.iloc[0] if not meta.empty else pd.Series()

        # Event dropdown with common list
        event_default = COMMON_EVENTS[0]
        event_pick = st.selectbox("Event 选择", [""] + COMMON_EVENTS, index=COMMON_EVENTS.index(event_default)+1)

        col_a, col_b = st.columns(2)
        with col_a:
            row_n = st.number_input("本次录入行数", min_value=1, max_value=20, value=2, step=1)
        with col_b:
            st.caption("时间格式可填 34.12 或 0:34.12（系统统一解析为 m:ss.xx 显示）。")

        # Dynamic rows
        inputs = []
        for i in range(int(row_n)):
            st.markdown(f"**记录 {i+1}**")
            c1, c2, c3, c4, c5 = st.columns([1.3, 2.0, 1.1, 1.0, 2.0])
            with c1:
                name = st.text_input(f"Name_{i}", value="")
            with c2:
                ev = st.selectbox(f"EventName_{i}", ["自定义"] + COMMON_EVENTS,
                                  index=(COMMON_EVENTS.index(event_pick)+1 if event_pick in COMMON_EVENTS else 0))
                if ev == "自定义":
                    ev = st.text_input(f"CustomEvent_{i}", value="")
            with c3:
                res = st.text_input(f"Result_{i}", value="")
            with c4:
                rank = st.number_input(f"Rank_{i}", min_value=0, max_value=999, value=0, step=1)
            with c5:
                note = st.text_input(f"Note_{i}", value="")
            inputs.append((name, ev, res, rank, note))

        save_local = st.checkbox("同时保存到本地 meets/ 目录（测试用）", value=False)

        if st.button("保存这些成绩", type="primary"):
            res_path = os.path.join(folder, "results.csv")
            base = read_csv_safe(res_path)
            if base.empty:
                base = pd.DataFrame(columns=["Name","EventName","Result","Rank","Note",
                                             "Date","City","MeetName","PoolName","LengthMeters"])
            new_rows = []
            for name, ev, res, rank, note in inputs:
                if not str(name).strip() or not str(ev).strip():
                    continue
                sec = parse_time_to_seconds(res)
                res_text = seconds_to_text(sec) if sec is not None else res
                row = {
                    "Name": name.strip(),
                    "EventName": ev.strip(),
                    "Result": res_text,
                    "Rank": int(rank),
                    "Note": note.strip(),
                    "Date": meta_row.get("Date", ""),
                    "City": meta_row.get("City", ""),
                    "MeetName": meta_row.get("MeetName", ""),
                    "PoolName": meta_row.get("PoolName", ""),
                    "LengthMeters": meta_row.get("LengthMeters", "")
                }
                new_rows.append(row)
            if new_rows:
                base = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
                write_csv_safe(base, res_path)
                st.success(f"已保存 {len(new_rows)} 条到：{res_path}")
                if st.session_state.get("push_github", False):
                    push_to_github_if_checked(res_path, os.path.relpath(res_path, "."))
                elif save_local:
                    st.info("已保存到服务器临时存储（meets/）。")

    with st.expander("③ 已登记记录（可编辑/删除）", expanded=True):
        # choose meet folder again
        meet_folders = [d for d in sorted(os.listdir(DATA_ROOT)) if os.path.isdir(os.path.join(DATA_ROOT, d))]
        if not meet_folders:
            st.info("还没有赛事。")
            return
        meet_choice2 = st.selectbox("选择赛事查看/编辑", meet_folders, index=len(meet_folders)-1, key="choose_meet_view")
        folder2 = os.path.join(DATA_ROOT, meet_choice2)
        res_path2 = os.path.join(folder2, "results.csv")
        df = read_csv_safe(res_path2)
        if df.empty:
            st.info("该赛事目前没有成绩记录。")
        else:
            # show editable grid
            st.caption("可直接在表格里编辑。勾选想删除的行后点“保存更改”。")
            df["__delete__"] = False
            edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, hide_index=True)
            if st.button("保存更改（写入 results.csv）"):
                keep = edited[edited["__delete__"] == False].drop(columns=["__delete__"])
                # normalize time
                if "Result" in keep.columns:
                    keep["Result"] = keep["Result"].apply(lambda x: seconds_to_text(parse_time_to_seconds(x)))
                write_csv_safe(keep, res_path2)
                st.success("更改已保存。")
                if st.session_state.get("push_github", False):
                    push_to_github_if_checked(res_path2, os.path.relpath(res_path2, "."))

# ------------------------
# App
# ------------------------
st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide")

st.sidebar.markdown("### 页面")
page = st.sidebar.radio("页面", ["查询/对比", "赛事管理/录入"], index=0)

if page == "查询/对比":
    page_query()
else:
    page_manage()
