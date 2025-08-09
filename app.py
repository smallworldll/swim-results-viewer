
# -*- coding: utf-8 -*-
import os
import re
import json
from pathlib import Path
from datetime import date, datetime
import base64
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide")

ROOT = Path(".")
MEETS = ROOT / "meets"
MEETS.mkdir(exist_ok=True)

EVENT_PRESETS = [
    "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle",
    "25m Backstroke","50m Backstroke","100m Backstroke","200m Backstroke",
    "25m Breaststroke","50m Breaststroke","100m Breaststroke","200m Breaststroke",
    "25m Butterfly","50m Butterfly","100m Butterfly","200m Butterfly",
    "200m IM","400m IM"
]

def sanitize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)  # remove illegal for paths
    s = re.sub(r"\s+", " ", s)
    return s

def meet_folder_name(d: str, city: str, pool: str) -> str:
    return f"{d}_{sanitize(city)}_{sanitize(pool)}"

def fmt_time(sec: float) -> str:
    if pd.isna(sec):
        return ""
    m = int(sec // 60)
    s = sec - 60*m
    return f"{m}:{s:05.2f}"

def parse_time(txt: str) -> float:
    """Accept 'm:ss.xx' OR 'ss.xx' -> seconds as float"""
    if txt is None:
        return float("nan")
    t = str(txt).strip()
    if t == "" or t.lower() == "none":
        return float("nan")
    try:
        # m:ss.xx
        if ":" in t:
            m, s = t.split(":", 1)
            return float(m)*60 + float(s)
        # ss.xx
        return float(t)
    except Exception:
        return float("nan")

@st.cache_data(ttl=30)
def list_meets() -> list[Path]:
    return sorted([p for p in MEETS.iterdir() if p.is_dir()], key=lambda p: p.name)

def load_meta(meet_dir: Path) -> pd.Series | None:
    f = meet_dir / "meta.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if not {"Date","City","MeetName","PoolName","LengthMeters"}.issubset(df.columns):
        return None
    return df.iloc[0]

def save_meta(date_str, city, meetname, poolname, length, push: bool):
    folder = meet_folder_name(date_str, city, poolname)
    meet_dir = MEETS / folder
    meet_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "Date": date_str,
        "City": city,
        "MeetName": meetname,
        "PoolName": poolname,
        "LengthMeters": int(length),
    }])
    df.to_csv(meet_dir / "meta.csv", index=False, encoding="utf-8-sig")
    if push:
        ok, msg = push_to_github(str(meet_dir / "meta.csv"))
        if ok:
            st.success(f"已推送到 GitHub：{folder}/meta.csv")
        else:
            st.warning(f"GitHub 推送失败：{msg}")
    st.success(f"已保存： {meet_dir}/meta.csv")

def load_results(meet_dir: Path) -> pd.DataFrame:
    f = meet_dir / "results.csv"
    if not f.exists():
        return pd.DataFrame(columns=[
            "Name","EventName","Result","Seconds","Rank","Note","Date","City","MeetName","PoolName","LengthMeters"
        ])
    df = pd.read_csv(f)
    # Ensure Seconds exists & consistent
    if "Seconds" not in df.columns:
        df["Seconds"] = df["Result"].map(parse_time)
    df["Result"] = df["Seconds"].map(fmt_time)
    return df

def write_results(meet_dir: Path, df: pd.DataFrame, push: bool):
    df = df.copy()
    df["Seconds"] = df["Result"].map(parse_time)
    df["Result"] = df["Seconds"].map(fmt_time)
    df.to_csv(meet_dir / "results.csv", index=False, encoding="utf-8-sig")
    if push:
        ok, msg = push_to_github(str(meet_dir / "results.csv"))
        if ok:
            st.success("results.csv 已推送 GitHub")
        else:
            st.warning(f"GitHub 推送失败：{msg}")
    st.success("results.csv 已写入")

def push_to_github(local_path: str) -> tuple[bool, str]:
    # Requires st.secrets["GITHUB_TOKEN"] and st.secrets["REPO"]
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo = st.secrets["REPO"]
    except Exception:
        return False, "缺少 Secrets：GITHUB_TOKEN / REPO"
    rel = Path(local_path).as_posix().split("meets/")[-1]
    # store under meets/ in repo
    repo_path = f"meets/{rel}"
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    api = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    # get sha if exists
    r = requests.get(api, headers=headers)
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {
        "message": f"Save {repo_path}",
        "content": content_b64,
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    r2 = requests.put(api, headers=headers, data=json.dumps(payload))
    if r2.status_code in (200,201):
        return True, "ok"
    return False, f"{r2.status_code} {r2.text}"

# ---------------- UI ----------------
st.title("🏊 游泳成绩系统（赛事制）")

page = st.sidebar.radio("页面", ["查询/对比", "赛事管理/录入"], index=0)

if page == "赛事管理/录入":
    st.header("🗂️ 新建/选择赛事（meta）")
    col1, col2, col3, col4, col5 = st.columns([1,1,2,2,1])
    with col1:
        d = st.date_input("Date", value=date.today())
    with col2:
        city = st.text_input("City", value="Chiang Mai")
    with col3:
        meetname = st.text_input("MeetName", value="")
    with col4:
        poolname = st.text_input("PoolName", value="")
    with col5:
        length = st.selectbox("LengthMeters", [25,50], index=0)

    push_meta = st.checkbox("保存时推送到 GitHub", value=True)
    if st.button("保存赛事信息（写入/推送 meta.csv）", type="primary"):
        save_meta(d.strftime("%Y-%m-%d"), city, meetname, poolname, length, push_meta)

    st.header("📝 新增成绩（results）")
    all_meets = list_meets()
    if not all_meets:
        st.info("当前还没有任何赛事，请先保存一条 meta。")
    else:
        # 默认最近一次赛事（列表最后一个，因为已排序）
        meet_dir = st.selectbox("选择赛事文件夹", all_meets, index=len(all_meets)-1, format_func=lambda p: p.name)
        meta = load_meta(meet_dir)
        # 事件下拉 + 自定义
        event_default = EVENT_PRESETS[0]
        event_sel = st.selectbox("Event 选择", ["自定义…"] + EVENT_PRESETS, index=1)
        if event_sel == "自定义…":
            event_name = st.text_input("自定义 EventName", value="100m Freestyle")
        else:
            event_name = event_sel
        # 录入行数
        n = st.number_input("本次录入行数", 1, 20, 2, step=1)
        rows = []
        for i in range(n):
            st.markdown(f"**记录 {i+1}**")
            c1, c2, c3, c4 = st.columns([1,2,1,2])
            name = c1.text_input(f"Name_{i+1}", key=f"name_{i}")
            res = c2.text_input(f"Result_{i+1}", placeholder="0:34.12 或 34.12", key=f"res_{i}")
            rank = c3.number_input(f"Rank_{i+1}", 0, 999, 0, key=f"rank_{i}")
            note = c4.text_input(f"Note_{i+1}", value="", key=f"note_{i}")
            if name.strip() or res.strip():
                rows.append({
                    "Name": name.strip(),
                    "EventName": event_name.strip(),
                    "Result": fmt_time(parse_time(res)),
                    "Seconds": parse_time(res),
                    "Rank": int(rank),
                    "Note": note.strip(),
                    "Date": meta["Date"] if meta is not None else "",
                    "City": meta["City"] if meta is not None else "",
                    "MeetName": meta["MeetName"] if meta is not None else "",
                    "PoolName": meta["PoolName"] if meta is not None else "",
                    "LengthMeters": int(meta["LengthMeters"]) if meta is not None else (25 if "25" in meet_dir.name else 50),
                })
        push_res = st.checkbox("提交到 GitHub（免下载上传）", value=True)
        save_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=True)
        if st.button("保存这些成绩", type="primary"):
            df = load_results(meet_dir)
            if rows:
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            if save_local:
                write_results(meet_dir, df, push_res)
            else:
                # 只推 GitHub 也需要本地文件存在，仍写入本地再推送
                write_results(meet_dir, df, push_res)

        st.header("🧾 已登记记录（可编辑/删除）")
        df2 = load_results(meet_dir)
        if df2.empty:
            st.info("该赛事还没有 results.csv。")
        else:
            # 供删除用的勾选列
            df2 = df2.copy()
            df2["删除?"] = False
            edited = st.data_editor(
                df2,
                hide_index=True,
                column_config={"Seconds": st.column_config.NumberColumn(format="%.2f", help="自动计算，不用手改")},
                use_container_width=True,
                key="editor_existing"
            )
            cdl, csp = st.columns([1,3])
            if cdl.button("🗑️ 删除选中行 并保存"):
                keep = edited[edited["删除?"] != True].drop(columns=["删除?"], errors="ignore")
                write_results(meet_dir, keep, push_res)
            if csp.button("💾 保存更改（写入 results.csv）"):
                keep = edited.drop(columns=["删除?"], errors="ignore")
                write_results(meet_dir, keep, push_res)

else:
    st.header("🔎 游泳成绩查询 / 对比")
    # 读取所有 results.csv
    all_rows = []
    for d in list_meets():
        f = d / "results.csv"
        if f.exists():
            try:
                df = pd.read_csv(f)
                df["Seconds"] = df["Result"].map(parse_time) if "Seconds" not in df.columns else df["Seconds"]
                df["Result"] = df["Seconds"].map(fmt_time)
                all_rows.append(df)
            except Exception as e:
                st.warning(f"{d.name}/results.csv 读取失败：{e}")
    if not all_rows:
        st.info("当前没有成绩数据。请先在“赛事管理/录入”中添加。")
    else:
        data = pd.concat(all_rows, ignore_index=True)
        # 过滤
        names = sorted([x for x in data["Name"].dropna().unique()])
        events = sorted([x for x in data["EventName"].dropna().unique()])
        lengths = sorted([int(x) for x in data["LengthMeters"].dropna().unique()])
        sel_names = st.multiselect("Name（可多选）", names, default=names[:1] if names else [])
        sel_event = st.selectbox("Event", ["全部"] + events, index=0)
        sel_len = st.selectbox("Length (Meters)", ["全部"] + [str(x) for x in lengths], index=0)
        q = data.copy()
        if sel_names:
            q = q[q["Name"].isin(sel_names)]
        if sel_event != "全部":
            q = q[q["EventName"] == sel_event]
        if sel_len != "全部":
            q = q[q["LengthMeters"] == int(sel_len)]
        # 排序：按 Seconds 由小到大
        q = q.sort_values(by=["Seconds","Date"], ascending=[True, True], na_position="last")
        st.dataframe(q.reset_index(drop=True), use_container_width=True)
