
import os
import io
import json
import base64
import datetime as dt
from urllib.parse import quote
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide")

# -------------------------
# Config & constants
# -------------------------
MEETS_ROOT = "meets"
DEFAULT_EVENTS = [
    "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle",
    "25m Backstroke","50m Backstroke","100m Backstroke","200m Backstroke",
    "25m Breaststroke","50m Breaststroke","100m Breaststroke","200m Breaststroke",
    "25m Butterfly","50m Butterfly","100m Butterfly","200m Butterfly",
    "200m IM","400m IM","自定义…"
]

# Secrets
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
REPO = st.secrets.get("REPO", "")  # "user/repo"

# -------------------------
# Utilities
# -------------------------
def safe_fmt_time(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    seconds = float(seconds)
    m = int(seconds // 60)
    s = seconds - 60*m
    return f"{m}:{s:05.2f}"

def parse_time_to_seconds(text: str) -> float:
    """Accept 'm:ss.xx' or 'ss.xx' (or 'm:ss') and return seconds float."""
    if text is None:
        return float('nan')
    s = str(text).strip()
    if not s:
        return float('nan')
    try:
        if ":" in s:
            m, rest = s.split(":", 1)
            return float(m)*60 + float(rest)
        else:
            return float(s)
    except Exception:
        return float('nan')

def gh_headers():
    if not GITHUB_TOKEN:
        return {}
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

def gh_contents_url(path: str) -> str:
    # Encode each segment but keep / as slash
    parts = [quote(p, safe="") for p in path.split("/") if p != ""]
    encoded = "/".join(parts)
    return f"https://api.github.com/repos/{REPO}/contents/{encoded}"

def gh_get(path: str):
    """GET GitHub content (directory or file). Return (status, json)."""
    url = gh_contents_url(path)
    r = requests.get(url, headers=gh_headers(), timeout=20)
    try:
        data = r.json()
    except Exception:
        data = None
    return r.status_code, data

def gh_put_file(path: str, content_bytes: bytes, message: str):
    """Create or update a file on GitHub Contents API."""
    # Check existing to get sha
    sha = None
    status, data = gh_get(path)
    if status == 200 and isinstance(data, dict):
        sha = data.get("sha")

    url = gh_contents_url(path)
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8")
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=gh_headers(), data=json.dumps(payload), timeout=20)
    ok = (200 <= r.status_code < 300)
    return ok, r.status_code, r.text

def gh_read_csv(path: str) -> pd.DataFrame:
    """Read CSV from repo. Return empty DataFrame if not found."""
    status, data = gh_get(path)
    if status == 200 and isinstance(data, dict) and data.get("encoding") == "base64":
        try:
            raw = base64.b64decode(data.get("content","").encode("utf-8"))
            return pd.read_csv(io.BytesIO(raw))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def gh_listdir(path: str):
    status, data = gh_get(path)
    if status == 200 and isinstance(data, list):
        return [item["name"] for item in data if item.get("type") == "dir"]
    return []

def ensure_local(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# -------------------------
# Data IO for meets
# -------------------------
def meet_dir_name(date_str: str, city: str, poolname: str) -> str:
    return f"{date_str}_{city}_{poolname}"

def write_meta_to_repo(date_str, city, pool, length_m, meetname):
    cols = ["Date","City","MeetName","PoolName","LengthMeters"]
    df = pd.DataFrame([[date_str, city, meetname, pool, length_m]], columns=cols)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    folder = meet_dir_name(date_str, city, pool)
    ok, sc, txt = gh_put_file(f"{MEETS_ROOT}/{folder}/meta.csv", csv_bytes, f"Save meta for {folder}")
    return ok, sc, txt, folder

def append_results_to_repo(folder: str, rows_df: pd.DataFrame):
    """Append rows to results.csv (create if not exists)."""
    existing = gh_read_csv(f"{MEETS_ROOT}/{folder}/results.csv")
    if not existing.empty:
        all_df = pd.concat([existing, rows_df], ignore_index=True)
    else:
        all_df = rows_df.copy()
    csv_bytes = all_df.to_csv(index=False).encode("utf-8")
    ok, sc, txt = gh_put_file(f"{MEETS_ROOT}/{folder}/results.csv", csv_bytes, f"Update results for {folder}")
    return ok, sc, txt

def load_all_results_from_repo():
    """Return (df, message) where df has combined results from all meets."""
    if not GITHUB_TOKEN or not REPO:
        return pd.DataFrame(), "未配置 GitHub Secrets（GITHUB_TOKEN / REPO）。"
    meet_dirs = gh_listdir(MEETS_ROOT)
    if not meet_dirs:
        return pd.DataFrame(), "远程无 meets 或无法访问。"
    rows = []
    for d in meet_dirs:
        meta = gh_read_csv(f"{MEETS_ROOT}/{d}/meta.csv")
        results = gh_read_csv(f"{MEETS_ROOT}/{d}/results.csv")
        if results.empty and meta.empty:
            continue
        if results.empty:
            # still expose meet info
            tmp = meta.copy()
            tmp["Name"] = None
            tmp["EventName"] = None
            tmp["Result"] = None
            tmp["Rank"] = None
            tmp["Note"] = None
            results = tmp[["Name","Date","City","EventName","Result","Rank","Note","MeetName","PoolName","LengthMeters"]]
        # ensure columns
        needed = ["Name","Date","City","EventName","Result","Rank","Note","MeetName","PoolName","LengthMeters"]
        for c in needed:
            if c not in results.columns:
                results[c] = None
        results = results[needed]
        results["__folder"] = d
        rows.append(results)
    if not rows:
        return pd.DataFrame(), "未在远程找到有效赛事数据。"
    all_df = pd.concat(rows, ignore_index=True)
    return all_df, ""

# -------------------------
# UI Pages
# -------------------------
def page_browse():
    st.header("🏊‍♀️ 游泳成绩查询 / 对比")
    df, msg = load_all_results_from_repo()
    if msg:
        st.info(f"GitHub 同步：{msg}")

    if df.empty:
        st.write("当前没有成绩数据。请先在“赛事管理/录入”中添加。")
        return

    # parse times
    df["Seconds"] = df["Result"].apply(parse_time_to_seconds)
    df["ResultFmt"] = df["Seconds"].apply(safe_fmt_time)

    # Filters
    names = sorted([x for x in df["Name"].dropna().unique().tolist()])
    default_names = [n for n in ["Anna"] if n in names] or names[:1]
    pick_names = st.multiselect("Name（可多选）", options=names, default=default_names)
    event_opts = ["全部"] + sorted([x for x in df["EventName"].dropna().unique().tolist()])
    pick_event = st.selectbox("Event", options=event_opts, index=0)
    len_opts = ["全部"] + sorted([int(x) for x in pd.Series(df["LengthMeters"]).dropna().unique()], key=int)
    pick_len = st.selectbox("Length (Meters)", options=len_opts, index=0)

    q = df.copy()
    if pick_names:
        q = q[q["Name"].isin(pick_names)]
    if pick_event != "全部":
        q = q[q["EventName"] == pick_event]
    if pick_len != "全部":
        q = q[q["LengthMeters"].astype("Int64") == int(pick_len)]

    if q.empty:
        st.warning("没有符合条件的记录。")
        return

    # Best (seed) per (Name, EventName, LengthMeters)
    best = (
        q.dropna(subset=["Seconds"])
         .sort_values("Seconds")
         .groupby(["Name","EventName","LengthMeters"], as_index=False)
         .first()[["Name","EventName","LengthMeters","Seconds"]]
         .rename(columns={"Seconds":"BestSeconds"})
    )
    qq = q.merge(best, on=["Name","EventName","LengthMeters"], how="left")
    qq["Seed"] = (qq["Seconds"] == qq["BestSeconds"]).fillna(False)
    qq["SeedMark"] = qq["Seed"].apply(lambda x: "⭐" if bool(x) else "")

    view_cols = ["SeedMark","Name","Date","EventName","ResultFmt","Rank","City","PoolName","LengthMeters","Note"]
    st.dataframe(qq[view_cols].sort_values(["Name","EventName","LengthMeters","Date"]),
                 use_container_width=True, hide_index=True)

    # Line chart by time
    if pick_event != "全部":
        st.subheader("按时间的成绩曲线（折线）")
        plot_df = qq.dropna(subset=["Seconds"]).copy()
        # Convert Date to datetime for ordering
        try:
            plot_df["Date"] = pd.to_datetime(plot_df["Date"])
        except Exception:
            pass
        plot_df = plot_df.sort_values("Date")
        if not plot_df.empty:
            chart_df = plot_df.pivot_table(index="Date", columns="Name", values="Seconds", aggfunc="min")
            st.line_chart(chart_df, height=320)
            st.caption("注：数值单位为秒，越低越好。")

def page_manage():
    st.header("📁 赛事管理 / 成绩录入")

    st.markdown("#### ① 新建/选择赛事（meta）")
    with st.form("meta_form", clear_on_submit=False):
        c1, c2, c3, c4, c5 = st.columns([1,1,2,2,1])
        date_str = c1.text_input("Date", dt.date.today().isoformat())
        city = c2.text_input("City", "Chiang Mai")
        meetname = c3.text_input("MeetName", "Local Meet")
        poolname = c4.text_input("PoolName", "National Sports University Chiang Mai Campus")
        length_m = c5.number_input("LengthMeters", min_value=15, max_value=100, step=5, value=25)
        push_meta = st.checkbox("保存时推送到 GitHub", value=True)
        submitted = st.form_submit_button("保存赛事信息（写入/推送 meta.csv）")
    current_folder = ""
    if submitted:
        if not (GITHUB_TOKEN and REPO):
            st.error("未配置 GitHub Secrets（GITHUB_TOKEN / REPO）。")
        else:
            ok, sc, txt, folder = write_meta_to_repo(date_str, city, poolname, int(length_m), meetname)
            current_folder = folder
            if ok:
                st.success(f"已保存：{MEETS_ROOT}/{folder}/meta.csv")
            else:
                st.warning(f"GitHub 推送失败（{sc}）：{txt}")

    st.markdown("---")
    st.markdown("#### ② 新增成绩（results）")

    # Pick an existing meet folder
    meet_dirs = gh_listdir(MEETS_ROOT) if (GITHUB_TOKEN and REPO) else []
    folder = st.selectbox("选择赛事文件夹", options=meet_dirs, index=meet_dirs.index(current_folder) if current_folder in meet_dirs else (0 if meet_dirs else 0))

    # Event selector
    ev = st.selectbox("Event 选择", options=DEFAULT_EVENTS, index=DEFAULT_EVENTS.index("100m Butterfly") if "100m Butterfly" in DEFAULT_EVENTS else 0)
    if ev == "自定义…":
        ev = st.text_input("自定义 EventName", value="100m Freestyle")

    st.caption("时间格式可填 34.12 或 0:34.12（系统会统一解析为 m:ss.xx 显示）。")

    n = st.number_input("本次录入行数", min_value=1, max_value=8, value=2, step=1)
    rows = []
    for i in range(1, int(n)+1):
        st.markdown(f"**记录 {i}**")
        c1, c2, c3, c4, c5 = st.columns([1,2,1,1,2])
        name = c1.text_input(f"Name_{i}", value="Anna" if i==1 else "")
        eventname = c2.text_input(f"EventName_{i}", value=ev)
        result = c3.text_input(f"Result_{i}", value="0:34.12")
        rank = c4.number_input(f"Rank_{i}", min_value=0, step=1, value=0)
        note = c5.text_input(f"Note_{i}", value="")
        rows.append({"Name":name, "EventName":eventname, "Result":result, "Rank":rank, "Note":note})

    push = st.checkbox("提交到 GitHub（免下载上传）", value=True)
    also_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=False)

    if st.button("保存这些成绩"):
        if not folder:
            st.error("请先选择（或在上面创建）赛事文件夹。")
            return
        # Try read meta for this folder to fill meet info
        meta = gh_read_csv(f"{MEETS_ROOT}/{folder}/meta.csv")
        if meta.empty:
            st.warning("未找到该赛事的 meta.csv，仍将保存成绩，但缺少赛事信息。")

        # Build rows df
        valid = [r for r in rows if r["Name"].strip() and r["Result"].strip()]
        if not valid:
            st.warning("没有有效行。")
            return
        r_df = pd.DataFrame(valid)
        # normalize time
        r_df["Seconds"] = r_df["Result"].apply(parse_time_to_seconds)
        r_df["Result"] = r_df["Seconds"].apply(safe_fmt_time)
        r_df.drop(columns=["Seconds"], inplace=True)
        # enrich with meta
        if not meta.empty:
            r_df["Date"] = meta.iloc[0].get("Date")
            r_df["City"] = meta.iloc[0].get("City")
            r_df["MeetName"] = meta.iloc[0].get("MeetName")
            r_df["PoolName"] = meta.iloc[0].get("PoolName")
            r_df["LengthMeters"] = int(meta.iloc[0].get("LengthMeters"))
        else:
            # minimal columns
            for c in ["Date","City","MeetName","PoolName","LengthMeters"]:
                if c not in r_df.columns:
                    r_df[c] = None

        # Save locally if chosen
        if also_local:
            local_path = os.path.join(MEETS_ROOT, folder, "results.csv")
            ensure_local(local_path)
            try:
                if os.path.exists(local_path):
                    prev = pd.read_csv(local_path)
                    r_df = pd.concat([prev, r_df], ignore_index=True)
                r_df.to_csv(local_path, index=False)
                st.success(f"已写入本地：{local_path}")
            except Exception as e:
                st.warning(f"本地写入失败：{e}")

        if push and GITHUB_TOKEN and REPO:
            ok, sc, txt = append_results_to_repo(folder, r_df)
            if ok:
                st.success(f"已推送：{MEETS_ROOT}/{folder}/results.csv")
            else:
                st.warning(f"GitHub 推送失败：{sc} {txt}")

        # Show current merged preview
        st.markdown("#### ④ 当前项目成绩（预览与临时排名）")
        preview = gh_read_csv(f"{MEETS_ROOT}/{folder}/results.csv")
        if not preview.empty:
            st.dataframe(preview, use_container_width=True, hide_index=True)
        else:
            st.info("远程还没有 results.csv。")

# -------------------------
# Main
# -------------------------
PAGES = {
    "查询 / 对比": page_browse,
    "赛事管理 / 录入": page_manage
}

with st.sidebar:
    st.markdown("## 页面")
    choice = st.radio("", list(PAGES.keys()), index=0)

PAGES[choice]()
