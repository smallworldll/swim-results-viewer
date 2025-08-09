# -*- coding: utf-8 -*-
import os
from pathlib import Path
from urllib.parse import quote
import base64
import json
import re
from datetime import datetime
import io

import requests
import pandas as pd
import streamlit as st

APP_TITLE = "🏊‍♀️ 游泳成绩系统（赛事制）"

# --------------------------
# Utils: time format
# --------------------------
TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2})(?:\.(\d{1,2}))?\s*$")

def parse_time_to_seconds(s: str) -> float | None:
    """
    Accepts 'm:ss.xx', 'ss.xx', 'm:ss', 'ss' and returns seconds (float).
    Returns None if empty or invalid.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    m = TIME_RE.match(s)
    if not m:
        return None
    mm = m.group(1)
    ss = int(m.group(2))
    xx = m.group(3)
    minutes = int(mm) if mm else 0
    frac = int(xx) / (10 ** len(xx)) if xx else 0.0
    return minutes * 60 + ss + frac

def format_seconds_to_time(sec: float | int | None) -> str:
    if sec is None:
        return ""
    try:
        sec = float(sec)
    except Exception:
        return ""
    if pd.isna(sec):
        return ""
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:05.2f}"

# --------------------------
# GitHub helpers
# --------------------------
def _get_secret(name: str, default: str | None = None) -> str | None:
    # streamlit secrets if available
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)

def gh_headers() -> dict:
    token = _get_secret("GITHUB_TOKEN", "")
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

def gh_repo_parts():
    repo = _get_secret("REPO", "")
    if not repo or "/" not in repo:
        return None, None
    owner, name = repo.split("/", 1)
    return owner, name

def gh_contents_url(path: str, branch: str = "main") -> str:
    owner, name = gh_repo_parts()
    if not owner:
        return ""
    # IMPORTANT: encode path but keep slashes
    enc = quote(path, safe="/")
    return f"https://api.github.com/repos/{owner}/{name}/contents/{enc}?ref={branch}"

def gh_put_file(path: str, content_bytes: bytes, message: str, branch: str="main") -> tuple[bool, str]:
    """
    Create or update a file in GitHub repo using contents API.
    """
    h = gh_headers()
    if not h:
        return False, "缺少 GITHUB_TOKEN 或 REPO"
    url = gh_contents_url(path, branch=branch)
    if not url:
        return False, "REPO 配置不正确"

    # Probe SHA if exists
    sha = None
    r_get = requests.get(url, headers=h)
    if r_get.status_code == 200:
        try:
            sha = r_get.json().get("sha")
        except Exception:
            sha = None
    elif r_get.status_code not in (200, 404):
        return False, f"读取远程文件失败: {r_get.status_code} {r_get.text}"

    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r_put = requests.put(url, headers=h, json=payload)
    if r_put.status_code in (200, 201):
        return True, "已推送至 GitHub"
    return False, f"推送失败: {r_put.status_code} {r_put.text}"

def gh_sync_meets_to_local(local_root: Path) -> str:
    """
    Best-effort: list 'meets' dir in GitHub and mirror down (meta.csv & results.csv).
    """
    h = gh_headers()
    owner, name = gh_repo_parts()
    if not h or not owner:
        return "未配置 GitHub，同步跳过"

    base_url = f"https://api.github.com/repos/{owner}/{name}/contents/meets?ref=main"
    r = requests.get(base_url, headers=h)
    if r.status_code != 200:
        return f"远程无 meets 或无法访问: {r.status_code}"

    local_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in r.json():
        if item.get("type") != "dir":
            continue
        sub_name = item.get("name")
        sub_url = item.get("url")
        rr = requests.get(sub_url, headers=h)
        if rr.status_code != 200:
            continue
        for file_item in rr.json():
            if file_item.get("type") != "file":
                continue
            fname = file_item.get("name")
            if fname not in ("meta.csv", "results.csv"):
                continue
            download_url = file_item.get("download_url")
            if not download_url:
                # Fallback via contents API to fetch base64
                fr = requests.get(file_item.get("url"), headers=h)
                if fr.status_code == 200:
                    data = fr.json().get("content", "")
                    if data:
                        content = base64.b64decode(data)
                        local_dir = local_root / sub_name
                        local_dir.mkdir(parents=True, exist_ok=True)
                        with open(local_dir / fname, "wb") as f:
                            f.write(content)
                            count += 1
                continue
            # Download raw
            fr = requests.get(download_url, headers=h)
            if fr.status_code == 200:
                local_dir = local_root / sub_name
                local_dir.mkdir(parents=True, exist_ok=True)
                with open(local_dir / fname, "wb") as f:
                    f.write(fr.content)
                    count += 1
    return f"已同步 {count} 个文件"

# --------------------------
# IO helpers
# --------------------------
MEETS_ROOT = Path("meets")

def meet_folder_name(date_str: str, city: str, pool: str) -> str:
    # keep long pool name as requested
    clean = lambda s: str(s).strip().replace("/", "_")
    return f"{clean(date_str)}_{clean(city)}_{clean(pool)}"

def ensure_meet_dir(meta: pd.Series) -> Path:
    folder = MEETS_ROOT / meet_folder_name(meta["Date"], meta["City"], meta["PoolName"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def write_meta(meta: pd.Series, push: bool) -> tuple[Path, list[str]]:
    msgs = []
    folder = ensure_meet_dir(meta)
    meta_path = folder / "meta.csv"
    meta.to_frame().T.to_csv(meta_path, index=False)
    msgs.append(f"已写入本地： {meta_path.as_posix()}")
    if push:
        ok, msg = gh_put_file(
            f"{folder.as_posix()}/meta.csv",
            meta.to_frame().T.to_csv(index=False).encode("utf-8"),
            f"Add/Update meta for {folder.name}"
        )
        msgs.append(msg)
    return folder, msgs

def read_meta(folder: Path) -> pd.Series | None:
    p = folder / "meta.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        return df.iloc[0]
    except Exception:
        return None

def read_results(folder: Path) -> pd.DataFrame:
    p = folder / "results.csv"
    if not p.exists():
        return pd.DataFrame(columns=["Name","EventName","Result","Rank","Note","Date","City","PoolName","LengthMeters"])
    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.DataFrame(columns=["Name","EventName","Result","Rank","Note","Date","City","PoolName","LengthMeters"])
    # ensure columns
    for col in ["Name","EventName","Result","Rank","Note","Date","City","PoolName","LengthMeters"]:
        if col not in df.columns:
            df[col] = None
    return df

def write_results(folder: Path, df: pd.DataFrame, push: bool) -> list[str]:
    msgs = []
    p = folder / "results.csv"
    df.to_csv(p, index=False)
    msgs.append(f"已写入本地： {p.as_posix()}")
    if push:
        ok, msg = gh_put_file(
            f"{folder.as_posix()}/results.csv",
            df.to_csv(index=False).encode("utf-8"),
            f"Add/Update results for {folder.name}"
        )
        msgs.append(msg)
    return msgs

# --------------------------
# UI Components
# --------------------------
COMMON_EVENTS = [
    "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle",
    "25m Breaststroke","50m Breaststroke","100m Breaststroke",
    "25m Backstroke","50m Backstroke","100m Backstroke",
    "25m Butterfly","50m Butterfly","100m Butterfly",
]

def page_query():
    st.subheader("🏊‍♀️ 游泳成绩查询 / 对比")

    # Try sync from GitHub (once per session)
    if st.session_state.get("synced") is None:
        msg = gh_sync_meets_to_local(MEETS_ROOT)
        st.session_state["synced"] = True
        st.info(f"GitHub 同步：{msg}")

    # Aggregate all results
    all_records = []
    for sub in sorted(MEETS_ROOT.glob("*")):
        if not sub.is_dir():
            continue
        meta = read_meta(sub)
        if meta is None:
            continue
        res = read_results(sub)
        if not res.empty:
            all_records.append(res)
    if all_records:
        data = pd.concat(all_records, ignore_index=True)
    else:
        st.info("当前没有成绩数据。请先在“赛事管理/成绩录入”中添加。")
        return

    # Filters
    names = sorted([x for x in data["Name"].dropna().unique().tolist()])
    events = ["全部"] + sorted([x for x in data["EventName"].dropna().unique().tolist()])
    lengths = ["全部"] + sorted([int(x) for x in pd.to_numeric(data["LengthMeters"], errors="coerce").dropna().unique().tolist()])

    sel_names = st.multiselect("Name（可多选）", names, default=names[:1] if names else [])
    sel_event = st.selectbox("Event", events, index=0)
    sel_length = st.selectbox("Length (Meters)", lengths, index=0)

    df = data.copy()
    if sel_names:
        df = df[df["Name"].isin(sel_names)]
    if sel_event != "全部":
        df = df[df["EventName"] == sel_event]
    if sel_length != "全部":
        df = df[pd.to_numeric(df["LengthMeters"], errors="coerce") == int(sel_length)]

    if df.empty:
        st.warning("没有匹配的记录。")
        return

    # Seed highlighting per (Name, EventName, LengthMeters)
    df["_sec"] = df["Result"].apply(parse_time_to_seconds)
    best = (
        df.groupby(["Name","EventName","LengthMeters"])["_sec"]
        .transform("min")
    )
    df["_is_seed"] = (df["_sec"] == best)

    # Display
    disp = df[["Name","Date","EventName","Result","Rank","Note","PoolName","City","LengthMeters"]].copy()
    def style_row(row):
        styles = []
        if row.get("_is_seed"):
            styles = [""]*disp.shape[1]
            # highlight Result cell (index 3)
            styles[3] = "color:#d00; font-weight:700;"
            return styles
        return [""]*disp.shape[1]

    styled = disp.style.apply(lambda x: [ "color:#d00; font-weight:700;" if df.loc[x.index, "_is_seed"].iloc[0] else "" ]*len(x), axis=1)
    # The above approach can be unstable; simpler per column:
    def highlight_seed(s):
        mask = df["_is_seed"].values
        return ['color:#d00; font-weight:700;' if m else '' for m in mask]
    styled = disp.style.apply(highlight_seed, subset=["Result"])

    st.dataframe(styled, use_container_width=True)

    # Line chart when a single event chosen
    if sel_event != "全部":
        # sort by date per person
        chart_df = df[["Name","Date","_sec"]].dropna().copy()
        if not chart_df.empty:
            # ensure date sortable
            chart_df["Date"] = pd.to_datetime(chart_df["Date"], errors="coerce")
            chart_df = chart_df.dropna(subset=["Date"]).sort_values("Date")
            # pivot
            pivot = chart_df.pivot_table(index="Date", columns="Name", values="_sec", aggfunc="min")
            if not pivot.empty:
                st.line_chart(pivot, height=260, use_container_width=True)
                st.caption("注：纵轴为秒（s），值越低代表速度越快。")

def page_manage():
    st.subheader("📁 赛事管理 / 成绩录入")

    # Section 1: 创建或选择赛事
    st.markdown("### ① 选择或新建赛事")
    colm1, colm2 = st.columns(2)
    with colm1:
        mode = st.radio("操作", ["选择已有赛事", "新建赛事"], horizontal=True)

    if mode == "新建赛事":
        date = st.date_input("Date", value=datetime.today())
        city = st.text_input("City", value="Chiang Mai")
        pool = st.text_input("PoolName", value="National Sports University Chiang Mai Campus")
        length = st.selectbox("LengthMeters", [25, 50], index=0)

        meta = pd.Series({
            "Date": date.strftime("%Y-%m-%d"),
            "City": city.strip(),
            "MeetName": st.text_input("MeetName", value="Swimming Championship"),
            "PoolName": pool.strip(),
            "LengthMeters": int(length)
        })
        push_meta = st.checkbox("提交到 GitHub（免下载上传）", value=True)
        if st.button("保存赛事信息"):
            folder, msgs = write_meta(meta, push=push_meta)
            for m in msgs:
                st.success(m)
            st.session_state["current_folder"] = folder.as_posix()
    else:
        # Select existing local meets; offer sync from GitHub
        if st.button("🔄 从 GitHub 同步到本地"):
            msg = gh_sync_meets_to_local(MEETS_ROOT)
            st.info(msg)

        options = [p for p in sorted(MEETS_ROOT.glob("*")) if p.is_dir()]
        labels = [p.name for p in options]
        idx = st.selectbox("选择赛事（文件夹）", list(range(len(labels))), format_func=lambda i: labels[i] if labels else "无", index=0 if labels else None)
        cur_folder = None
        if labels:
            cur_folder = options[idx]
            st.session_state["current_folder"] = cur_folder.as_posix()
            meta = read_meta(cur_folder)
            if meta is not None:
                st.dataframe(meta.to_frame().T, use_container_width=True)
            else:
                st.warning("该赛事缺少 meta.csv")

    # If we have a current folder, show result editor
    cur_path = Path(st.session_state.get("current_folder", "")) if st.session_state.get("current_folder") else None
    if not cur_path:
        return

    # Read current meta & results
    meta = read_meta(cur_path)
    if meta is None:
        st.warning("请先保存赛事信息（meta.csv）。")
        return
    results_df = read_results(cur_path)

    st.markdown("### ② 新增成绩")
    # Event selection with dropdown & custom
    use_custom = st.checkbox("自定义项目", value=False)
    if use_custom:
        event_template = st.text_input("EventName（自定义）", value="100m Freestyle")
    else:
        event_template = st.selectbox("Event 选择", COMMON_EVENTS, index=COMMON_EVENTS.index("100m Freestyle"))

    rows = st.number_input("本次录入行数", min_value=1, max_value=20, value=2, step=1)

    input_rows = []
    for i in range(rows):
        st.markdown(f"**记录 {i+1}**")
        c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
        name = c1.text_input(f"Name_{i+1}", value="Anna" if i == 0 else "", key=f"name_{i}")
        ev = c2.text_input(f"EventName_{i+1}", value=event_template, key=f"ev_{i}")
        result = c3.text_input(f"Result_{i+1}", placeholder="如 34.12 或 0:34.12", key=f"res_{i}")
        rank = int(c4.number_input(f"Rank_{i+1}", min_value=0, max_value=999, value=0, step=1, key=f"rank_{i}"))
        note = st.text_input(f"Note_{i+1}", value="", key=f"note_{i}")
        input_rows.append((name, ev, result, rank, note))

    colsave1, colsave2 = st.columns(2)
    push = colsave1.checkbox("提交到 GitHub（免下载上传）", value=True)
    save_local = colsave2.checkbox("同时保存到本地 meets/ 目录（调试用）", value=True)

    if st.button("保存这些成绩"):
        new_rows = []
        for name, ev, res, rank, note in input_rows:
            name = str(name).strip()
            ev = str(ev).strip()
            res = str(res).strip()
            if not name or not ev or not res:
                continue
            # normalize time
            sec = parse_time_to_seconds(res)
            if sec is None:
                continue
            res_fmt = format_seconds_to_time(sec)
            new_rows.append({
                "Name": name,
                "EventName": ev,
                "Result": res_fmt,
                "Rank": rank,
                "Note": note,
                "Date": meta["Date"],
                "City": meta["City"],
                "PoolName": meta["PoolName"],
                "LengthMeters": meta["LengthMeters"],
            })
        if not new_rows:
            st.warning("没有有效的新记录。")
        else:
            new_df = pd.DataFrame(new_rows)
            out_df = pd.concat([results_df, new_df], ignore_index=True)
            # always write meta first (in case user仅创建赛事就离开)
            _, msgs1 = write_meta(meta, push=push)
            msgs2 = write_results(cur_path, out_df, push=push)
            st.success(f"已保存 {len(new_rows)} 条。")
            for m in msgs1 + msgs2:
                st.info(m)

    st.markdown("### ③ 该赛事已录入成绩")
    results_df = read_results(cur_path)
    if results_df.empty:
        st.info("暂无成绩记录。")
    else:
        st.dataframe(results_df, use_container_width=True)

        # 简单删除功能
        st.markdown("**批量删除（勾选后保存）**")
        results_df["_del"] = False
        edited = st.data_editor(results_df, num_rows="dynamic", use_container_width=True, key="editor")
        if st.button("保存修改（删除打勾的行）"):
            keep_df = edited[edited["_del"] != True].drop(columns=["_del"], errors="ignore")
            msgs = write_results(cur_path, keep_df, push=True)
            st.success("已保存修改")
            for m in msgs:
                st.info(m)

def main():
    st.set_page_config(page_title="Swim Results", layout="wide")
    st.title(APP_TITLE)

    page = st.sidebar.radio("页面", ["查询/对比", "赛事管理/录入"], index=0)

    if page == "查询/对比":
        page_query()
    else:
        page_manage()

if __name__ == "__main__":
    main()
