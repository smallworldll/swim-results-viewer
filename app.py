
# -*- coding: utf-8 -*-
import os
import re
import base64
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st
import requests

APP_TITLE = "🏊‍♀️ 游泳成绩系统（赛事制）"
MEETS_ROOT = Path("meets")


# ------------------------
# Utilities
# ------------------------

def ensure_meets_root() -> Path:
    MEETS_ROOT.mkdir(parents=True, exist_ok=True)
    return MEETS_ROOT


def slugify(s: str) -> str:
    """Safe-ish for folder part (keep spaces and letters)."""
    s = s.strip().replace("/", "-").replace("\\", "-")
    return re.sub(r"\s+", " ", s)


def parse_time_to_seconds(s: str) -> Optional[float]:
    """
    Accepts 'm:ss.xx', 'ss.xx', 'm:ss', 'ss' and returns seconds (float).
    Returns None if cannot parse.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s:
        return None
    # normalize comma decimal
    s = s.replace(",", ".")
    # m:ss.xx ?
    m = re.match(r"^(\d+):(\d{1,2})(?:\.(\d{1,2}))?$", s)
    if m:
        mins = int(m.group(1))
        secs = int(m.group(2))
        hund = int(m.group(3)) if m.group(3) else 0
        return mins * 60 + secs + hund / (10 ** len(m.group(3)) if m.group(3) else 1)
    # ss.xx ?
    m = re.match(r"^(\d+)(?:\.(\d{1,2}))?$", s)
    if m:
        secs = int(m.group(1))
        hund = int(m.group(2)) if m.group(2) else 0
        return secs + hund / (10 ** len(m.group(2)) if m.group(2) else 1)
    return None


def format_seconds(sec: Optional[float]) -> str:
    if sec is None or pd.isna(sec):
        return ""
    sec = float(sec)
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:05.2f}"  # m:ss.xx


def list_meet_dirs() -> List[Path]:
    ensure_meets_root()
    dirs = [p for p in MEETS_ROOT.iterdir() if p.is_dir()]
    # sort by date desc if possible
    def key(p: Path):
        try:
            parts = p.name.split("_")
            return datetime.strptime(parts[0], "%Y-%m-%d")
        except Exception:
            return datetime.min
    return sorted(dirs, key=key, reverse=True)


def build_meet_folder_name(d: date, city: str, pool_name: str) -> str:
    return f"{d.strftime('%Y-%m-%d')}_{slugify(city)}_{slugify(pool_name)}"


def read_meta(folder: Path) -> pd.Series:
    meta_fp = folder / "meta.csv"
    if meta_fp.exists():
        df = pd.read_csv(meta_fp)
        if not df.empty:
            return df.iloc[0]
    # default empty
    return pd.Series({"Date": "", "City": "", "MeetName": "", "PoolName": "", "LengthMeters": ""})


def write_meta(folder: Path, meta: pd.Series):
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([meta])
    df.to_csv(folder / "meta.csv", index=False)


def read_results(folder: Path) -> pd.DataFrame:
    res_fp = folder / "results.csv"
    if res_fp.exists():
        try:
            df = pd.read_csv(res_fp)
        except Exception:
            df = pd.read_csv(res_fp, encoding="utf-8-sig")
    else:
        df = pd.DataFrame(columns=["Name", "EventName", "Result", "Rank", "Note"])
    return df


def write_results(folder: Path, df: pd.DataFrame):
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "results.csv", index=False)


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[name]
    except Exception:
        return default


def github_upsert_file(repo: str, path: str, content_bytes: bytes, message: str) -> Tuple[bool, str]:
    """
    Create or update a file in GitHub repo via REST API.
    repo: 'owner/repo'
    path: repo path (e.g., 'meets/.../meta.csv')
    """
    token = get_secret("GITHUB_TOKEN")
    if not token or not repo:
        return False, "Missing GITHUB_TOKEN or REPO secret."

    base = "https://api.github.com"
    url = f"{base}/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    # Check existing sha
    sha = None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        try:
            sha = r.json().get("sha")
        except Exception:
            sha = None
    elif r.status_code not in (404,):
        return False, f"GET for SHA failed: {r.status_code} {r.text}"

    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, data=json.dumps(payload))
    if r.status_code in (200, 201):
        return True, "OK"
    else:
        return False, f"{r.status_code} {r.text}"


def push_file_if_checked(meet_folder: Path, fname: str, content_bytes: bytes, commit_msg: str, push: bool) -> Optional[str]:
    if not push:
        return None
    repo = get_secret("REPO", "")
    rel_path = str((meet_folder / fname).as_posix())
    ok, msg = github_upsert_file(repo, rel_path, content_bytes, commit_msg)
    return f"GitHub 推送：{msg}"


def event_list() -> List[str]:
    return [
        "25m Freestyle", "50m Freestyle", "100m Freestyle", "200m Freestyle", "400m Freestyle",
        "25m Backstroke", "50m Backstroke", "100m Backstroke",
        "25m Breaststroke", "50m Breaststroke", "100m Breaststroke",
        "25m Butterfly", "50m Butterfly", "100m Butterfly",
        "200m IM"
    ]


# ------------------------
# UI helpers
# ------------------------

def section_new_meet():
    st.subheader("① 新建/选择赛事（meta）")

    today = date.today()
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 2, 1])
    with col1:
        the_date = st.date_input("Date", value=today, format="YYYY-MM-DD")
    with col2:
        city = st.text_input("City", value="Chiang Mai")
    with col3:
        meet_name = st.text_input("MeetName", value="")
    with col4:
        pool_name = st.text_input("PoolName", value="")
    with col5:
        length = st.selectbox("LengthMeters", options=[25, 50], index=0)

    push = st.checkbox("保存时推送到 GitHub", value=True)

    if st.button("保存赛事信息（写入/推送 meta.csv）", use_container_width=True):
        folder_name = build_meet_folder_name(the_date, city, pool_name or "Pool")
        folder = ensure_meets_root() / folder_name
        meta = pd.Series({
            "Date": the_date.strftime("%Y-%m-%d"),
            "City": city,
            "MeetName": meet_name,
            "PoolName": pool_name,
            "LengthMeters": int(length),
        })
        write_meta(folder, meta)

        # push meta if enabled
        msg = push_file_if_checked(folder, "meta.csv", pd.DataFrame([meta]).to_csv(index=False).encode("utf-8"), f"Save meta for {folder_name}", push)
        st.success(f"已保存： {folder / 'meta.csv'}")
        if msg:
            st.info(msg)


def section_add_results():
    st.subheader("② 新增成绩（results）")

    dirs = list_meet_dirs()
    if not dirs:
        st.info("还没有赛事，请先在上面新建并保存。")
        return

    # 默认选择最近的赛事
    meet_folder = st.selectbox("选择赛事文件夹", options=dirs, format_func=lambda p: p.name, index=0)
    meta = read_meta(meet_folder)

    # 顶部选择 Event（行内不再重复）
    event_options = ["（自定义…）"] + event_list()
    event_choice = st.selectbox("Event 选择", options=event_options, index=1 if len(event_options) > 1 else 0)
    if event_choice == "（自定义…）":
        event_name = st.text_input("自定义 EventName", value="", placeholder="例如：100m Freestyle")
    else:
        event_name = event_choice

    st.caption("时间格式可填 34.12 或 0:34.12（系统会统一解析为 m:ss.xx 显示）。")

    n = st.number_input("本次录入行数", min_value=1, max_value=20, value=2, step=1)

    rows = []
    for i in range(1, n + 1):
        st.markdown(f"**记录 {i}**")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            name = st.text_input(f"Name_{i}", key=f"name_{i}")
        with c2:
            result = st.text_input(f"Result_{i}", key=f"result_{i}")
        with c3:
            rank = st.number_input(f"Rank_{i}", min_value=0, value=0, step=1, key=f"rank_{i}")
        with c4:
            note = st.text_input(f"Note_{i}", key=f"note_{i}")
        if name.strip() and event_name.strip() and result.strip():
            sec = parse_time_to_seconds(result)
            rows.append({
                "Name": name.strip(),
                "EventName": event_name.strip(),
                "Result": format_seconds(sec) if sec is not None else result.strip(),
                "Rank": int(rank),
                "Note": note.strip(),
            })
        st.divider()

    push = st.checkbox("提交到 GitHub（免下载上传）", value=True)
    also_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=False)

    if st.button("保存这些成绩", type="primary"):
        if not rows:
            st.warning("没有有效记录（需要至少填写 Name、Event、Result）。")
            return
        df_old = read_results(meet_folder)
        df_new = pd.DataFrame(rows, columns=["Name", "EventName", "Result", "Rank", "Note"])
        df = pd.concat([df_old, df_new], ignore_index=True)
        # 保存
        write_results(meet_folder, df)
        st.success(f"已保存 {len(df_new)} 条。")

        if push:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            msg = push_file_if_checked(meet_folder, "results.csv", csv_bytes, f"Add results for {meet_folder.name}", push=True)
            if msg:
                st.info(msg)
        if also_local:
            st.info(f"文件已写入：{meet_folder / 'results.csv'}")

    # ③ 已登记记录（可编辑/删除）
    st.subheader("③ 已登记记录（可编辑/删除）")
    # 选择比赛查看/编辑（默认当前选择）
    folder_for_edit = st.selectbox("选择赛事查看/编辑", options=dirs, index=dirs.index(meet_folder), format_func=lambda p: p.name)
    df_view = read_results(folder_for_edit).copy()

    # 附带元信息，便于观察（只展示，不保存这些列）
    mt = read_meta(folder_for_edit)
    if not df_view.empty:
        df_view["Date"] = mt.get("Date", "")
        df_view["City"] = mt.get("City", "")
        df_view["MeetName"] = mt.get("MeetName", "")
        df_view["PoolName"] = mt.get("PoolName", "")
        df_view["LengthMeters"] = mt.get("LengthMeters", "")

    edited_df = st.data_editor(
        df_view,
        use_container_width=True,
        num_rows="dynamic",
        key=f"editor_{folder_for_edit.name}",
        column_order=["Name", "EventName", "Result", "Rank", "Note", "Date", "City", "MeetName", "PoolName", "LengthMeters"],
        disabled=["Date", "City", "MeetName", "PoolName", "LengthMeters"],
    )

    save_col1, save_col2 = st.columns([1, 1])
    with save_col1:
        if st.button("保存更改（写入 results.csv）", use_container_width=True):
            # 仅保留基本五列写回
            to_save = edited_df[["Name", "EventName", "Result", "Rank", "Note"]].copy()
            write_results(folder_for_edit, to_save)
            # 推送
            push = st.session_state.get("push_last", True)
            if push:
                msg = push_file_if_checked(folder_for_edit, "results.csv", to_save.to_csv(index=False).encode("utf-8"), f"Edit results for {folder_for_edit.name}", push=True)
                if msg:
                    st.info(msg)
            st.success("更改已保存。")
    with save_col2:
        if st.button("删除选中行（先在表格左侧勾选）", use_container_width=True):
            st.info("请在表格中直接删除行（点击行号右键 -> Delete row），然后点击“保存更改”。")


def section_query():
    st.header("🏊‍♀️ 游泳成绩查询 / 对比")

    # 汇总所有比赛
    dirs = list_meet_dirs()
    if not dirs:
        st.info("当前没有成绩数据。请先在“赛事管理/成绩录入”中添加。")
        return

    records = []
    for d in dirs:
        meta = read_meta(d)
        df = read_results(d)
        if df.empty:
            continue
        df = df.copy()
        df["Date"] = meta.get("Date", "")
        df["City"] = meta.get("City", "")
        df["MeetName"] = meta.get("MeetName", "")
        df["PoolName"] = meta.get("PoolName", "")
        df["LengthMeters"] = meta.get("LengthMeters", "")
        records.append(df)

    if not records:
        st.info("没有成绩数据。")
        return

    all_df = pd.concat(records, ignore_index=True)
    all_df["Seconds"] = all_df["Result"].apply(parse_time_to_seconds)
    all_df["Result"] = all_df["Seconds"].apply(format_seconds)

    names = sorted([n for n in all_df["Name"].dropna().unique().tolist() if str(n).strip()])
    sel_names = st.multiselect("Name（可多选）", names, default=names[:1] if names else [])
    events = ["全部"] + sorted([e for e in all_df["EventName"].dropna().unique().tolist() if str(e).strip()])
    sel_event = st.selectbox("Event", events, index=0)
    lengths = ["全部", 25, 50]
    sel_len = st.selectbox("Length (Meters)", lengths, index=0)

    df = all_df.copy()
    if sel_names:
        df = df[df["Name"].isin(sel_names)]
    if sel_event != "全部":
        df = df[df["EventName"] == sel_event]
    if sel_len != "全部":
        df = df[df["LengthMeters"].astype(str) == str(sel_len)]

    df = df.sort_values(by=["Seconds"], ascending=True, na_position="last")
    disp = df[["Name", "Date", "EventName", "Result", "Rank", "City", "PoolName", "LengthMeters"]]
    st.dataframe(disp, use_container_width=True)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏊‍♀️", layout="wide")
    st.title(APP_TITLE)

    tab1, tab2 = st.tabs(["查询 / 对比", "赛事管理 / 录入"])
    with tab1:
        section_query()
    with tab2:
        section_new_meet()
        st.divider()
        section_add_results()


if __name__ == "__main__":
    main()
