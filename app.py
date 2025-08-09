
import os
import io
import json
import base64
import requests
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide")

MEETS_ROOT = Path("meets")  # relative to repo root

# Standard events for dropdowns
STANDARD_EVENTS = [
    "25m Freestyle", "25m Backstroke", "25m Breaststroke", "25m Butterfly",
    "50m Freestyle", "50m Backstroke", "50m Breaststroke", "50m Butterfly",
    "100m Freestyle", "100m Backstroke", "100m Breaststroke", "100m Butterfly",
    "200m Freestyle", "200m Backstroke", "200m Breaststroke", "200m Butterfly",
    "200m IM", "400m Freestyle", "400m IM"
]

# ---------------------------
# Helpers
# ---------------------------

def ensure_meets_root():
    MEETS_ROOT.mkdir(exist_ok=True)

def meet_folder_name(date_str: str, city: str, poolname: str) -> str:
    # 2025-08-09_Chiang Mai_National Sports University Chiang Mai Campus
    safe_city = city.replace("/", "_").strip()
    safe_pool = poolname.replace("/", "_").strip()
    return f"{date_str}_{safe_city}_{safe_pool}"

def parse_time_to_display_and_seconds(s: str):
    """Accept '34.12' or '0:34.12' or '1:02.45'. Return ('m:ss.xx', seconds float)."""
    s = (s or "").strip()
    if not s:
        return "", None
    # normalize decimals
    try:
        if ":" in s:
            # m:ss.xx
            parts = s.split(":")
            if len(parts) == 2:
                m = int(parts[0])
                sec = float(parts[1])
            else:
                # h:mm:ss.xx -> convert to minutes
                h = int(parts[0])
                m = int(parts[1]) + 60 * h
                sec = float(parts[2])
        else:
            # "34.12" -> 0:34.12
            m = 0
            sec = float(s)
        total = m * 60 + sec
        # format
        mm = int(total // 60)
        ss = total - mm * 60
        display = f"{mm}:{ss:05.2f}"
        return display, round(total, 2)
    except Exception:
        return s, None

def gh_headers():
    token = st.secrets.get("GITHUB_TOKEN", "")
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

def gh_repo():
    return st.secrets.get("REPO", "")  # e.g., "smallworlddll/swim-results-viewer"

def gh_api_url(path: str):
    return f"https://api.github.com/repos/{gh_repo()}/contents/{path}"

def push_to_github(path: str, content_bytes: bytes, commit_message: str):
    """Create or update a file via GitHub API. Always attempt when token/repo present."""
    repo = gh_repo()
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not repo or not token:
        # show a gentle warn
        st.info("⚠️ 未配置 GITHUB_TOKEN/REPO，已跳过 GitHub 推送（本地已保存）。")
        return False, "NO_SECRET"

    url = gh_api_url(path)
    headers = gh_headers()

    # Check if exists to get sha
    resp = requests.get(url, headers=headers)
    sha = None
    if resp.status_code == 200:
        try:
            sha = resp.json().get("sha")
        except Exception:
            sha = None

    payload = {
        "message": commit_message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    put = requests.put(url, headers=headers, data=json.dumps(payload))
    if put.status_code in (200, 201):
        return True, "OK"
    else:
        try:
            msg = put.json()
        except Exception:
            msg = {"status": put.status_code, "text": put.text}
        st.warning(f"GitHub 推送失败: {msg}")
        return False, msg

def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def list_meets():
    ensure_meets_root()
    items = []
    for p in sorted(MEETS_ROOT.glob("*")):
        if not p.is_dir():
            continue
        meta = p / "meta.csv"
        row = {"folder": p.name}
        if meta.exists():
            try:
                m = pd.read_csv(meta)
                if not m.empty:
                    row.update(m.iloc[0].to_dict())
            except Exception:
                pass
        items.append(row)
    # sort by Date desc
    def _key(x):
        d = x.get("Date") or ""
        try:
            return dt.datetime.strptime(d, "%Y-%m-%d")
        except Exception:
            return dt.datetime.min
    items.sort(key=_key, reverse=True)
    return items

def load_meet_results(folder: str) -> pd.DataFrame:
    path = MEETS_ROOT / folder / "results.csv"
    df = read_csv_if_exists(path)
    return df

def load_meet_meta(folder: str) -> pd.Series:
    path = MEETS_ROOT / folder / "meta.csv"
    m = read_csv_if_exists(path)
    if m.empty:
        return pd.Series(dtype=object)
    return m.iloc[0]

def build_event_options(existing_df: pd.DataFrame) -> list:
    options = list(STANDARD_EVENTS)
    if not existing_df.empty and "EventName" in existing_df.columns:
        extra = [x for x in existing_df["EventName"].dropna().unique().tolist() if x not in options]
        options.extend(extra)
    return options

def add_rows_and_save(folder: str, rows: list):
    """rows: list of dicts with required fields"""
    df = load_meet_results(folder)
    base_cols = [
        "Name", "EventName", "Result", "Rank", "Note",
        "Seconds", "Date", "City", "MeetName", "PoolName", "LengthMeters"
    ]
    if df.empty:
        df = pd.DataFrame(columns=base_cols)
    # Append
    add_df = pd.DataFrame(rows, columns=base_cols)
    # Deduplicate on a set of columns to avoid accidental duplicate save clicks
    subset = ["Name", "EventName", "Result", "Date", "City", "MeetName", "PoolName", "LengthMeters"]
    merged = pd.concat([df, add_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=subset, keep="first")
    # Sort (optional) by Seconds asc if present
    if "Seconds" in merged.columns:
        merged = merged.sort_values(by=["EventName", "Seconds"], ascending=[True, True], na_position="last")
    # Save local
    local_path = MEETS_ROOT / folder / "results.csv"
    write_csv(merged, local_path)
    # Push GitHub
    with open(local_path, "rb") as f:
        ok, msg = push_to_github(str(local_path).replace("\\", "/"), f.read(), f"Save results for {folder}")
    return merged, ok

def delete_selected_and_save(folder: str, indices: list):
    df = load_meet_results(folder)
    if df.empty or not indices:
        return df, False
    keep = df.index.difference(indices)
    new_df = df.loc[keep].reset_index(drop=True)
    local_path = MEETS_ROOT / folder / "results.csv"
    write_csv(new_df, local_path)
    with open(local_path, "rb") as f:
        ok, msg = push_to_github(str(local_path).replace("\\", "/"), f.read(), f"Delete rows in {folder}")
    return new_df, ok

def save_meta_and_push(date_str: str, city: str, meetname: str, poolname: str, length: int):
    folder = meet_folder_name(date_str, city, poolname)
    meta_path = MEETS_ROOT / folder / "meta.csv"
    row = {
        "Date": date_str,
        "City": city,
        "MeetName": meetname,
        "PoolName": poolname,
        "LengthMeters": int(length)
    }
    write_csv(pd.DataFrame([row]), meta_path)
    with open(meta_path, "rb") as f:
        ok, msg = push_to_github(str(meta_path).replace("\\", "/"), f.read(), f"Save meta for {folder}")
    return folder, ok

# ---------------------------
# UI Sections
# ---------------------------

def section_meta():
    st.markdown("## 🗂️ ① 新建/选择赛事（meta）")

    today = dt.date.today().strftime("%Y-%m-%d")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        date_str = st.text_input("Date", value=today)
    with col2:
        city = st.text_input("City", value="Chiang Mai")
    with col3:
        length = st.selectbox("LengthMeters", options=[25,50], index=0)

    meetname = st.text_input("MeetName（必填）", value="")
    poolname = st.text_input("PoolName（必填）", value="")

    if st.button("保存赛事信息（写入/推送 meta.csv）", type="primary"):
        if not meetname.strip() or not poolname.strip():
            st.error("MeetName 与 PoolName 均为必填。")
        else:
            folder, ok = save_meta_and_push(date_str, city, meetname.strip(), poolname.strip(), length)
            st.success(f"已保存：{MEETS_ROOT / folder / 'meta.csv'}")

def section_results_entry_and_manage():
    st.markdown("## 📝 ② 新增成绩 / 管理")
    meets = list_meets()
    folders = [x["folder"] for x in meets] if meets else []
    if not folders:
        st.info("当前没有赛事，请先在上方创建赛事。")
        return
    # 默认选择最近（列表已按日期倒序）
    folder = st.selectbox("选择赛事文件夹", options=folders, index=0)
    meta = load_meet_meta(folder)
    # Show existing first
    st.markdown("### ③ 已登记记录（可编辑/删除）")
    df_exist = load_meet_results(folder)
    if df_exist.empty:
        st.info("本赛事暂无记录。")
    else:
        # Allow multi-select delete
        # show essential columns
        show_cols = ["Name", "EventName", "Result", "Rank", "Note", "Seconds", "Date", "City", "MeetName", "PoolName", "LengthMeters"]
        for c in show_cols:
            if c not in df_exist.columns:
                df_exist[c] = ""
        # Use selection box for indices
        st.caption("勾选需要删除的记录（支持多选），然后点击下方“删除选中并保存”。")
        selected = st.dataframe(df_exist[show_cols], use_container_width=True)
        # Provide multi-select by indices via text input for now (Streamlit's DF selection is limited)
        del_idxs_text = st.text_input("要删除的行号（用逗号分隔，例如：2,5,7）", value="")
        if st.button("🗑️ 删除选中并保存（写入 results.csv 并推送 GitHub）"):
            try:
                idxs = [int(x.strip())-1 for x in del_idxs_text.split(",") if x.strip().isdigit()]
            except Exception:
                idxs = []
            new_df, ok = delete_selected_and_save(folder, idxs)
            st.success("已删除并保存（本地 & GitHub）。")
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("### ④ 新增成绩")
    # build event options
    event_options = build_event_options(df_exist)
    default_event = event_options[0] if event_options else "50m Freestyle"
    event_selected = st.selectbox("Event 选择", options=event_options, index=event_options.index(default_event) if default_event in event_options else 0)

    rows = st.number_input("本次录入行数", min_value=1, max_value=10, value=1, step=1)
    inputs = []
    for i in range(1, rows+1):
        st.markdown(f"#### 记录 {i}")
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            name = st.text_input(f"Name_{i}", value="")
        with c2:
            eventname = st.text_input(f"EventName_{i}", value=event_selected)
        with c3:
            result = st.text_input(f"Result_{i}", value="")
        with c4:
            rank = st.text_input(f"Rank_{i}", value="")
        note = st.text_input(f"Note_{i}", value="可留空")
        inputs.append({"Name": name, "EventName": eventname, "Result": result, "Rank": rank, "Note": note})

    if st.button("保存这些成绩（写入 results.csv 并推送 GitHub）", type="primary"):
        # prepare rows
        if meta.empty:
            st.error("未找到赛事 meta，请返回上方保存赛事信息。")
        else:
            prepared = []
            for r in inputs:
                if not r["Name"].strip() or not r["EventName"].strip() or not r["Result"].strip():
                    # skip incomplete
                    continue
                display, secs = parse_time_to_display_and_seconds(r["Result"])
                row = {
                    "Name": r["Name"].strip(),
                    "EventName": r["EventName"].strip(),
                    "Result": display if display else r["Result"].strip(),
                    "Rank": r["Rank"].strip() if r["Rank"] is not None else "",
                    "Note": r["Note"].strip() if r["Note"] else "",
                    "Seconds": secs,
                    "Date": meta.get("Date", ""),
                    "City": meta.get("City", ""),
                    "MeetName": meta.get("MeetName", ""),
                    "PoolName": meta.get("PoolName", ""),
                    "LengthMeters": meta.get("LengthMeters", "")
                }
                prepared.append(row)
            if not prepared:
                st.warning("没有可保存的完整记录（Name / Event / Result 必填）。")
            else:
                merged, ok = add_rows_and_save(folder, prepared)
                st.success(f"已保存 {len(prepared)} 条到 {MEETS_ROOT / folder / 'results.csv'}（并已推送）。")
                # clear inputs by rerun
                st.experimental_rerun()

def section_query():
    st.markdown("## 🔎 ⑤ 成绩查询 / 对比")
    # gather all results
    ensure_meets_root()
    all_rows = []
    for p in MEETS_ROOT.glob("*/*"):
        if p.name == "results.csv":
            df = read_csv_if_exists(p)
            if not df.empty:
                all_rows.append(df)
    if not all_rows:
        st.info("未找到任何成绩数据。")
        return
    data = pd.concat(all_rows, ignore_index=True)
    # filters
    names = sorted([x for x in data["Name"].dropna().unique().tolist() if str(x).strip()])
    name_sel = st.multiselect("Name（可多选）", names, default=names[:1] if names else [])
    events_all = sorted([x for x in data["EventName"].dropna().unique().tolist() if str(x).strip()])
    event_sel = st.selectbox("Event", options=["全部"] + events_all, index=0)
    len_options = sorted([int(x) for x in data["LengthMeters"].dropna().unique().tolist()])
    length_sel = st.selectbox("Length (Meters)", options=["全部"] + len_options, index=0)

    df = data.copy()
    if name_sel:
        df = df[df["Name"].isin(name_sel)]
    if event_sel != "全部":
        df = df[df["EventName"] == event_sel]
    if length_sel != "全部":
        df = df[df["LengthMeters"] == length_sel]

    # order by Seconds asc
    if "Seconds" in df.columns:
        df = df.sort_values(by=["Seconds"], ascending=[True], na_position="last")

    st.dataframe(df, use_container_width=True)

# ---------------------------
# Main
# ---------------------------

def main():
    st.title("🏊 游泳成绩系统（赛事制）")
    section_meta()
    st.divider()
    section_results_entry_and_manage()
    st.divider()
    section_query()

if __name__ == "__main__":
    main()
