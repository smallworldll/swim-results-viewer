# -*- coding: utf-8 -*-
"""
Swim Results (Meet Mode) - Stable build
- No EventName in meta (MeetName/PoolName required; Length 25/50)
- After save: clear inputs & rerun (using safe_rerun for new Streamlit)
- Deletions/edits/new rows: save locally AND auto-push to GitHub (if secrets present)
- Dedup writes to avoid accidental double-saves
- Results shown shortest time first; global query with filters
"""

from __future__ import annotations
import re, json, base64
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------- Setup ----------------
st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide", initial_sidebar_state="expanded")
APP_TITLE = "🏊‍♀️ 游泳成绩系统（赛事制）"
MEETS_ROOT = Path("meets")
MEETS_ROOT.mkdir(parents=True, exist_ok=True)

# ------------- Compatibility -------------
def safe_rerun():
    """Streamlit rerun across versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ------------- Time helpers -------------
TIME_RE = re.compile(r"^\s*(?:(\d{1,2}):)?(\d{1,2})(?:[.:](\d{1,2}))?\s*$")

def parse_time_to_seconds(s: str) -> Optional[float]:
    """Accept '34.12' or '0:34.12' or '1:05.3' -> seconds(float)."""
    if not isinstance(s, str):
        return None
    s = s.strip().replace("：", ":").replace("，", ".")
    if not s:
        return None
    m = TIME_RE.match(s)
    if not m:
        return None
    mm = int(m.group(1)) if m.group(1) else 0
    ss = int(m.group(2))
    ff = m.group(3)
    hundredths = int(ff) if ff else 0
    if ff and len(ff) == 1:
        hundredths *= 10
    return mm * 60 + ss + hundredths / 100.0

def seconds_to_mssxx(sec: Optional[float]) -> str:
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return ""
    sec = float(sec)
    m = int(sec // 60)
    s = sec - m * 60
    whole = int(s)
    hund = int(round((s - whole) * 100))
    if hund == 100:
        hund = 0
        whole += 1
        if whole == 60:
            whole = 0
            m += 1
    return f"{m}:{whole:02d}.{hund:02d}"

# ------------- GitHub push -------------
def _gh_headers(token: str):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

def push_to_github(path_in_repo: str, content: bytes, message: str) -> Tuple[bool, str]:
    """Create/update a file at path_in_repo (e.g., 'meets/2025-08-09_X_Y/meta.csv')."""
    repo = st.secrets.get("REPO", "")
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not repo or not token:
        st.info("⚠️ 未配置 GITHUB_TOKEN/REPO，已跳过 GitHub 推送（本地已保存）。")
        return False, "NO_SECRET"
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = _gh_headers(token)
    # check if exists for sha
    r = requests.get(url, headers=headers, timeout=15)
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha
    r2 = requests.put(url, headers=headers, data=json.dumps(payload), timeout=20)
    if r2.status_code in (200,201):
        return True, "OK"
    try:
        return False, f"{r2.status_code} {r2.json()}"
    except Exception:
        return False, str(r2.status_code)

# ------------- IO helpers -------------
def sanitize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).replace("/", "_").strip()

def meet_folder_name(date_str: str, city: str, pool: str) -> str:
    return f"{date_str}_{sanitize(city)}_{sanitize(pool)}"

def list_meets_sorted() -> List[Path]:
    items = [p for p in MEETS_ROOT.iterdir() if p.is_dir() and (p / "meta.csv").exists()]
    def k(p: Path):
        try:
            return datetime.strptime(p.name.split("_",1)[0], "%Y-%m-%d")
        except Exception:
            return datetime.min
    return sorted(items, key=k, reverse=True)

def load_meta(meet_dir: Path) -> pd.Series:
    p = meet_dir / "meta.csv"
    if not p.exists():
        return pd.Series(dtype=object)
    df = pd.read_csv(p)
    return df.iloc[0] if not df.empty else pd.Series(dtype=object)

def save_meta_and_push(date_str: str, city: str, meet: str, pool: str, length: int) -> Path:
    folder = meet_folder_name(date_str, city, pool)
    meet_dir = MEETS_ROOT / folder
    meet_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame([{
        "Date": date_str,
        "City": city,
        "MeetName": meet,
        "PoolName": pool,
        "LengthMeters": int(length),
    }])[["Date","City","MeetName","PoolName","LengthMeters"]]
    meta_path = meet_dir / "meta.csv"
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
    # push
    push_to_github(str(meta_path).replace("\\","/"), meta_path.read_bytes(), f"Save meta for {folder}")
    return meet_dir

def load_results(meet_dir: Path) -> pd.DataFrame:
    p = meet_dir / "results.csv"
    if not p.exists():
        return pd.DataFrame(columns=["Name","EventName","Result","Rank","Note","Seconds",
                                     "Date","City","MeetName","PoolName","LengthMeters"])
    df = pd.read_csv(p)
    if "Seconds" not in df.columns and "Result" in df.columns:
        df["Seconds"] = df["Result"].map(parse_time_to_seconds)
    return df

def write_results_and_push(meet_dir: Path, df: pd.DataFrame, message: str):
    p = meet_dir / "results.csv"
    df = df.copy()
    # normalize time columns
    df["Seconds"] = df["Seconds"] if "Seconds" in df.columns else df["Result"].map(parse_time_to_seconds)
    df["Result"] = df["Seconds"].map(seconds_to_mssxx)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    push_to_github(str(p).replace("\\","/"), p.read_bytes(), message)

def append_rows_dedup_and_push(meet_dir: Path, rows: List[dict]):
    base = load_results(meet_dir)
    add = pd.DataFrame(rows, columns=["Name","EventName","Result","Rank","Note","Seconds","Date","City","MeetName","PoolName","LengthMeters"])
    merged = pd.concat([base, add], ignore_index=True)
    # dedup key
    def norm(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).str.strip().str.lower()
    key = (
        norm(merged["Name"]) + "|" + norm(merged["EventName"]) + "|" + norm(merged["Result"]) + "|" +
        norm(merged["Date"]) + "|" + norm(merged["City"]) + "|" + norm(merged["MeetName"]) + "|" +
        norm(merged["PoolName"]) + "|" + norm(merged["LengthMeters"])
    )
    merged["__k__"] = key
    merged = merged.drop_duplicates("__k__", keep="first").drop(columns="__k__")
    write_results_and_push(meet_dir, merged, f"Save results for {meet_dir.name}")
    return merged

def delete_indices_and_push(meet_dir: Path, indices: List[int]):
    base = load_results(meet_dir)
    if base.empty or not indices:
        return base
    keep = base.index.difference(indices)
    out = base.loc[keep].reset_index(drop=True)
    write_results_and_push(meet_dir, out, f"Delete rows in {meet_dir.name}")
    return out

# ------------- UI Sections -------------
DEFAULT_EVENTS = [
    "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle","400m Freestyle",
    "25m Backstroke","50m Backstroke","100m Backstroke","200m Backstroke",
    "25m Breaststroke","50m Breaststroke","100m Breaststroke","200m Breaststroke",
    "25m Butterfly","50m Butterfly","100m Butterfly","200m Butterfly",
    "100m IM","200m IM","400m IM",
]

def section_meta():
    st.subheader("① 新建/选择赛事（meta）")
    with st.form("meta_form", clear_on_submit=False):
        d = st.date_input("Date", value=date.today(), format="YYYY-MM-DD", key="meta_date")
        city = st.text_input("City", value="Chiang Mai", key="meta_city")
        meet_name = st.text_input("MeetName（必填）", value="", key="meta_meet")
        pool_name = st.text_input("PoolName（必填）", value="", key="meta_pool")
        length = st.selectbox("LengthMeters", [25,50], index=0, key="meta_len")
        submitted = st.form_submit_button("保存赛事信息（写入/推送 meta.csv）")
        if submitted:
            if not meet_name.strip() or not pool_name.strip():
                st.error("❌ MeetName 与 PoolName 必填。")
            else:
                meet_dir = save_meta_and_push(d.isoformat(), city.strip(), meet_name.strip(), pool_name.strip(), int(length))
                st.success(f"✅ 已保存并推送：{(meet_dir / 'meta.csv').as_posix()}")

def section_results_and_manage():
    st.subheader("② 已登记记录（先看后改/删），然后新增成绩")

    meets = list_meets_sorted()
    if not meets:
        st.info("暂无赛事，请先在上方创建 meta。")
        return
    meet_dir = st.selectbox("选择赛事文件夹", options=meets, index=0, key="sel_meet", format_func=lambda p: p.name)
    meta = load_meta(meet_dir)

    # ------ Existing records (edit/delete) ------
    df = load_results(meet_dir)
    st.caption("下面是该赛事已有记录。可编辑单元格；要删除请勾选“删除？”后点击按钮。保存将写回本地并推送到 GitHub。")
    if df.empty:
        st.info("该赛事暂无记录。")
    else:
        show = df.copy()
        if "删除？" not in show.columns:
            show.insert(0, "删除？", False)
        show = show.sort_values(by=["Seconds","Name"], ascending=[True, True], na_position="last")
        edited = st.data_editor(
            show,
            key=f"editor_{meet_dir.name}",
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "删除？": st.column_config.CheckboxColumn("删除？", help="勾选并点击下方删除按钮")
            }
        )

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("💾 保存更改（写回并推送）", key="btn_save_edits"):
                out = edited.drop(columns=["删除？"], errors="ignore")
                write_results_and_push(meet_dir, out, f"Edit results for {meet_dir.name}")
                st.success("已保存并推送。")
                safe_rerun()
        with c2:
            if st.button("🗑️ 删除勾选行 并保存（写回并推送）", key="btn_delete_rows"):
                mask = edited.get("删除？", False)
                if isinstance(mask, pd.Series) and mask.any():
                    keep_df = edited.loc[~mask].drop(columns=["删除？"], errors="ignore")
                    write_results_and_push(meet_dir, keep_df, f"Delete rows in {meet_dir.name}")
                    st.success(f"已删除 {int(mask.sum())} 行并推送。")
                    safe_rerun()
                else:
                    st.info("未勾选任何行。")

    st.markdown("---")
    # ------ Add results ------
    st.subheader("③ 新增成绩（results）")

    # event dropdown from defaults + existing
    existing_events = sorted([x for x in df["EventName"].dropna().unique().tolist()]) if not df.empty else []
    ev_options = sorted(set(DEFAULT_EVENTS + existing_events))
    selected_event = st.selectbox("Event 选择", options=ev_options, index=ev_options.index("100m Freestyle") if "100m Freestyle" in ev_options else 0, key="ev_pick")

    rows_n = st.number_input("本次录入行数", min_value=1, max_value=10, value=1, step=1, key="rows_n")
    # ensure keys exist to avoid state errors
    for i in range(1, int(rows_n)+1):
        for f in ("Name","EventName","Result","Rank","Note"):
            st.session_state.setdefault(f"{f}_{i}", "")

    inputs = []
    for i in range(1, int(rows_n)+1):
        st.markdown(f"**记录 {i}**")
        c1, c2, c3, c4 = st.columns([1.2,1.4,1.0,1.4])
        name = c1.text_input(f"Name_{i}", key=f"Name_{i}", value=st.session_state.get(f"Name_{i}", "Anna"), placeholder="选手名")
        event_name = c2.text_input(f"EventName_{i}", key=f"EventName_{i}", value=selected_event)
        result = c3.text_input(f"Result_{i}", key=f"Result_{i}", placeholder="34.12 或 0:34.12")
        rank = c4.text_input(f"Rank_{i}（可空）", key=f"Rank_{i}", value="")
        note = st.text_input(f"Note_{i}", key=f"Note_{i}", placeholder="可留空")
        inputs.append((name, event_name, result, rank, note))

    if st.button("保存这些成绩（写回并推送）", type="primary", key="btn_save_new"):
        rows = []
        for (name, ev, res, rk, note) in inputs:
            if not str(name).strip() or not str(ev).strip() or not str(res).strip():
                continue
            secs = parse_time_to_seconds(str(res))
            if secs is None:
                st.warning(f"时间格式不合法：{res}（已跳过）")
                continue
            rows.append({
                "Name": str(name).strip(),
                "EventName": str(ev).strip(),
                "Result": seconds_to_mssxx(secs),
                "Rank": str(rk).strip() if str(rk).strip() else "",
                "Note": str(note).strip(),
                "Seconds": secs,
                "Date": str(meta.get('Date','')),
                "City": str(meta.get('City','')),
                "MeetName": str(meta.get('MeetName','')),
                "PoolName": str(meta.get('PoolName','')),
                "LengthMeters": int(meta.get('LengthMeters', 25)) if str(meta.get('LengthMeters','')).strip() else 25,
            })
        if not rows:
            st.info("没有有效行可保存。")
        else:
            append_rows_dedup_and_push(meet_dir, rows)
            st.success(f"已保存并推送到 GitHub：{(meet_dir/'results.csv').as_posix()}")
            # clear inputs
            for i in range(1, int(rows_n)+1):
                for f in ("Name","EventName","Result","Rank","Note"):
                    st.session_state.pop(f"{f}_{i}", None)
            safe_rerun()

# --------- Query page ---------
def page_query():
    st.header("🔎 成绩查询 / 对比")
    frames = []
    for d in list_meets_sorted():
        p = d / "results.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                frames.append(df)
            except Exception:
                pass
    if not frames:
        st.info("暂无数据。")
        return
    data = pd.concat(frames, ignore_index=True)
    if "Seconds" in data.columns:
        data["Seconds"] = data["Seconds"].where(data["Seconds"].notna(), data["Result"].map(parse_time_to_seconds))
    else:
        data["Seconds"] = data["Result"].map(parse_time_to_seconds)
    data["Result"] = data["Seconds"].map(seconds_to_mssxx)

    names = sorted([x for x in data["Name"].dropna().unique().tolist() if str(x).strip()])
    events = ["全部"] + sorted([x for x in data["EventName"].dropna().unique().tolist() if str(x).strip()])
    lengths = ["全部"] + [str(int(x)) for x in pd.to_numeric(data["LengthMeters"], errors="coerce").dropna().unique().tolist()]

    pick_names = st.multiselect("Name（可多选）", names, default=names[:1] if names else [])
    pick_event = st.selectbox("Event", events, index=0)
    pick_len = st.selectbox("Length (Meters)", lengths, index=0)

    q = data.copy()
    if pick_names:
        q = q[q["Name"].isin(pick_names)]
    if pick_event != "全部":
        q = q[q["EventName"] == pick_event]
    if pick_len != "全部":
        # 强健的泳池长度筛选（兼容 25/50 和 25.0/50.0 等）
        try:
            pick_len_int = int(pick_len)
            q = q[pd.to_numeric(q["LengthMeters"], errors="coerce").round().astype("Int64") == pick_len_int]
        except Exception:
            pass

    q = q.sort_values(by=["Seconds","Date","Name"], ascending=[True, True, True])
    show_cols = ["Name","Date","EventName","Result","Rank","Note","City","PoolName","LengthMeters","MeetName"]
    show_cols = [c for c in show_cols if c in q.columns]
    st.dataframe(q[show_cols], use_container_width=True, hide_index=True)

# ---------------- Main ----------------
def main():
    st.title(APP_TITLE)
    tab1, tab2 = st.tabs(["赛事管理 / 成绩录入", "查询 / 对比"])
    with tab1:
        section_meta()
        st.markdown("---")
        section_results_and_manage()
    with tab2:
        page_query()

if __name__ == "__main__":
    main()