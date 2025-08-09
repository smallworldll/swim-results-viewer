
# -*- coding: utf-8 -*-
import os
import re
import base64
import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import requests

APP_TITLE = "🏊‍♀️ 游泳成绩系统（赛事制）"
ROOT = Path("meets")
ROOT.mkdir(exist_ok=True, parents=True)

# ---- Constants ----
RES_COLS = ["Name","EventName","Result","Rank","Note","Date","City","MeetName","PoolName","LengthMeters"]
COMMON_EVENTS = [
    "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle","400m Freestyle",
    "25m Backstroke","50m Backstroke","100m Backstroke","200m Backstroke",
    "25m Breaststroke","50m Breaststroke","100m Breaststroke","200m Breaststroke",
    "25m Butterfly","50m Butterfly","100m Butterfly","200m Butterfly",
    "200m IM","400m IM"
]

# ---- Helpers ----
def sanitize(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def meet_folder_name(d: str, city: str, pool: str) -> str:
    return f"{d}_{sanitize(city)}_{sanitize(pool)}"

def parse_time_to_seconds(txt: str) -> Optional[float]:
    if txt is None:
        return None
    s = str(txt).strip().replace("，", ".").replace("：", ":")
    if s == "":
        return None
    # m:ss.xx or m:ss
    m = re.fullmatch(r"(\d+):(\d{1,2})(?:\.(\d{1,2}))?", s)
    if m:
        mm = int(m.group(1))
        ss = int(m.group(2))
        frac = int(m.group(3)) if m.group(3) else 0
        return mm*60 + ss + frac / (10 ** (len(m.group(3)) if m.group(3) else 1))
    # ss.xx or ss
    m = re.fullmatch(r"(\d+)(?:\.(\d{1,2}))?", s)
    if m:
        ss = int(m.group(1))
        frac = int(m.group(2)) if m.group(2) else 0
        return ss + frac / (10 ** (len(m.group(2)) if m.group(2) else 1))
    return None

def seconds_to_text(sec: Optional[float]) -> str:
    if sec is None or pd.isna(sec):
        return ""
    sec = float(sec)
    m = int(sec // 60)
    s = sec - m*60
    return f"{m}:{s:05.2f}"

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = "" if c != "Rank" else ""
    return out[cols]

def write_results(folder: str, df: pd.DataFrame):
    path = ROOT / folder / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # normalize time + order columns
    df = df.copy()
    if "Result" in df.columns:
        secs = df["Result"].apply(parse_time_to_seconds)
        df["Result"] = secs.apply(seconds_to_text)
    # Rank keep as raw text (empty allowed)
    df = ensure_cols(df, RES_COLS)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def read_results(folder: str) -> pd.DataFrame:
    path = ROOT / folder / "results.csv"
    if not path.exists():
        return pd.DataFrame(columns=RES_COLS)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df = ensure_cols(df, RES_COLS)
    return df

def read_meta(folder: str) -> pd.Series:
    p = ROOT / folder / "meta.csv"
    if not p.exists():
        return pd.Series({c:"" for c in ["Date","City","MeetName","PoolName","LengthMeters","EventName"]})
    df = pd.read_csv(p)
    if df.empty:
        return pd.Series({c:"" for c in ["Date","City","MeetName","PoolName","LengthMeters","EventName"]})
    # Ensure EventName column exists (may be blank if not used)
    for c in ["EventName"]:
        if c not in df.columns:
            df[c] = ""
    return df.iloc[0]

def write_meta(date_str: str, city: str, meet: str, pool: str, length: int):
    folder = meet_folder_name(date_str, city, pool)
    d = ROOT / folder
    d.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame([{
        "Date": date_str,
        "City": city,
        "MeetName": meet,
        "PoolName": pool,
        "LengthMeters": int(length),
    }], columns=["Date","City","MeetName","PoolName","LengthMeters"])
    meta.to_csv(d / "meta.csv", index=False, encoding="utf-8-sig")
    return folder

def list_meet_folders_sorted() -> List[str]:
    if not ROOT.exists():
        return []
    items = []
    for f in ROOT.iterdir():
        if f.is_dir():
            name = f.name
            try:
                dt = datetime.strptime(name.split("_",1)[0], "%Y-%m-%d")
            except Exception:
                dt = datetime.min
            items.append((dt, name))
    items.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in items]

# ---- GitHub push (optional) ----
def github_upsert(rel_path: str, content_bytes: bytes, message: str) -> Tuple[bool, str]:
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("REPO", "")
    if not token or not repo:
        return False, "Missing GITHUB_TOKEN/REPO"
    url = f"https://api.github.com/repos/{repo}/contents/{rel_path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    # get sha
    r = requests.get(url, headers=headers, timeout=15)
    sha = r.json().get("sha") if r.status_code == 200 else None
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
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

# ---- Pages ----
def page_manage():
    st.header("📁 赛事管理 / 成绩录入")

    # ① 新建/选择赛事（meta）
    st.subheader("① 新建/选择赛事（meta）")
    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        d = st.date_input("Date", value=date.today(), format="YYYY-MM-DD")
    with c2:
        city = st.text_input("City", value="Chiang Mai")
    with c3:
        meet = st.text_input("MeetName", value="")

    c4,c5,c6 = st.columns([2,1,2])
    with c4:
        pool = st.text_input("PoolName（必填）", value="")
    with c5:
        length = st.selectbox("LengthMeters", [25,50], index=0)

    push_meta = st.checkbox("保存时推送到 GitHub", value=False)

    if st.button("保存赛事信息（写入/推送 meta.csv)"):
        if not meet or not pool:
            st.error("请填写 MeetName 和 PoolName 后再保存")
        else:
            folder = write_meta(date_str, city, meet, pool, length)
            st.success(f"已保存： meets/{folder}/meta.csv")
            if push_meta:
                ok, msg = github_upsert(
                    f"meets/{folder}/meta.csv",
                    (ROOT / folder / "meta.csv").read_bytes(),
                    f"Save meta for {folder}"
                )
                if ok:
                    st.info("GitHub: 已推送")
                else:
                    st.warning(f"GitHub 推送失败（{msg}）")

    st.divider()

    # ② 已登记记录（先看后改/删，再新增）
    st.subheader("② 已登记记录（可编辑/删除）")
    meets = list_meet_folders_sorted()
    if not meets:
        st.info("暂无赛事，请先创建 meta。")
        return
    sel_meet = st.selectbox("选择赛事文件夹", meets, index=0)
    meta = read_meta(sel_meet)
    df = read_results(sel_meet).copy()

    # Always show EventName in list (you asked to show it)
    if df.empty:
        st.caption("该赛事尚无记录。")
    else:
        # add a delete checkbox column
        if "Delete?" not in df.columns:
            df.insert(0, "Delete?", False)
        # sort by time
        df["_sec"] = df["Result"].apply(parse_time_to_seconds)
        df = df.sort_values(by=["_sec","Name"], ascending=[True, True]).drop(columns=["_sec"])

        st.caption("在表格里可直接编辑；勾选要删除的行后点“🗑️ 删除选中 并保存”。")
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{sel_meet}",
        )

        cA,cB = st.columns([1,1])
        with cA:
            if st.button("💾 保存更改（写回 results.csv）", type="secondary"):
                out = edited.drop(columns=["Delete?"], errors="ignore")
                write_results(sel_meet, out)
                st.success("更改已保存。")
        with cB:
            if st.button("🗑️ 删除选中 并保存", type="primary"):
                keep = edited[edited.get("Delete?", False) != True].drop(columns=["Delete?"], errors="ignore")
                write_results(sel_meet, keep)  # persists deletion
                st.success("已删除并保存。")
                st.rerun()

    st.divider()

    # ③ 新增成绩（results）
    st.subheader("③ 新增成绩（results）")

    # Event dropdown: presets + events already in this meet + meta default
    existing_events = sorted([x for x in read_results(sel_meet)["EventName"].dropna().unique().tolist() if str(x).strip()])
    meta_default_ev = [meta.get("EventName")] if str(meta.get("EventName","")).strip() else []
    ev_options = sorted(set(COMMON_EVENTS + existing_events + meta_default_ev))
    ev_choice = st.selectbox("Event 选择", options=ev_options + ["自定义…"], index= (ev_options.index(meta_default_ev[0]) if meta_default_ev and meta_default_ev[0] in ev_options else 0))
    if ev_choice == "自定义…":
        event_name = st.text_input("自定义 EventName", value="", placeholder="如：100m Butterfly")
    else:
        event_name = ev_choice

    n = st.number_input("本次录入行数", min_value=1, max_value=20, value=1, step=1)
    rows = []
    for i in range(1, int(n)+1):
        st.markdown(f"**记录 {i}**")
        c1,c2,c3,c4 = st.columns([1.2,1.2,1.0,2.0])
        name = c1.text_input(f"Name_{i}", value="", key=f"name_{i}")
        result = c2.text_input(f"Result_{i}", value="", placeholder="m:ss.xx 或 ss.xx", key=f"result_{i}")
        rank = c3.text_input(f"Rank_{i}（可空）", value="", key=f"rank_{i}")
        note = c4.text_input(f"Note_{i}", value="", key=f"note_{i}")
        if name.strip() or result.strip() or rank.strip() or note.strip():
            rows.append({"Name": name.strip(), "Result": result.strip(), "Rank": rank.strip(), "Note": note.strip()})

    push_res = st.checkbox("保存时推送到 GitHub", value=False)

    if st.button("保存这些成绩", type="primary"):
        # build valid rows
        valid = []
        for r in rows:
            if not r["Name"] or not r["Result"]:
                continue
            sec = parse_time_to_seconds(r["Result"])
            if sec is None:
                st.warning(f"时间格式不合法：{r['Result']}（{r['Name']}）")
                continue
            # rank
            rank_val = r["Rank"].strip()
            if rank_val != "":
                try:
                    int(rank_val)
                except Exception:
                    st.warning(f"名次不是整数，已置空：{rank_val}（{r['Name']}）")
                    rank_val = ""
            valid.append({
                "Name": r["Name"],
                "EventName": event_name.strip(),
                "Result": seconds_to_text(sec),
                "Rank": rank_val,
                "Note": r["Note"],
                "Date": meta.get("Date",""),
                "City": meta.get("City",""),
                "MeetName": meta.get("MeetName",""),
                "PoolName": meta.get("PoolName",""),
                "LengthMeters": meta.get("LengthMeters",""),
            })
        if not valid:
            st.warning("没有有效记录可保存。")
        else:
            base = read_results(sel_meet)
            out = pd.concat([base, pd.DataFrame(valid)], ignore_index=True)
            write_results(sel_meet, out)
            if push_res:
                ok, msg = github_upsert(f"meets/{sel_meet}/results.csv", (ROOT/sel_meet/"results.csv").read_bytes(), f"Save results for {sel_meet}")
                st.info("GitHub: " + ("已推送" if ok else f"失败：{msg}"))
            st.success(f"已保存 {len(valid)} 条。")
            # clear inputs to avoid duplicate save
            for i in range(1, int(n)+1):
                for key in [f"name_{i}", f"result_{i}", f"rank_{i}", f"note_{i}"]:
                    if key in st.session_state:
                        st.session_state[key] = ""

def page_query():
    st.header("🔍 游泳成绩查询 / 对比")
    folders = list_meet_folders_sorted()
    if not folders:
        st.info("暂无数据。")
        return
    frames = []
    for f in folders:
        df = read_results(f)
        if not df.empty:
            frames.append(df)
    if not frames:
        st.info("暂无数据。")
        return
    data = pd.concat(frames, ignore_index=True)
    data["Seconds"] = data["Result"].apply(parse_time_to_seconds)

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
        q = q[q["LengthMeters"].astype(str) == pick_len]

    q = q.sort_values(by=["Seconds","Date","Name"], ascending=[True, True, True])
    st.dataframe(q[["Name","Date","EventName","Result","Rank","City","PoolName","LengthMeters"]], use_container_width=True, hide_index=True)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    tab = st.sidebar.radio("页面", ["赛事管理 / 录入", "查询 / 对比"], index=0)
    if tab.startswith("赛事管理"):
        page_manage()
    else:
        page_query()

if __name__ == "__main__":
    main()