
# -*- coding: utf-8 -*-
import os
import io
import json
import base64
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st
import altair as alt
import requests

APP_TITLE = "🏊‍♂️ 游泳成绩系统（赛事制）"
MEETS_ROOT = "meets"

RESULT_COLS = [
    "Name", "EventName", "Result", "Rank", "Note",
    "Date", "City", "MeetName", "PoolName", "LengthMeters"
]

def ensure_meets_root():
    os.makedirs(MEETS_ROOT, exist_ok=True)

def list_meets() -> List[str]:
    ensure_meets_root()
    folders = []
    for name in sorted(os.listdir(MEETS_ROOT)):
        d = os.path.join(MEETS_ROOT, name)
        if os.path.isdir(d):
            folders.append(name)
    return folders

def meta_path(meet_folder: str) -> str:
    return os.path.join(MEETS_ROOT, meet_folder, "meta.csv")

def results_path(meet_folder: str) -> str:
    return os.path.join(MEETS_ROOT, meet_folder, "results.csv")

def load_meta(meet_folder: str) -> pd.Series:
    p = meta_path(meet_folder)
    if os.path.exists(p):
        df = pd.read_csv(p)
        if not df.empty:
            return df.iloc[0]
    # empty fallback
    return pd.Series({
        "Date": None, "City": None, "MeetName": None,
        "PoolName": None, "LengthMeters": None
    })

def load_results(meet_folder: str) -> pd.DataFrame:
    p = results_path(meet_folder)
    if os.path.exists(p):
        df = pd.read_csv(p)
        # normalize missing columns
        for c in RESULT_COLS:
            if c not in df.columns:
                df[c] = None
        return df[RESULT_COLS]
    else:
        return pd.DataFrame(columns=RESULT_COLS)

def save_meta(meet_folder: str, meta: dict):
    d = os.path.join(MEETS_ROOT, meet_folder)
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame([meta])
    df.to_csv(meta_path(meet_folder), index=False, encoding="utf-8-sig")

def save_results(meet_folder: str, df: pd.DataFrame):
    # Save strictly columns we know, in order
    d = os.path.join(MEETS_ROOT, meet_folder)
    os.makedirs(d, exist_ok=True)
    out = df.copy()
    # Drop any helper columns like "Del"
    if "Del" in out.columns:
        out = out[out["Del"] != True]  # filter any row marked for delete (should already be filtered)
        out = out.drop(columns=["Del"])
    for c in RESULT_COLS:
        if c not in out.columns:
            out[c] = None
    out = out[RESULT_COLS]
    out.to_csv(results_path(meet_folder), index=False, encoding="utf-8-sig")

def normalize_event_name(raw: str) -> str:
    # keep as-is, but strip spaces
    return (raw or "").strip()

def parse_time_input(text: str) -> str:
    """Accept '34.12' or '0:34.12' or '1:04.3' and return m:ss.xx"""
    if text is None:
        return None
    s = str(text).strip()
    if s == "" or s.lower() == "none":
        return None
    if ":" not in s:
        # assume seconds.hundredths
        # if "34.1" -> 34.10, if "34" -> 34.00
        if "." in s:
            sec, frac = s.split(".", 1)
            frac = (frac + "00")[:2]
        else:
            sec, frac = s, "00"
        m = 0
        ssec = int(sec)
        mm = ssec // 60
        ss = ssec % 60
        return f"{mm}:{ss:02d}.{frac:0<2}"
    # variants like m:ss, m:ss.xx, mm:ss.x
    try:
        m, rest = s.split(":", 1)
        if "." in rest:
            sec, frac = rest.split(".", 1)
            frac = (frac + "00")[:2]
        else:
            sec, frac = rest, "00"
        mm = int(m)
        ss = int(sec)
        return f"{mm}:{ss:02d}.{frac:0<2}"
    except Exception:
        return s  # keep raw, better than crashing

def to_time_seconds(s: str) -> float:
    if not s or s is None or s == "None":
        return None
    s = str(s)
    try:
        m, rest = s.split(":")
        if "." in rest:
            sec, frac = rest.split(".")
            return int(m)*60 + int(sec) + int(frac)/100.0
        else:
            return int(m)*60 + int(rest)
    except Exception:
        return None

def best_by_person_event_length(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Seconds"] = df["Result"].apply(to_time_seconds)
    df = df.dropna(subset=["Seconds"])
    # The best per (Name, EventName, LengthMeters)
    idx = df.groupby(["Name", "EventName", "LengthMeters"])["Seconds"].idxmin()
    best = df.loc[idx].copy()
    best["is_best"] = True
    # Merge flag back
    merged = df.merge(best[["Name","EventName","LengthMeters","Seconds","is_best"]],
                      on=["Name","EventName","LengthMeters","Seconds"],
                      how="left")
    merged["is_best"] = merged["is_best"].fillna(False)
    return merged

def github_headers():
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token:
        return None
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

def github_repo():
    return st.secrets.get("REPO", "")  # format owner/repo

def github_put_file(path_in_repo: str, content_bytes: bytes, commit_message: str) -> Tuple[bool, str]:
    headers = github_headers()
    repo = github_repo()
    if not headers or not repo:
        return False, "GitHub Token/Repo 未配置"
    # Get sha if exists
    api = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    # First check if exists
    resp = requests.get(api, headers=headers)
    sha = None
    if resp.status_code == 200:
        try:
            sha = resp.json().get("sha")
        except Exception:
            sha = None

    b64 = base64.b64encode(content_bytes).decode("utf-8")
    payload = {
        "message": commit_message,
        "content": b64
    }
    if sha:
        payload["sha"] = sha

    put = requests.put(api, headers=headers, json=payload)
    if put.status_code in (200, 201):
        return True, "OK"
    return False, f"{put.status_code} {put.text}"

# =============== UI sections ===============

def section_query():
    st.header("🏊‍♀️ 游泳成绩查询 / 对比")
    # Load all results from all meets
    rows = []
    for folder in list_meets():
        meta = load_meta(folder)
        df = load_results(folder)
        if df.empty:
            continue
        # fill meta columns if missing
        df["Date"] = df["Date"].fillna(meta.get("Date"))
        df["City"] = df["City"].fillna(meta.get("City"))
        df["MeetName"] = df["MeetName"].fillna(meta.get("MeetName"))
        df["PoolName"] = df["PoolName"].fillna(meta.get("PoolName"))
        df["LengthMeters"] = df["LengthMeters"].fillna(meta.get("LengthMeters"))
        rows.append(df)
    if rows:
        data = pd.concat(rows, ignore_index=True)
    else:
        st.info("当前没有成绩数据。请先在“赛事管理/成绩录入”中添加。")
        return

    # Filters
    names = sorted([x for x in data["Name"].dropna().unique().tolist()])
    events = ["全部"] + sorted([x for x in data["EventName"].dropna().unique().tolist()])
    lengths = ["全部"] + sorted([str(int(x)) for x in pd.to_numeric(data["LengthMeters"], errors="coerce").dropna().unique()])

    sel_names = st.multiselect("Name（可多选）", options=names, default=names[:1] if names else [])
    sel_event = st.selectbox("Event", options=events, index=0)
    sel_len = st.selectbox("Length (Meters)", options=lengths, index=0)

    df = data.copy()
    if sel_names:
        df = df[df["Name"].isin(sel_names)]
    if sel_event != "全部":
        df = df[df["EventName"] == sel_event]
    if sel_len != "全部":
        df = df[df["LengthMeters"].astype(str) == sel_len]

    df = df.sort_values(["Name","EventName","Date"])
    st.dataframe(df[["Name","Date","EventName","LengthMeters","Result","Rank","City","MeetName","PoolName"]], use_container_width=True)

    # Line chart by Result time
    if not df.empty:
        plot = df.copy()
        plot["Seconds"] = plot["Result"].apply(to_time_seconds)
        plot = plot.dropna(subset=["Seconds"])
        if not plot.empty:
            chart = alt.Chart(plot).mark_line(point=True).encode(
                x="Date:T",
                y="Seconds:Q",
                color="Name:N",
                tooltip=["Name","Date","EventName","LengthMeters","Result"]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)


def section_manage():
    st.header("📁 赛事管理 / 成绩录入")

    st.subheader("① 新建/选择赛事（meta）")
    date = st.text_input("Date", value=datetime.today().strftime("%Y-%m-%d"))
    city = st.text_input("City", value="Chiang Mai")
    meetname = st.text_input("MeetName", value="Local Meet")
    poolname = st.text_input("PoolName", value="National Sports University Chiang Mai Campus")
    length = st.number_input("LengthMeters", min_value=0, step=1, value=25)
    folder_name = f"{date}_{city}_{poolname}"
    folder_name = folder_name.replace("/", "-")

    if st.button("保存赛事信息（写入/推送 meta.csv）", type="primary"):
        meta = {
            "Date": date, "City": city, "MeetName": meetname,
            "PoolName": poolname, "LengthMeters": int(length)
        }
        save_meta(folder_name, meta)
        # also push to GitHub if configured
        repo_path = f"{MEETS_ROOT}/{folder_name}/meta.csv"
        content_bytes = pd.DataFrame([meta]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        ok, msg = github_put_file(repo_path, content_bytes, f"Save meta for {folder_name}")
        if ok:
            st.success(f"已保存：{repo_path}（已推送 GitHub）")
        else:
            st.warning(f"已保存：{repo_path}（GitHub 推送失败：{msg}）")

    st.divider()

    st.subheader("② 新增成绩（results）")
    existing = list_meets()
    if not existing:
        st.info("还没有任何赛事，请先保存一条赛事信息（上面的按钮）。")
        return
    sel_meet = st.selectbox("选择赛事文件夹", options=existing, index=0, key="meet_for_add")

    evts_default = [
        "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle",
        "25m Backstroke","50m Backstroke","100m Backstroke","200m Backstroke",
        "25m Breaststroke","50m Breaststroke","100m Breaststroke","200m Breaststroke",
        "25m Butterfly","50m Butterfly","100m Butterfly","200m Butterfly",
        "100m IM","200m IM","400m IM"
    ]
    evt = st.selectbox("Event 选择", options=sorted(set(evts_default)), index=2)

    nrows = st.number_input("本次录入行数", min_value=1, max_value=20, step=1, value=1)
    meta = load_meta(sel_meet)

    add_rows = []
    for i in range(int(nrows)):
        st.markdown(f"**记录 {i+1}**")
        c1,c2,c3,c4 = st.columns([1,2,1,2])
        with c1:
            name = st.text_input(f"Name_{i}", value="Anna", key=f"nm_{i}")
        with c2:
            ev = st.text_input(f"EventName_{i}", value=evt, key=f"ev_{i}")
        with c3:
            res_in = st.text_input(f"Result_{i}", value="34.12", key=f"rs_{i}")
        with c4:
            rank = st.number_input(f"Rank_{i}", min_value=0, step=1, value=0, key=f"rk_{i}")
        note = st.text_input(f"Note_{i}", value="", key=f"nt_{i}")

        add_rows.append({
            "Name": name.strip(),
            "EventName": normalize_event_name(ev),
            "Result": parse_time_input(res_in),
            "Rank": int(rank),
            "Note": note.strip(),
            "Date": meta.get("Date"),
            "City": meta.get("City"),
            "MeetName": meta.get("MeetName"),
            "PoolName": meta.get("PoolName"),
            "LengthMeters": int(meta.get("LengthMeters")) if pd.notna(meta.get("LengthMeters")) else None
        })

    colA, colB = st.columns(2)
    push_gh = colA.checkbox("提交到 GitHub（免下载上传）", value=True)
    save_local = colB.checkbox("同时保存到本地 meets/ 目录（调试用）", value=True)

    if st.button("保存这些成绩", type="primary"):
        df_old = load_results(sel_meet)
        df_new = pd.DataFrame(add_rows)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # write local
        if save_local:
            save_results(sel_meet, df_all)
        # push both meta and results to github
        if push_gh:
            repo_meta = f"{MEETS_ROOT}/{sel_meet}/meta.csv"
            repo_results = f"{MEETS_ROOT}/{sel_meet}/results.csv"
            meta_bytes = pd.DataFrame([load_meta(sel_meet)]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            okm, msgm = github_put_file(repo_meta, meta_bytes, f"Save meta for {sel_meet}")
            res_bytes = df_all[RESULT_COLS].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            okr, msgr = github_put_file(repo_results, res_bytes, f"Save results for {sel_meet}")
            if okm and okr:
                st.success(f"已保存并推送：{repo_results}")
            else:
                st.warning(f"本地已保存；GitHub 推送部分失败 meta:{okm} results:{okr} -> {msgm} / {msgr}")
        else:
            st.success("本地已保存。")

    st.divider()
    st.subheader("③ 已登记记录（可编辑/删除）")
    sel_edit_meet = st.selectbox("选择赛事查看/编辑", options=existing, index=existing.index(sel_meet) if sel_meet in existing else 0, key="meet_for_edit")
    df = load_results(sel_edit_meet).copy()
    if df.empty:
        st.info("该赛事暂无记录。")
    else:
        # Add helper delete column
        df["Del"] = False
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Del": st.column_config.CheckboxColumn("删除", help="勾选后点击下方‘保存更改’即可真正删除该行。")
            },
            key="editor_existing_results"
        )
        if st.button("保存更改（写入 results.csv）", type="primary"):
            # 1) Normalize Result format
            edited["Result"] = edited["Result"].apply(parse_time_input)
            # 2) Actually delete rows marked for deletion
            keep = edited[edited.get("Del", False) != True].copy()
            if "Del" in keep.columns:
                keep = keep.drop(columns=["Del"])
            # 3) enforce columns/order and write
            for c in RESULT_COLS:
                if c not in keep.columns:
                    keep[c] = None
            keep = keep[RESULT_COLS]
            save_results(sel_edit_meet, keep)
            # Optionally push to GitHub as well
            repo_results = f"{MEETS_ROOT}/{sel_edit_meet}/results.csv"
            res_bytes = keep.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            ok, msg = github_put_file(repo_results, res_bytes, f"Update results for {sel_edit_meet}")
            if ok:
                st.success("更改已保存（本地 & GitHub）。")
            else:
                st.success("更改已保存到本地。GitHub 推送未配置或失败。")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    page = st.radio("页面", ["查询/对比", "赛事管理/录入"], horizontal=False, index=0)
    ensure_meets_root()

    if page == "查询/对比":
        section_query()
    else:
        section_manage()

if __name__ == "__main__":
    main()
