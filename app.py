
# -*- coding: utf-8 -*-
"""
Swim Results Viewer / Recorder
- Storage: meets/YYYY-MM-DD_City/{meta.csv, results.csv}
- Time input accepted: "m:ss.xx" (e.g., 1:02.45) OR "ss.xx" (e.g., 34.12)
  We normalize and store as "m:ss.xx".
- Rank is manual (optional). No auto-ranking on save.
"""

import os
import re
import base64
from datetime import date
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import requests

APP_TITLE = "🏊 游泳成绩系统（赛事制）"
MEETS_DIR = "meets"


# ---------------- Helpers ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def time_to_seconds(tstr: str) -> float:
    """
    Convert time string to seconds.
    Accepts either:
      - m:ss.xx (e.g., 1:02.45)
      - ss.xx   (e.g., 34.12) -> interpreted as 0:34.12
    Returns np.nan if invalid.
    """
    if not isinstance(tstr, str):
        return np.nan
    t = tstr.strip()
    # m:ss.xx
    m1 = re.match(r'^(\d+):([0-5]\d)\.(\d{2})$', t)
    if m1:
        mm = int(m1.group(1)); ss = int(m1.group(2)); cs = int(m1.group(3))
        return mm * 60 + ss + cs / 100.0
    # ss.xx (0-59.99)
    m2 = re.match(r'^([0-5]?\d)\.(\d{2})$', t)
    if m2:
        ss = int(m2.group(1)); cs = int(m2.group(2))
        return ss + cs / 100.0
    return np.nan


def seconds_to_time(secs: float) -> str:
    if pd.isna(secs):
        return ""
    m = int(secs // 60)
    s = int(secs % 60)
    cs = int(round((secs - int(secs)) * 100))
    return f"{m}:{s:02d}.{cs:02d}"


def normalize_time_str(tstr: str) -> str:
    """Parse with time_to_seconds and reformat as m:ss.xx; raise ValueError if invalid."""
    sec = time_to_seconds(tstr)
    if pd.isna(sec):
        raise ValueError("时间格式需为 m:ss.xx 或 ss.xx")
    return seconds_to_time(sec)


def list_meets() -> List[str]:
    if not os.path.isdir(MEETS_DIR):
        return []
    names = []
    for x in os.listdir(MEETS_DIR):
        p = os.path.join(MEETS_DIR, x)
        if os.path.isdir(p):
            names.append(x)
    names.sort()
    return names


def meta_path(meet_folder: str) -> str:
    return os.path.join(MEETS_DIR, meet_folder, "meta.csv")


def results_path(meet_folder: str) -> str:
    return os.path.join(MEETS_DIR, meet_folder, "results.csv")


def read_meta(meet_folder: str) -> pd.DataFrame:
    p = meta_path(meet_folder)
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame(columns=["Date", "City", "MeetName", "PoolName", "LengthMeters"])


def read_results(meet_folder: str) -> pd.DataFrame:
    p = results_path(meet_folder)
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame(columns=["Name", "EventName", "Date", "City", "PoolName", "LengthMeters", "Result", "Rank", "Note"])


def write_meta(meet_folder: str, df: pd.DataFrame):
    df = df[["Date", "City", "MeetName", "PoolName", "LengthMeters"]]
    df.to_csv(meta_path(meet_folder), index=False)


def write_results(meet_folder: str, df: pd.DataFrame):
    df = df[["Name", "EventName", "Date", "City", "PoolName", "LengthMeters", "Result", "Rank", "Note"]]
    df.to_csv(results_path(meet_folder), index=False)


def push_to_github_if_needed(local_path: str, repo_path: str) -> Tuple[bool, str]:
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo_full = st.secrets.get("REPO", None)
    if not token or not repo_full:
        return False, "GitHub Secrets 未配置，跳过推送。"

    try:
        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return False, f"读取文件失败：{e}"

    api_base = "https://api.github.com"
    url = f"{api_base}/repos/{repo_full}/contents/{repo_path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    sha = None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        try:
            sha = r.json().get("sha")
        except Exception:
            sha = None

    data = {"message": f"update {repo_path} via app", "content": content_b64, "branch": "main"}
    if sha:
        data["sha"] = sha

    r = requests.put(url, headers=headers, json=data)
    if r.status_code in (200, 201):
        return True, "已推送到 GitHub。"
    return False, f"GitHub 推送失败：{r.status_code} {r.text}"


@st.cache_data
def load_all_results() -> pd.DataFrame:
    rows = []
    for m in list_meets():
        df = read_results(m)
        rows.append(df)
    if rows:
        big = pd.concat(rows, ignore_index=True)
    else:
        big = pd.DataFrame(columns=["Name", "EventName", "Date", "City", "PoolName", "LengthMeters", "Result", "Rank", "Note"])
    return big


# ---------------- Pages ----------------
def page_manage():
    st.subheader("📁 赛事管理 / 成绩录入")

    # 选择或新建赛事
    with st.expander("① 选择或新建赛事", expanded=True):
        all_meets = list_meets()
        mode = st.radio("操作", ["选择已有赛事", "新建赛事"], horizontal=True)
        if mode == "选择已有赛事":
            if not all_meets:
                st.info("目前没有赛事，请先新建。")
                return
            meet = st.selectbox("选择赛事（文件夹）", all_meets)
            if not meet:
                return
            meta = read_meta(meet)
            st.dataframe(meta, hide_index=True, use_container_width=True)
        else:
            d = st.date_input("比赛日期", value=date.today(), format="YYYY-MM-DD")
            city = st.text_input("城市（英文/拼音）", value="ChiangMai")
            meet_name = st.text_input("赛事名称", value="Local Meet")
            pool_name = st.text_input("泳池名称", value="Kawila")
            length = st.selectbox("泳池长度（米）", [25, 50], index=1)
            meet = f"{d.strftime('%Y-%m-%d')}_{city}"
            st.caption(f"将创建赛事文件夹：`{MEETS_DIR}/{meet}`")

            if st.button("创建/更新赛事元信息"):
                ensure_dir(os.path.join(MEETS_DIR, meet))
                meta = pd.DataFrame([{
                    "Date": d.strftime("%Y-%m-%d"),
                    "City": city,
                    "MeetName": meet_name,
                    "PoolName": pool_name,
                    "LengthMeters": int(length),
                }])
                write_meta(meet, meta)
                st.success("meta.csv 已写入。")

    if not locals().get("meet"):
        return

    # 选择或创建项目 EventName
    with st.expander("② 选择或创建项目（EventName）", expanded=True):
        res = read_results(meet)
        existed_events = sorted(res["EventName"].dropna().unique().tolist()) if not res.empty else []
        event_mode = st.radio("方式", ["选择已有", "新建"], horizontal=True)
        if event_mode == "选择已有":
            if not existed_events:
                st.info("赛事下还没有项目，请先新建。")
                return
            event = st.selectbox("EventName", existed_events)
        else:
            event = st.text_input("新项目名称（如 '100m Freestyle'）", value="100m Freestyle")

        if not event:
            return

    # 成绩录入（不自动排名；时间可填 ss.xx 或 m:ss.xx，统一存 m:ss.xx）
    with st.expander("③ 录入本项目成绩（不自动排名）", expanded=True):
        meta_df = read_meta(meet)
        if meta_df.empty:
            st.warning("请先完善赛事元信息。")
            return
        meta_row = meta_df.iloc[0].to_dict()

        if "entries" not in st.session_state:
            st.session_state.entries = [{"Name": "", "Result": "", "Rank": "", "Note": ""}]

        def add_line():
            st.session_state.entries.append({"Name": "", "Result": "", "Rank": "", "Note": ""})

        def clear_lines():
            st.session_state.entries = [{"Name": "", "Result": "", "Rank": "", "Note": ""}]

        cols = st.columns([2, 2, 1, 3])
        cols[0].markdown("**Name**")
        cols[1].markdown("**Result (m:ss.xx 或 ss.xx)**")
        cols[2].markdown("**Rank (可空)**")
        cols[3].markdown("**Note (可空)**")

        for i, row in enumerate(st.session_state.entries):
            c1, c2, c3, c4 = st.columns([2, 2, 1, 3])
            row["Name"] = c1.text_input(f"Name_{i}", value=row.get("Name", ""))
            row["Result"] = c2.text_input(f"Result_{i}", value=row.get("Result", ""), placeholder="如 1:02.45 或 34.12")
            row["Rank"] = c3.text_input(f"Rank_{i}", value=row.get("Rank", ""))
            row["Note"] = c4.text_input(f"Note_{i}", value=row.get("Note", ""))

        c_add, c_clear = st.columns(2)
        with c_add:
            st.button("＋ 再加一行", on_click=add_line, type="secondary")
        with c_clear:
            st.button("清空以上行", on_click=clear_lines)

        push_github = st.checkbox("提交到 GitHub（免下载上传）", value=False)
        save_local = st.checkbox("同时保存到本地 meets/ 目录", value=True)

        if st.button("保存这些成绩"):
            new_rows = []
            for row in st.session_state.entries:
                name = row["Name"].strip()
                result_raw = row["Result"].strip()
                rank = row["Rank"].strip()
                note = row["Note"].strip()

                if not name and not result_raw:
                    continue
                if not name:
                    st.error("Name 不能为空")
                    return

                # parse & normalize time
                sec = time_to_seconds(result_raw)
                if pd.isna(sec):
                    st.error(f"时间格式错误：{result_raw}，需要 m:ss.xx 或 ss.xx")
                    return
                result_norm = seconds_to_time(sec)  # store as m:ss.xx

                # rank 可空或数字
                if rank and not re.match(r"^\d+$", rank):
                    st.error(f"Rank 必须是数字或留空（行：{name}）")
                    return

                new_rows.append({
                    "Name": name,
                    "EventName": event,
                    "Date": meta_row["Date"],
                    "City": meta_row["City"],
                    "PoolName": meta_row["PoolName"],
                    "LengthMeters": int(meta_row["LengthMeters"]),
                    "Result": result_norm,
                    "Rank": int(rank) if rank else "",
                    "Note": note,
                })

            if not new_rows:
                st.info("没有可保存的记录。")
                return

            df_old = read_results(meet)
            df_new = pd.DataFrame(new_rows)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            ensure_dir(os.path.join(MEETS_DIR, meet))
            write_results(meet, df_all)
            st.success(f"已保存 {len(new_rows)} 条。")

            if push_github:
                ok, msg = push_to_github_if_needed(
                    results_path(meet),
                    f"{MEETS_DIR}/{meet}/results.csv"
                )
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

    with st.expander("④ 当前项目成绩（预览与临时排序）", expanded=False):
        df = read_results(meet)
        df_ev = df[df["EventName"] == event].copy()
        st.dataframe(df_ev, hide_index=True, use_container_width=True)

        if not df_ev.empty:
            if st.button("按成绩临时计算排名（仅预览，不写回）"):
                tmp = df_ev.copy()
                tmp["secs"] = tmp["Result"].map(time_to_seconds)
                tmp = tmp.sort_values("secs", ascending=True, na_position="last")
                tmp["Rank*"] = range(1, len(tmp) + 1)
                st.dataframe(tmp.drop(columns=["secs"]), hide_index=True, use_container_width=True)


def highlight_best(df_filtered: pd.DataFrame) -> pd.io.formats.style.Styler:
    if df_filtered.empty:
        return df_filtered.style

    df = df_filtered.copy()
    df["secs"] = df["Result"].map(time_to_seconds)

    mins = df.groupby(["Name", "EventName", "LengthMeters"])["secs"].transform("min")

    def _style_row(row):
        styles = [""] * len(df.columns)
        best = pd.notna(row["secs"]) and (row["secs"] == mins.loc[row.name])
        if best:
            idx = df.columns.get_loc("Result")
            styles[idx] = "color: red; font-weight: 700;"
        return styles

    return df.style.apply(_style_row, axis=1).hide(axis="index")


def page_browse():
    st.subheader("🔎 查询与对比")

    big = load_all_results()
    if big.empty:
        st.info("暂无数据，请先录入。")
        return

    names = sorted(big["Name"].dropna().unique().tolist())
    events = sorted(big["EventName"].dropna().unique().tolist())
    lengths = sorted(big["LengthMeters"].dropna().unique().astype(int).tolist())
    pools = sorted(big["PoolName"].dropna().unique().tolist())
    cities = sorted(big["City"].dropna().unique().tolist())
    dates = sorted(big["Date"].dropna().unique().tolist())

    st.markdown("### 请选择筛选条件")
    sel_names = st.multiselect("Name（可多选）", options=names, default=["Anna"] if "Anna" in names else [])
    sel_event = st.selectbox("Event", options=["全部"] + events, index=0)
    sel_len = st.selectbox("Length (Meters)", options=["全部"] + [str(x) for x in lengths], index=0)
    sel_pool = st.selectbox("Pool Name", options=["全部"] + pools, index=0)
    sel_city = st.selectbox("City", options=["全部"] + cities, index=0)
    sel_date = st.selectbox("Date", options=["全部"] + dates, index=0)

    df = big.copy()
    if sel_names:
        df = df[df["Name"].isin(sel_names)]
    if sel_event != "全部":
        df = df[df["EventName"] == sel_event]
    if sel_len != "全部":
        df = df[df["LengthMeters"].astype(int) == int(sel_len)]
    if sel_pool != "全部":
        df = df[df["PoolName"] == sel_pool]
    if sel_city != "全部":
        df = df[df["City"] == sel_city]
    if sel_date != "全部":
        df = df[df["Date"] == sel_date]

    st.markdown("### 比赛记录")
    if df.empty:
        st.info("empty")
        return

    styled = highlight_best(df)
    try:
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)

    st.markdown("### 成绩趋势（越低越好）")
    dfc = df.copy()
    dfc["secs"] = dfc["Result"].map(time_to_seconds)
    dfc = dfc.dropna(subset=["secs"])
    if dfc.empty:
        st.info("没有可绘制的有效成绩。")
        return

    dfc["Date2"] = pd.to_datetime(dfc["Date"], errors="coerce")
    dfc = dfc.dropna(subset=["Date2"])
    if dfc.empty:
        st.info("日期格式异常，无法绘图。")
        return

    pivot = dfc.pivot_table(index="Date2", columns="Name", values="secs", aggfunc="min").sort_index()
    st.line_chart(pivot, height=300)


def main():
    st.set_page_config(page_title="Swim Results", layout="wide")
    st.title(APP_TITLE)

    tab = st.sidebar.radio("功能", ["赛事管理", "查询与对比"], index=0)
    if tab == "赛事管理":
        page_manage()
    else:
        page_browse()


if __name__ == "__main__":
    main()
