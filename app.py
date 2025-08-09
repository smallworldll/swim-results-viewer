# app.py
# -*- coding: utf-8 -*-
import os
import re
import io
import json
import base64
import datetime as dt
from typing import List, Tuple, Optional, Dict, Set

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# 基本设置
# -----------------------------
st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide")

MEETS_ROOT = "meets"  # 赛事根目录
META_FILE = "meta.csv"
RESULTS_FILE = "results.csv"

# 预置项目（可自行扩展）
DEFAULT_EVENTS = [
    # Freestyle
    "25m Freestyle",
    "50m Freestyle",
    "100m Freestyle",
    "200m Freestyle",
    "400m Freestyle",
    # Backstroke
    "25m Backstroke",
    "50m Backstroke",
    "100m Backstroke",
    "200m Backstroke",
    # Breaststroke
    "25m Breaststroke",
    "50m Breaststroke",
    "100m Breaststroke",
    "200m Breaststroke",
    # Butterfly
    "25m Butterfly",
    "50m Butterfly",
    "100m Butterfly",
    "200m Butterfly",
    # IM
    "100m IM",
    "200m IM",
    "400m IM",
]

# 高亮颜色（不同选手）
HIGHLIGHT_COLORS = [
    "#d62728",  # red
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # teal
]

# -----------------------------
# 小工具：时间解析/格式化
# -----------------------------
def parse_time_to_seconds(x: str) -> Optional[float]:
    """
    接受 "m:ss.xx"、"mm:ss"、"ss.xx"、"ss" 等形式，返回秒(float)；解析失败返回 None
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        if ":" in s:
            # m:ss.xx 或 mm:ss
            m, rest = s.split(":", 1)
            m = int(m)
            sec = float(rest)
            return m * 60 + sec
        else:
            # 纯秒，可能是 "34.12"
            return float(s)
    except Exception:
        return None


def format_seconds_to_mmss(seconds: Optional[float]) -> str:
    if seconds is None:
        return "None"
    try:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m}:{s:05.2f}"
    except Exception:
        return "None"


# -----------------------------
# 路径/命名工具
# -----------------------------
def sanitize_segment(seg: str) -> str:
    """
    保留中文/英文/数字/空格/下划线/连字符/括号/点号，把多空格压成一个，再把空格换成下划线。
    """
    seg = (seg or "").strip()
    seg = re.sub(r"\s+", " ", seg)
    seg = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9 _\-\(\)\.]", "", seg)
    return seg.replace(" ", "_")


# -----------------------------
# 文件/数据 IO
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_meets() -> List[str]:
    """列出现有赛事文件夹（按名称排序）"""
    if not os.path.isdir(MEETS_ROOT):
        return []
    items = [
        d for d in os.listdir(MEETS_ROOT)
        if os.path.isdir(os.path.join(MEETS_ROOT, d))
    ]
    items.sort()
    return items


def read_meta(meet_dir: str) -> pd.Series:
    """读取 meta.csv -> Series：Date, City, MeetName, PoolName, LengthMeters"""
    p = os.path.join(MEETS_ROOT, meet_dir, META_FILE)
    if not os.path.isfile(p):
        return pd.Series(dtype="object")
    df = pd.read_csv(p)
    if df.empty:
        return pd.Series(dtype="object")
    # 只取第一行，确保列顺序
    cols = ["Date", "City", "MeetName", "PoolName", "LengthMeters"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].iloc[0]


def write_meta(meet_dir: str, meta: Dict) -> None:
    """写 meta.csv"""
    ensure_dir(os.path.join(MEETS_ROOT, meet_dir))
    df = pd.DataFrame([meta])
    # 统一列顺序
    df = df[["Date", "City", "MeetName", "PoolName", "LengthMeters"]]
    df.to_csv(os.path.join(MEETS_ROOT, meet_dir, META_FILE), index=False, encoding="utf-8-sig")


def read_results(meet_dir: str) -> pd.DataFrame:
    """读取 results.csv；若不存在返回空 DataFrame"""
    p = os.path.join(MEETS_ROOT, meet_dir, RESULTS_FILE)
    if not os.path.isfile(p):
        return pd.DataFrame(columns=["Name", "EventName", "Result", "Rank", "Note"])
    df = pd.read_csv(p)
    for c in ["Name", "EventName", "Result", "Rank", "Note"]:
        if c not in df.columns:
            df[c] = None
    return df[["Name", "EventName", "Result", "Rank", "Note"]]


def write_results(meet_dir: str, df: pd.DataFrame) -> None:
    p = os.path.join(MEETS_ROOT, meet_dir, RESULTS_FILE)
    df.to_csv(p, index=False, encoding="utf-8-sig")


def append_results(meet_dir: str, rows: List[Dict]) -> None:
    """追加写入 results.csv"""
    df_old = read_results(meet_dir)
    df_new = pd.DataFrame(rows)
    df = pd.concat([df_old, df_new], ignore_index=True)
    write_results(meet_dir, df)


def aggregate_all_results() -> pd.DataFrame:
    """
    汇总所有赛事的 results + meta，返回统一 DataFrame：
    [Name, EventName, Date, City, PoolName, LengthMeters, Result, ResultSeconds, Rank, Note, MeetFolder]
    """
    meets = list_meets()
    rows = []
    for md in meets:
        meta = read_meta(md)
        if meta.empty:
            continue
        res = read_results(md)
        if res.empty:
            continue
        for _, r in res.iterrows():
            sec = parse_time_to_seconds(r.get("Result"))
            rows.append({
                "Name": r.get("Name"),
                "EventName": r.get("EventName"),
                "Date": meta.get("Date"),
                "City": meta.get("City"),
                "PoolName": meta.get("PoolName"),
                "LengthMeters": meta.get("LengthMeters"),
                "Result": r.get("Result"),
                "ResultSeconds": sec,
                "Rank": r.get("Rank"),
                "Note": r.get("Note"),
                "MeetFolder": md,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        # 转日期，保证排序
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            pass
        # 排序：日期升序
        df = df.sort_values(["Date", "Name", "EventName"], kind="mergesort")
    return df


# -----------------------------
# GitHub 上传（可选）
# -----------------------------
def _github_headers() -> Optional[Dict]:
    try:
        token = st.secrets["GITHUB_TOKEN"]
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
    except Exception:
        return None


def _repo_path() -> Optional[Tuple[str, str]]:
    """
    返回 (owner, repo)；st.secrets["REPO"] 形如 "user/repo"
    """
    try:
        full = st.secrets["REPO"]
        owner, repo = full.split("/", 1)
        return owner, repo
    except Exception:
        return None


def _github_get_file_sha(owner: str, repo: str, path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = _github_headers()
    if headers is None:
        return None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        return data.get("sha")
    return None


def _github_put_file(owner: str, repo: str, path: str, content_bytes: bytes, message: str) -> Tuple[bool, str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = _github_headers()
    if headers is None:
        return False, "未配置 GITHUB_TOKEN/REPO"

    sha = _github_get_file_sha(owner, repo, path)
    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, data=json.dumps(payload))
    if 200 <= r.status_code < 300:
        return True, "OK"
    else:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        return False, f"{r.status_code} {msg}"


def push_meet_to_github(meet_dir: str) -> Tuple[bool, str]:
    """
    将 meet_dir 下的 meta.csv、results.csv 上传到仓库同路径（meets/...）
    """
    rep = _repo_path()
    if rep is None:
        return False, "未配置 REPO"

    owner, repo = rep
    base = os.path.join(MEETS_ROOT, meet_dir)

    # meta.csv
    meta_path = os.path.join(base, META_FILE)
    if os.path.isfile(meta_path):
        with open(meta_path, "rb") as f:
            ok, msg = _github_put_file(
                owner, repo, os.path.join(MEETS_ROOT, meet_dir, META_FILE),
                f.read(), f"update {meet_dir}/{META_FILE}"
            )
        if not ok:
            return False, f"meta.csv 推送失败：{msg}"

    # results.csv（可能不存在）
    res_path = os.path.join(base, RESULTS_FILE)
    if os.path.isfile(res_path):
        with open(res_path, "rb") as f:
            ok, msg = _github_put_file(
                owner, repo, os.path.join(MEETS_ROOT, meet_dir, RESULTS_FILE),
                f.read(), f"update {meet_dir}/{RESULTS_FILE}"
            )
        if not ok:
            return False, f"results.csv 推送失败：{msg}"

    return True, "已推送"


# -----------------------------
# 高亮与图表辅助
# -----------------------------
def compute_seed_indices(df: pd.DataFrame) -> Set[int]:
    """
    返回当前 DataFrame 中“最佳成绩”的行索引集合：
    按 (Name, EventName, LengthMeters) 分组，ResultSeconds 最小的行视为种子成绩。
    """
    seeds: Set[int] = set()
    if df.empty or "ResultSeconds" not in df.columns:
        return seeds
    g = df.dropna(subset=["ResultSeconds"]).groupby(["Name", "EventName", "LengthMeters"])
    for _, sub in g:
        idx = sub["ResultSeconds"].idxmin()
        seeds.add(idx)
    return seeds


def apply_seed_style(df_disp: pd.DataFrame, seed_indices: Set[int]) :
    """
    仅对 Result 列应用高亮；不同选手不同颜色（按 Name）。
    df_disp 必须包含 Name 列；其 index 与源 df 一致。
    """
    if df_disp.empty:
        return df_disp.style

    people = df_disp["Name"].fillna("Unknown").unique().tolist()
    color_map = {n: HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)] for i, n in enumerate(people)}

    def row_style(row):
        styles = []
        for col in df_disp.columns:
            if (row.name in seed_indices) and (col == "Result"):
                c = color_map.get(row.get("Name"), "#d62728")
                styles.append(f"color: {c}; font-weight: 700")
            else:
                styles.append("")
        return styles

    return df_disp.style.apply(row_style, axis=1)


# -----------------------------
# UI：查询/对比
# -----------------------------
def page_browse():
    st.header("🏊 游泳成绩查询 / 对比")

    df = aggregate_all_results()
    if df.empty:
        st.info("当前没有成绩数据。请先在“赛事管理/成绩录入”中添加。")
        return

    # 过滤器
    with st.expander("🔎 请选择筛选条件", expanded=True):
        all_names = sorted([x for x in df["Name"].dropna().unique().tolist()])
        default_names = ["Anna"] if "Anna" in all_names else (all_names[:1] if all_names else [])
        names = st.multiselect("Name（可多选）", options=all_names, default=default_names)

        events = ["All"] + sorted([x for x in df["EventName"].dropna().unique().tolist()])
        event = st.selectbox("Event", events, index=events.index("All") if "All" in events else 0)

        lengths = ["All"] + sorted([int(x) for x in df["LengthMeters"].dropna().unique().tolist()])
        length = st.selectbox("Length (Meters)", lengths, index=0)

        poolnames = ["All"] + sorted([x for x in df["PoolName"].dropna().unique().tolist()])
        poolname = st.selectbox("Pool Name", poolnames, index=0)

        cities = ["All"] + sorted([x for x in df["City"].dropna().unique().tolist()])
        city = st.selectbox("City", cities, index=0)

        dates = ["All"] + sorted([str(x.date()) for x in df["Date"].dropna().unique().tolist()])
        date_sel = st.selectbox("Date", dates, index=0)

    f = df.copy()
    if names:
        f = f[f["Name"].isin(names)]
    if event != "All":
        f = f[f["EventName"] == event]
    if length != "All":
        f = f[f["LengthMeters"] == int(length)]
    if poolname != "All":
        f = f[f["PoolName"] == poolname]
    if city != "All":
        f = f[f["City"] == city]
    if date_sel != "All":
        try:
            dt_obj = pd.to_datetime(date_sel)
            f = f[f["Date"] == dt_obj]
        except Exception:
            pass

    # 展示表格（带高亮）
    if not f.empty:
        # 准备显示列
        disp = f.copy()
        disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
        disp["Result"] = disp["ResultSeconds"].apply(format_seconds_to_mmss)
        disp = disp[["Name", "Date", "EventName", "Result", "Rank", "Note", "PoolName", "City", "LengthMeters", "MeetFolder"]]
        # 计算种子索引，并套用到 disp（索引一致）
        seeds = compute_seed_indices(f)
        styled = apply_seed_style(disp, seeds)
        st.subheader("📑 比赛记录")
        st.dataframe(styled, use_container_width=True)
    else:
        st.warning("当前条件下没有数据。")

    # 折线图（同一项目的趋势图）
    if not f.empty:
        st.subheader("📈 成绩折线图（单位：秒，越低越好）")
        if event == "All":
            st.info("请选择具体的 Event 以绘制趋势图。")
        else:
            f2 = f.dropna(subset=["ResultSeconds"]).copy()
            if f2.empty:
                st.info("没有可绘图的成绩。")
            else:
                pivot = (
                    f2.pivot_table(index="Date", columns="Name", values="ResultSeconds", aggfunc="min")
                    .sort_index()
                )
                st.line_chart(pivot, height=320)


# -----------------------------
# UI：赛事管理 / 成绩录入
# -----------------------------
def page_manage():
    st.header("📁 赛事管理 / 成绩录入")

    # 选择模式：已有赛事 / 新建赛事
    mode = st.radio("操作", ["选择已有赛事", "新建赛事"], horizontal=True)

    # 新建赛事表单
    if mode == "新建赛事":
        with st.form("new_meet"):
            date_val = st.date_input("Date", value=dt.date.today())
            city_raw = st.text_input("City", value="")
            pool_raw = st.text_input("PoolName", value="")
            length = st.number_input("LengthMeters", min_value=10, max_value=100, value=25, step=1)
            meet_name = st.text_input("MeetName", value="")
            push_on_create = st.checkbox("创建后立即推送到 GitHub（推荐，避免丢失）", value=True)
            submit = st.form_submit_button("创建/保存赛事")

        if submit:
            if not city_raw or not pool_raw:
                st.error("City / PoolName 不能为空。")
            else:
                city = city_raw.strip()
                poolname = pool_raw.strip()
                meet_dir = f"{date_val.isoformat()}_{sanitize_segment(city)}_{sanitize_segment(poolname)}"
                meta = {
                    "Date": str(date_val),
                    "City": city,
                    "MeetName": meet_name if meet_name else f"{city} Meet",
                    "PoolName": poolname,
                    "LengthMeters": int(length),
                }
                write_meta(meet_dir, meta)
                # 确保 results.csv 存在（空表）
                if not os.path.exists(os.path.join(MEETS_ROOT, meet_dir, RESULTS_FILE)):
                    write_results(meet_dir, pd.DataFrame(columns=["Name","EventName","Result","Rank","Note"]))
                st.success(f"已创建赛事：{meet_dir}")
                if push_on_create:
                    ok, msg = push_meet_to_github(meet_dir)
                    if ok:
                        st.success("（创建）已推送到 GitHub。")
                    else:
                        st.warning(f"GitHub 推送失败：{msg}")

    # 选择已有赛事
    else:
        meets = list_meets()
        if not meets:
            st.info("当前无赛事文件夹，请切换到“新建赛事”创建。")
            return

        meet_dir = st.selectbox("选择已有赛事（文件夹）", options=meets)
        meta = read_meta(meet_dir)
        if meta.empty:
            st.warning("该赛事缺少 meta.csv。")
            return

        colm = st.columns([1, 1, 2, 2, 1])
        with colm[0]:
            st.write("**Date**")
            st.info(str(meta.get("Date")))
        with colm[1]:
            st.write("**City**")
            st.info(str(meta.get("City")))
        with colm[2]:
            st.write("**PoolName**")
            st.info(str(meta.get("PoolName")))
        with colm[3]:
            st.write("**MeetName**")
            st.info(str(meta.get("MeetName")))
        with colm[4]:
            st.write("**Length**")
            st.info(str(meta.get("LengthMeters")))

        # 显示现有成绩
        df_exist = read_results(meet_dir)
        st.subheader("📜 该赛事已有成绩")
        if df_exist.empty:
            st.info("暂无成绩记录。")
        else:
            tdf = df_exist.copy()
            tdf["ResultSeconds"] = tdf["Result"].apply(parse_time_to_seconds)
            tdf["ResultFmt"] = tdf["ResultSeconds"].apply(format_seconds_to_mmss)
            tdf = tdf[["Name", "EventName", "ResultFmt", "Rank", "Note"]].rename(columns={"ResultFmt": "Result"})
            st.dataframe(tdf, use_container_width=True)

        st.markdown("---")

        # 录入成绩
        st.subheader("📝 新增成绩")

        left, right = st.columns([1.2, 0.8])
        with left:
            # 组合“已录项目” + “常用项目”（去重）
            exist_events = sorted([x for x in df_exist["EventName"].dropna().unique().tolist() if x])
            candidates = ["（自定义…）"] + sorted(set(exist_events + DEFAULT_EVENTS))
            event_choice = st.selectbox("Event 选择", options=candidates, index=1 if len(candidates) > 1 else 0)
            if event_choice == "（自定义…）":
                event_name = st.text_input("自定义 EventName")
            else:
                event_name = event_choice

        with right:
            batch_n = st.number_input("本次录入行数", 1, 20, value=3, step=1)

        st.caption("时间格式可填 34.12 或 0:34.12（系统会统一解析为秒再格式化显示）。")

        rows = []
        for i in range(int(batch_n)):
            c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1, 0.7, 1.2])
            name = c1.text_input(f"Name_{i+1}", value="", key=f"name_{i}")
            ev = c2.text_input(f"EventName_{i+1}", value=event_name, key=f"ev_{i}")
            result = c3.text_input(f"Result_{i+1}", value="", key=f"res_{i}", placeholder="34.12 或 0:34.12")
            rank = c4.number_input(f"Rank_{i+1}", min_value=0, max_value=999, value=0, step=1, key=f"rank_{i}")
            note = c5.text_input(f"Note_{i+1}", value="", key=f"note_{i}")
            rows.append({"Name": name, "EventName": ev, "Result": result, "Rank": int(rank), "Note": note})

        push_flag = st.checkbox("提交到 GitHub（免下载上传）", value=True)
        also_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=True)

        if st.button("保存这些成绩", type="primary"):
            # 过滤有效行：Name + EventName + Result 至少不空
            valid_rows = []
            for r in rows:
                if (r["Name"] or r["EventName"] or r["Result"]):
                    valid_rows.append(r)
            if not valid_rows:
                st.warning("没有有效行，无需保存。")
            else:
                append_results(meet_dir, valid_rows)
                st.success(f"已保存 {len(valid_rows)} 条。")

                if push_flag:
                    ok, msg = push_meet_to_github(meet_dir)
                    if ok:
                        st.success("已推送到 GitHub。")
                    else:
                        st.warning(f"GitHub 推送失败：{msg}")

                if also_local:
                    st.info(f"文件已写入：{os.path.join(MEETS_ROOT, meet_dir)}")


# -----------------------------
# 主路由
# -----------------------------
def main():
    with st.sidebar:
        st.title("页面")
        page = st.radio("",
                        ["查询 / 对比", "赛事管理 / 录入"],
                        index=0)

    if page == "查询 / 对比":
        page_browse()
    else:
        page_manage()


if __name__ == "__main__":
    os.makedirs(MEETS_ROOT, exist_ok=True)
    main()
