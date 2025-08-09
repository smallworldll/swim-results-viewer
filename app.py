# -*- coding: utf-8 -*-
import os
import re
import io
import json
import base64
from datetime import datetime, date
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
import requests

APP_TITLE = "🏊‍♀️ 游泳成绩系统（赛事制）"
MEETS_ROOT = "meets"  # 所有赛事文件夹的根目录

# ------------- 工具函数 -------------

def ensure_root():
    os.makedirs(MEETS_ROOT, exist_ok=True)


def sanitize_segment(seg: str) -> str:
    """
    清理路径片段：保留常见中英文、数字、下划线、空格和部分符号；将多余空白压缩为单个空格。
    不做截断，尽可能保留原始信息。
    """
    seg = seg.strip()
    seg = re.sub(r"\s+", " ", seg)
    # 允许汉字、英文、数字、空格、下划线、连字符、括号、点号
    allowed = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9 _\-\(\)\.]+")
    seg = allowed.sub("", seg)
    return seg


def folder_from_meta(d: str, city: str, pool: str) -> str:
    # 目录名：YYYY-MM-DD_City_PoolName
    return f"{d}_{city}_{pool}"


def parse_time_input(x: str) -> Optional[float]:
    """
    接受 'm:ss.xx' 或 'ss.xx' 两种格式，返回秒(float)；非法返回 None。
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        # 如果包含冒号，按 m:ss.xx 解析
        if ":" in s:
            m, rest = s.split(":", 1)
            minutes = int(m)
            seconds = float(rest)
            if not (0 <= seconds < 60):
                return None
            return minutes * 60 + seconds
        else:
            # 纯秒（含小数）
            return float(s)
    except Exception:
        return None


def format_time(sec: Optional[float]) -> str:
    if sec is None:
        return ""
    try:
        sec = float(sec)
        m = int(sec // 60)
        s = sec - m * 60
        return f"{m}:{s:05.2f}"
    except Exception:
        return ""


def read_meta(dirpath: str) -> pd.DataFrame:
    meta_fp = os.path.join(dirpath, "meta.csv")
    if os.path.exists(meta_fp):
        df = pd.read_csv(meta_fp)
    else:
        df = pd.DataFrame(columns=["Date", "City", "MeetName", "PoolName", "LengthMeters"])
    return df


def write_meta(dirpath: str, meta_df: pd.DataFrame):
    meta_fp = os.path.join(dirpath, "meta.csv")
    meta_df.to_csv(meta_fp, index=False, encoding="utf-8-sig")


def read_results(dirpath: str) -> pd.DataFrame:
    fp = os.path.join(dirpath, "results.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
    else:
        df = pd.DataFrame(columns=["Name", "EventName", "Result", "Rank", "Note"])
    # 统一字符串类型，避免 NaN 带来的问题
    for c in ["Name", "EventName", "Result", "Note"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")
    if "Rank" in df.columns:
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    return df


def write_results(dirpath: str, res_df: pd.DataFrame):
    fp = os.path.join(dirpath, "results.csv")
    res_df.to_csv(fp, index=False, encoding="utf-8-sig")


def list_meets() -> pd.DataFrame:
    """
    扫描 meets 目录，读取每个赛事的 meta.csv，返回整表
    """
    ensure_root()
    rows = []
    for name in sorted(os.listdir(MEETS_ROOT)):
        dirpath = os.path.join(MEETS_ROOT, name)
        if not os.path.isdir(dirpath):
            continue
        meta_df = read_meta(dirpath)
        if len(meta_df) == 0:
            # 空 meta，跳过
            continue
        m = meta_df.iloc[0].to_dict()
        m["Folder"] = name
        rows.append(m)
    if rows:
        df = pd.DataFrame(rows)
        # 尝试按日期排序
        try:
            df["_d"] = pd.to_datetime(df["Date"])
            df = df.sort_values("_d").drop(columns=["_d"])
        except Exception:
            pass
        return df
    return pd.DataFrame(columns=["Date", "City", "MeetName", "PoolName", "LengthMeters", "Folder"])


def highlighted_style(df: pd.DataFrame, meta: pd.Series) -> object:
    """
    对最佳成绩高亮：同一个人、同一项目、同一池长（LengthMeters）的最小秒数标红。
    """
    work = df.copy()
    work["Seconds"] = work["Result"].apply(parse_time_input)
    styles = pd.DataFrame("", index=work.index, columns=work.columns)

    group_cols = ["Name", "EventName"]
    if "LengthMeters" in meta.index:
        # meta 只有单值，统一注入列
        work["LengthMeters"] = meta["LengthMeters"]
        group_cols = ["Name", "EventName", "LengthMeters"]

    if len(work) > 0:
        mins = work.groupby(group_cols)["Seconds"].transform("min")
        mask = work["Seconds"] == mins
        styles.loc[mask, "Result"] = "color: red; font-weight: 700;"
    return df.style.set_properties(**{"font-size": "14px"}).set_table_styles(
        [{"selector": "th", "props": [("font-size", "14px"), ("text-align", "center")]}]
    ).apply(lambda _: styles, axis=None)


# ---------- GitHub 推送 ----------
def github_put_file(repo: str, token: str, branch: str, repo_path: str, content_bytes: bytes, commit_msg: str):
    """
    使用 GitHub API PUT /repos/{owner}/{repo}/contents/{path} 创建或更新文件
    """
    api = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    # 先获取 sha（如果已存在，需要更新）
    sha = None
    r = requests.get(api, headers=headers, params={"ref": branch})
    if r.status_code == 200:
        try:
            sha = r.json().get("sha")
        except Exception:
            sha = None

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch
    }
    if sha:
        payload["sha"] = sha

    r2 = requests.put(api, headers=headers, data=json.dumps(payload))
    if r2.status_code not in (200, 201):
        raise RuntimeError(f"GitHub 推送失败：{r2.status_code} {r2.text}")


def push_meet_to_github(folder: str, message: str = "Update meet files"):
    repo = st.secrets.get("REPO")
    token = st.secrets.get("GITHUB_TOKEN")
    if not repo or not token:
        st.warning("未配置 REPO / GITHUB_TOKEN，已跳过推送。")
        return

    branch = "main"
    local_dir = os.path.join(MEETS_ROOT, folder)
    for fname in ["meta.csv", "results.csv"]:
        local_fp = os.path.join(local_dir, fname)
        if not os.path.exists(local_fp):
            continue
        with open(local_fp, "rb") as f:
            content = f.read()
        repo_path = f"{MEETS_ROOT}/{folder}/{fname}"
        github_put_file(repo=repo, token=token, branch=branch,
                       repo_path=repo_path, content_bytes=content, commit_msg=message)
    st.success("已推送至 GitHub ✅")


# ------------- 页面：赛事管理/录入 -------------

def page_manage():
    st.header("📁 赛事管理 / 成绩录入")

    tab_sel, tab_new = st.tabs(["选择已有赛事/录入", "新建赛事"])

    
with tab_new:
    st.subheader("新建赛事")
    c1, c2 = st.columns(2)
    dt = c1.date_input("Date", value=date.today())
    city = c2.text_input("City", value="")
    meet_name = st.text_input("MeetName", value="")
    pool_name = st.text_input("PoolName", value="")
    length = st.number_input("LengthMeters", min_value=0, step=25, value=50)
    push_now = st.checkbox("创建后立即推送到 GitHub（推荐，避免丢失）", value=True,
                           help="需要在 Secrets 配置 GITHUB_TOKEN 与 REPO；会创建 meta.csv 与空的 results.csv 并提交。")

    if st.button("创建赛事文件夹", type="primary"):
        ensure_root()
        dstr = dt.strftime("%Y-%m-%d")
        folder = folder_from_meta(dstr, sanitize_segment(city), sanitize_segment(pool_name))
        dirpath = os.path.join(MEETS_ROOT, folder)
        os.makedirs(dirpath, exist_ok=True)

        meta_df = pd.DataFrame([{
            "Date": dstr,
            "City": city,
            "MeetName": meet_name,
            "PoolName": pool_name,
            "LengthMeters": int(length)
        }])
        write_meta(dirpath, meta_df)
        # 初始化空 results.csv
        write_results(dirpath, read_results(dirpath))

        st.success(f"已创建：{dirpath}")
        # 可选：立即推送到 GitHub，确保即使没有录入成绩也会持久保存
        if push_now:
            try:
                push_meet_to_github(folder, message=f"Create {folder}")
            except Exception as e:
                st.warning(f"GitHub 推送失败：{e}")

        st.info("现在切换到“选择已有赛事/录入”标签进行成绩录入。")


    with tab_sel:
        st.subheader("选择已有赛事并录入")
        meets_df = list_meets()
        if len(meets_df) == 0:
            st.info("当前没有赛事，请先到“新建赛事”创建。")
            return

        # 选择赛事
        folder_options = meets_df["Folder"].tolist()
        folder = st.selectbox("赛事文件夹", folder_options)

        # 展示 meta
        meta_row = meets_df[meets_df["Folder"] == folder].iloc[0]
        st.table(meta_row.to_frame(name="Value"))

        # 读取当前赛事 results
        dirpath = os.path.join(MEETS_ROOT, folder)
        results_df = read_results(dirpath)

        # 选择/新建 EventName
        event_names = sorted([x for x in results_df["EventName"].dropna().unique().tolist() if x])
        st.markdown("### 选择或新建项目（EventName）")
        mode = st.radio("方式", ["选择已有", "新建"], horizontal=True)
        if mode == "新建":
            new_evt = st.text_input("新建 EventName", value="")
            if new_evt:
                event_sel = new_evt
            else:
                st.stop()
        else:
            event_sel = st.selectbox("EventName", options=event_names or ["(暂无，改用“新建”)"])

        # 该项目历史记录
        st.markdown("### 该项目历史记录")
        cur_evt_df = results_df[results_df["EventName"] == event_sel].copy()
        if len(cur_evt_df) == 0:
            st.info("暂无记录")
        else:
            # 展示表 + 删除选中
            cur_evt_df_display = cur_evt_df.reset_index(drop=False).rename(columns={"index": "RowID"})
            st.dataframe(cur_evt_df_display, use_container_width=True)
            del_ids = st.multiselect("选择要删除的 RowID", cur_evt_df_display["RowID"].tolist(), key="del_ids")
            if st.button("删除选中记录", type="secondary", disabled=len(del_ids) == 0):
                results_df = results_df.drop(index=del_ids).reset_index(drop=True)
                write_results(dirpath, results_df)
                st.success("已删除并保存")
                st.experimental_rerun()

        # 新增一条记录
        st.markdown("### 新增记录")
        col1, col2 = st.columns(2)
        name_input = col1.text_input("Name", value="Anna")
        result_input = col2.text_input("Result（支持 34.12 或 0:34.12）", value="0:20.00")
        col3, col4 = st.columns(2)
        rank_input = col3.number_input("Rank（可留空）", min_value=0, step=1)
        note_input = col4.text_input("Note", value="")

        add_btn = st.button("添加到当前项目")
        if add_btn:
            sec = parse_time_input(result_input)
            if sec is None:
                st.error("成绩格式不合法，请输入 34.12 或 0:34.12 或 m:ss.xx")
            else:
                row = {
                    "Name": name_input.strip(),
                    "EventName": event_sel,
                    "Result": format_time(sec),
                    "Rank": int(rank_input) if rank_input else "",
                    "Note": note_input.strip(),
                }
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                write_results(dirpath, results_df)
                st.success("已添加并保存")
                st.experimental_rerun()

        # 保存 & 推送
        st.markdown("---")
        push = st.checkbox("提交到 GitHub（免下载上传）", value=False, help="需要在 Secrets 配置 GITHUB_TOKEN 与 REPO")
        if st.button("保存/提交"):
            # 本地早已保存；若勾选推送则推送
            if push:
                try:
                    push_meet_to_github(folder, message=f"Update {folder}")
                except Exception as e:
                    st.error(f"GitHub 推送失败：{e}")
            st.success("操作完成 ✅")


# ------------- 页面：查询 / 对比 -------------

def page_browse():
    st.header("🔎 游泳成绩查询 / 对比")

    meets_df = list_meets()
    if len(meets_df) == 0:
        st.info("当前没有成绩数据。请先在“赛事管理/成绩录入”中添加。")
        return

    # 汇总所有 results，加上 meta 字段以便筛选
    all_rows = []
    for _, row in meets_df.iterrows():
        dirpath = os.path.join(MEETS_ROOT, row["Folder"])
        res = read_results(dirpath)
        if len(res) == 0:
            continue
        res = res.copy()
        res["Date"] = row["Date"]
        res["City"] = row["City"]
        res["PoolName"] = row["PoolName"]
        res["LengthMeters"] = row["LengthMeters"]
        res["Folder"] = row["Folder"]
        all_rows.append(res)
    if not all_rows:
        st.info("没有可用成绩。")
        return
    df = pd.concat(all_rows, ignore_index=True)

    # 侧边筛选
    with st.sidebar:
        st.subheader("筛选条件")
        names = sorted([x for x in df["Name"].dropna().unique().tolist() if x])
        name_sel = st.multiselect("Name（可多选）", options=names, default=[])

        events = sorted([x for x in df["EventName"].dropna().unique().tolist() if x])
        event_sel = st.selectbox("Event", options=["All"] + events)

        lengths = sorted([int(x) for x in df["LengthMeters"].dropna().unique().tolist() if str(x) != ""])
        length_sel = st.selectbox("Length (Meters)", options=["All"] + [str(x) for x in lengths])

        pools = sorted([x for x in df["PoolName"].dropna().unique().tolist() if x])
        pool_sel = st.selectbox("Pool Name", options=["All"] + pools)

        cities = sorted([x for x in df["City"].dropna().unique().tolist() if x])
        city_sel = st.selectbox("City", options=["All"] + cities)

        dates = sorted([x for x in df["Date"].dropna().unique().tolist() if x])
        dmin, dmax = (dates[0], dates[-1]) if dates else ("", "")
        date_range = st.text_input("Date 范围（YYYY-MM-DD~YYYY-MM-DD）", value=f"{dmin}~{dmax}" if dmin else "")

    q = df.copy()
    if name_sel:
        q = q[q["Name"].isin(name_sel)]
    if event_sel != "All":
        q = q[q["EventName"] == event_sel]
    if length_sel != "All":
        q = q[q["LengthMeters"] == int(length_sel)]
    if pool_sel != "All":
        q = q[q["PoolName"] == pool_sel]
    if city_sel != "All":
        q = q[q["City"] == city_sel]
    if date_range.strip():
        m = re.match(r"\s*(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})\s*", date_range)
        if m:
            a, b = m.group(1), m.group(2)
            q = q[(q["Date"] >= a) & (q["Date"] <= b)]

    if len(q) == 0:
        st.info("没有匹配的记录。")
        return

    # 展示结果表（带高亮）
    meta_for_style = pd.Series({"LengthMeters": q["LengthMeters"].iloc[0] if "LengthMeters" in q.columns and len(q)>0 else None})
    st.markdown("### 比赛记录")
    st.dataframe(q, use_container_width=True)

    # 生成折线图：按日期画每个人成绩（秒数越低越好）
    st.markdown("### 成绩趋势（越低越好）")
    qq = q.copy()
    qq["Seconds"] = qq["Result"].apply(parse_time_input)
    try:
        qq["_dt"] = pd.to_datetime(qq["Date"])
    except Exception:
        qq["_dt"] = qq["Date"]
    chart_df = qq[["_dt", "Name", "Seconds"]].sort_values(["Name", "_dt"])
    # 使用 Streamlit 原生 line_chart
    st.line_chart(data=chart_df.rename(columns={"_dt": "Date"}), x="Date", y="Seconds", color="Name")

    # 下载筛选结果
    csv_bytes = q.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载当前筛选 CSV", data=csv_bytes, file_name="filtered_results.csv", mime="text/csv")


# ------------- 主入口 -------------

def main():
    st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide", page_icon="🏊‍♀️")
    st.title(APP_TITLE)

    page = st.sidebar.radio("页面", ["查询/对比", "赛事管理/录入"])

    ensure_root()

    if page == "查询/对比":
        page_browse()
    else:
        page_manage()


if __name__ == "__main__":
    main()
def common_events() -> list:
    """Return a list of commonly used swimming events."""
    base = []
    # Freestyle
    for d in [25, 50, 100, 200, 400]:
        base.append(f"{d}m Freestyle")
    # Backstroke / Breaststroke / Butterfly
    for stroke in ["Backstroke", "Breaststroke", "Butterfly"]:
        for d in [25, 50, 100, 200]:
            base.append(f"{d}m {stroke}")
    # Individual Medley
    for d in [100, 200, 400]:
        base.append(f"{d}m IM")
    return base

