
import os
import io
import base64
import json
from datetime import date
from typing import Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import requests

APP_TITLE = "🏊‍♀️ 游泳成绩查询系统（录入 + 查询）"


# ----------------------------
# Utilities
# ----------------------------
def hide_index_compat(styler: "pd.io.formats.style.Styler"):
    """
    pandas 2.x: Styler.hide(axis="index")
    pandas 1.x: Styler.hide_index()
    """
    try:
        return styler.hide(axis="index")
    except Exception:
        return styler.hide_index()


@st.cache_data(show_spinner=False)
def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_time_to_seconds(s: str) -> float:
    """Parse 'm:ss.xx' or 'mm:ss' or 'ss.xx' to seconds (float)."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    try:
        if ":" in s:
            parts = s.split(":")
            parts = [p.strip() for p in parts]
            if len(parts) == 2:
                m, rest = parts
                sec = float(rest)
                return int(m) * 60 + sec
            elif len(parts) == 3:
                h, m, rest = parts
                return int(h) * 3600 + int(m) * 60 + float(rest)
        # fall back: seconds
        return float(s)
    except Exception:
        return np.nan


def load_all_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load results from meets/*/results.csv and meta.csv; join to one table.

    Returns: (results_df, meta_df)
    """
    meets_dir = "meets"
    rows = []
    metas = []
    if os.path.isdir(meets_dir):
        for root, dirs, files in os.walk(meets_dir):
            if "results.csv" in files:
                path = os.path.join(root, "results.csv")
                df = read_csv_safe(path)
                if not df.empty:
                    df.columns = [c.strip() for c in df.columns]
                    # --- normalize results to EventName ---
                    res_ren = {c: c for c in df.columns}
                    for c in df.columns:
                        lc = c.lower()
                        if lc == "event":
                            res_ren[c] = "EventName"
                        elif lc == "eventname":
                            res_ren[c] = "EventName"
                        elif lc == "name":
                            res_ren[c] = "Name"
                        elif lc == "result":
                            res_ren[c] = "Result"
                        elif lc == "rank":
                            res_ren[c] = "Rank"
                        elif lc == "note":
                            res_ren[c] = "Note"
                    df = df.rename(columns=res_ren)
                    for need in ["Name","EventName","Result","Rank","Note"]:
                        if need not in df.columns:
                            df[need] = np.nan
                    df["__meet_root__"] = root
                    rows.append(df)
            if "meta.csv" in files:
                mpath = os.path.join(root, "meta.csv")
                md = read_csv_safe(mpath)
                if not md.empty:
                    md.columns = [c.strip() for c in md.columns]
                    meta_ren = {}
                    for c in md.columns:
                        lc = c.lower()
                        if lc == "date":
                            meta_ren[c] = "Date"
                        elif lc == "city":
                            meta_ren[c] = "City"
                        elif lc == "meetname":
                            meta_ren[c] = "MeetName"
                        elif lc == "poolname":
                            meta_ren[c] = "PoolName"
                        elif lc == "lengthmeters":
                            meta_ren[c] = "LengthMeters"
                    md = md.rename(columns=meta_ren)
                    for need in ["Date","City","MeetName","PoolName","LengthMeters"]:
                        if need not in md.columns:
                            md[need] = np.nan
                    md["__meet_root__"] = root
                    metas.append(md)

    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()

    # merge by folder
    if not results.empty and not meta.empty:
        meta_small = meta[["__meet_root__", "Date", "City", "MeetName", "PoolName", "LengthMeters"]].drop_duplicates()
        merged = results.merge(meta_small, on="__meet_root__", how="left")
    else:
        merged = results.copy()

    # parse types
    if not merged.empty:
        merged["Seconds"] = merged["Result"].apply(parse_time_to_seconds)
        try:
            merged["Date"] = pd.to_datetime(merged["Date"]).dt.date
        except Exception:
            pass

    return merged, meta


def color_for_name(name: str) -> str:
    palette = [
        "#d62728",  # red
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#17becf",  # cyan
        "#e377c2",  # pink
    ]
    if not name:
        return "#d62728"
    idx = (sum(ord(c) for c in name) % len(palette))
    return palette[idx]


def style_best_rows(df_display: pd.DataFrame, best_mask: pd.Series, name_col="Name") -> "pd.io.formats.style.Styler":
    """Highlight best rows in df_display using color per Name."""
    colors = {n: color_for_name(n) for n in df_display[name_col].unique()}

    def style_row(row):
        if best_mask.loc[row.name]:
            c = colors.get(row[name_col], "#d62728")
            return [f"color:{c}; font-weight:700" for _ in row]
        return [""] * len(row)

    styler = df_display.style.apply(style_row, axis=1)
    return hide_index_compat(styler)


def push_to_github(token: str, repo: str, branch: str, path: str, content_bytes: bytes, commit_msg: str):
    """Create or update a file via GitHub API. Return (ok, message/url)."""
    api_base = "https://api.github.com"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    # get existing sha (if exists)
    get_url = f"{api_base}/repos/{repo}/contents/{path}?ref={branch}"
    sha = None
    r = requests.get(get_url, headers=headers)
    if r.status_code == 200:
        try:
            sha = r.json().get("sha")
        except Exception:
            sha = None

    b64 = base64.b64encode(content_bytes).decode("utf-8")
    put_url = f"{api_base}/repos/{repo}/contents/{path}"
    payload = {"message": commit_msg, "content": b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    resp = requests.put(put_url, headers=headers, data=json.dumps(payload))
    if resp.status_code in (200, 201):
        try:
            data = resp.json()
            html_url = data.get("content", {}).get("html_url", "")
        except Exception:
            html_url = ""
        return True, html_url or f"https://github.com/{repo}/blob/{branch}/{path}"
    return False, f"{resp.status_code}: {resp.text}"


# ----------------------------
# UI PAGES
# ----------------------------
def page_browse():
    st.header("🔎 请选择筛选条件")
    data, meta = load_all_results()

    if data.empty:
        st.info("当前仓库里还没有 `meets/*/results.csv` 数据。请先到“➕ 表单录入”页新增记录。")
        return

    # Filters
    names = sorted([x for x in data["Name"].dropna().unique().tolist()])
    events = sorted([x for x in data["EventName"].dropna().unique().tolist()]) if "EventName" in data.columns else []
    lengths = ["All"] + sorted([int(x) for x in pd.to_numeric(data["LengthMeters"], errors="coerce").dropna().unique().tolist()])
    pools = ["All"] + sorted([x for x in data["PoolName"].dropna().unique().tolist()])
    cities = ["All"] + sorted([x for x in data["City"].dropna().unique().tolist()])
    dates_list = sorted([str(x) for x in data["Date"].dropna().unique().tolist()])

    # selection widgets
    default_names = [n for n in names if isinstance(n, str) and n.lower() == "anna"] or (names[:1] if names else [])
    sel_names = st.multiselect("Name（可多选）", options=names, default=default_names, placeholder="选择选手")
    sel_event = st.selectbox("EventName", options=["All"] + events, index=0)
    sel_length = st.selectbox("Length (Meters)", options=lengths, index=0)
    sel_pool = st.selectbox("Pool Name", options=pools, index=0)
    sel_city = st.selectbox("City", options=cities, index=0)
    sel_date = st.selectbox("Date", options=["All"] + dates_list, index=0)

    # Apply filters
    disp = data.copy()
    if sel_names:
        disp = disp[disp["Name"].isin(sel_names)]
    if sel_event != "All" and "EventName" in disp.columns:
        disp = disp[disp["EventName"] == sel_event]
    if sel_length != "All":
        disp = disp[pd.to_numeric(disp["LengthMeters"], errors="coerce") == int(sel_length)]
    if sel_pool != "All":
        disp = disp[disp["PoolName"] == sel_pool]
    if sel_city != "All":
        disp = disp[disp["City"] == sel_city]
    if sel_date != "All":
        try:
            dt = pd.to_datetime(sel_date).date()
            disp = disp[disp["Date"] == dt]
        except Exception:
            pass

    if disp.empty:
        st.warning("没有匹配的记录。换个筛选条件试试～")
        return

    disp = disp.sort_values(["Name", "EventName", "Date"] if "EventName" in disp.columns else ["Name", "Date"]).reset_index(drop=True)

    # Determine seed/best rows
    group_cols = ["Name"]
    if "EventName" in disp.columns:
        group_cols.append("EventName")
    if sel_length == "All":
        group_cols.append("LengthMeters")
    best_mask = disp.groupby(group_cols)["Seconds"].transform(lambda s: s == s.min()) if "Seconds" in disp.columns else pd.Series(False, index=disp.index)

    # Display table with style
    base_cols = ["Name", "Date", "Result", "Rank", "Note", "PoolName", "City", "LengthMeters"]
    show_cols = base_cols.copy()
    if "EventName" in disp.columns:
        show_cols.insert(2, "EventName")
    for col in show_cols:
        if col not in disp.columns:
            disp[col] = np.nan
    styled = style_best_rows(disp[show_cols], best_mask)
    st.subheader("🏅 比赛记录")
    st.dataframe(styled, use_container_width=True)

    # CSV download
    csv_bytes = disp[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ 下载当前筛选结果 CSV", data=csv_bytes, file_name="filtered_results.csv", mime="text/csv")

    # Line chart of trends (seconds; lower is better)
    st.subheader("📈 成绩趋势（单位：秒，越低越好）")
    chart_df = disp.dropna(subset=["Date", "Seconds"]).copy() if "Seconds" in disp.columns else pd.DataFrame()
    if not chart_df.empty:
        chart_df = chart_df.sort_values("Date")
        pivot = chart_df.pivot_table(index="Date", columns="Name", values="Seconds", aggfunc="min")
        st.line_chart(pivot, height=320, use_container_width=True)
    else:
        st.info("当前筛选下没有可绘制的时间序列数据。")


def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen and x is not None and str(x).strip():
            out.append(x)
            seen.add(x)
    return out


def page_form():
    st.header("📝 表单录入（支持自动提交到 GitHub）")
    st.markdown("先填写**赛事信息**，再点击 **“➕ 添加一条记录”** 逐条录入成绩。")

    # 赛事信息
    with st.form("meet_form_top", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            m_date = st.date_input("Date", value=date.today())
            m_city = st.text_input("City", value="ChiangMai")
            m_pool = st.text_input("PoolName", value="")
        with col2:
            m_meet = st.text_input("MeetName", value="Local Meet")
            m_length = st.selectbox("LengthMeters", options=[25, 50], index=1)
        submitted_top = st.form_submit_button("保存赛事信息（继续录入成绩）")
    # 记录列表在 session_state 里维护
    if "rows" not in st.session_state:
        st.session_state.rows = 1  # 默认一条
    if "records_removed" not in st.session_state:
        st.session_state.records_removed = False

    st.markdown("---")
    st.subheader("成绩记录")

    data_existing, _ = load_all_results()
    events_from_repo = sorted(data_existing["EventName"].dropna().unique().tolist()) if ("EventName" in data_existing.columns and not data_existing.empty) else []
    common_events = ["25m Freestyle","50m Freestyle","25m Breaststroke","50m Breaststroke","25m Butterfly","50m Butterfly","25m Backstroke","50m Backstroke"]
    event_options = _unique_preserve(events_from_repo + common_events + ["自定义..."])

    # 控制按钮
    col_btn1, col_btn2 = st.columns([1,1])
    with col_btn1:
        if st.button("➕ 添加一条记录"):
            st.session_state.rows += 1
    with col_btn2:
        if st.button("🧹 清空所有记录"):
            st.session_state.rows = 1
            # 也清理可能的键值，避免残留
            for k in list(st.session_state.keys()):
                if k.startswith(("name_", "eventopt_", "eventcustom_", "result_", "rank_", "note_")):
                    del st.session_state[k]

    to_delete = []

    # 渲染每条记录
    for i in range(st.session_state.rows):
        st.markdown(f"**记录 {i+1}**")
        c1, c2, c3, c4, c5, c6 = st.columns([1.2, 1.3, 1.0, 0.8, 1.2, 0.6])
        with c1:
            name = st.text_input(f"Name_{i+1}", value=st.session_state.get(f"name_{i}", "Anna"), key=f"name_{i}")
        with c2:
            chosen = st.selectbox(f"EventName_{i+1}", options=event_options, index=0 if event_options else 0, key=f"eventopt_{i}")
            if chosen == "自定义...":
                eventname = st.text_input(f"自定义 EventName_{i+1}", value=st.session_state.get(f"eventcustom_{i}", ""), key=f"eventcustom_{i}")
            else:
                eventname = chosen
        with c3:
            result = st.text_input(f"Result_{i+1}", value=st.session_state.get(f"result_{i}", ""), key=f"result_{i}", help="示例：0:20.42 或 1:02.45")
        with c4:
            rank = st.number_input(f"Rank_{i+1}", min_value=0, max_value=999, value=int(st.session_state.get(f"rank_{i}", 0)), step=1, key=f"rank_{i}")
        with c5:
            note = st.text_input(f"Note_{i+1}", value=st.session_state.get(f"note_{i}", ""), key=f"note_{i}")
        with c6:
            if st.button("🗑️ 删除", key=f"del_{i}"):
                to_delete.append(i)

        # 存到 session（保证 rerun 后值还在）
        st.session_state[f"name_{i}"] = name
        st.session_state[f"eventcustom_{i}"] = st.session_state.get(f"eventcustom_{i}", "") if chosen == "自定义..." else ""
        st.session_state[f"result_{i}"] = result
        st.session_state[f"rank_{i}"] = rank
        st.session_state[f"note_{i}"] = note

    # 处理删除（从后往前删索引才安全）
    if to_delete:
        for idx in sorted(to_delete, reverse=True):
            # 清除该条的 state
            for k in [f"name_{idx}", f"eventopt_{idx}", f"eventcustom_{idx}", f"result_{idx}", f"rank_{idx}", f"note_{idx}", f"del_{idx}"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.rows -= 1
        st.experimental_rerun()

    # 提交
    push_github = st.checkbox("提交到 GitHub（免下载上传）", value=True, help="需要在 Settings → Secrets 配置 GITHUB_TOKEN 与 REPO。")
    save_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=False)

    if st.button("生成/提交"):
        # 汇总记录
        recs = []
        for i in range(st.session_state.rows):
            name = st.session_state.get(f"name_{i}", "").strip()
            eventopt = st.session_state.get(f"eventopt_{i}", "").strip()
            eventcustom = st.session_state.get(f"eventcustom_{i}", "").strip()
            eventname = eventcustom if eventopt == "自定义..." and eventcustom else eventopt
            result = st.session_state.get(f"result_{i}", "").strip()
            rank = st.session_state.get(f"rank_{i}", 0)
            note = st.session_state.get(f"note_{i}", "").strip()
            # 过滤空的
            if name and eventname and result:
                recs.append({"Name": name, "EventName": eventname, "Result": result, "Rank": int(rank), "Note": note})

        if not recs:
            st.error("请至少填写一条完整记录（Name、EventName、Result 必填）。")
            return

        # 赛事信息
        meta_df = pd.DataFrame([{
            "Date": m_date.strftime("%Y-%m-%d"),
            "City": m_city,
            "MeetName": m_meet,
            "PoolName": m_pool,
            "LengthMeters": int(m_length),
        }])
        results_df = pd.DataFrame(recs, columns=["Name","EventName","Result","Rank","Note"])

        # 文件夹
        folder = f"meets/{m_date.strftime('%Y-%m-%d')}_{m_city}"
        os.makedirs(folder, exist_ok=True)

        # 本地保存
        if save_local:
            meta_df.to_csv(os.path.join(folder, "meta.csv"), index=False)
            results_df.to_csv(os.path.join(folder, "results.csv"), index=False)
            st.success(f"已写入本地： {folder}/meta.csv, results.csv")

        # 推 GitHub
        if push_github:
            token = st.secrets.get("GITHUB_TOKEN", "")
            repo = st.secrets.get("REPO", "")
            branch = st.secrets.get("BRANCH", "main")
            if not token or not repo:
                st.error("缺少 Secrets：['GITHUB_TOKEN', 'REPO']。请在 Streamlit Cloud - App - Settings - Secrets 中配置。")
            else:
                ok_all = True
                ok1, msg1 = push_to_github(token, repo, branch, f"{folder}/meta.csv", meta_df.to_csv(index=False).encode("utf-8"),
                                           f"Add/Update {folder}/meta.csv")
                if not ok1:
                    ok_all = False
                ok2, msg2 = push_to_github(token, repo, branch, f"{folder}/results.csv", results_df.to_csv(index=False).encode("utf-8"),
                                           f"Add/Update {folder}/results.csv")
                if not ok2:
                    ok_all = False
                if ok_all:
                    st.success(f"已写入 GitHub：{folder}/meta.csv, results.csv")
                    if msg1:
                        st.write(msg1)
                    if msg2:
                        st.write(msg2")
                else:
                    st.error("推送到 GitHub 失败：")
                    st.code(f"meta → {ok1}: {msg1}\nresults → {ok2}: {msg2}")


def main():
    st.set_page_config(page_title="Swim Results", layout="wide")
    st.title(APP_TITLE)

    tab1, tab2 = st.tabs(["📊 浏览 / 分析", "➕ 表单录入"])
    with tab1:
        page_browse()
    with tab2:
        page_form()


if __name__ == "__main__":
    main()
