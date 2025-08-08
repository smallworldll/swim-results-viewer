# -*- coding: utf-8 -*-
"""
Swim Results System (meet-based)
- Folder layout: meets/YYYY-MM-DD_City/{meta.csv, results.csv}
- meta.csv columns: Date, City, MeetName, PoolName, LengthMeters
- results.csv columns: Name, EventName, Result, Rank, Note, Date, City, PoolName, LengthMeters, ResultSeconds
"""
import os
import io
import math
import json
import base64
import datetime as dt
from typing import List, Optional, Dict

import pandas as pd
import streamlit as st

# ---------- Utilities ----------

MEETS_ROOT = "meets"


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_time_to_seconds(s: Optional[str]) -> Optional[float]:
    """Accept: 'm:ss.xx', 'ss.xx', 'ss', 'm:ss', 'm:ss.s', etc. Return total seconds as float."""
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return None
    try:
        if ":" in s:
            m_str, sec_str = s.split(":", 1)
            m = int(m_str.strip())
            sec = float(sec_str.strip())
            return m * 60.0 + sec
        # pure seconds
        return float(s)
    except Exception:
        return None


def format_m_ss_xx(seconds: Optional[float]) -> str:
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or math.isinf(seconds))):
        return ""
    try:
        seconds = float(seconds)
        m = int(seconds // 60)
        s = seconds - m * 60
        # Always two decimals, seconds padded to 2 digits (including decimals)
        # Example: 64.12 -> '1:04.12'
        return f"{m}:{s:05.2f}"
    except Exception:
        return ""


def read_meta(meet_dir: str) -> pd.DataFrame:
    meta_path = os.path.join(meet_dir, "meta.csv")
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        # Normalize expected columns / order
        for col in ["Date", "City", "MeetName", "PoolName", "LengthMeters"]:
            if col not in df.columns:
                df[col] = None
        return df[["Date", "City", "MeetName", "PoolName", "LengthMeters"]].head(1)
    return pd.DataFrame(columns=["Date", "City", "MeetName", "PoolName", "LengthMeters"])


def read_results(meet_dir: str) -> pd.DataFrame:
    res_path = os.path.join(meet_dir, "results.csv")
    if not os.path.exists(res_path):
        return pd.DataFrame(columns=[
            "Name", "EventName", "Result", "Rank", "Note",
            "Date", "City", "PoolName", "LengthMeters", "ResultSeconds"
        ])
    df = pd.read_csv(res_path)
    # Ensure required columns
    for col in ["Name", "EventName", "Result", "Rank", "Note", "Date", "City", "PoolName", "LengthMeters", "ResultSeconds"]:
        if col not in df.columns:
            df[col] = None
    # Compute ResultSeconds if missing
    if df["ResultSeconds"].isna().any() or not pd.api.types.is_numeric_dtype(df["ResultSeconds"]):
        df["ResultSeconds"] = df["Result"].map(parse_time_to_seconds)
    # Normalize Result formatting
    df["Result"] = df["ResultSeconds"].map(format_m_ss_xx)
    return df


def load_all_results() -> pd.DataFrame:
    """Aggregate all meets results with meta attached."""
    rows = []
    if not os.path.exists(MEETS_ROOT):
        return pd.DataFrame(columns=[
            "Name", "EventName", "Result", "Rank", "Note",
            "Date", "City", "PoolName", "LengthMeters", "ResultSeconds", "MeetFolder"
        ])
    for folder in sorted(os.listdir(MEETS_ROOT)):
        meet_dir = os.path.join(MEETS_ROOT, folder)
        if not os.path.isdir(meet_dir):
            continue
        meta = read_meta(meet_dir)
        if meta.empty:
            continue
        meta_row = meta.iloc[0].to_dict()
        df = read_results(meet_dir)
        if df.empty:
            continue
        df = df.copy()
        for k in ["Date", "City", "PoolName", "LengthMeters"]:
            df[k] = meta_row.get(k)
        df["MeetFolder"] = folder
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=[
            "Name", "EventName", "Result", "Rank", "Note",
            "Date", "City", "PoolName", "LengthMeters", "ResultSeconds", "MeetFolder"
        ])
    out = pd.concat(rows, ignore_index=True)
    # Coerce numeric
    with pd.option_context('mode.use_inf_as_na', True):
        out["LengthMeters"] = pd.to_numeric(out["LengthMeters"], errors="coerce").astype("Int64")
        out["Rank"] = pd.to_numeric(out["Rank"], errors="coerce").astype("Int64")
        out["ResultSeconds"] = pd.to_numeric(out["ResultSeconds"], errors="coerce")
    return out


def github_put(owner_repo: str, path_in_repo: str, content_bytes: bytes, message: str, token: str) -> Dict:
    """Create or update file via GitHub API. owner_repo='user/repo' . Returns JSON response."""
    import requests

    url = f"https://api.github.com/repos/{owner_repo}/contents/{path_in_repo}"
    b64 = base64.b64encode(content_bytes).decode("utf-8")

    # Try to get existing SHA (for update)
    sha = None
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r_get = requests.get(url, headers=headers)
    if r_get.status_code == 200:
        try:
            sha = r_get.json().get("sha")
        except Exception:
            sha = None

    payload = {"message": message, "content": b64}
    if sha:
        payload["sha"] = sha

    r_put = requests.put(url, headers=headers, json=payload)
    try:
        return r_put.json()
    except Exception:
        return {"status": r_put.status_code, "text": r_put.text}


# ---------- UI Helpers ----------

def pick_meet_folder() -> Optional[str]:
    """Let user pick existing meet folder or create a new one. Returns meet_dir or None."""
    st.markdown("### ① 选择或新建赛事")
    ensure_dir(MEETS_ROOT)
    mode = st.radio("操作", ["选择已有赛事", "新建赛事"], horizontal=True, index=0)

    existing = sorted([f for f in os.listdir(MEETS_ROOT) if os.path.isdir(os.path.join(MEETS_ROOT, f))])

    if mode == "选择已有赛事":
        if not existing:
            st.info("当前没有任何赛事文件夹，请新建一个。")
            return None
        sel = st.selectbox("选择赛事（文件夹）", existing, index=len(existing)-1, key="sel_meet")
        meet_dir = os.path.join(MEETS_ROOT, sel)
        # Show meta preview
        meta = read_meta(meet_dir)
        st.dataframe(meta, use_container_width=True, hide_index=True)
        return meet_dir

    # Create new meet
    col1, col2, col3, col4 = st.columns([1.2, 1, 1.2, 1])
    with col1:
        date = st.date_input("日期 (Date)", value=dt.date.today())
    with col2:
        city = st.text_input("城市 (City)", value="Chiang Mai").strip()
    with col3:
        pool = st.text_input("泳池名 (PoolName)", value="").strip()
    with col4:
        length = st.number_input("泳池长度 (LengthMeters)", min_value=10, max_value=100, value=50, step=5)

    meet_name = st.text_input("赛事名称 (MeetName)", value="New Meet")
    folder_name = f"{date.strftime('%Y-%m-%d')}_{city.replace(' ', '')}"
    st.caption(f"将创建文件夹：`{MEETS_ROOT}/{folder_name}`")

    if st.button("创建赛事"):
        meet_dir = os.path.join(MEETS_ROOT, folder_name)
        if os.path.exists(meet_dir):
            st.warning("同名赛事已存在，已为你打开它。")
        ensure_dir(meet_dir)
        meta = pd.DataFrame([{
            "Date": date.strftime("%Y-%m-%d"),
            "City": city,
            "MeetName": meet_name,
            "PoolName": pool,
            "LengthMeters": length
        }])
        meta.to_csv(os.path.join(meet_dir, "meta.csv"), index=False)
        st.success("创建成功！已写入 meta.csv")
        return meet_dir

    return None


def render_history_manager(meet_dir: str):
    """View/Edit/Delete results of a specific meet."""
    st.markdown("### ② 历史成绩管理 / 编辑")
    res_path = os.path.join(meet_dir, "results.csv")
    if not os.path.exists(res_path):
        st.info("这个赛事还没有 `results.csv`，保存一次新成绩后会自动创建。")
        return

    try:
        df = pd.read_csv(res_path)
    except Exception as e:
        st.error(f"读取 results.csv 失败：{e}")
        return

    if df.empty:
        st.info("暂无成绩记录。")
        return

    # Filters in manager
    c1, c2 = st.columns(2)
    with c1:
        ev_opt = ["All"] + sorted(df["EventName"].dropna().unique().tolist()) if "EventName" in df.columns else ["All"]
        ev = st.selectbox("按项目筛选 (EventName)", ev_opt, index=0, key="hist_ev")
    with c2:
        name_kw = st.text_input("按姓名模糊搜索 (Name)", "", key="hist_kw")

    view = df.copy()
    if ev != "All" and "EventName" in view.columns:
        view = view[view["EventName"] == ev]
    if name_kw.strip() and "Name" in view.columns:
        view = view[view["Name"].astype(str).str.contains(name_kw.strip(), case=False, na=False)]

    st.caption(f"当前筛选：{len(view)} 条")

    edit_mode = st.toggle("✏️ 编辑模式", value=False)
    if not edit_mode:
        st.dataframe(view, use_container_width=True)
        return

    st.caption("可直接修改表格；新增行请在底部 +；删除勾选后点击保存。")
    edited = st.data_editor(view, num_rows="dynamic", use_container_width=True, key="hist_editor")
    to_delete = st.multiselect("删除这些索引（当前视图的 index）", edited.index.tolist(), key="hist_del")

    col_s, col_r = st.columns(2)
    with col_s:
        if st.button("💾 保存到 results.csv", type="primary"):
            base = df.copy()
            # update existing
            common_idx = base.index.intersection(edited.index)
            base.loc[common_idx, edited.columns] = edited.loc[common_idx, edited.columns]
            # add new rows
            new_rows = edited.loc[edited.index.difference(df.index)]
            if not new_rows.empty:
                base = pd.concat([base, new_rows], ignore_index=True)
            # delete
            if to_delete:
                base = base.drop(index=to_delete, errors="ignore").reset_index(drop=True)
            # normalize Result / ResultSeconds
            base["ResultSeconds"] = base["Result"].map(parse_time_to_seconds)
            base["Result"] = base["ResultSeconds"].map(format_m_ss_xx)
            base.to_csv(res_path, index=False)
            st.success("已保存。")
            st.rerun()
    with col_r:
        if st.button("↩️ 放弃修改（刷新）"):
            st.rerun()


def render_add_results(meet_dir: str):
    """Add multiple result rows for a given meet; no auto-ranking."""
    st.markdown("### ③ 录入本次赛事某个项目的成绩")
    meta = read_meta(meet_dir)
    if meta.empty:
        st.warning("meta.csv 缺失或为空。")
        return
    meta_row = meta.iloc[0].to_dict()

    # Select or create event
    res_path = os.path.join(meet_dir, "results.csv")
    existing_events: List[str] = []
    if os.path.exists(res_path):
        try:
            ex = pd.read_csv(res_path)
            if "EventName" in ex.columns:
                existing_events = sorted(ex["EventName"].dropna().unique().tolist())
        except Exception:
            pass

    event_mode = st.radio("方式", ["选择已有", "新建"], horizontal=True, index=0)
    if event_mode == "选择已有" and existing_events:
        event_name = st.selectbox("EventName", existing_events, key="add_event_sel")
    elif event_mode == "选择已有" and not existing_events:
        st.info("暂无已录入的项目，改为新建。")
        event_name = st.text_input("新建 EventName", value="100m Freestyle", key="add_event_new")
    else:
        event_name = st.text_input("新建 EventName", value="100m Freestyle", key="add_event_new")

    # Editable table for new rows
    st.caption("在下表中填写选手成绩；Result 支持 34.12 或 0:34.12 等格式。")
    new_rows = st.data_editor(
        pd.DataFrame([{"Name": "Anna", "Result": "", "Rank": None, "Note": ""}]),
        num_rows="dynamic",
        use_container_width=True,
        key="new_rows_editor"
    )

    col_left, col_right = st.columns(2)
    with col_left:
        push_github = st.checkbox("提交到 GitHub（免下载上传）", value=False)
    with col_right:
        save_local = st.checkbox("同时保存到本地 meets/ 目录", value=True)

    if st.button("保存这些成绩", type="primary"):
        # Prepare DataFrame
        rows = []
        for _, r in new_rows.iterrows():
            name = str(r.get("Name", "")).strip()
            if not name:
                continue
            result_raw = str(r.get("Result", "")).strip()
            sec = parse_time_to_seconds(result_raw)
            rows.append({
                "Name": name,
                "EventName": event_name,
                "Result": format_m_ss_xx(sec),
                "Rank": r.get("Rank", None),
                "Note": r.get("Note", ""),
                "Date": meta_row.get("Date"),
                "City": meta_row.get("City"),
                "PoolName": meta_row.get("PoolName"),
                "LengthMeters": meta_row.get("LengthMeters"),
                "ResultSeconds": sec
            })

        if not rows:
            st.warning("没有有效记录需要保存。")
            return

        # Merge into results.csv
        df_new = pd.DataFrame(rows)
        if os.path.exists(res_path):
            base = pd.read_csv(res_path)
            merged = pd.concat([base, df_new], ignore_index=True)
        else:
            merged = df_new

        # Normalize and save local
        merged["ResultSeconds"] = merged["ResultSeconds"].map(lambda x: x if x is None else float(x))
        merged["Result"] = merged["ResultSeconds"].map(format_m_ss_xx)

        if save_local:
            merged.to_csv(res_path, index=False)
            st.success(f"已保存 {len(df_new)} 条到本地。")

        # Push to GitHub
        if push_github:
            token = st.secrets.get("GITHUB_TOKEN", None)
            repo = st.secrets.get("REPO", None)  # e.g., "smallworldll/swim-results-viewer"
            if not token or not repo:
                st.error("缺少 Secrets：'GITHUB_TOKEN' 或 'REPO'。请在 Streamlit Cloud -> App -> Settings -> Secrets 中配置。")
            else:
                # Upload meta.csv and results.csv
                folder = os.path.basename(meet_dir)
                # meta.csv
                meta_bytes = pd.DataFrame([meta_row]).to_csv(index=False).encode("utf-8")
                r1 = github_put(repo, f"{MEETS_ROOT}/{folder}/meta.csv", meta_bytes, f"Update meta {folder}", token)
                # results.csv
                res_bytes = merged.to_csv(index=False).encode("utf-8")
                r2 = github_put(repo, f"{MEETS_ROOT}/{folder}/results.csv", res_bytes, f"Update results {folder}", token)
                st.success("已尝试推送到 GitHub。")
                st.code(json.dumps({"meta": r1, "results": r2}, ensure_ascii=False, indent=2))

        # Refresh page
        st.rerun()


def styled_seed_table(df: pd.DataFrame, selected_event: Optional[str], selected_length: Optional[str | int], selected_names: List[str]):
    """Return a styled dataframe: mark best (min) ResultSeconds per Name (and per length if 'All')."""
    if df.empty:
        return df

    df = df.copy()
    # Build color palette for names
    name_list = sorted(df["Name"].dropna().unique().tolist())
    palette = [
        "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#e377c2", "#8c564b", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    name_color = {n: palette[i % len(palette)] for i, n in enumerate(name_list)}

    # Determine seeds (index set)
    seeds = set()
    # Work only on selected names if provided
    name_scope = selected_names if selected_names else name_list

    # If a specific event is selected, seed by that event; otherwise seed within each event separately.
    events_scope = [selected_event] if selected_event and selected_event != "All" else sorted(df["EventName"].dropna().unique().tolist())

    for ev in events_scope:
        sub_e = df[df["EventName"] == ev]
        if sub_e.empty:
            continue

        if str(selected_length).lower() != "all" and pd.notna(selected_length):
            try:
                L = int(selected_length)
            except Exception:
                L = None
            sub_L = sub_e if L is None else sub_e[sub_e["LengthMeters"] == L]
            for nm in name_scope:
                g = sub_L[sub_L["Name"] == nm]
                g = g.dropna(subset=["ResultSeconds"])
                if not g.empty:
                    idx = g["ResultSeconds"].idxmin()
                    seeds.add(idx)
        else:
            # mark best per length (two seeds possible if both 25/50 exist)
            for nm in name_scope:
                for L in sorted(sub_e["LengthMeters"].dropna().unique().tolist()):
                    g = sub_e[(sub_e["Name"] == nm) & (sub_e["LengthMeters"] == L)].dropna(subset=["ResultSeconds"])
                    if not g.empty:
                        idx = g["ResultSeconds"].idxmin()
                        seeds.add(idx)

    def _highlight(row):
        color = ""
        if row.name in seeds:
            color = name_color.get(row["Name"], "#d62728")
        if color:
            return [f"color: {color}; font-weight: 700" for _ in row]
        return [""] * len(row)

    styled = df.style.apply(_highlight, axis=1)
    return styled


def page_browse():
    st.header("🏊‍♀️ 游泳成绩查询 / 对比")
    df = load_all_results()
    if df.empty:
        st.info("当前没有成绩数据。请先在“赛事管理/成绩录入”中添加。")
        return

    # ---- Filters ----
    st.markdown("#### 选择筛选条件")
    # Names
    all_names = sorted(df["Name"].dropna().unique().tolist())
    default_names = ["Anna"] if "Anna" in all_names else (all_names[:1] if all_names else [])
    names = st.multiselect("Name (可多选)", all_names, default=default_names)

    # Events
    events = ["All"] + sorted(df["EventName"].dropna().unique().tolist())
    event = st.selectbox("Event", events, index=events.index("All") if "All" in events else 0)

    # Length
    lengths = sorted([int(x) for x in df["LengthMeters"].dropna().unique().tolist()])
    length_opts = ["All"] + [str(x) for x in lengths]
    len_sel = st.selectbox("Length (Meters)", length_opts, index=0)

    # PoolName
    pools = ["All"] + sorted(df["PoolName"].dropna().unique().tolist())
    pool_sel = st.selectbox("Pool Name", pools, index=0)

    # City
    cities = ["All"] + sorted(df["City"].dropna().unique().tolist())
    city_sel = st.selectbox("City", cities, index=0)

    # Date (All or choose range)
    dates = sorted(df["Date"].dropna().unique().tolist())
    date_sel = st.selectbox("Date", ["All"] + dates, index=0)

    # ---- Apply filters ----
    view = df.copy()
    if names:
        view = view[view["Name"].isin(names)]
    if event != "All":
        view = view[view["EventName"] == event]
    if len_sel != "All":
        try:
            L = int(len_sel)
            view = view[view["LengthMeters"] == L]
        except Exception:
            pass
    if pool_sel != "All":
        view = view[view["PoolName"] == pool_sel]
    if city_sel != "All":
        view = view[view["City"] == city_sel]
    if date_sel != "All":
        view = view[view["Date"] == date_sel]

    # Sort by Date then ResultSeconds
    if not view.empty:
        view = view.sort_values(by=["Date", "EventName", "Name"]).reset_index(drop=True)

    st.markdown("### 比赛记录")
    # Display styled table
    styled = styled_seed_table(view, event, len_sel, names)
    try:
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(view, use_container_width=True, hide_index=True)

    # ---- Chart: line (time series) ----
    st.markdown("### 成绩折线图（越低越好）")
    import plotly.express as px
    # pick x-axis: Date; y: ResultSeconds; color: Name; facet by Event maybe
    chart_df = view.dropna(subset=["ResultSeconds"]).copy()
    if chart_df.empty:
        st.info("筛选后无可绘图数据。")
    else:
        # Use Date as datetime for better ordering
        try:
            chart_df["Date_dt"] = pd.to_datetime(chart_df["Date"], errors="coerce")
        except Exception:
            chart_df["Date_dt"] = chart_df["Date"]
        chart_df["ResultLabel"] = chart_df["ResultSeconds"].map(format_m_ss_xx)
        fig = px.line(
            chart_df,
            x="Date_dt",
            y="ResultSeconds",
            color="Name",
            markers=True,
            hover_data=["EventName", "PoolName", "LengthMeters", "ResultLabel"],
        )
        fig.update_yaxes(autorange="reversed", title="Seconds (lower is better)")
        fig.update_xaxes(title="Date")
        st.plotly_chart(fig, use_container_width=True)

    # Download filtered CSV
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button("下载筛选后的 CSV", data=csv_bytes, file_name="filtered_results.csv", mime="text/csv")


def page_manage():
    st.header("🗂️ 赛事管理 / 成绩录入")
    meet_dir = pick_meet_folder()
    if not meet_dir:
        return

    with st.expander("查看/编辑/删除历史成绩", expanded=True):
        render_history_manager(meet_dir)

    st.divider()
    with st.expander("新增成绩（按项目录入）", expanded=True):
        render_add_results(meet_dir)


def main():
    st.set_page_config(page_title="Swim Results", layout="wide")
    st.title("🏊‍♀️ 游泳成绩系统（赛事制）")

    page = st.sidebar.radio("页面", ["查询 / 对比", "赛事管理 / 录入"], index=0)
    if page == "查询 / 对比":
        page_browse()
    else:
        page_manage()


if __name__ == "__main__":
    main()
