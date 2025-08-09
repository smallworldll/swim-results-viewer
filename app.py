
import os
import io
import json
from datetime import datetime, date
from typing import List, Tuple

import pandas as pd
import streamlit as st

APP_TITLE = "🏊‍♀️ 游泳成绩系统（赛事制）"
MEETS_ROOT = "meets"


# ---------- Utilities ----------
def ensure_meets_root():
    os.makedirs(MEETS_ROOT, exist_ok=True)


def norm_pool_name(name: str) -> str:
    # sanitize folder name
    return name.replace("/", "_").replace("\\", "_").strip()


def meet_folder_name(dt: str, city: str, pool_name: str) -> str:
    folder = f"{dt}_{city}_{norm_pool_name(pool_name)}"
    return folder


def list_meets() -> List[str]:
    ensure_meets_root()
    items = []
    for x in os.listdir(MEETS_ROOT):
        p = os.path.join(MEETS_ROOT, x)
        if os.path.isdir(p):
            items.append(x)
    items.sort(reverse=True)  # newest first by folder string (starts with date)
    return items


def parse_time_to_seconds(s: str) -> float:
    """
    Accepts 'm:ss.xx' or 'ss.xx' and returns seconds as float.
    """
    s = (s or "").strip()
    if not s:
        return None
    # Standardize delimiter for millisecond part
    s = s.replace("：", ":")
    if ":" in s:
        m, rest = s.split(":", 1)
        try:
            m = int(m)
        except Exception:
            return None
        try:
            sec = float(rest)
        except Exception:
            return None
        return m * 60 + sec
    else:
        # only seconds
        try:
            sec = float(s)
            return sec
        except Exception:
            return None


def seconds_to_mmssxx(sec: float) -> str:
    if pd.isna(sec) or sec is None:
        return ""
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:05.2f}"


def read_meta(folder: str) -> pd.Series:
    meta_path = os.path.join(MEETS_ROOT, folder, "meta.csv")
    if not os.path.exists(meta_path):
        return pd.Series(dtype=object)
    df = pd.read_csv(meta_path)
    if df.empty:
        return pd.Series(dtype=object)
    return df.iloc[0]


def write_meta(dt_str: str, city: str, meet_name: str, pool_name: str, length_m: int) -> str:
    ensure_meets_root()
    folder = meet_folder_name(dt_str, city, pool_name)
    path = os.path.join(MEETS_ROOT, folder)
    os.makedirs(path, exist_ok=True)
    df = pd.DataFrame(
        [{
            "Date": dt_str,
            "City": city,
            "MeetName": meet_name,
            "PoolName": pool_name,
            "LengthMeters": int(length_m),
        }]
    )
    df.to_csv(os.path.join(path, "meta.csv"), index=False, encoding="utf-8")
    return folder


def read_results(folder: str) -> pd.DataFrame:
    res_path = os.path.join(MEETS_ROOT, folder, "results.csv")
    if not os.path.exists(res_path):
        return pd.DataFrame(columns=[
            "Name", "EventName", "Result", "Rank", "Note",
            "Date", "City", "MeetName", "PoolName", "LengthMeters"
        ])
    df = pd.read_csv(res_path)
    return df


def append_results(folder: str, rows: List[dict]):
    path = os.path.join(MEETS_ROOT, folder, "results.csv")
    old = read_results(folder)
    new = pd.DataFrame(rows)
    out = pd.concat([old, new], ignore_index=True)
    out.to_csv(path, index=False, encoding="utf-8")


def save_results_df(folder: str, df: pd.DataFrame):
    path = os.path.join(MEETS_ROOT, folder, "results.csv")
    df.to_csv(path, index=False, encoding="utf-8")


# ---------- UI helpers ----------
def toast_success(msg: str):
    st.success(msg, icon="✅")


def toast_warning(msg: str):
    st.warning(msg, icon="⚠️")


def toast_error(msg: str):
    st.error(msg, icon="❌")


# ---------- Pages ----------
def page_manage():
    st.header("📁 赛事管理 / 成绩录入")

    st.subheader("① 新建/选择赛事（meta）")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        dt_val = st.date_input("Date", value=date.today(), format="YYYY-MM-DD", key="meta_date")
    with col2:
        city_val = st.text_input("City", value="Chiang Mai", key="meta_city")
    with col3:
        meet_val = st.text_input("MeetName（必填）", value="", key="meta_meetname", placeholder="请输入赛事名称")

    colp1, colp2 = st.columns([2,1])
    with colp1:
        pool_val = st.text_input("PoolName（必填）", value="", key="meta_poolname", placeholder="请输入泳池名称")
    with colp2:
        length_val = st.selectbox("LengthMeters", [25, 50], index=0, key="meta_length")

    if st.button("保存赛事信息（写入/更新 meta.csv）", type="primary", use_container_width=False):
        if not meet_val.strip() or not pool_val.strip():
            toast_error("MeetName 和 PoolName 为必填，不能为空。")
        else:
            folder = write_meta(dt_val.strftime("%Y-%m-%d"), city_val.strip(), meet_val.strip(), pool_val.strip(), int(length_val))
            toast_success(f"已保存：{MEETS_ROOT}/{folder}/meta.csv")

    st.divider()

    # ② 新增成绩 + 已登记记录（优先展示已登记记录）
    st.subheader("② 已登记记录（先看后改/删，再决定是否新增）")

    meets = list_meets()
    if not meets:
        st.info("当前还没有赛事，请先在上面创建。")
        return

    # 默认选择最近一次赛事
    default_index = 0
    sel_meet = st.selectbox("选择赛事文件夹", meets, index=default_index, key="sel_meet_for_results")

    meta = read_meta(sel_meet)
    results_df = read_results(sel_meet).copy()

    # 统一提供可编辑/可删除的表格（简化操作）
    if results_df.empty:
        st.caption("该赛事尚无登记记录。")
    else:
        # 在编辑表格里不必展示 EventName（按你的要求）
        show_cols = ["Name", "Result", "Rank", "Note", "Date", "City", "MeetName", "PoolName", "LengthMeters"]
        exist_cols = [c for c in show_cols if c in results_df.columns]
        editable_df = results_df[exist_cols].copy()
        # 追加一个删除列，默认 False
        editable_df["删除?"] = False

        st.write("👇 直接在表格里编辑需要修改的字段；勾选“删除?”后点击下方按钮即可删除。")
        edited = st.data_editor(
            editable_df,
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor_existing",
        )

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("保存更改（写入 results.csv）", type="secondary", use_container_width=True, key="btn_save_edits"):
                # 删除被勾选的行
                keep = ~edited["删除?"].fillna(False)
                new_df = edited.loc[keep, exist_cols].copy()

                # 合并回原始（因为用户可能改动了部分列；我们用 exist_cols 替换原对应列）
                # 简化：直接用 new_df 覆盖（只保留展示列）。
                save_results_df(sel_meet, new_df)
                toast_success("更改已保存。")

        with colB:
            if st.button("删除勾选行 并 保存", type="primary", use_container_width=True, key="btn_delete_rows"):
                keep = ~edited["删除?"].fillna(False)
                new_df = edited.loc[keep, exist_cols].copy()
                save_results_df(sel_meet, new_df)
                toast_success("已删除并保存。")

    st.divider()
    st.subheader("③ 新增成绩（results）")

    # event 下拉来源：该赛事已存在的 EventName 列 + 自定义
    existing_events = []
    if "EventName" in results_df.columns and not results_df["EventName"].dropna().empty:
        existing_events = sorted(results_df["EventName"].dropna().unique().tolist())
    event_choice = st.selectbox("Event 选择", options=["（自定义）"] + existing_events, index=0, key="add_event_choice")
    if event_choice == "（自定义）":
        event_name = st.text_input("自定义 EventName", value="", placeholder="如：100m Butterfly", key="add_event_custom")
    else:
        event_name = event_choice

    n_rows = st.number_input("本次录入行数", min_value=1, max_value=20, value=1, step=1, key="add_n_rows")

    add_rows = []
    # Build inputs; rank default empty (no +/-), allow direct input
    for i in range(1, n_rows + 1):
        st.markdown(f"**记录 {i}**")
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 2.0])
        name = c1.text_input(f"Name_{i}", value="", key=f"add_name_{i}")
        # Show selected event name (read-only)
        _ = c2.text_input(f"EventName_{i}", value=event_name, key=f"add_eventname_{i}", disabled=True)
        result_str = c3.text_input(f"Result_{i}", value="", placeholder="m:ss.xx 或 ss.xx", key=f"add_result_{i}")
        rank_str = c4.text_input(f"Rank_{i}（可空）", value="", placeholder="整数，可空", key=f"add_rank_{i}")
        note = st.text_input(f"Note_{i}", value="", key=f"add_note_{i}")

        if name.strip() or result_str.strip():
            # Build row but validation deferred until save
            add_rows.append({
                "Name": name.strip(),
                "EventName": event_name.strip(),
                "Result": result_str.strip(),
                "Rank": rank_str.strip(),
                "Note": note.strip(),
            })

    # Save button
    col_left, col_right = st.columns([1,1])
    with col_left:
        do_push = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=True, key="add_save_local")

    if st.button("保存这些成绩", type="primary", key="btn_save_new_results"):
        if not add_rows:
            toast_warning("没有可保存的内容。")
        else:
            # Validate
            valid_rows = []
            missing = 0
            for row in add_rows:
                if not row["Name"] or not row["Result"]:
                    missing += 1
                    continue
                sec = parse_time_to_seconds(row["Result"])
                if sec is None:
                    toast_warning(f"时间格式无效：{row['Result']}（{row['Name']}）")
                    continue

                # Parse rank
                rank_val = None
                if row["Rank"] != "":
                    try:
                        rank_val = int(row["Rank"])
                    except Exception:
                        toast_warning(f"名次非整数：{row['Rank']}（{row['Name']}），已置空。")
                        rank_val = None

                # Meta info
                if meta.empty:
                    # fallback if meta missing
                    meta_dict = {"Date": "", "City": "", "MeetName": "", "PoolName": "", "LengthMeters": ""}
                else:
                    meta_dict = meta.to_dict()

                valid_rows.append({
                    "Name": row["Name"],
                    "EventName": row["EventName"],
                    "Result": seconds_to_mmssxx(sec),
                    "Rank": rank_val if rank_val is not None else "",
                    "Note": row["Note"],
                    "Date": meta_dict.get("Date", ""),
                    "City": meta_dict.get("City", ""),
                    "MeetName": meta_dict.get("MeetName", ""),
                    "PoolName": meta_dict.get("PoolName", ""),
                    "LengthMeters": meta_dict.get("LengthMeters", ""),
                })

            if not valid_rows:
                toast_warning("没有通过校验的记录可保存。")
            else:
                if do_push:
                    append_results(sel_meet, valid_rows)
                toast_success(f"已保存 {len(valid_rows)} 条。")

                # 清空历史输入，避免重复保存
                for i in range(1, n_rows + 1):
                    for k in [f"add_name_{i}", f"add_eventname_{i}", f"add_result_{i}", f"add_rank_{i}", f"add_note_{i}"]:
                        if k in st.session_state:
                            st.session_state[k] = ""

    st.caption("提示：名次默认空，不再提供 + / - 按钮；成绩支持 '34.12' 输入并自动解析为 '0:34.12'。")


def page_query():
    st.header("🔍 游泳成绩查询 / 对比")

    meets = list_meets()
    if not meets:
        st.info("暂无数据，请先到“赛事管理/成绩录入”页添加。")
        return

    # 读取所有 results 汇总
    all_rows = []
    for folder in meets:
        df = read_results(folder)
        if not df.empty:
            all_rows.append(df)
    if not all_rows:
        st.info("暂无成绩数据。")
        return
    data = pd.concat(all_rows, ignore_index=True)

    # 处理秒数列用于排序
    data["Seconds"] = data["Result"].map(parse_time_to_seconds)
    data_sorted = data.sort_values(by=["Seconds"], ascending=True, na_position="last").copy()

    # 筛选器
    names = sorted([x for x in data_sorted["Name"].dropna().unique().tolist() if x != ""])
    sel_names = st.multiselect("Name（可多选）", options=names, default=names[:1] if names else [])
    events = sorted([x for x in data_sorted["EventName"].dropna().unique().tolist() if x != ""])
    sel_event = st.selectbox("Event", options=["全部"] + events, index=0)

    # 过滤
    dfq = data_sorted.copy()
    if sel_names:
        dfq = dfq[dfq["Name"].isin(sel_names)]
    if sel_event != "全部":
        dfq = dfq[dfq["EventName"] == sel_event]

    # 展示
    show_cols = ["Name", "Date", "EventName", "Result", "Rank", "City", "PoolName", "LengthMeters"]
    show_cols = [c for c in show_cols if c in dfq.columns]
    st.dataframe(dfq[show_cols], use_container_width=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    tab = st.sidebar.radio("页面", ["赛事管理 / 录入", "查询 / 对比"], index=0)
    ensure_meets_root()

    if tab == "赛事管理 / 录入":
        page_manage()
    else:
        page_query()


if __name__ == "__main__":
    main()
