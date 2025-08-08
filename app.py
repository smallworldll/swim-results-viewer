
# -*- coding: utf-8 -*-
import os
import io
import re
import json
import base64
import uuid
import datetime as dt
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ------------------------------
# Constants & Helpers
# ------------------------------
MEETS_ROOT = "meets"  # base folder

RESULT_COLS = ["RowID", "Name", "EventName", "Result", "Rank", "Note"]
META_COLS = ["Date", "City", "MeetName", "PoolName", "LengthMeters"]

TIME_PATTERN = re.compile(r"^\d+:\d{2}\.\d{2}$")  # m:ss.xx


def ensure_dirs():
    os.makedirs(MEETS_ROOT, exist_ok=True)


def list_meets() -> List[str]:
    ensure_dirs()
    items = []
    for name in sorted(os.listdir(MEETS_ROOT)):
        p = os.path.join(MEETS_ROOT, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "meta.csv")):
            items.append(name)
    return items


def meet_paths(meet_folder: str) -> Tuple[str, str]:
    folder = os.path.join(MEETS_ROOT, meet_folder)
    meta = os.path.join(folder, "meta.csv")
    results = os.path.join(folder, "results.csv")
    return meta, results


def parse_time_str(t: str) -> float:
    """Parse m:ss.xx -> seconds (float). Strict format."""
    if t is None:
        raise ValueError("Empty time string")
    t = str(t).strip()
    if not TIME_PATTERN.match(t):
        raise ValueError(f"时间格式必须是 m:ss.xx，例如 1:02.45；收到: {t}")
    minutes, sec_ms = t.split(":")
    seconds, hundredths = sec_ms.split(".")
    total = int(minutes) * 60 + int(seconds) + int(hundredths) / 100.0
    return total


def to_time_str(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    hs = int(round((seconds - int(seconds)) * 100))
    return f"{m}:{s:02d}.{hs:02d}"


def load_meta(meta_csv: str) -> pd.DataFrame:
    if os.path.isfile(meta_csv):
        df = pd.read_csv(meta_csv)
        return df[META_COLS]
    else:
        return pd.DataFrame(columns=META_COLS)


def load_results(results_csv: str) -> pd.DataFrame:
    if os.path.isfile(results_csv):
        df = pd.read_csv(results_csv)
        if "RowID" not in df.columns:
            # backfill RowID if user has old file
            df["RowID"] = [str(uuid.uuid4()) for _ in range(len(df))]
        # Ensure cols order and types
        for col in RESULT_COLS:
            if col not in df.columns:
                df[col] = ""
        return df[RESULT_COLS]
    else:
        return pd.DataFrame(columns=RESULT_COLS)


def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def combine_all_meets() -> pd.DataFrame:
    """Join results with meta for browsing."""
    rows = []
    for folder in list_meets():
        meta_csv, res_csv = meet_paths(folder)
        meta = load_meta(meta_csv)
        res = load_results(res_csv)
        if meta.empty:
            continue
        # Use first (only) row in meta
        m = meta.iloc[0].to_dict()
        if not res.empty:
            df = res.copy()
            for k, v in m.items():
                df[k] = v
            df["MeetFolder"] = folder
            rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=RESULT_COLS + META_COLS + ["MeetFolder"])


def github_push_file(local_path: str, dest_path: str, message: str) -> Tuple[bool, str]:
    """Push a single file to GitHub using REST API.
       Requires secrets: GITHUB_TOKEN, REPO (e.g., 'username/repo').
    """
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo = st.secrets.get("REPO", None)
    if not token or not repo:
        return False, "缺少 Secrets：GITHUB_TOKEN 或 REPO"

    import requests

    url = f"https://api.github.com/repos/{repo}/contents/{dest_path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Check if file exists to get sha
    r = requests.get(url, headers=headers)
    sha = None
    if r.status_code == 200:
        sha = r.json().get("sha")

    data = {"message": message, "content": content_b64, "branch": "main"}
    if sha:
        data["sha"] = sha

    r = requests.put(url, headers=headers, data=json.dumps(data))
    if r.status_code in (200, 201):
        return True, "已提交到 GitHub"
    else:
        try:
            err = r.json()
        except Exception:
            err = r.text
        return False, f"GitHub 提交失败: {err}"


# ------------------------------
# UI Helpers
# ------------------------------
def event_name_options() -> List[str]:
    # 常见项目，可自行扩展
    base = [
        "25m Freestyle", "50m Freestyle", "100m Freestyle", "200m Freestyle",
        "25m Backstroke", "50m Backstroke", "100m Backstroke", "200m Backstroke",
        "25m Breaststroke", "50m Breaststroke", "100m Breaststroke", "200m Breaststroke",
        "25m Butterfly", "50m Butterfly", "100m Butterfly", "200m Butterfly",
        "100m IM", "200m IM", "400m IM"
    ]
    return base


def highlight_best(row, best_map):
    key = (row["Name"], row["EventName"], row["LengthMeters"])
    is_best = False
    try:
        is_best = parse_time_str(row["Result"]) == best_map[key]
    except Exception:
        is_best = False
    return ["color: red; font-weight:700" if is_best else "" for _ in row]


def display_trend_chart(df: pd.DataFrame):
    # Prepare time series by date for each person
    if df.empty:
        return
    df2 = df.copy()
    # parse date as datetime for sorting
    try:
        df2["Date"] = pd.to_datetime(df2["Date"])
    except Exception:
        pass
    # seconds for plotting
    df2["Seconds"] = df2["Result"].apply(lambda x: parse_time_str(x))
    for name in sorted(df2["Name"].unique()):
        sub = df2[df2["Name"] == name].sort_values("Date")
        if sub.empty:
            continue
        st.line_chart(sub.set_index("Date")["Seconds"], height=220, use_container_width=True)
        st.caption(f"折线图（单位：秒） - {name}（越低越好）")


# ------------------------------
# Pages
# ------------------------------
def page_browse():
    st.header("🔍 查询与对比")

    all_df = combine_all_meets()
    if all_df.empty:
        st.info("还没有任何比赛数据。请先在『赛事管理』里创建并录入。")
        return

    # Filters
    names = sorted(all_df["Name"].dropna().unique().tolist())
    events = sorted(all_df["EventName"].dropna().unique().tolist())
    lengths = sorted(all_df["LengthMeters"].dropna().unique().tolist())
    pools = sorted(all_df["PoolName"].dropna().unique().tolist())
    cities = sorted(all_df["City"].dropna().unique().tolist())
    dates = sorted(all_df["Date"].dropna().unique().tolist())

    c1, c2 = st.columns(2)
    sel_names = c1.multiselect("Name（可多选）", names, default=[n for n in names if n.lower() == "anna"] or [])
    sel_event = c2.selectbox("Event", ["All"] + events, index=0)

    c3, c4, c5 = st.columns(3)
    sel_len = c3.selectbox("Length (Meters)", ["All"] + [str(x) for x in lengths], index=0)
    sel_pool = c4.selectbox("Pool Name", ["All"] + pools, index=0)
    sel_city = c5.selectbox("City", ["All"] + cities, index=0)

    sel_date = st.selectbox("Date", ["All"] + dates, index=0)

    df = all_df.copy()
    if sel_names:
        df = df[df["Name"].isin(sel_names)]
    if sel_event != "All":
        df = df[df["EventName"] == sel_event]
    if sel_len != "All":
        df = df[df["LengthMeters"].astype(str) == sel_len]
    if sel_pool != "All":
        df = df[df["PoolName"] == sel_pool]
    if sel_city != "All":
        df = df[df["City"] == sel_city]
    if sel_date != "All":
        df = df[df["Date"] == sel_date]

    if df.empty:
        st.warning("没有匹配的数据。调整一下条件试试。")
        return

    # 计算每个 Name+EventName+Length 的最佳（最小秒数）
    df["_sec"] = df["Result"].apply(parse_time_str)
    best_map = (
        df.groupby(["Name", "EventName", "LengthMeters"])["_sec"]
        .min()
        .to_dict()
    )

    disp_cols = ["Name", "Date", "EventName", "Result", "Rank", "Note",
                 "PoolName", "City", "LengthMeters"]
    disp = df.sort_values(["Name", "EventName", "Date"])[disp_cols].reset_index(drop=True)

    styled = disp.style.apply(lambda r: highlight_best(r, best_map), axis=1)
    st.dataframe(styled, use_container_width=True)

    st.subheader("📈 趋势图")
    display_trend_chart(df)


def page_manage():
    st.header("🛠️ 赛事管理 / 按项目录入")

    ensure_dirs()
    meets = list_meets()

    st.markdown("### 选择或新建赛事")
    mode = st.radio("操作", ["选择已有赛事", "新建赛事"], horizontal=True)

    meet_folder = None
    if mode == "选择已有赛事" and meets:
        meet_folder = st.selectbox("已有赛事", meets)
    elif mode == "新建赛事":
        with st.form("new_meet"):
            c1, c2 = st.columns(2)
            date_str = c1.text_input("Date (YYYY-MM-DD)", dt.date.today().isoformat())
            city = c2.text_input("City", "")
            meet_name = st.text_input("MeetName", "")
            pool = st.text_input("PoolName", "")
            length = st.selectbox("LengthMeters", [25, 50], index=1)
            submit = st.form_submit_button("创建赛事")
        if submit:
            # create folder
            folder = f"{date_str}_{city}".replace(" ", "")
            meta_csv, res_csv = meet_paths(folder)
            if os.path.exists(os.path.dirname(meta_csv)) and os.path.isfile(meta_csv):
                st.error("该日期/城市的赛事已存在。")
            else:
                os.makedirs(os.path.dirname(meta_csv), exist_ok=True)
                meta_df = pd.DataFrame([{
                    "Date": date_str, "City": city, "MeetName": meet_name,
                    "PoolName": pool, "LengthMeters": int(length)
                }])[META_COLS]
                save_csv(meta_df, meta_csv)
                save_csv(pd.DataFrame(columns=RESULT_COLS), res_csv)
                st.success(f"已创建赛事：{folder}")
                meets = list_meets()
                meet_folder = folder

    if not meet_folder:
        st.stop()

    meta_csv, res_csv = meet_paths(meet_folder)
    meta = load_meta(meta_csv)
    res = load_results(res_csv)

    st.markdown(f"**当前赛事：** `{meet_folder}`")
    st.dataframe(meta, use_container_width=True)

    # Select or create EventName
    st.markdown("### 选择或创建单项（Event）")
    existing_events = sorted(res["EventName"].dropna().unique().tolist())
    c1, c2 = st.columns(2)
    event = c1.selectbox("选择已有 Event", ["<新建>"] + existing_events, index=1 if existing_events else 0)
    new_event = c2.selectbox("快速选择常用项目（用于新建）", ["(不使用)"] + event_name_options(), index=0)
    if event == "<新建>":
        event = st.text_input("输入新 EventName", ("" if new_event == "(不使用)" else new_event))

    if not event:
        st.info("先选择或输入一个 EventName。")
        st.stop()

    st.markdown(f"**正在录入项目：** `{event}`")

    # ---- Batch paste ----
    st.markdown("#### 批量粘贴名次表（Name, Result[, Note]）")
    st.caption("每行一位选手：例如 `Anna, 1:02.45, 下雨`；`Rank` 会按成绩自动计算，也可在下方编辑/重排。")
    pasted = st.text_area("在此粘贴（可多行）", height=150, placeholder="Name, m:ss.xx, Note（可选）")

    if "batch_preview" not in st.session_state:
        st.session_state["batch_preview"] = pd.DataFrame(columns=RESULT_COLS)

    if st.button("预览粘贴内容"):
        rows = []
        for ln in pasted.splitlines():
            if not ln.strip():
                continue
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) < 2:
                st.error(f"无法解析：{ln}")
                continue
            name = parts[0]
            result = parts[1]
            note = parts[2] if len(parts) > 2 else ""
            # validate result
            try:
                _sec = parse_time_str(result)
            except Exception as e:
                st.error(f"{ln} -> {e}")
                continue
            rows.append({
                "RowID": str(uuid.uuid4()),
                "Name": name,
                "EventName": event,
                "Result": result,
                "Rank": 0,  # will set later
                "Note": note
            })
        prev = pd.DataFrame(rows, columns=RESULT_COLS)
        if not prev.empty:
            prev["_sec"] = prev["Result"].apply(parse_time_str)
            prev = prev.sort_values("_sec").drop(columns="_sec")
            prev["Rank"] = range(1, len(prev) + 1)
        st.session_state["batch_preview"] = prev

    prev = st.session_state["batch_preview"]
    if not prev.empty:
        st.dataframe(prev[["Name", "Result", "Rank", "Note"]], use_container_width=True)
        c1, c2 = st.columns(2)
        if c1.button("追加到本场 results.csv"):
            new_df = pd.concat([res, prev], ignore_index=True)
            save_csv(new_df, res_csv)
            st.success(f"已写入 {len(prev)} 条记录。")
            st.session_state["batch_preview"] = pd.DataFrame(columns=RESULT_COLS)
            res = load_results(res_csv)  # refresh
        if c2.button("清空预览区"):
            st.session_state["batch_preview"] = pd.DataFrame(columns=RESULT_COLS)

    st.markdown("#### 逐条添加")
    with st.form("add_one"):
        c1, c2, c3 = st.columns([2, 1, 2])
        name_one = c1.text_input("Name", "")
        result_one = c2.text_input("Result (m:ss.xx)", "")
        note_one = c3.text_input("Note", "")
        add = st.form_submit_button("添加一条")
    if add:
        try:
            parse_time_str(result_one)
            row = pd.DataFrame([{
                "RowID": str(uuid.uuid4()),
                "Name": name_one,
                "EventName": event,
                "Result": result_one,
                "Rank": 0,
                "Note": note_one
            }])[RESULT_COLS]
            new_df = pd.concat([res, row], ignore_index=True)
            save_csv(new_df, res_csv)
            st.success("已添加。")
            res = load_results(res_csv)
        except Exception as e:
            st.error(f"添加失败：{e}")

    # Show current event table & tools
    st.markdown("### 当前项目成绩（本场）")
    event_df = res[res["EventName"] == event].copy()
    if event_df.empty:
        st.info("这个项目还没有成绩。")
    else:
        # Re-rank by time
        if st.button("按成绩重排名次（1为最好）"):
            tmp = event_df.copy()
            tmp["_sec"] = tmp["Result"].apply(parse_time_str)
            tmp = tmp.sort_values("_sec").drop(columns="_sec")
            tmp["Rank"] = range(1, len(tmp) + 1)
            # write back to res
            res_update = res.copy()
            for _, r in tmp.iterrows():
                res_update.loc[res_update["RowID"] == r["RowID"], "Rank"] = r["Rank"]
            save_csv(res_update, res_csv)
            st.success("已重排名次。")
            res = load_results(res_csv)
            event_df = res[res["EventName"] == event].copy()

        st.dataframe(event_df[["Name", "Result", "Rank", "Note"]], use_container_width=True)

        # Edit
        st.markdown("#### 编辑单条")
        choices = [f'{r.Name} | {r.Result} | Rank {int(r.Rank)} | {r.RowID[:8]}' for _, r in event_df.iterrows()]
        if choices:
            pick = st.selectbox("选择要编辑的记录", ["(不编辑)"] + choices, index=0)
            if pick != "(不编辑)":
                idx = choices.index(pick)
                row = event_df.iloc[idx]
                with st.form("edit_row"):
                    c1, c2, c3 = st.columns([2,1,2])
                    new_name = c1.text_input("Name", row["Name"])
                    new_result = c2.text_input("Result (m:ss.xx)", row["Result"])
                    new_note = c3.text_input("Note", row["Note"])
                    new_rank = st.number_input("Rank", value=int(row["Rank"]), min_value=0)
                    ok = st.form_submit_button("保存修改")
                if ok:
                    try:
                        parse_time_str(new_result)
                        res2 = res.copy()
                        mask = res2["RowID"] == row["RowID"]
                        res2.loc[mask, "Name"] = new_name
                        res2.loc[mask, "Result"] = new_result
                        res2.loc[mask, "Rank"] = int(new_rank)
                        res2.loc[mask, "Note"] = new_note
                        save_csv(res2, res_csv)
                        st.success("已保存。")
                        res = load_results(res_csv)
                        event_df = res[res["EventName"] == event].copy()
                    except Exception as e:
                        st.error(f"保存失败：{e}")

        # Delete
        st.markdown("#### 删除记录")
        del_choices = [f'{r.Name} | {r.Result} | RowID {r.RowID[:8]}' for _, r in event_df.iterrows()]
        del_pick = st.selectbox("选择要删除的记录", ["(不删除)"] + del_choices, index=0)
        if del_pick != "(不删除)":
            idx = del_choices.index(del_pick)
            row = event_df.iloc[idx]
            if st.button("确认删除这条记录"):
                res2 = res[res["RowID"] != row["RowID"]].copy()
                save_csv(res2, res_csv)
                st.success("已删除。")
                res = load_results(res_csv)

    st.markdown("---")
    st.markdown("### 保存/同步")
    push_gh = st.checkbox("提交到 GitHub（免下载上传）", value=False)
    save_local = st.checkbox("同时保存到本地 meets/ 目录（已默认）", value=True, disabled=True)

    if st.button("保存/同步文件"):
        # local already saved on operations; here we just optionally push
        ok1 = True
        msg1 = "本地已保存。"
        # push two files
        if push_gh:
            meta_ok, meta_msg = github_push_file(meta_csv, f"{MEETS_ROOT}/{meet_folder}/meta.csv", f"[{meet_folder}] update meta")
            res_ok, res_msg = github_push_file(res_csv, f"{MEETS_ROOT}/{meet_folder}/results.csv", f"[{meet_folder}] update results")
            ok1 = meta_ok and res_ok
            msg1 = meta_msg + " | " + res_msg
        if ok1:
            st.success(msg1)
        else:
            st.error(msg1)


# ------------------------------
# Main
# ------------------------------
def main():
    st.set_page_config(page_title="Swim Results", layout="wide")
    st.title("🏊 游泳成绩系统")

    tabs = st.tabs(["查询与对比", "赛事管理"])
    with tabs[0]:
        page_browse()
    with tabs[1]:
        page_manage()


if __name__ == "__main__":
    main()
