
import os, re, io, json, base64, shutil
from datetime import date, datetime
from typing import List, Tuple
import pandas as pd
import streamlit as st
import requests

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide")
APP_TITLE = "🏊‍♀️ 游泳成绩系统（赛事制）"

# ------------------------------
# Helpers
# ------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^0-9A-Za-z\- _]+", "", s)
    return s.replace(" ", "_")

def parse_time_to_secs(txt: str) -> float:
    """Accept formats like 'm:ss.xx', 'mm:ss', 'ss.xx'. Return seconds(float).
    Raise ValueError if not parsable."""
    t = (txt or "").strip()
    if not t:
        raise ValueError("empty")
    if re.fullmatch(r"\d{1,2}:\d{1,2}(\.\d{1,3})?", t):
        # m:ss(.xx)
        left, right = t.split(":")
        m = int(left)
        if "." in right:
            s, frac = right.split(".")
            s = int(s)
            cs = float("0." + frac)
            return m*60 + s + cs
        else:
            s = int(right)
            return m*60 + s
    elif re.fullmatch(r"\d{1,3}(\.\d{1,3})?", t):
        # ss(.xx)
        return float(t)
    else:
        raise ValueError(f"无法识别时间格式: {txt}")

def format_secs_to_mssx(x: float) -> str:
    if pd.isna(x):
        return ""
    x = max(0.0, float(x))
    m = int(x // 60)
    s = x - m*60
    return f"{m}:{s:05.2f}"  # mm:ss.xx

def load_meta(folder: str) -> pd.Series:
    meta_path = os.path.join(folder, "meta.csv")
    if os.path.exists(meta_path):
        m = pd.read_csv(meta_path)
        # Expect columns: Date, City, MeetName, PoolName, LengthMeters
        return m.iloc[0]
    else:
        return pd.Series(dtype=object)

def load_results(folder: str) -> pd.DataFrame:
    p = os.path.join(folder, "results.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df
    else:
        return pd.DataFrame(columns=[
            "Name","Date","City","EventName","Result","Rank","Note",
            "PoolName","LengthMeters","MeetName"
        ])

def write_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8")

def list_meet_folders() -> List[str]:
    base = "meets"
    if not os.path.exists(base):
        return []
    sub = []
    for d in os.listdir(base):
        full = os.path.join(base, d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "meta.csv")):
            sub.append(d)
    # sort by date prefix if possible
    def dkey(x):
        try:
            dd = x.split("_")[0]
            return datetime.strptime(dd, "%Y-%m-%d")
        except Exception:
            return datetime.min
    sub.sort(key=dkey, reverse=True)
    return sub

# ---------- GitHub push ----------
def _gh_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

def github_put_file(repo: str, path_in_repo: str, content_bytes: bytes, token: str) -> Tuple[bool, str]:
    """Create or update file in GitHub repo using contents API."""
    try:
        # get sha if exists
        url_get = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
        r = requests.get(url_get, headers=_gh_headers(token))
        sha = None
        if r.status_code == 200:
            sha = r.json().get("sha")

        url_put = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
        b64 = base64.b64encode(content_bytes).decode("utf-8")
        payload = {
            "message": f"Save {path_in_repo}",
            "content": b64
        }
        if sha:
            payload["sha"] = sha
        r2 = requests.put(url_put, headers=_gh_headers(token), json=payload)
        if r2.status_code in (200,201):
            return True, "ok"
        else:
            return False, f"{r2.status_code} {r2.text}"
    except Exception as e:
        return False, str(e)

def try_push(repo_path: str) -> Tuple[bool, str]:
    token = st.secrets.get("GITHUB_TOKEN", None)
    repo = st.secrets.get("REPO", None)
    if not token or not repo:
        return False, "未配置 GITHUB_TOKEN / REPO"
    if not os.path.exists(repo_path):
        return False, f"本地文件不存在：{repo_path}"
    with open(repo_path, "rb") as f:
        ok, msg = github_put_file(repo, repo_path, f.read(), token)
    return ok, msg

# ------------------------------
# UI building blocks
# ------------------------------
def section_create_meet():
    st.subheader("① 新建/选择赛事（meta）")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        d = st.date_input("Date", value=date.today(), key="meta_date")
    with c2:
        city = st.text_input("City", value="Chiang Mai", key="meta_city")
    with c3:
        meet_name = st.text_input("MeetName（必填）", value="", placeholder="例：Chiang Mai Swimming Championship", key="meta_meetname")

    c4, c5, c6 = st.columns([2,1,2])
    with c4:
        pool_name = st.text_input("PoolName（必填）", value="", placeholder="例：National Sports University Chiang Mai Campus", key="meta_poolname")
    with c5:
        length = st.selectbox("LengthMeters", options=[25,50], index=0, key="meta_length")
    with c6:
        # Removed EventName from meta per user's request
        st.markdown(" ")

    # checkboxes must have unique keys
    colb1, colb2 = st.columns([1,1])
    with colb1:
        push_meta = st.checkbox("保存时推送到 GitHub", value=False, key="meta_push")
    with colb2:
        save_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=False, key="meta_local")

    if st.button("保存赛事信息（写入/推送 meta.csv）", use_container_width=False, key="btn_save_meta"):
        if not meet_name.strip() or not pool_name.strip():
            st.error("MeetName 与 PoolName 为必填。")
        else:
            folder = f"meets/{d.isoformat()}_{sanitize(city)}_{sanitize(pool_name)}"
            ensure_dir(folder)
            meta_df = pd.DataFrame([{
                "Date": d.isoformat(),
                "City": city.strip(),
                "MeetName": meet_name.strip(),
                "PoolName": pool_name.strip(),
                "LengthMeters": int(length)
            }])[["Date","City","MeetName","PoolName","LengthMeters"]]
            write_csv(meta_df, os.path.join(folder, "meta.csv"))
            st.success(f"已保存：{folder}/meta.csv")

            if push_meta:
                ok, msg = try_push(os.path.join(folder, "meta.csv"))
                if ok:
                    st.info("GitHub 已推送 meta.csv")
                else:
                    st.warning(f"GitHub 推送失败：{msg}")

def section_results_entry_and_manage():
    st.subheader("② 新增成绩（results）")

    # pick meet (default latest)
    meets = list_meet_folders()
    default_idx = 0
    sel = ""
    if meets:
        default_idx = 0
        sel = st.selectbox("选择赛事文件夹", meets, index=default_idx, key="res_meet_sel")
    else:
        st.info("暂时没有赛事，请先在上方创建。")
        return

    meta = load_meta(os.path.join("meets", sel))
    if meta.empty:
        st.warning("该赛事缺少 meta.csv")
        return

    # event selection (predefined + custom)
    EVENT_OPTIONS = [
        "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle","400m Freestyle","800m Freestyle","1500m Freestyle",
        "25m Backstroke","50m Backstroke","100m Backstroke","200m Backstroke",
        "25m Breaststroke","50m Breaststroke","100m Breaststroke","200m Breaststroke",
        "25m Butterfly","50m Butterfly","100m Butterfly","200m Butterfly",
        "100m IM","200m IM","400m IM","Other..."
    ]
    event_pick = st.selectbox("Event 选择", options=EVENT_OPTIONS, index=EVENT_OPTIONS.index("100m Freestyle"), key="res_event_pick")
    custom_event = ""
    if event_pick == "Other...":
        custom_event = st.text_input("自定义项目名", key="res_event_custom")
    real_event = custom_event.strip() if event_pick == "Other..." else event_pick

    # rows count
    n = st.number_input("本次录入行数", min_value=1, max_value=20, value=1, step=1, key="res_rows")

    rows = []
    for i in range(1, n+1):
        st.markdown(f"**记录 {i}**")
        c1, c2, c3, c4 = st.columns([1.2,1.8,1,2])
        with c1:
            name = st.text_input(f"Name_{i}", key=f"res_name_{i}")
        with c2:
            # event per row follows overall event, but still editable if needed
            ev = st.text_input(f"EventName_{i}", value=real_event, key=f"res_event_{i}")
        with c3:
            res = st.text_input(f"Result_{i}", placeholder="0:34.12 或 34.12", key=f"res_time_{i}")
        with c4:
            rank = st.text_input(f"Rank_{i}", placeholder="", key=f"res_rank_{i}")
        c5 = st.text_input(f"Note_{i}", placeholder="可留空", key=f"res_note_{i}")
        rows.append((name, ev, res, rank, c5))

    colx1, colx2 = st.columns([1,1])
    with colx1:
        push_res = st.checkbox("保存时推送到 GitHub（免下载上传）", value=True, key="res_push")
    with colx2:
        save_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=False, key="res_local")

    if st.button("保存这些成绩", key="btn_save_results"):
        folder = os.path.join("meets", sel)
        res_path = os.path.join(folder, "results.csv")
        # Load existing
        df_old = load_results(folder)

        # build new rows
        add_rows = []
        errors = []
        for (name, ev, r, rk, note) in rows:
            if not name.strip() or not ev.strip() or not r.strip():
                # ignore empty
                continue
            try:
                secs = parse_time_to_secs(r)
            except Exception as e:
                errors.append(f"{name} / {ev} 时间格式错误：{r}")
                continue
            add_rows.append({
                "Name": name.strip(),
                "Date": meta["Date"],
                "City": meta["City"],
                "EventName": ev.strip(),
                "Result": format_secs_to_mssx(secs),
                "Rank": (rk.strip() if isinstance(rk,str) else rk),
                "Note": note.strip(),
                "PoolName": meta["PoolName"],
                "LengthMeters": int(meta["LengthMeters"]),
                "MeetName": meta["MeetName"]
            })
        if errors:
            for e in errors:
                st.error(e)
        if add_rows:
            df_add = pd.DataFrame(add_rows)
            df_new = pd.concat([df_old, df_add], ignore_index=True)
            write_csv(df_new, res_path)
            st.success(f"已保存 {len(df_add)} 条到 {res_path}")
            if push_res:
                ok, msg = try_push(res_path)
                if ok:
                    st.info("GitHub 已推送 results.csv")
                else:
                    st.warning(f"GitHub 推送失败：{msg}")

            # clear inputs to防重复提交
            for i in range(1, n+1):
                st.session_state[f"res_name_{i}"] = ""
                st.session_state[f"res_event_{i}"] = real_event
                st.session_state[f"res_time_{i}"] = ""
                st.session_state[f"res_rank_{i}"] = ""
                st.session_state[f"res_note_{i}"] = ""
        else:
            st.info("无有效行被写入。")

    # --- Manage existing ---
    st.subheader("③ 已登记记录（可编辑/删除）")
    df = load_results(os.path.join("meets", sel))
    if df.empty:
        st.info("该赛事暂无 results 记录。")
        return

    # show editor with a 删除 列更直观
    df_show = df.copy()
    df_show["删除"] = False
    edited = st.data_editor(
        df_show,
        key="results_editor",
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "删除": st.column_config.CheckboxColumn("删除", help="勾选后点下面按钮删除并保存")
        }
    )

    cdel1, cdel2 = st.columns([1,2])
    with cdel1:
        do_del = st.button("删除勾选行并保存（先勾选左侧“删除”）", key="btn_delete_rows")
    with cdel2:
        do_save = st.button("保存更改（写入 results.csv）", key="btn_save_edit")

    if do_del or do_save:
        # use edited content
        df_edited = pd.DataFrame(edited)
        # drop 删除=True
        if "删除" in df_edited.columns:
            df_edited = df_edited[df_edited["删除"] == False].drop(columns=["删除"])
        # ensure columns order
        cols = ["Name","Date","City","EventName","Result","Rank","Note","PoolName","LengthMeters","MeetName"]
        df_edited = df_edited.reindex(columns=cols)
        res_path = os.path.join("meets", sel, "results.csv")
        write_csv(df_edited, res_path)
        st.success("更改已保存。")
        if st.checkbox("保存后推送 GitHub", value=False, key="edit_push"):
            ok, msg = try_push(res_path)
            if ok:
                st.info("GitHub 已推送 results.csv")
            else:
                st.warning(f"GitHub 推送失败：{msg}")

def section_query_compare():
    st.subheader("游泳成绩查询 / 对比")
    # assemble all results
    meets = list_meet_folders()
    frames = []
    for m in meets:
        df = load_results(os.path.join("meets", m))
        if not df.empty:
            frames.append(df)
    if not frames:
        st.info("当前没有成绩数据。请先在“赛事管理/录入”中添加。")
        return
    data = pd.concat(frames, ignore_index=True)
    # numeric seconds
    def _to_sec(s):
        try:
            return parse_time_to_secs(str(s))
        except:
            return None
    data["secs"] = data["Result"].apply(_to_sec)

    # filters
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        names = sorted(data["Name"].dropna().unique().tolist())
        selected_names = st.multiselect("Name（可多选）", names, default=[])
    with c2:
        events = ["全部"] + sorted(data["EventName"].dropna().unique().tolist())
        event = st.selectbox("Event", events, index=0)
    with c3:
        length = st.selectbox("Length (Meters)", ["全部", 25, 50], index=0)
    with c4:
        pool = st.selectbox("PoolName", ["全部"] + sorted(data["PoolName"].dropna().unique().tolist()))
    with c5:
        city = st.selectbox("City", ["全部"] + sorted(data["City"].dropna().unique().tolist()))

    q = data.copy()
    if selected_names:
        q = q[q["Name"].isin(selected_names)]
    if event != "全部":
        q = q[q["EventName"]==event]
    if length != "全部":
        q = q[q["LengthMeters"]==int(length)]
    if pool != "全部":
        q = q[q["PoolName"]==pool]
    if city != "全部":
        q = q[q["City"]==city]

    # sort by time ascending
    q = q.sort_values(by=["secs"], ascending=True, na_position="last")

    # seed marking per person per (EventName, LengthMeters)
    q["Seed"] = ""
    if not q.empty:
        grp = q.groupby(["Name","EventName","LengthMeters"], dropna=False)
        idx = grp["secs"].idxmin()
        seeds = set(idx.dropna().tolist())
        q.loc[q.index.isin(seeds), "Seed"] = "★"

    show_cols = ["Seed","Name","Date","EventName","Result","Rank","Note","PoolName","City","LengthMeters","MeetName"]
    st.dataframe(q[show_cols], use_container_width=True, hide_index=True)

    # line chart when a specific event chosen
    if event != "全部" and not q.empty:
        chart_df = q.dropna(subset=["secs"]).copy()
        chart_df["Date2"] = pd.to_datetime(chart_df["Date"])
        chart_df = chart_df.sort_values("Date2")
        # pivot to one line per name
        pivot = chart_df.pivot_table(index="Date2", columns="Name", values="secs", aggfunc="min")
        st.markdown("**成绩折线图（越低越好）**")
        st.line_chart(pivot)

# ------------------------------
# Main navigation
# ------------------------------
st.title(APP_TITLE)
page = st.sidebar.radio("页面", ["查询/对比", "赛事管理/录入"], index=1)

if page == "赛事管理/录入":
    section_create_meet()
    st.markdown("---")
    section_results_entry_and_manage()
else:
    section_query_compare()
