# -*- coding: utf-8 -*-
"""
游泳成绩系统（赛事制） - 精简稳固版
修复点：
- 保存后清空输入，不再出现 Streamlit safe_session_state 相关报错
- 结果写入采用合并去重，防止重复保存
- 支持 34.12 / 0:34.12 两种时间输入，统一规范化
- “已登记记录”放在新增成绩之前；删除更直观
- meta 不含 EventName；PoolName/MeetName 为必填；LengthMeters 只允许 25/50
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import re
import base64
import json
import requests

APP_TITLE = "🏊‍♂️ 游泳成绩系统（赛事制）"
MEETS_ROOT = Path("meets")

# =====================
# 辅助：时间解析/规范化
# =====================
TIME_RE = re.compile(r"^\s*(?:(\d{1,2}):)?(\d{1,2})(?:[.:](\d{1,2}))?\s*$")

def parse_time_to_seconds(s: str) -> float | None:
    """接受 '34.12' 或 '0:34.12' 或 '1:05.3'，返回秒(float)，不合法返回 None"""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    m = TIME_RE.match(s)
    if not m:
        return None
    mm = m.group(1)
    ss = m.group(2)
    ff = m.group(3)
    minutes = int(mm) if mm is not None else 0
    seconds = int(ss)
    hundredths = int(ff) if ff is not None else 0
    # 如果给了 1 位小数，视为十分位（补 0）；给 2 位按百分位；超过 2 位截断
    if ff is not None and len(ff) == 1:
        hundredths *= 10
    return minutes * 60 + seconds + hundredths / 100.0

def seconds_to_mssxx(sec: float) -> str:
    """将秒转成 m:ss.xx 统一显示"""
    if sec is None or np.isnan(sec):
        return ""
    sec = float(sec)
    minutes = int(sec // 60)
    rest = sec - minutes * 60
    s = int(rest)
    hundredths = int(round((rest - s) * 100))
    # 有时四舍五入会到 .100，进位处理
    if hundredths == 100:
        hundredths = 0
        s += 1
        if s == 60:
            s = 0
            minutes += 1
    return f"{minutes}:{s:02d}.{hundredths:02d}"

# =====================
# 文件 I/O
# =====================
def load_meta(meet_dir: Path) -> pd.Series:
    p = meet_dir / "meta.csv"
    if not p.exists():
        return pd.Series(dtype=object)
    df = pd.read_csv(p)
    if df.empty:
        return pd.Series(dtype=object)
    return df.iloc[0]

def save_meta(meet_dir: Path, date_str: str, city: str, meet_name: str, pool_name: str, length_m: int):
    meet_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame([{
        "Date": date_str,
        "City": city,
        "MeetName": meet_name,
        "PoolName": pool_name,
        "LengthMeters": int(length_m),
    }])
    meta.to_csv(meet_dir / "meta.csv", index=False, encoding="utf-8-sig")

def load_results(meet_dir: Path) -> pd.DataFrame:
    p = meet_dir / "results.csv"
    if not p.exists():
        return pd.DataFrame(columns=["Name","EventName","Result","Rank","Note",
                                     "Date","City","MeetName","PoolName","LengthMeters","Seconds"])
    df = pd.read_csv(p)
    # 确保 Seconds 存在
    if "Seconds" not in df.columns and "Result" in df.columns:
        secs = df["Result"].map(parse_time_to_seconds)
        df["Seconds"] = secs
    return df

def _norm_col(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()

def write_results_dedup(meet_dir: Path, df_new: pd.DataFrame):
    """合并去重写入 results.csv"""
    p = meet_dir / "results.csv"
    if p.exists():
        df_all = pd.concat([pd.read_csv(p), df_new], ignore_index=True)
    else:
        df_all = df_new.copy()

    # 去重键：人+项目+成绩+日期+城市+场馆+池长
    parts = [
        _norm_col(df_all["Name"]),
        _norm_col(df_all["EventName"]),
        _norm_col(df_all["Result"]),
        _norm_col(df_all.get("Date","")),
        _norm_col(df_all.get("City","")),
        _norm_col(df_all.get("MeetName","")),
        _norm_col(df_all.get("PoolName","")),
        _norm_col(df_all.get("LengthMeters","")),
    ]
    df_all["__k__"] = "|".join(parts) if isinstance(parts, str) else parts[0]
    # 如果不是字符串拼接，上面返回的是 Series 列表，正确写法：
    if "__k__" in df_all.columns and not isinstance(df_all["__k__"], pd.Series):
        df_all["__k__"] = (
            _norm_col(df_all["Name"]) + "|" +
            _norm_col(df_all["EventName"]) + "|" +
            _norm_col(df_all["Result"]) + "|" +
            _norm_col(df_all.get("Date","")) + "|" +
            _norm_col(df_all.get("City","")) + "|" +
            _norm_col(df_all.get("MeetName","")) + "|" +
            _norm_col(df_all.get("PoolName","")) + "|" +
            _norm_col(df_all.get("LengthMeters",""))
        )

    df_all = df_all.drop_duplicates("__k__", keep="first").drop(columns="__k__")
    df_all.to_csv(p, index=False, encoding="utf-8-sig")

# =====================
# GitHub 推送（可选）
# =====================
def github_upsert(repo: str, token: str, path_in_repo: str, content_bytes: bytes, commit_msg: str):
    """用 GitHub API create/update contents"""
    api = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    # 先查 sha（判断是否已存在）
    r = requests.get(api, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")
    else:
        sha = None

    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha

    r2 = requests.put(api, headers=headers, data=json.dumps(payload))
    if r2.status_code not in (200, 201):
        raise RuntimeError(f"GitHub 推送失败：{r2.status_code} {r2.text}")

def try_push_to_github(local_path: Path, rel_path_in_repo: str, commit_msg: str):
    token = st.secrets.get("GITHUB_TOKEN")
    repo = st.secrets.get("REPO")
    if not token or not repo:
        return  # 未配置就忽略
    github_upsert(repo, token, rel_path_in_repo, local_path.read_bytes(), commit_msg)

# =====================
# UI 辅助
# =====================
def list_meets() -> list[Path]:
    if not MEETS_ROOT.exists():
        return []
    ds = [p for p in MEETS_ROOT.iterdir() if p.is_dir()]
    return sorted(ds)

def latest_meet_dir() -> Path | None:
    ms = list_meets()
    return ms[-1] if ms else None

def ensure_state_defaults(row_count: int):
    """确保控件 key 存在（避免 safe_session_state 赋值报错）"""
    for i in range(1, row_count+1):
        for f in ("Name","EventName","Result","Rank","Note"):
            st.session_state.setdefault(f"{f}_{i}", "")

def clear_result_inputs(row_count: int):
    """保存后清空输入并刷新"""
    for i in range(1, row_count+1):
        for f in ("Name","EventName","Result","Rank","Note"):
            st.session_state.pop(f"{f}_{i}", None)
    st.session_state.pop("rows_count", None)
    st.rerun()

DEFAULT_EVENTS = [
    # 25m池典型项目
    "25m Freestyle","50m Freestyle","100m Freestyle","200m Freestyle",
    "25m Backstroke","50m Backstroke","100m Backstroke","200m Backstroke",
    "25m Breaststroke","50m Breaststroke","100m Breaststroke","200m Breaststroke",
    "25m Butterfly","50m Butterfly","100m Butterfly","200m Butterfly",
    "100m IM","200m IM","400m Freestyle",
]

def section_meta():
    st.subheader("① 新建/选择赛事（meta）")
    with st.form("meta_form", clear_on_submit=False):
        d = st.date_input("Date", value=date.today(), format="YYYY-MM-DD")
        city = st.text_input("City", value="Chiang Mai")
        meet_name = st.text_input("MeetName（必填）", value="", placeholder="例：Chiang Mai Local Meet")
        pool_name = st.text_input("PoolName（必填）", value="", placeholder="例：National Sports University Chiang Mai Campus")
        length_m = st.selectbox("LengthMeters", options=[25, 50], index=0)

        push = st.checkbox("保存时推送到 GitHub（可选）", value=False)
        submitted = st.form_submit_button("保存赛事信息（写入/推送 meta.csv）")
        if submitted:
            if not meet_name.strip() or not pool_name.strip():
                st.error("❌ MeetName 与 PoolName 必填。")
            else:
                meet_dir = MEETS_ROOT / f"{d}_{city}_{pool_name}"
                save_meta(meet_dir, d.isoformat(), city.strip(), meet_name.strip(), pool_name.strip(), int(length_m))
                st.success(f"✅ 已保存：{meet_dir/'meta.csv'}")
                if push:
                    rel = f"{meet_dir}/meta.csv".replace("\\", "/")
                    try:
                        try_push_to_github(meet_dir/"meta.csv", rel, f"Save meta for {meet_dir.name}")
                        st.info("GitHub 已推送 meta.csv")
                    except Exception as e:
                        st.warning(f"GitHub 推送失败：{e}")

def section_results_entry_and_manage():
    st.subheader("② 新增成绩 & 管理")

    meets = list_meets()
    if not meets:
        st.info("当前尚无赛事，请先在上方创建 meta。")
        return

    default_idx = len(meets)-1
    meet_dir = st.selectbox("选择赛事文件夹", options=meets, index=default_idx, format_func=lambda p: p.name)
    meta = load_meta(meet_dir)

    # ========== 已登记记录（先展示） ==========
    st.markdown("**已登记记录（可编辑/删除）**")
    df = load_results(meet_dir)

    # EventName 可见
    display_cols = ["Name","EventName","Result","Rank","Note","Seconds","Date","City","MeetName","PoolName","LengthMeters"]
    for c in display_cols:
        if c not in df.columns:
            df[c] = ""

    # 操作区域：删除勾选
    if not df.empty:
        _sel = st.multiselect("选择要删除的行（按行号）", options=list(df.index), format_func=lambda i: f"{i+1}：{df.loc[i,'Name']} - {df.loc[i,'EventName']} - {df.loc[i,'Result']}")
        col_del, col_save = st.columns([1,1])
        with col_del:
            if st.button("🗑️ 删除所选行（需要再点一次下方“保存更改”才会落盘）", type="secondary", use_container_width=True, disabled=len(_sel)==0):
                st.session_state["__to_delete__"] = list(_sel)
                st.info(f"已标记待删除：{len(_sel)} 行，请点击下方 **保存更改**。")

        with col_save:
            if st.button("💾 保存更改（写入 results.csv）", use_container_width=True):
                to_delete = st.session_state.pop("__to_delete__", [])
                if to_delete:
                    df2 = df.drop(index=to_delete).reset_index(drop=True)
                else:
                    df2 = df
                # 统一 Seconds/Result 格式
                df2["Seconds"] = df2["Result"].map(parse_time_to_seconds)
                df2["Result"] = df2["Seconds"].map(seconds_to_mssxx)
                df2.to_csv(meet_dir/"results.csv", index=False, encoding="utf-8-sig")
                st.success("更改已保存。")
                st.rerun()

        # 显示（按成绩由短到长）
        df_show = df.copy()
        df_show["Seconds"] = df_show["Seconds"].map(lambda x: parse_time_to_seconds(str(x)) if pd.notna(x) else np.nan)
        df_show = df_show.sort_values("Seconds", na_position="last").reset_index(drop=True)
        st.dataframe(df_show[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("该赛事暂时没有成绩记录。")

    st.markdown("---")
    # ========== 新增成绩（在已登记后面） ==========

    # 提供项目下拉（从默认 + 现有项目并集）
    existing_events = sorted(df["EventName"].dropna().unique().tolist()) if not df.empty else []
    options = sorted(set(DEFAULT_EVENTS + existing_events))
    selected_event = st.selectbox("Event 选择", options=options, index=options.index("100m Freestyle") if "100m Freestyle" in options else 0)
    rows_count = st.number_input("本次录入行数", min_value=1, max_value=10, value=1, step=1, key="rows_count")

    ensure_state_defaults(int(rows_count))

    # 输入区
    rows = []
    for i in range(1, int(rows_count)+1):
        st.markdown(f"**记录 {i}**")
        cols = st.columns([1,1,1,1])
        name = cols[0].text_input(f"Name_{i}", key=f"Name_{i}", placeholder="选手名")
        ev = cols[1].text_input(f"EventName_{i}", key=f"EventName_{i}", value=selected_event)
        res = cols[2].text_input(f"Result_{i}", key=f"Result_{i}", placeholder="34.12 或 0:34.12")
        rank = cols[3].text_input(f"Rank_{i}（可空）", key=f"Rank_{i}", value="")
        note = st.text_input(f"Note_{i}", key=f"Note_{i}", placeholder="可留空")
        rows.append((name, ev, res, rank, note))

    col_left, col_right = st.columns([1,1])
    push_now = col_left.checkbox("保存时推送到 GitHub（免下载上传）", value=False)
    save_clicked = col_right.button("保存这些成绩", use_container_width=True)

    if save_clicked:
        # 组装新增 df
        records = []
        for name, ev, res, rank, note in rows:
            if not str(name).strip() or not str(ev).strip() or not str(res).strip():
                # 空行跳过
                continue
            seconds = parse_time_to_seconds(str(res))
            if seconds is None:
                st.warning(f"时间格式不合法：{res}（已跳过）")
                continue
            records.append({
                "Name": str(name).strip(),
                "EventName": str(ev).strip(),
                "Result": seconds_to_mssxx(seconds),
                "Rank": str(rank).strip() if str(rank).strip() else "",
                "Note": str(note).strip(),
                "Date": str(meta.get("Date", "")),
                "City": str(meta.get("City","")),
                "MeetName": str(meta.get("MeetName","")),
                "PoolName": str(meta.get("PoolName","")),
                "LengthMeters": int(meta.get("LengthMeters", 25)) if str(meta.get("LengthMeters","")).strip() else 25,
                "Seconds": seconds,
            })
        if not records:
            st.warning("没有有效数据可保存。")
        else:
            df_new = pd.DataFrame(records)
            write_results_dedup(meet_dir, df_new)
            st.success(f"✅ 已保存 {len(records)} 条到 {meet_dir/'results.csv'}")

            # 推送到 GitHub（可选）
            if push_now:
                rel = f"{meet_dir}/results.csv".replace("\\", "/")
                try:
                    try_push_to_github(meet_dir/"results.csv", rel, f"Append results for {meet_dir.name}")
                    st.info("GitHub 已推送 results.csv")
                except Exception as e:
                    st.warning(f"GitHub 推送失败：{e}")

            # 清空输入并刷新，避免重复保存
            clear_result_inputs(int(rows_count))

# =====================
# 页面：查询 / 对比
# =====================
def page_browse():
    st.header("🏊‍♂️ 游泳成绩查询 / 对比")

    # 合并所有 meets 的 results 进行查询
    frames = []
    for m in list_meets():
        p = m/"results.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["__meet__"] = m.name
            frames.append(df)
    if not frames:
        st.info("当前没有可查询的成绩，请先录入。")
        return

    data = pd.concat(frames, ignore_index=True)
    # 统一 Seconds、Result
    data["Seconds"] = data["Seconds"] if "Seconds" in data.columns else data["Result"].map(parse_time_to_seconds)
    data["Result"] = data["Seconds"].map(seconds_to_mssxx)

    names = sorted(data["Name"].dropna().unique().tolist())
    events = sorted(data["EventName"].dropna().unique().tolist())
    lengths = ["全部"] + sorted(data["LengthMeters"].dropna().astype(str).unique().tolist())

    pick_names = st.multiselect("Name（可多选）", options=names, default=names[:1] if names else [])
    pick_event = st.selectbox("Event", options=["全部"]+events, index=0)
    pick_len = st.selectbox("Length (Meters)", options=lengths, index=0)

    q = data.copy()
    if pick_names:
        q = q[q["Name"].isin(pick_names)]
    if pick_event != "全部":
        q = q[q["EventName"]==pick_event]
    if pick_len != "全部":
        q = q[q["LengthMeters"].astype(str)==pick_len]

    if q.empty:
        st.info("没有符合条件的记录。")
        return

    q = q.sort_values("Seconds", na_position="last")
    show_cols = ["Name","EventName","Result","Rank","Note","Date","City","MeetName","PoolName","LengthMeters"]
    for c in show_cols:
        if c not in q.columns:
            q[c] = ""
    st.dataframe(q[show_cols], use_container_width=True, hide_index=True)

# =====================
# 主页面
# =====================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    tab1, tab2 = st.tabs(["赛事管理 / 成绩录入", "查询 / 对比"])
    with tab1:
        section_meta()
        st.markdown("---")
        section_results_entry_and_manage()
    with tab2:
        page_browse()

if __name__ == "__main__":
    main()
