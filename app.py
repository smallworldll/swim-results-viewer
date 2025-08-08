# -*- coding: utf-8 -*-
import os
import io
import json
import base64
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd
import streamlit as st

# -----------------------------
# Helpers
# -----------------------------

MEETS_DIR = "meets"  # all data lives here
META_FILE = "meta.csv"
RESULTS_FILE = "results.csv"

def ensure_meet_folder(meet_folder: str):
    os.makedirs(meet_folder, exist_ok=True)

def load_meta(meet_folder: str) -> pd.DataFrame:
    path = os.path.join(meet_folder, META_FILE)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=["Date","City","MeetName","PoolName","LengthMeters"])

def load_results(meet_folder: str) -> pd.DataFrame:
    path = os.path.join(meet_folder, RESULTS_FILE)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # normalize columns in case of older files
        expected_cols = ["Name","EventName","Date","City","PoolName","LengthMeters","Result","Rank","Note"]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = None
        return df[expected_cols]
    else:
        return pd.DataFrame(columns=["Name","EventName","Date","City","PoolName","LengthMeters","Result","Rank","Note"])

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8")

def list_meet_folders() -> list:
    if not os.path.isdir(MEETS_DIR):
        return []
    subs = [d for d in os.listdir(MEETS_DIR) if os.path.isdir(os.path.join(MEETS_DIR, d))]
    subs.sort(reverse=True)
    return subs

# --- time parsing: accept "m:ss.xx" or "ss.xx" or "ss" ---
def normalize_time(s: str) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # If only seconds (with or without .xx)
    if ":" not in s:
        # seconds.xx or seconds
        try:
            sec = float(s)
            # turn into m:ss.xx
            m = int(sec // 60)
            r = sec - m*60
            return f"{m}:{r:05.2f}"
        except:
            return s  # leave as is
    # already m:ss.xx or m:ss
    try:
        m, rest = s.split(":", 1)
        m = int(m)
        sec = float(rest)
        return f"{m}:{sec:05.2f}"
    except:
        return s

# --- GitHub push ---
def push_to_github(path: str, message: str) -> Tuple[bool, str]:
    """
    Push a local file to GitHub repo using PAT in st.secrets.
    Secrets required:
      GITHUB_TOKEN, REPO, optional BRANCH (default 'main')
    """
    try:
        import requests
    except Exception as e:
        return False, f"缺少 requests：{e}"

    token = st.secrets.get("GITHUB_TOKEN")
    repo  = st.secrets.get("REPO")
    branch = st.secrets.get("BRANCH", "main")

    if not token or not repo:
        return False, "未配置 GITHUB_TOKEN / REPO（App → Settings → Secrets）"

    with open(path, "rb") as f:
        content = f.read()
    b64 = base64.b64encode(content).decode("utf-8")

    api = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    # get sha if file exists
    sha = None
    r = requests.get(api, headers=headers, params={"ref": branch})
    if r.status_code == 200:
        sha = r.json().get("sha")
    elif r.status_code not in (404, 200):
        return False, f"获取 SHA 失败：{r.status_code} {r.text}"

    payload = {
        "message": message,
        "content": b64,
        "branch": branch
    }
    if sha:
        payload["sha"] = sha

    r2 = requests.put(api, headers=headers, data=json.dumps(payload))
    if r2.status_code in (200,201):
        return True, "推送成功"
    else:
        return False, f"GitHub 推送失败： {r2.status_code} {r2.text}"

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="游泳成绩系统（赛事制）", layout="wide")
st.title("🏊‍♀️ 游泳成绩系统（赛事制）")

# ① 选择或新建赛事（文件夹）
with st.expander("① 选择或新建赛事", expanded=True):
    mode = st.radio("操作", ["选择已有赛事","新建赛事"], horizontal=True, index=0)
    meet_list = list_meet_folders()
    selected_folder = None

    if mode == "选择已有赛事":
        if not meet_list:
            st.info("还没有任何赛事文件夹，请先新建。")
        else:
            selected_folder = st.selectbox("选择赛事（文件夹）", meet_list, index=0)
    else:
        c1, c2 = st.columns(2)
        with c1:
            date = st.date_input("日期", value=datetime.today())
        with c2:
            city = st.text_input("城市", value="Chiang Mai")
        new_folder_name = f"{date.strftime('%Y-%m-%d')}_{city.replace(' ', '')}"
        st.write(f"即将创建文件夹：`{MEETS_DIR}/{new_folder_name}`")
        if st.button("创建赛事文件夹"):
            ensure_meet_folder(os.path.join(MEETS_DIR, new_folder_name))
            # 写入一个空 meta.csv 作为示例
            meta_df = pd.DataFrame([{
                "Date": date.strftime("%Y-%m-%d"),
                "City": city,
                "MeetName": "",
                "PoolName": "",
                "LengthMeters": ""
            }])
            save_csv(meta_df, os.path.join(MEETS_DIR, new_folder_name, META_FILE))
            st.success("已创建。请在下方完善 meta 并保存。")
            selected_folder = new_folder_name

    if selected_folder:
        meet_path = os.path.join(MEETS_DIR, selected_folder)
        ensure_meet_folder(meet_path)
        meta_df = load_meta(meet_path)

        st.write("当前赛事 meta：")
        meta_ed = st.data_editor(meta_df, num_rows="dynamic", key="meta_ed")
        btns = st.columns([1,1,6])
        with btns[0]:
            if st.button("💾 保存 meta.csv"):
                save_csv(meta_ed, os.path.join(meet_path, META_FILE))
                st.success("已保存 meta.csv")

        # ② 历史成绩管理（新增：选中已有赛事后实时显示/编辑/删除）
        st.divider()
        st.subheader("② 历史成绩管理")
        results_df = load_results(meet_path)

        # 过滤器
        c1,c2,c3 = st.columns([2,2,1])
        with c1:
            events = ["All"] + sorted([e for e in results_df["EventName"].dropna().unique().tolist()])
            f_event = st.selectbox("按 EventName 过滤", events, index=0)
        with c2:
            f_name = st.text_input("按 Name 搜索（包含匹配）", value="")
        with c3:
            edit_mode = st.toggle("编辑模式", value=False)

        df_view = results_df.copy()
        if f_event != "All":
            df_view = df_view[df_view["EventName"]==f_event]
        if f_name.strip():
            df_view = df_view[df_view["Name"].astype(str).str.contains(f_name.strip(), case=False, na=False)]

        st.caption("提示：Result 支持 `m:ss.xx` 以及 `ss.xx`（会自动转成 `m:ss.xx`）")
        if edit_mode:
            # 允许编辑
            edited = st.data_editor(
                df_view,
                key="results_editor",
                num_rows="dynamic",
                use_container_width=True
            )
            col_save, col_del, _ = st.columns([1,1,6])
            with col_save:
                if st.button("💾 保存结果"):
                    # 写回原数据（只替换被筛选到的 index 部分）
                    # 规范化 Result
                    edited = edited.copy()
                    edited["Result"] = edited["Result"].apply(normalize_time)
                    # 将编辑内容覆盖原 results_df 对应行，未在筛选内的行保持不变
                    results_df.loc[edited.index, results_df.columns] = edited[results_df.columns]
                    # 保存
                    save_csv(results_df, os.path.join(meet_path, RESULTS_FILE))
                    st.success("已保存 results.csv")
                    # 可选推送 GitHub
                    if st.checkbox("同时推送到 GitHub（免下载上传）", value=False, key="push_after_save"):
                        ok, msg = push_to_github(os.path.join(meet_path, RESULTS_FILE), f"update results {selected_folder}")
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(msg)
            with col_del:
                # 选择删除：用多选行号
                to_delete = st.multiselect("选择要删除的行号", edited.index.tolist(), key="rows_to_delete")
                if st.button("🗑️ 删除选中行"):
                    if to_delete:
                        results_df = results_df.drop(index=to_delete)
                        save_csv(results_df, os.path.join(meet_path, RESULTS_FILE))
                        st.success(f"已删除 {len(to_delete)} 行，并保存。")
                        if st.checkbox("删除后推送到 GitHub", value=False, key="push_after_delete"):
                            ok, msg = push_to_github(os.path.join(meet_path, RESULTS_FILE), f"delete rows {selected_folder}")
                            if ok:
                                st.success(msg)
                            else:
                                st.warning(msg)
                    else:
                        st.info("未选择任何行。")
        else:
            st.dataframe(df_view, use_container_width=True)

        st.markdown("— End —")

else:
    st.info("请先在上面选择或新建一个赛事。")
