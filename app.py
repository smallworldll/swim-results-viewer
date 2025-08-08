
import os, glob, io, zipfile, re, base64, requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="🏊‍♀️ 游泳成绩查询系统（赛事文件夹 + 表单 + GitHub提交）", layout="wide")

HELP = """
**赛事文件夹命名（建议）**：`meets/YYYY-MM-DD_City/`  
**meta.csv 列**（顺序随意，列名必须一致）：`Date, City, MeetName, PoolName, LengthMeters`  
**results.csv 列**：`Name, EventName, Result, Rank, Note`  
> 程序按**列名**读取，不依赖列顺序。`Result` 建议用 `mm:ss` 或 `mm:ss.ms`（如 `01:02.45`）。
"""

st.title("🏊‍♀️ 游泳成绩查询系统")
st.caption(HELP)

# ----------------- helpers -----------------
def parse_time_to_seconds(ts: str):
    """支持 mm:ss 或 mm:ss.xxx（毫秒/百分秒）"""
    if not isinstance(ts, str):
        return None
    ts = ts.strip()
    if not ts:
        return None
    try:
        if ":" in ts:
            m, s = ts.split(":", 1)   # 允许 s 部分带小数
            return int(m) * 60 + float(s)
        return float(ts)
    except Exception:
        return None

def load_all_meets(base_dir="meets"):
    rows = []
    for folder in sorted(glob.glob(os.path.join(base_dir, "*"))):
        meta_path = os.path.join(folder, "meta.csv")
        res_path  = os.path.join(folder, "results.csv")
        if os.path.isfile(meta_path) and os.path.isfile(res_path):
            try:
                meta = pd.read_csv(meta_path)
                res  = pd.read_csv(res_path)
                required_meta = ["Date","City","MeetName","PoolName","LengthMeters"]
                required_res  = ["Name","EventName","Result","Rank","Note"]
                if not all(col in meta.columns for col in required_meta):
                    st.warning(f"{folder} 的 meta.csv 列名不完整，请包含：{required_meta}")
                    continue
                if not all(col in res.columns for col in required_res):
                    st.warning(f"{folder} 的 results.csv 列名不完整，请包含：{required_res}")
                    continue
                meta = meta[required_meta].copy()
                res  = res[required_res].copy()
                merged = res.assign(**{col: meta.iloc[0][col] for col in required_meta})
                merged["MeetKey"] = os.path.basename(folder)
                rows.append(merged)
            except Exception as e:
                st.warning(f"读取 {folder} 出错：{e}")
                continue
    if rows:
        df = pd.concat(rows, ignore_index=True)
        df["LengthMeters"] = pd.to_numeric(df["LengthMeters"], errors="coerce").fillna(0).astype(int)
        df["Seconds"] = df["Result"].apply(parse_time_to_seconds)
        return df
    return pd.DataFrame(columns=["Name","EventName","Result","Rank","Note","Date","City","MeetName","PoolName","LengthMeters","MeetKey","Seconds"])

# ---- GitHub helpers ----
def have_github_secrets():
    needed = ["GITHUB_TOKEN", "REPO"]
    missing = [k for k in needed if k not in st.secrets]
    return len(missing) == 0, missing

def gh_put(path, content_bytes, message):
    token  = st.secrets["GITHUB_TOKEN"]
    repo   = st.secrets["REPO"]
    branch = st.secrets.get("BRANCH", "main")

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com/repos/{repo}/contents/{path}"

    # check existing file to get sha
    r = requests.get(url, headers=headers, params={"ref": branch})
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode(),
        "branch": branch
    }
    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=headers, json=payload, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"GitHub API 错误：{resp.status_code} {resp.text}")
    return resp.json()

def sanitize_folder_name(date_str, city):
    # Keep date as-is, sanitize city (letters, numbers, underscore/dash allowed)
    city_clean = re.sub(r"[^A-Za-z0-9_\-]", "", str(city).strip().replace(" ", ""))
    return f"{date_str}_{city_clean}"

# ----------------- tabs -----------------
tab1, tab2 = st.tabs(["📊 浏览/分析", "➕ 表单录入（生成赛事并可自动提交到 GitHub）"])

with tab1:
    data = load_all_meets()
    if data.empty:
        st.info("meets/ 目录下暂未发现有效赛事文件夹。请到“表单录入”标签页新建，或手动上传。")
    else:
        # Filters
        all_names = sorted(data["Name"].dropna().unique())
        sel_names = st.multiselect("Name（可多选）", all_names, default=["Anna"] if "Anna" in all_names else all_names)

        events = ["All"] + sorted(data["EventName"].dropna().unique())
        sel_event = st.selectbox("Event", events)

        lengths = ["All"] + sorted(data["LengthMeters"].unique().tolist())
        sel_len = st.selectbox("Length (Meters)", lengths)

        pools = ["All"] + sorted(data["PoolName"].dropna().unique())
        sel_pool = st.selectbox("Pool Name", pools)

        cities = ["All"] + sorted(data["City"].dropna().unique())
        sel_city = st.selectbox("City", cities)

        dates = ["All"] + sorted(data["Date"].dropna().unique())
        sel_date = st.selectbox("Date", dates)

        meets = ["All"] + sorted(data["MeetName"].dropna().unique())
        sel_meet = st.selectbox("Meet Name", meets)

        df = data.copy()
        if sel_names: df = df[df["Name"].isin(sel_names)]
        if sel_event != "All": df = df[df["EventName"] == sel_event]
        if sel_len   != "All": df = df[df["LengthMeters"] == sel_len]
        if sel_pool  != "All": df = df[df["PoolName"] == sel_pool]
        if sel_city  != "All": df = df[df["City"] == sel_city]
        if sel_date  != "All": df = df[df["Date"] == sel_date]
        if sel_meet  != "All": df = df[df["MeetName"] == sel_meet]

        st.subheader("🏅 比赛记录")
        st.dataframe(df)

        # Seed highlight logic
        if not df.empty:
            palette = ["red","blue","green","purple","orange","brown","teal","magenta","olive","navy"]
            name_order = sel_names if sel_names else all_names
            name_colors = {name: palette[i % len(palette)] for i, name in enumerate(name_order)}

            tmp = df.copy()
            group_cols = ["Name","LengthMeters"] if sel_event != "All" else ["Name","EventName","LengthMeters"]
            tmp["BestSeconds"] = tmp.groupby(group_cols)["Seconds"].transform("min")
            tmp["IsBest"] = tmp["Seconds"] == tmp["BestSeconds"]
            disp = tmp[["Name","EventName","Result","Rank","Note","PoolName","City","LengthMeters","Date"]]

            def style_row(row):
                styles = []
                is_best = tmp.loc[row.name, "IsBest"]
                for col in disp.columns:
                    if col == "Result" and is_best:
                        styles.append(f"color: {name_colors.get(row['Name'], 'black')}")
                    else:
                        styles.append("")
                return styles

            styled = disp.style.apply(style_row, axis=1).hide_index()
            st.markdown("**说明：每位选手在不同泳池长度（或不同项目×长度）下的最佳成绩已用该选手的固定颜色高亮在 Result 列。**")
            st.write(styled.to_html(escape=False), unsafe_allow_html=True)

            st.subheader("📈 成绩折线图（单位：秒）")
            chart_df = tmp.pivot_table(index="Date", columns="Name", values="Seconds", aggfunc="min")
            st.line_chart(chart_df)

        st.download_button("📥 下载当前筛选结果", df.to_csv(index=False).encode("utf-8-sig"), file_name="filtered_results.csv")

with tab2:
    st.subheader("步骤 1：填写赛事信息（meta.csv）")
    with st.form("meta_form", clear_on_submit=False):
        date_val = st.date_input("Date（日期）")
        city_val = st.text_input("City（城市）", placeholder="ChiangMai")
        meet_name = st.text_input("MeetName（赛事名称）", placeholder="Chiang Open")
        pool_name = st.text_input("PoolName（泳池名）", placeholder="Kawila")
        length_val = st.selectbox("LengthMeters（泳池长度）", [25, 50])
        meta_submitted = st.form_submit_button("保存赛事信息")
    if meta_submitted:
        st.session_state["meta"] = {
            "Date": str(date_val),
            "City": city_val.strip(),
            "MeetName": meet_name.strip(),
            "PoolName": pool_name.strip(),
            "LengthMeters": int(length_val),
        }
        st.success("✅ 已保存赛事信息")
        st.table(pd.DataFrame([st.session_state["meta"]]))

    st.markdown("---")
    st.subheader("步骤 2：录入成绩（results.csv）")
    default_events = [
        "25m Freestyle","25m Backstroke","25m Breaststroke","25m Butterfly",
        "50m Freestyle","50m Backstroke","50m Breaststroke","50m Butterfly",
        "100m Freestyle","100m Backstroke","100m Breaststroke","100m Butterfly",
        "200m Freestyle","200m Backstroke","200m Breaststroke","200m Butterfly",
        "400m Freestyle"
    ]

    if "result_rows" not in st.session_state:
        st.session_state["result_rows"] = 1
    cols = st.columns(3)
    if cols[0].button("➕ 添加一行"):
        st.session_state["result_rows"] += 1
    if cols[1].button("➖ 删除一行") and st.session_state["result_rows"] > 1:
        st.session_state["result_rows"] -= 1
    st.caption(f"当前行数：{st.session_state['result_rows']}")

    results_rows = []
    with st.form("results_form", clear_on_submit=False):
        for i in range(st.session_state["result_rows"]):
            st.markdown(f"**记录 {i+1}**")
            c1, c2, c3, c4, c5 = st.columns([1.2,1.4,1,0.8,1.2])
            name_i = c1.text_input(f"Name_{i}", key=f"name_{i}")
            ev_choice = c2.selectbox(f"Event_{i}", options=["（自定义）"] + default_events, key=f"event_{i}")
            custom_event = ""
            if ev_choice == "（自定义）":
                custom_event = c2.text_input(f"自定义项目_{i}", key=f"cust_event_{i}", placeholder="100m IM 等")
            result_i = c3.text_input(f"Result_{i}", key=f"result_{i}", placeholder="mm:ss 或 mm:ss.ms，比如 01:02.45")
            rank_i = c4.number_input(f"Rank_{i}", key=f"rank_{i}", min_value=0, step=1)
            note_i = c5.text_input(f"Note_{i}", key=f"note_{i}", placeholder="可留空")

            final_event = custom_event.strip() if ev_choice == "（自定义）" else ev_choice
            results_rows.append({
                "Name": name_i.strip(),
                "EventName": final_event,
                "Result": result_i.strip(),
                "Rank": int(rank_i) if rank_i is not None else 0,
                "Note": note_i.strip(),
            })

        auto_push = st.checkbox("✅ 提交到 GitHub（免下载上传）", value=False,
                                help="需要在 Streamlit Secrets 里配置 GITHUB_TOKEN、REPO（和可选的 BRANCH）。")
        also_write_local = st.checkbox("同时保存到本地 meets/ 目录（调试用）", value=False)
        gen = st.form_submit_button("生成/提交")

    if gen:
        if "meta" not in st.session_state:
            st.error("请先完成“步骤 1：填写赛事信息”。")
        else:
            meta = st.session_state["meta"]
            if not meta["Date"] or not meta["City"] or not meta["MeetName"]:
                st.error("日期 / 城市 / 赛事名称 不能为空")
                st.stop()
            clean_rows = [r for r in results_rows if r["Name"] and r["EventName"] and r["Result"]]
            if not clean_rows:
                st.error("请至少填写一条完整的成绩记录（Name、Event、Result 必填）。")
                st.stop()

            meta_df = pd.DataFrame([meta], columns=["Date","City","MeetName","PoolName","LengthMeters"])
            results_df = pd.DataFrame(clean_rows, columns=["Name","EventName","Result","Rank","Note"])

            folder_key = sanitize_folder_name(meta["Date"], meta["City"])
            # 本地保存（可选）
            if also_write_local:
                local_dir = os.path.join("meets", folder_key)
                os.makedirs(local_dir, exist_ok=True)
                meta_df.to_csv(os.path.join(local_dir, "meta.csv"), index=False, encoding="utf-8-sig")
                results_df.to_csv(os.path.join(local_dir, "results.csv"), index=False, encoding="utf-8-sig")
                st.success(f"已写入本地：{local_dir}/meta.csv, results.csv")

            # GitHub 提交或生成ZIP
            if auto_push:
                ok, missing = have_github_secrets()
                if not ok:
                    st.error(f"缺少 Secrets：{missing}。请在 Streamlit Cloud - App - Settings - Secrets 中配置。")
                else:
                    try:
                        folder = f"meets/{folder_key}"
                        r1 = gh_put(f"{folder}/meta.csv", meta_df.to_csv(index=False).encode("utf-8-sig"),
                                    f"Add meta for {folder_key}")
                        r2 = gh_put(f"{folder}/results.csv", results_df.to_csv(index=False).encode("utf-8-sig"),
                                    f"Add results for {folder_key}")
                        st.success("✅ 已提交到 GitHub！仓库收到新文件后会自动触发重部署，稍等片刻刷新即可看到最新数据。")
                        st.markdown(f"[查看 meta.csv]({r1.get('content',{}).get('html_url','')})  |  [查看 results.csv]({r2.get('content',{}).get('html_url','')})")
                    except Exception as e:
                        st.error(f"提交失败：{e}")
            else:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
                    z.writestr(f"meets/{folder_key}/meta.csv", meta_df.to_csv(index=False, encoding="utf-8-sig"))
                    z.writestr(f"meets/{folder_key}/results.csv", results_df.to_csv(index=False, encoding="utf-8-sig"))
                st.success(f"✅ 已生成赛事文件夹：meets/{folder_key}/ （内含 meta.csv 与 results.csv）")
                st.download_button("📦 下载 ZIP（手动上传到 GitHub）", data=zip_buffer.getvalue(),
                                   file_name=f"{folder_key}.zip", mime="application/zip")
