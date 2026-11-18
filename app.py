"""
app.py：精油生理數據整合平台（第一版）

執行方式（在終端機裡，先 cd 到這個資料夾）：
    streamlit run app.py
瀏覽器會自動打開一個網頁，就是這個平台。

這個檔案負責「網頁長什麼樣子」和「畫圖、算統計」。
讀檔和清理的細節在 loaders.py。
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import loaders

HERE = Path(__file__).parent
SAMPLE_WATCH = HERE / "sample_data" / "watch_blood_pressure.csv"
SAMPLE_EEG = HERE / "sample_data" / "eeg_summary.xlsx"

# 圖表顏色：固定順序，第 1 種一定是藍、第 2 種一定是橘，不會換來換去。
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#b8b7b2"

WATCH_METRICS = {
    "sys": "收縮壓 (mmHg)",
    "dia": "舒張壓 (mmHg)",
    "hr": "心跳 (次/分)",
    "rmssd": "RMSSD (ms，越高通常越放鬆)",
    "stress": "壓力指數",
}

st.set_page_config(page_title="香當有感 ScentiQ", layout="wide")
st.title("香當有感 ScentiQ")
st.caption("整合 ASUS Watch 血壓資料與腦電波資料，比較使用精油前後的變化。")


# ---------------------------------------------------------------------------
# 小工具：成對比較（同一場實驗的「前」跟「後」比）
# ---------------------------------------------------------------------------
def paired_compare(before: pd.Series, after: pd.Series) -> dict:
    """
    before、after 是同樣順序的兩串數字（第 i 個 before 跟第 i 個 after 來自同一場/同一天）。
    回傳：有幾組、平均變化多少、兩種檢定的 p 值。
    """
    pair = pd.DataFrame({"b": before.values, "a": after.values}).dropna()
    diff = pair["a"] - pair["b"]
    result = {"組數 n": len(pair), "前 平均": pair["b"].mean(), "後 平均": pair["a"].mean(),
              "平均變化(後-前)": diff.mean(), "t 檢定 p 值": None, "Wilcoxon p 值": None}
    if len(pair) >= 3:
        result["t 檢定 p 值"] = stats.ttest_rel(pair["a"], pair["b"]).pvalue
        if (diff != 0).any():
            result["Wilcoxon p 值"] = stats.wilcoxon(pair["a"], pair["b"]).pvalue
    return result


def explain_p(p) -> str:
    """把 p 值翻成白話。"""
    if p is None or pd.isna(p):
        return "資料太少（至少要 3 組），還不能做檢定"
    if p < 0.05:
        return "差異「不太像只是巧合」（p < 0.05）"
    return "目前看不出確定的差異（可能真的沒差，也可能資料還不夠多）"


def style_fig(fig, y_title, x_title=""):
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#fcfcfb",
        yaxis_title=y_title, xaxis_title=x_title, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#ececea")
    return fig


# ---------------------------------------------------------------------------
# 左側欄：上傳檔案和設定
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. 上傳資料")
    watch_files = st.file_uploader("ASUS Watch 血壓 CSV（可一次選多個）", type="csv", accept_multiple_files=True)
    eeg_file = st.file_uploader("腦電波整理 Excel (.xlsx)", type="xlsx")
    st.caption("沒有上傳的話，會先用範例資料示範。")

    st.header("2. 顯示更多")
    show_advanced = st.toggle("顯示進階分析（後台）", value=False,
                              help="平常只看總結就好。要檢查細節、看每場實驗或做統計時再打開。")

    # 清理設定收在折疊區塊裡：大部分時候用預設值就好，不用動。
    with st.expander("清理設定（通常不用改）"):
        drop_invalid = st.checkbox("移除「壓力指數空白 / RMSSD=0」的量測", value=True,
                                   help="這些通常是量測失敗。取消勾選可以看看保留它們會怎樣。")
        gap = st.number_input("手動量測間隔超過幾分鐘，算新的一場實驗", 10, 240, 30, step=5)
        n_pre = st.number_input("每場實驗的前幾筆是「擦精油前」", 1, 5, 1,
                                help="預設第 1 筆是擦精油前的基準值，後面都是擦精油後。")


# ---------------------------------------------------------------------------
# 讀資料
# ---------------------------------------------------------------------------
if watch_files:
    raw_watch = pd.concat([loaders.load_watch_csv(f) for f in watch_files]).drop_duplicates()
    raw_watch = raw_watch.sort_values("time").reset_index(drop=True)
else:
    raw_watch = loaders.load_watch_csv(SAMPLE_WATCH)
    st.info("目前顯示的是範例手錶資料。在左邊上傳你自己的 CSV 就會換成你的資料。")

watch, removed = loaders.clean_watch(raw_watch, drop_invalid_hrv=drop_invalid)
sessions = loaders.find_sessions(raw_watch, watch, gap_minutes=int(gap))

try:
    eeg = loaders.load_eeg_excel(eeg_file if eeg_file else SAMPLE_EEG)
except ValueError as e:
    eeg = pd.DataFrame()
    st.error(f"腦電波檔案讀不懂：{e}")

# ---------------------------------------------------------------------------
# 總結頁（前台）：一眼看出「擦精油前 vs 擦精油後」變化多少
# ---------------------------------------------------------------------------
# 想法：評審或一般使用者不需要看五個分頁，只想知道「有沒有變、變多少」。
# 所以這裡只挑 4 個最好懂的指標，每個指標一張卡片：擦前平均 → 擦後平均。
# 完整的圖和統計表都還在下面的「進階分析」，沒有刪掉。

def before_after_watch(metric: str):
    """手錶：每場實驗先算「擦前平均」和「擦後平均」，再把所有場次放在一起比。"""
    if sessions.empty:
        return None
    s = sessions.assign(階段=sessions["order"].apply(lambda k: "擦前" if k < n_pre else "擦後"))
    per = s.pivot_table(index="session", columns="階段", values=metric)
    if "擦前" not in per or "擦後" not in per:
        return None
    return paired_compare(per["擦前"], per["擦後"])


def before_after_eeg(metric: str):
    """腦波：每天的 pre 跟「擦後所有時間點的平均」比。"""
    if eeg.empty or metric not in eeg:
        return None
    wide = eeg.pivot_table(index="date", columns="stage", values=metric, observed=True)
    if "pre" not in wide:
        return None
    after = wide.drop(columns="pre").mean(axis=1)
    return paired_compare(wide["pre"], after)


def summary_card(col, title, unit, r, good_when_down, meaning):
    """畫一張卡片。good_when_down=True 代表數字變低是「比較放鬆」的方向。"""
    with col.container(border=True):
        st.markdown(f"**{title}**")
        if r is None or r["組數 n"] == 0:
            st.write("還沒有資料")
            return
        change = r["平均變化(後-前)"]
        # delta_color="inverse"：數字變低時顯示綠色（血壓變低是好事）
        st.metric("擦精油後", f"{r['後 平均']:.1f} {unit}", f"{change:+.1f} {unit}",
                  delta_color="inverse" if good_when_down else "normal")
        st.caption(f"擦精油前：{r['前 平均']:.1f} {unit}　｜　{meaning}")
        p = r["t 檢定 p 值"]
        if p is None or pd.isna(p):
            st.caption(f"共 {r['組數 n']} 組，資料太少，還不能判斷")
        elif p < 0.05:
            st.caption(f"共 {r['組數 n']} 組，✅ 這個變化不太像只是巧合")
        else:
            st.caption(f"共 {r['組數 n']} 組，⚪ 目前還看不出確定的差異")


st.header("結論：擦精油前後，身體有什麼變化？")
cards = st.columns(4)
summary_card(cards[0], "收縮壓（血壓）", "mmHg", before_after_watch("sys"), True, "變低 = 血壓下降")
summary_card(cards[1], "心跳", "次/分", before_after_watch("hr"), True, "變低 = 心跳變慢")
summary_card(cards[2], "HRV 心率變異度（RMSSD）", "ms", before_after_watch("rmssd"), False, "變高 = 通常較放鬆")
summary_card(cards[3], "腦波放鬆度（Meditation）", "分", before_after_eeg("Meditation"), False, "變高 = 較放鬆")
st.caption("綠色箭頭 = 往「放鬆」的方向變。手錶數據以每場手動量測實驗為一組，腦波以每天為一組。"
           "本平台只整理與呈現數據，不用來判斷療效或診斷疾病。")

if not show_advanced:
    st.info("想看每場實驗的圖表和完整統計？在左邊打開「顯示進階分析（後台）」。")
    st.stop()   # 停在這裡，下面的進階分頁就不會顯示

st.divider()
st.header("進階分析（後台）")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① 資料總覽與清理", "② 手錶：每場擦前 vs 擦後", "③ 手錶：時段比較（如睡眠）", "④ 腦電波：擦前 vs 擦後", "⑤ 同一天合併比對"]
)

# ---------------------------------------------------------------------------
# ① 資料總覽
# ---------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("手錶原始筆數", len(raw_watch))
    c2.metric("清理後筆數", len(watch))
    c3.metric("被移除", len(removed))
    c4.metric("找到幾場手動量測實驗", sessions["session"].nunique() if len(sessions) else 0)

    st.subheader("血壓隨時間變化")
    fig = go.Figure()
    fig.add_scatter(x=watch["time"], y=watch["sys"], name="收縮壓", mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=5))
    fig.add_scatter(x=watch["time"], y=watch["dia"], name="舒張壓", mode="lines+markers",
                    line=dict(color=ORANGE, width=2), marker=dict(size=5))
    st.plotly_chart(style_fig(fig, "mmHg"), width="stretch")

    with st.expander(f"看被移除的 {len(removed)} 筆資料和原因"):
        st.dataframe(removed[["time", "type_name", "sys", "dia", "hr", "stress", "rmssd", "reason"]],
                     width="stretch")
    with st.expander("看清理後的全部手錶資料"):
        st.dataframe(watch, width="stretch")
    if not eeg.empty:
        with st.expander(f"看讀到的腦電波資料（{eeg['date'].nunique()} 天）"):
            st.dataframe(eeg, width="stretch")

# ---------------------------------------------------------------------------
# ② 每場手動量測實驗：擦精油前 vs 擦精油後
# ---------------------------------------------------------------------------
with tab2:
    if sessions.empty:
        st.warning("沒有找到手動量測（Type=1）的資料。")
    else:
        st.markdown(
            f"平台把**手動量測**按時間分成一場一場的實驗。每場的前 **{n_pre}** 筆當作「擦精油前」，"
            "其餘當作「擦精油後」。請在下表**勾選哪幾場有用精油**（沒用精油的場次可以當對照組）。"
        )
        summary = sessions.groupby("session").agg(
            日期=("time", lambda t: t.min().strftime("%Y-%m-%d")),
            開始時間=("time", lambda t: t.min().strftime("%H:%M")),
            筆數=("time", "size"),
        ).reset_index().rename(columns={"session": "場次"})
        summary["有用精油"] = True
        summary["備註"] = ""
        edited = st.data_editor(summary, disabled=["場次", "日期", "開始時間", "筆數"],
                                hide_index=True, key="session_editor", width="stretch")
        oil_map = dict(zip(edited["場次"], edited["有用精油"]))
        sessions["有用精油"] = sessions["session"].map(oil_map)
        sessions["階段"] = sessions["order"].apply(lambda k: "擦前" if k < n_pre else "擦後")

        metric = st.selectbox("要看哪個指標", list(WATCH_METRICS), format_func=WATCH_METRICS.get, key="m2")

        # 圖：x 軸 = 這場的第幾筆；每場一條細線，藍 = 有精油，橘 = 沒精油
        fig = go.Figure()
        for sid, g in sessions.groupby("session"):
            oil = bool(g["有用精油"].iloc[0])
            fig.add_scatter(x=g["order"] + 1, y=g[metric], mode="lines+markers",
                            name="有用精油" if oil else "沒用精油", legendgroup=str(oil),
                            showlegend=bool(sid == sessions[sessions["有用精油"] == oil]["session"].min()),
                            line=dict(color=BLUE if oil else ORANGE, width=1.5), opacity=0.55,
                            marker=dict(size=7), hovertemplate=f"第{sid}場 %{{y}}<extra></extra>")
        fig.add_vline(x=n_pre + 0.5, line_dash="dot", line_color="#52514e",
                      annotation_text="← 擦前 ｜ 擦後 →", annotation_position="top")
        st.plotly_chart(style_fig(fig, WATCH_METRICS[metric], "這場的第幾筆量測"), width="stretch")

        # 統計：每場算一個「擦前平均」和「擦後平均」，再做成對比較
        per_session = sessions.pivot_table(index=["session", "有用精油"], columns="階段", values=list(WATCH_METRICS))
        st.subheader("統計結果")
        rows = []
        for oil_flag, label in [(True, "有用精油的場次"), (False, "沒用精油的場次")]:
            part = per_session.xs(oil_flag, level="有用精油") if oil_flag in per_session.index.get_level_values(1) else None
            if part is None:
                continue
            for m in WATCH_METRICS:
                if (m, "擦前") in part and (m, "擦後") in part:
                    r = paired_compare(part[(m, "擦前")], part[(m, "擦後")])
                    rows.append({"組別": label, "指標": WATCH_METRICS[m], **r, "白話解讀": explain_p(r["t 檢定 p 值"])})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        st.caption("「組數 n」= 有擦前也有擦後資料的場次數。平均變化是負的，代表擦後數值變低。")

# ---------------------------------------------------------------------------
# ③ 時段比較：例如「有擦精油睡覺的晚上」vs「沒擦的晚上」
# ---------------------------------------------------------------------------
with tab3:
    st.markdown(
        "在下表填入**有用精油的時段**（例如某天晚上 22:00 到隔天 07:00）。"
        "平台會把落在這些時段的量測標成「有用精油」，其他標成「未標記」，然後比較兩組。"
        "\n\n點表格最下面的空白列就能新增一列。標籤可以自己取名，例如「薰衣草」「對照」。"
    )
    empty_periods = pd.DataFrame({"開始": pd.Series(dtype="datetime64[ns]"),
                                  "結束": pd.Series(dtype="datetime64[ns]"),
                                  "標籤": pd.Series(dtype="str")})
    periods = st.data_editor(
        empty_periods, num_rows="dynamic", key="period_editor", width="stretch",
        column_config={"開始": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                       "結束": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                       "標籤": st.column_config.TextColumn(default="有用精油")},
    )
    c1, c2 = st.columns(2)
    only_auto = c1.checkbox("只用手錶自動量測（Type 129）", value=True,
                            help="手動量測多半是白天的實驗，比較睡眠時通常只看自動量測。")
    night = c2.checkbox("只看睡覺時段（22:00–07:00）", value=False)

    data = watch[watch["type"] == 129] if only_auto else watch
    if night:
        h = data["time"].dt.hour
        data = data[(h >= 22) | (h < 7)]
    data = data.assign(組別=loaders.label_by_periods(data, periods))

    metric3 = st.selectbox("要看哪個指標", list(WATCH_METRICS), format_func=WATCH_METRICS.get, key="m3")
    groups = [g for g in data["組別"].unique()]
    fig = go.Figure()
    for i, g in enumerate(sorted(groups, key=lambda x: x == "未標記")):
        color = GRAY if g == "未標記" else [BLUE, AQUA, ORANGE][i % 3]
        fig.add_box(y=data.loc[data["組別"] == g, metric3], name=g, boxpoints="all", jitter=0.4,
                    marker=dict(color=color, size=6), line=dict(color=color, width=2))
    st.plotly_chart(style_fig(fig, WATCH_METRICS[metric3]), width="stretch")

    table = data.groupby("組別")[list(WATCH_METRICS)].agg(["count", "mean"]).round(1)
    st.dataframe(table, width="stretch")
    labelled = [g for g in groups if g != "未標記"]
    if labelled and "未標記" in groups:
        a = data.loc[data["組別"] == labelled[0], metric3].dropna()
        b = data.loc[data["組別"] == "未標記", metric3].dropna()
        if len(a) >= 3 and len(b) >= 3:
            p = stats.mannwhitneyu(a, b).pvalue
            st.write(f"**{labelled[0]}** vs **未標記**：平均差 {a.mean() - b.mean():+.1f}，"
                     f"Mann-Whitney p = {p:.3f} → {explain_p(p)}")
    elif not labelled:
        st.info("還沒填任何時段，所以全部都是「未標記」。")

# ---------------------------------------------------------------------------
# ④ 腦電波：擦前 vs 擦後各時間點
# ---------------------------------------------------------------------------
with tab4:
    if eeg.empty:
        st.warning("沒有腦電波資料。")
    else:
        metric_cols = [c for c in eeg.columns if c not in ("date", "stage")]
        m4 = st.selectbox("要看哪個腦波指標", metric_cols,
                          index=metric_cols.index("Meditation") if "Meditation" in metric_cols else 0)
        fig = go.Figure()
        for d, g in eeg.groupby("date"):
            fig.add_scatter(x=g["stage"].astype(str), y=g[m4], mode="lines", line=dict(color=GRAY, width=1),
                            name=d.strftime("%m/%d"), showlegend=False,
                            hovertemplate=d.strftime("%m/%d") + "：%{y:.1f}<extra></extra>")
        avg = eeg.groupby("stage", observed=True)[m4].agg(["mean", "std"]).reset_index()
        fig.add_scatter(x=avg["stage"].astype(str), y=avg["mean"], mode="lines+markers", name="每天平均（±標準差）",
                        error_y=dict(type="data", array=avg["std"], color=BLUE, thickness=1.5),
                        line=dict(color=BLUE, width=3), marker=dict(size=9))
        st.plotly_chart(style_fig(fig, m4, "階段（pre = 擦精油前）"), width="stretch")
        st.caption("灰線 = 每一天；藍線 = 所有天的平均。")

        st.subheader("每個時間點跟「擦前」比")
        wide = eeg.pivot_table(index="date", columns="stage", values=m4, observed=True)
        rows = []
        for stg in wide.columns:
            if stg == "pre":
                continue
            r = paired_compare(wide["pre"], wide[stg])
            rows.append({"比較": f"pre → {stg}", **r, "白話解讀": explain_p(r["t 檢定 p 值"])})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        st.caption("Meditation（放鬆度）上升、Pressure 下降，通常解讀為比較放鬆。"
                   "注意：範例中 Pressure 剛好等於 100 − Meditation，兩者其實是同一件事。")

# ---------------------------------------------------------------------------
# ⑤ 同一天的手錶 + 腦電波合併
# ---------------------------------------------------------------------------
with tab5:
    st.markdown(
        "這裡把**同一天**的手錶實驗和腦電波放在一起。對齊方式（假設）："
        "手錶那場的第 1 筆 ↔ 腦波 pre，第 2 筆 ↔ 0 min，第 3 筆 ↔ 5 min…依此類推。"
    )
    if sessions.empty or eeg.empty:
        st.warning("需要手錶手動量測和腦電波兩種資料都有。")
    else:
        s = sessions.copy()
        s["date"] = s["time"].dt.normalize()
        s = s[s["session"] == s.groupby("date")["session"].transform("min")]   # 一天只取第一場
        s = s[s["order"] < len(loaders.STAGE_ORDER)]
        s["stage"] = s["order"].map(dict(enumerate(loaders.STAGE_ORDER)))
        e = eeg.copy()
        e["stage"] = e["stage"].astype(str)
        merged = s.merge(e, on=["date", "stage"], how="inner")
        watch_days = set(s["date"].dt.date)
        eeg_days = set(eeg["date"].dt.date)
        st.write(f"手錶實驗日期：{sorted(map(str, watch_days))}")
        st.write(f"腦電波日期：{sorted(map(str, eeg_days))}")
        if merged.empty:
            st.warning("兩份資料沒有重疊的日期，所以還無法合併。")
        else:
            st.success(f"有 {merged['date'].nunique()} 天兩邊都有資料。")
            show = ["date", "stage", "time", "sys", "dia", "hr", "rmssd"] + \
                   [c for c in ("Attention", "Meditation", "Pressure") if c in merged]
            st.dataframe(merged[show], hide_index=True, width="stretch")
            if merged["date"].nunique() < 3:
                st.info("重疊的天數少於 3 天，先只列表格；累積更多天後才適合算相關性。")
            else:
                x_col = st.selectbox("手錶指標", list(WATCH_METRICS), format_func=WATCH_METRICS.get, key="m5x")
                y_col = st.selectbox("腦波指標", [c for c in e.columns if c not in ("date", "stage")], key="m5y")
                r, p = stats.spearmanr(merged[x_col], merged[y_col], nan_policy="omit")
                fig = go.Figure(go.Scatter(x=merged[x_col], y=merged[y_col], mode="markers",
                                           marker=dict(color=BLUE, size=9)))
                st.plotly_chart(style_fig(fig, y_col, WATCH_METRICS[x_col]), width="stretch")
                st.write(f"Spearman 相關係數 = {r:.2f}，p = {p:.3f} → {explain_p(p)}")
