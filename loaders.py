"""
loaders.py：負責「讀檔」和「清理資料」。

這個檔案不畫圖、不做網頁，只做一件事：
把 ASUS Watch 的 CSV 和腦電波的 Excel，變成整齊的表格（pandas DataFrame）。

小知識：DataFrame 就像 Excel 的一張工作表，有欄（column）有列（row）。
"""

import re
import datetime as dt

import pandas as pd
import openpyxl


# ---------------------------------------------------------------------------
# 第一部分：ASUS Watch 血壓 CSV
# ---------------------------------------------------------------------------

# CSV 的欄位名稱長這樣：「收縮壓 (Sys)」。我們取括號裡的英文，換成簡短好打的名字。
WATCH_COLUMN_NAMES = {
    "Time": "time",            # 量測時間
    "DeviceId": "device",      # 手錶編號
    "Type": "type",            # 量測類型：129 = 手錶自動量（約每小時一次），1 = 手動量
    "Sys": "sys",              # 收縮壓（mmHg）
    "Dia": "dia",              # 舒張壓（mmHg）
    "HR": "hr",                # 心跳（次/分）
    "deStressIndex": "stress",  # 手錶算的壓力指數
    "RMSSD": "rmssd",          # 心率變異度（越高通常代表越放鬆）
}

TYPE_NAMES = {129: "自動量測（約每小時）", 1: "手動量測"}


def load_watch_csv(file) -> pd.DataFrame:
    """讀一個 ASUS Watch 血壓 CSV，回傳整理好欄位名稱的表格。"""
    # encoding="utf-8-sig"：這個 CSV 開頭有一個看不見的 BOM 字元，用這個設定才不會讀錯。
    df = pd.read_csv(file, encoding="utf-8-sig")

    new_names = {}
    for col in df.columns:
        found = re.search(r"\(([^)]+)\)", col)          # 找括號裡的字，例如 "Sys"
        key = found.group(1).strip() if found else col.strip()
        new_names[col] = WATCH_COLUMN_NAMES.get(key, key)
    df = df.rename(columns=new_names)

    df["time"] = pd.to_datetime(df["time"])              # 文字 → 真正的時間
    for col in ["type", "sys", "dia", "hr", "stress", "rmssd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # 空白會變成 NaN（代表「沒有值」）
    df["type_name"] = df["type"].map(TYPE_NAMES).fillna("其他")
    return df.sort_values("time").reset_index(drop=True)


def clean_watch(df: pd.DataFrame, drop_invalid_hrv: bool = True):
    """
    把不可靠的量測挑出來。回傳 (乾淨的資料, 被移除的資料)。
    被移除的資料會多一欄 reason，寫明為什麼被移除，讓你可以自己檢查。
    """
    reasons = pd.Series("", index=df.index)

    if drop_invalid_hrv:
        # 範例檔裡有些列：壓力指數是空白、RMSSD 是 0。
        # RMSSD 是 0 在真人身上幾乎不可能，通常代表這次量測失敗（例如手沒放好）。
        bad = df["stress"].isna() | (df["rmssd"] <= 0)
        reasons[bad] += "壓力指數空白或 RMSSD=0（量測可能失敗）；"

    # 生理上不合理的數字（例如收縮壓 300）也剔除。
    reasons[(df["sys"] < 70) | (df["sys"] > 220)] += "收縮壓超出 70–220；"
    reasons[(df["dia"] < 30) | (df["dia"] > 140)] += "舒張壓超出 30–140；"
    reasons[(df["hr"] < 30) | (df["hr"] > 220)] += "心跳超出 30–220；"

    removed = df[reasons != ""].copy()
    removed["reason"] = reasons[reasons != ""]
    clean = df[reasons == ""].copy()
    return clean.reset_index(drop=True), removed.reset_index(drop=True)


def find_sessions(raw: pd.DataFrame, clean: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """
    找出「一次實驗」。
    觀察範例檔：手動量測（Type=1）總是一群一群出現，每群約 6 筆、每筆間隔約 10 分鐘，
    很像「擦精油前量一次，擦完後再量好幾次」。
    規則：手動量測之間如果隔超過 gap_minutes 分鐘，就算新的一場。

    為什麼要同時給 raw（原始）和 clean（清理後）？
    分場次要用「原始」資料，因為失敗的量測也代表「那時候有人在量」，
    如果先刪掉它們，時間空隙會變大，一場實驗可能被誤切成兩場。
    分好場次之後，才只留下清理後的那些筆，再重新數「第幾筆」。

    回傳的表格多了：session（第幾場）、order（這場的第幾筆有效量測，從 0 開始）、
    minutes（距離這場第一筆有效量測幾分鐘）。
    """
    m = raw[raw["type"] == 1].sort_values("time").copy()
    if m.empty:
        return m.assign(session=[], order=[], minutes=[])
    new_session = m["time"].diff() > pd.Timedelta(minutes=gap_minutes)
    m["session"] = new_session.cumsum() + 1
    m = m[m["time"].isin(clean.loc[clean["type"] == 1, "time"])]   # 只留有效量測
    m["session"] = m["session"].rank(method="dense").astype(int)  # 場次重新編號成 1, 2, 3...
    m["order"] = m.groupby("session").cumcount()
    first_time = m.groupby("session")["time"].transform("min")
    m["minutes"] = (m["time"] - first_time).dt.total_seconds() / 60
    return m.reset_index(drop=True)


def label_by_periods(df: pd.DataFrame, periods: pd.DataFrame) -> pd.Series:
    """
    periods 是你在網頁上填的表格，每一列有「開始」「結束」「標籤」。
    這個函式幫每一筆量測找到它落在哪個時段，回傳對應的標籤；都沒落到就是「未標記」。
    """
    labels = pd.Series("未標記", index=df.index)
    for _, p in periods.dropna(subset=["開始", "結束"]).iterrows():
        inside = (df["time"] >= pd.Timestamp(p["開始"])) & (df["time"] <= pd.Timestamp(p["結束"]))
        labels[inside] = p.get("標籤") or "有用精油"
    return labels


# ---------------------------------------------------------------------------
# 第二部分：腦電波（EEG）整理好的 Excel
# ---------------------------------------------------------------------------

STAGE_ORDER = ["pre", "0 min", "5 min", "10 min", "15 min", "20 min"]


def _normalize_stage(text: str):
    """把 'pre'、'post-0 min'、'Pre'、'0 min' 這些寫法統一成 STAGE_ORDER 裡的樣子。"""
    t = str(text).strip().lower()
    if t == "pre":
        return "pre"
    found = re.search(r"(\d+)\s*min", t)
    if found:
        return f"{int(found.group(1))} min"
    return None


def load_eeg_excel(file) -> pd.DataFrame:
    """
    讀「數據整理」那種格式的 Excel（像範例檔的「工作表3」）：
        一個日期
        一列標題：Attention, Meditation, Pressure, Delta, ...
        幾列資料：pre, post-0 min, post-5 min, ...
        下一個日期 ... 一直重複

    回傳「長格式」表格，每一列 = 某一天、某個階段、所有腦波指標。
    """
    wb = openpyxl.load_workbook(file, data_only=True)  # data_only=True：讀公式算出來的值
    best = pd.DataFrame()
    for ws in wb.worksheets:                            # 每張工作表都試試看，取讀到最多資料的那張
        rows = []
        current_date, header = None, None
        for row in ws.iter_rows(values_only=True):
            cells = list(row)
            dates = [c for c in cells if isinstance(c, (dt.datetime, dt.date))]
            if dates:                                    # 這列有日期 → 換到新的一天
                current_date, header = pd.Timestamp(dates[0]).normalize(), None
                continue
            if "Attention" in cells:                     # 這列是標題
                header = {i: str(c).strip() for i, c in enumerate(cells) if isinstance(c, str)}
                continue
            if current_date is None or header is None:
                continue
            stage = next((_normalize_stage(c) for c in cells if isinstance(c, str) and _normalize_stage(c)), None)
            if stage is None:
                continue
            record = {"date": current_date, "stage": stage}
            for i, name in header.items():
                if i < len(cells) and isinstance(cells[i], (int, float)):
                    record[name] = float(cells[i])
            if len(record) > 2:
                rows.append(record)
        sheet_df = pd.DataFrame(rows)
        if len(sheet_df) > len(best):
            best = sheet_df

    if best.empty:
        raise ValueError("在這個 Excel 裡找不到「日期 → 標題列(含 Attention) → pre/post 資料」的格式。")
    best["stage"] = pd.Categorical(best["stage"], categories=STAGE_ORDER, ordered=True)
    return best.sort_values(["date", "stage"]).reset_index(drop=True)
