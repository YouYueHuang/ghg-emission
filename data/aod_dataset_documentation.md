# GK-2 Satellite Aerosol Optical Depth (AOD) Dataset

本資料集包含由 **GEO-KOMPSAT-2 (GK-2)** 靜止氣象衛星反演之台灣周邊區域氣膠光學厚度（AOD）網格資料。本文件旨在說明 AOD 之大氣物理意義、資料集結構維度、已知之座標軸異常狀況、數值統計特徵以及建議的資料處理與載入方式。

---

## 1. 大氣科學背景與物理意義 (AOD Background)

**AOD (Aerosol Optical Depth，氣膠光學厚度)** 是大氣科學與衛星遙測中用以衡量大氣中懸浮微粒（如 PM2.5、沙塵、海鹽、硫酸鹽、煙霧等）總量與大氣透明度最重要的無因次（Dimensionless）物理量。

* **物理定義**：代表太陽輻射（或特定波長電磁波）從大氣頂端垂直穿透至地表過程中，因大氣氣膠粒子之**散射（Scattering）與吸收（Absorption）**作用導致光強減弱的總累計程度。
* **數值範圍指標**：
  * **$AOD \approx 0.01 \sim 0.05$**：大氣極度乾淨與透明（如高山或純淨海洋）。
  * **$AOD \approx 0.2 \sim 0.4$**：大氣含有中等濃度懸浮微粒或輕度空氣汙染。
  * **$AOD > 0.8$ 甚至 $> 1.0$**：大氣極度混濁（常見於嚴重霧霾、沙塵暴侵襲或森林大火濃煙覆蓋）。

---

## 2. 資料集維度與空間範圍 (Dataset Architecture)

* **變數名稱**：`GK2_D_AOD`
* **資料維度大小**：`(day: 31, time: 8, latitude: 167, longitude: 117)`
* **總像素數（Total Pixels）**：$31 \times 8 \times 167 \times 117 = 4,845,672$ 個像素點。
* **空間涵蓋範圍 (Spatial Coverage)**：
  * **經度 (Longitude)**：$119.0^\circ\text{E} \sim 122.48^\circ\text{E}$
  * **緯度 (Latitude)**：$21.0^\circ\text{N} \sim 25.98^\circ\text{N}$
  * **網格解析度 (Spatial Resolution)**：$0.03^\circ \times 0.03^\circ$（涵蓋台灣全島及周邊海域）
* **時間涵蓋範圍 (Temporal Coverage)**：31 天，每天包含 8 個時次（Time Slots）。

---

## 3. 已知座標問題與修正指引 (Known Issues & Fixes)

在使用 `xarray` 或 `netCDF4` 載入本資料集時，原始 NetCDF 標頭存在以下座標與維度命名問題，若未手動修正可能導致分析腳本錯誤地將 8 個時次合併為單一維度：

### ⚠️ 問題說明
1. **`day` 座標填補值問題**：`day` 座標軸內部數值均為 `9.969e+36`（NetCDF 預設 Fill Value），未寫入真實日期（1–31）。
2. **`time` 維度名稱與型態問題**：
   * 檔案時間維度名為 `time`，若處理腳本僅尋找 `hour` 維度名稱，會導致維度辨識失敗。
   * `time` 座標型態為 `<U1`（單字元字串）且內容無法順利讀取，無法直接對應 UTC 小時。

### 🔧 建議 Python 修正程式碼
在對資料進行 reshape 或涵蓋率統計之前，請依下述範例手動重置 `day` 與 `time` 座標：

```python
import numpy as np
import xarray as xr

# 1. 讀取 NetCDF 檔案
ds = xr.open_dataset("GK2_AOD_data.nc")
da = ds["GK2_D_AOD"]

# 2. 修正 day 座標（重新指定為 1 到 31 天）
da = da.assign_coords(day=np.arange(1, da.sizes["day"] + 1))

# 3. 修正 time 座標（重置時次索引 0 到 7，對應當天 8 個觀測時次）
da = da.assign_coords(time=np.arange(da.sizes["time"]))

# 4. 指定 time 為時間維度進行後續分析
HOUR_DIM = "time"
```

---

## 4. 數據統計與品質摘要 (Data Statistics)

本資料集針對所有有效像素進行之品質統計結果如下：

### 4.1 缺值與覆蓋率 (Missing Values & Coverage)
* **無效值/缺值率 (NaN Rate)**：**86.30%**
* **有效值比例 (Valid Data Rate)**：**13.70%**（共計 663,626 筆有效觀測）
* **缺值原因分析**：衛星 AOD 反演受雲層遮蔽（Cloud Masking）、夜間無日照（Solar Zenith Angle 過大）以及地表高反射率過濾影響，86.3% 的缺值屬光學遙測之正常現象。
* **資料填補狀況**：無未知或未定義之 `-999` 填補碼，缺值均已統一標註為 standard `NaN`。

### 4.2 AOD 數值分佈 (AOD Value Distribution)
* **最小值 (Min)**：`0.0121`
* **最大值 (Max)**：`3.8848`（極端高值可能為強污染事件、境外沙塵傳輸或邊緣雲污染殘留）
* **平均值 (Mean)**：`0.2135`（符合台灣周邊區域之背景大氣量級）
* **中位數 (Median)**：`0.1752`
* **百分位數 (Percentiles)**：
  * **95th Percentile**：`0.47`（95% 之有效觀測值小於等於 0.47）
  * **99th Percentile**：`0.65`
  * **$AOD > 1.0$ 比例**：`0.12%`
  * **$AOD > 5.0$ 比例**：`0.00%`

---

## 5. 聯絡與補充資訊

若需確認 8 個觀測時次（`time` 0~7）所對應之精確 UTC/CST 時間戳記，請參閱原始 GEO-KOMPSAT-2 衛星 L2 產品產製日誌或檔名時間註記。