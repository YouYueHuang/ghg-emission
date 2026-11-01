# Comprehensive Spatiotemporal Dataset Documentation & Variable Dictionary

本文件為整合性網格資料集（Gridded Dataset）之完整說明文件，內容涵蓋資料集之 **時空維度結構（Dimensions & Coordinates）**、**四大模組應用機制** 以及 **19 個特徵變數之分類**。

---

## 1. Dataset Architecture

本資料集建立了一個高解析度的 **時空特徵資料集**，將氣象動力、空氣品質、遙測代理變數與人為排放特徵完全對齊於相同的網格陣列中。

* **整體資料大小**：約 **368 MB** (單一變數約佔 19 MB)
* **單一變數陣列維度 (Shape)**：`(day: 31, hour: 8, lat: 167, lon: 117)`
* **總像素/網格點數量 (Total Pixels)**：$31 \times 8 \times 167 \times 117 = 4,845,672$ 筆像素資料

### Coordinates Overview

| 座標軸 (Coord) | 維度大小 | 涵蓋範圍 / 數值範圍 | 說明與解析度 |
| :--- | :--- | :--- | :--- |
| **`day`** | 31 | `1` ~ `31` | 一個月中的第 1 至第 31 天。 |
| **`hour`** | 8 | `8, 9, 10, 11, 12, 13, 14, 15` | 白天關鍵日照與對流時段（8 到 15 時）。 |
| **`lat`** | 167 | $21.0^\circ\text{N} \sim 25.98^\circ\text{N}$ | 緯度座標，網格解析度約 $0.03^\circ$（涵蓋台灣全島及周遭海域）。 |
| **`lon`** | 117 | $119.0^\circ\text{E} \sim 122.5^\circ\text{E}$ | 經度座標，網格解析度約 $0.03^\circ$。 |

---

## 2. 19 Features

資料集內含的 19 個變數依其**大氣物理、遙測與社會經濟屬性**分為五大類別：

### 2.1 大氣動力與氣象條件 (Meteorological & Atmospheric Dynamics)
控制大氣熱力穩定度、污染擴散/輸送以及物理/化學反應之關鍵因子。

| 變數名稱 (Variable) | 全名描述 (Full Name) | 物理意義與環境效應 (Physical & Environmental Significance) |
| :--- | :--- | :--- |
| **`U`** | 東西向風速 ($u$-wind) | 東西向風速分量 (m/s)。正值代表西風，負值代表東風。主導污染物的水平平流輸送方向。 |
| **`V`** | 南北向風速 ($v$-wind) | 南北向風速分量 (m/s)。正值代表南風，負值代表北風。決定污染物沿緯度方向的移動軌跡。 |
| **`Temp`** | 氣溫 (Temperature) | 近地表或低層大氣溫度 ($^\circ\text{C}$ 或 $\text{K}$)。直接驅動大氣熱力穩定度與光化學反應速率。 |
| **`Hum`** | 相對濕度 (Relative Humidity) | 空氣水氣飽和度 (%)。高濕度會引發懸浮微粒吸濕膨脹（Hygroscopic Growth），改變光散射效率並促進二次氣膠生成。 |
| **`PBLH`** | 行星邊界層高度 (Boundary Layer Height) | 地面混合層高度 (m)。**邊界層越低，污染物越容易被壓縮在近地表**，導致嚴重積聚事件。 |
| **`SWD`** | 短波向下輻射 (Shortwave Radiation) | 到達地表之太陽輻射強度 ($W/m^2$)。驅動臭氧（$\text{O}_3$）及二次有機氣膠 (SOA) 光化學生成反應的核心能量源。 |

### 2.2 空氣品質與化學成分 (Air Quality & Chemical Components)
大氣中即時觀測或模擬之氣態與顆粒態污染物濃度。

| 變數名稱 (Variable) | 全名描述 (Full Name) | 物理意義與環境效應 (Physical & Environmental Significance) |
| :--- | :--- | :--- |
| **`troposphere_ozone`**| 對流層臭氧總量 | 對流層內臭氧之柱濃度或特定層濃度，代表區域大氣的總體氧化能力。 |
| **`ozone`** | 近地表臭氧 ($\text{O}_3$) | 由氮氧化物 ($\text{NO}_x$) 與揮發性有機物 (VOCs) 在強太陽輻射 (`SWD`) 照射下生成的強氧化性二次污染物。 |
| **`no2`** | 二氧化氮 ($\text{NO}_2$) | 主要空氣污染物與臭氧前驅物，源自車輛排氣、工廠及發電廠等化石燃料燃燒。 |

### 2.3 地表環境與衛星遙測特徵 (Remote Sensing & Surface Proxies)
反映地表覆蓋與人類夜間活動強度的遙測代理指標。

| 變數名稱 (Variable) | 全名描述 (Full Name) | 物理意義與環境效應 (Physical & Environmental Significance) |
| :--- | :--- | :--- |
| **`VIIRS`** | VIIRS 夜間燈光指數 | 來自衛星日夜波段 (DNB) 之夜間燈光輻射值。作為城市化程度、用電量與經濟活動強度的空間代理變數。 |
| **`NDVI`** | 歸一化植被指數 | 綠色植被覆蓋率指數 ($[-1, 1]$)。高值代表森林/綠地，低值代表水泥化都市區或工業區。 |

### 2.4 人類活動與土地利用代理變數 (Anthropogenic Proxies & Land Use)
代表局部污染源排放強度的空間代理指標。

| 變數名稱 (Variable) | 全名描述 (Full Name) | 物理意義與環境效應 (Physical & Environmental Significance) |
| :--- | :--- | :--- |
| **`Industry`** | 工業區指標 | 網格內工業用地佔比或活動強度，代表點源與固定源工業排放。 |
| **`Road total length`**| 道路總長度 | 網格內道路網總長度 (m 或 km)，用以估算移動源（車輛廢氣、煞車與輪胎磨損）的排放強度。 |
| **`Business`** | 商業區指標 | 商業用地佔比或商業活動密集度，代表商業營運與周邊交通活動產生的排放。 |
| **`Residential`** | 住宅區指標 | 人口居住密度或住宅區佔比，反映民生烹飪、暖氣與日常生活產生的排放特徵。 |

### 2.5 時空定位特徵 (Spatio-temporal Coordinates)
供機器學習與深度學習模型捕捉時空異質性與週期性規律的特徵變數。

| 變數名稱 (Variable) | 全名描述 (Full Name) | 物理意義與環境效應 (Physical & Environmental Significance) |
| :--- | :--- | :--- |
| **`Lon`** | 經度陣列 (Longitude) | 每個網格點之絕對經度數值 ($^\circ\text{E}$)，提供模型空間定位特徵。 |
| **`Lat`** | 緯度陣列 (Latitude) | 每個網格點之絕對緯度數值 ($^\circ\text{N}$)，提供模型空間定位特徵。 |
| **`Hr`** | 時次特徵 (Hour of Day) | 一天中觀測的時間點 ($8 \sim 15$)。協助模型捕捉日夜循環與車流尖峰等日規律。 |
| **`MM`** | 月份特徵 (Month of Year) | 日曆月份。協助模型捕捉季節交替（如東北季風與西南季風轉換）帶來的背景空氣品質變化。 |

---

## 3. 機器學習模型構建與整合分析總結 (Summary)

本資料集經過精心設計，將空氣品質預測模型所需的四大核心模組完美對齊於統一的時空網格中：

```
+-----------------------------------------------------------------------------------+
|                            Spatiotemporal Data Cube                               |
|                  (31 Days x 8 Hours x 167 Latitudes x 117 Longitudes)             |
+-----------------------------------------------------------------------------------+
        |                       |                         |                     |
        v                       v                         v                     v
[1. 大氣擴散條件]       [2. 熱力與光化學驅動]     [3. 區域排放/人類活動]   [4. 時空定位資訊]
 (U, V, PBLH)             (Temp, Hum, SWD,         (Industry, Road,         (Lat, Lon,
                           troposphere_ozone,       Business, Residential,   Hr, MM)
                           ozone, no2)              VIIRS, NDVI)
```
