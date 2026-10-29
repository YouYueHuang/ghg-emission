import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ===== 路徑設定 =====
input_dir = r"../data/raw/nc_datas/data_2023/GEMS_NO2"
output_dir = r"../results/tables/GEMS_NO2"
filename = "202301.nc"
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(input_dir, filename)
tag = os.path.splitext(filename)[0]

# ===== 1. 基本結構 =====
ds = xr.open_dataset(file_path)
print("=" * 60)
print("1. 檔案結構")
print("=" * 60)
print(ds)
no2 = ds['no2']
print("\nno2 屬性:", no2.attrs)
print("no2 維度:", dict(no2.sizes))

# ===== 2. 經緯度資訊 =====
lon = ds['longitude'].values
lat = ds['latitude'].values
print("\n" + "=" * 60)
print("2. 經緯度")
print("=" * 60)
print(f"longitude shape={lon.shape}, 範圍 {np.nanmin(lon):.4f} ~ {np.nanmax(lon):.4f}")
print(f"latitude  shape={lat.shape}, 範圍 {np.nanmin(lat):.4f} ~ {np.nanmax(lat):.4f}")
if lon.ndim == 2:
    print(f"經度方向間距(中位數): {np.nanmedian(np.diff(lon, axis=1)):.4f}")
    print(f"緯度方向間距(中位數): {np.nanmedian(np.diff(lat, axis=0)):.4f}")
else:
    print(f"經度間距(中位數): {np.nanmedian(np.diff(lon)):.4f}")
    print(f"緯度間距(中位數): {np.nanmedian(np.diff(lat)):.4f}")

# ===== 3. 整理成 (day, hour, 空間像素) =====
no2 = no2.transpose('day', 'hour', ...)
day_vals = no2['day'].values
hour_vals = no2['hour'].values
nd, nh = len(day_vals), len(hour_vals)
arr = no2.values.reshape(nd, nh, -1).astype('float64')
npix = arr.shape[2]

with np.errstate(invalid='ignore'):
    is_nan = ~np.isfinite(arr)
    is_fill = np.isfinite(arr) & (arr <= -1e30)
    is_nonpos = np.isfinite(arr) & (arr > -1e30) & (arr <= 0)
    is_valid = np.isfinite(arr) & (arr > 0)

print("\n" + "=" * 60)
print("3. 缺值分類統計（全部像素）")
print("=" * 60)
total = arr.size
print(f"總像素數      : {total}")
print(f"NaN           : {is_nan.sum()} ({is_nan.mean()*100:.2f}%)")
print(f"fill value    : {is_fill.sum()} ({is_fill.mean()*100:.2f}%)")
print(f"0 或負值      : {is_nonpos.sum()} ({is_nonpos.mean()*100:.2f}%)")
print(f"有效值(>0)    : {is_valid.sum()} ({is_valid.mean()*100:.2f}%)")

# ===== 4. day x hour 有效像素矩陣 =====
valid_count = is_valid.sum(axis=2)                      # (day, hour)
coverage = valid_count / npix * 100                     # 涵蓋率 %

count_df = pd.DataFrame(valid_count, index=day_vals, columns=hour_vals)
count_df.index.name = 'day'
count_df.columns.name = 'hour'

cov_df = pd.DataFrame(coverage, index=day_vals, columns=hour_vals).round(1)
cov_df.index.name = 'day'
cov_df.columns.name = 'hour'

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

print("\n" + "=" * 60)
print("4. 各 (day, hour) 有效像素涵蓋率 % （列=day, 欄=hour）")
print("=" * 60)
print(cov_df)

# ===== 5. 哪些 hour / day 完全沒有 value =====
hour_total = valid_count.sum(axis=0)    # 各 hour 在整個月的有效像素
day_total = valid_count.sum(axis=1)     # 各 day 在所有 hour 的有效像素

empty_hours = hour_vals[hour_total == 0]
empty_days = day_vals[day_total == 0]
hours_with_data = hour_vals[hour_total > 0]
days_with_data = day_vals[day_total > 0]

print("\n" + "=" * 60)
print("5. 完全沒有 value 的 hour / day")
print("=" * 60)
print(f"【整個月都沒有任何資料的 hour】({len(empty_hours)}/{nh}): {list(empty_hours)}")
print(f"【有資料的 hour】({len(hours_with_data)}/{nh}): {list(hours_with_data)}")
print(f"【整天都沒有任何資料的 day】({len(empty_days)}/{nd}): {list(empty_days)}")
print(f"【有資料的 day】({len(days_with_data)}/{nd}): {list(days_with_data)}")

print("\n各 hour 在整月的有效天數（該 hour 至少有 1 個有效像素的天數）:")
hour_days = (valid_count > 0).sum(axis=0)
print(pd.Series(hour_days, index=hour_vals, name='days_with_data').to_string())

print("\n各 day 有資料的 hour 數:")
day_hours = (valid_count > 0).sum(axis=1)
print(pd.Series(day_hours, index=day_vals, name='hours_with_data').to_string())

# ===== 6. 數值統計 =====
print("\n" + "=" * 60)
print("6. 有效值統計（單位: x1e16 molecules/cm²）")
print("=" * 60)
vals = arr[is_valid] / 1e16
if vals.size > 0:
    q = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
    print(f"筆數 {vals.size}, min {vals.min():.4f}, max {vals.max():.4f}, mean {vals.mean():.4f}")
    print("百分位 1/5/25/50/75/95/99: " + ", ".join(f"{x:.4f}" for x in q))

    # 各 hour 平均值
    hour_mean = []
    for j in range(nh):
        v = arr[:, j, :][is_valid[:, j, :]] / 1e16
        hour_mean.append(v.mean() if v.size else np.nan)
    print("\n各 hour 有效值平均 (x1e16):")
    print(pd.Series(hour_mean, index=hour_vals, name='mean').round(4).to_string())
else:
    print("沒有任何有效值！")

# ===== 7. 建議用來畫圖的 day / hour =====
print("\n" + "=" * 60)
print("7. 建議挑選")
print("=" * 60)
if valid_count.max() > 0:
    best_day_i = int(np.argmax(day_hours))               # 有資料 hour 最多的一天
    best_day = day_vals[best_day_i]
    best_hour_j = int(np.argmax(valid_count[best_day_i]))
    best_hour = hour_vals[best_hour_j]
    print(f"有資料的 hour 最多的一天: day={best_day} ({day_hours[best_day_i]} 個 hour)")
    print(f"該天涵蓋率最高的 hour   : hour={best_hour} "
          f"(涵蓋率 {coverage[best_day_i, best_hour_j]:.1f}%)")
    gi, gj = np.unravel_index(valid_count.argmax(), valid_count.shape)
    print(f"整月涵蓋率最高的 (day, hour): day={day_vals[gi]}, hour={hour_vals[gj]} "
          f"(涵蓋率 {coverage[gi, gj]:.1f}%)")

# ===== 8. 輸出檔案 =====
count_df.to_csv(os.path.join(output_dir, f'explore_{tag}_valid_count.csv'))
cov_df.to_csv(os.path.join(output_dir, f'explore_{tag}_coverage_pct.csv'))

summary = pd.DataFrame({
    'type': ['hour'] * nh + ['day'] * nd,
    'value': list(hour_vals) + list(day_vals),
    'valid_pixels_total': list(hour_total) + list(day_total),
    'n_slices_with_data': list(hour_days) + list(day_hours),
})
summary['has_data'] = summary['valid_pixels_total'] > 0
summary.to_csv(os.path.join(output_dir, f'explore_{tag}_summary.csv'), index=False)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cov_df.values, aspect='auto', cmap='viridis', vmin=0, vmax=100)
ax.set_xticks(range(nh))
ax.set_xticklabels(hour_vals)
ax.set_yticks(range(nd))
ax.set_yticklabels(day_vals)
ax.set_xlabel('hour')
ax.set_ylabel('day')
ax.set_title(f'GEMS NO2 valid pixel coverage (%) - {tag}')
fig.colorbar(im, ax=ax, label='coverage (%)')
fig.savefig(os.path.join(output_dir, f'explore_{tag}_coverage_heatmap.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\n已輸出 CSV 與熱圖至: {output_dir}")