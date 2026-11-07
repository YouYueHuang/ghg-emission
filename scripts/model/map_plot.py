"""
把預測結果畫成空間分佈圖（疊上縣市邊界）。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config


def plot_no2_map(pred_df, lat_col, lon_col, title, out_path, shp_path=config.SHP_PATH):
    grid = (pred_df.assign(**{lat_col: pred_df[lat_col].round(5), lon_col: pred_df[lon_col].round(5)})
            .pivot_table(index=lat_col, columns=lon_col, values='pred_no2', aggfunc='mean'))
    lon, lat, Z = grid.columns.to_numpy(), grid.index.to_numpy(), grid.to_numpy()

    vmin = config.MAP_VMIN
    vmax = config.MAP_VMAX if config.MAP_VMAX is not None else float(np.nanpercentile(Z, 99))
    print(f"   - 色階範圍: {vmin} ~ {vmax:.3g}；有預測值的像素: {int(np.isfinite(Z).sum())}")

    fig, ax = plt.subplots(figsize=(7.5, 9), dpi=config.FIG_DPI)
    im = ax.pcolormesh(lon, lat, Z, cmap=config.MAP_CMAP, vmin=vmin, vmax=vmax, shading='auto')

    try:
        import geopandas as gpd
        counties = gpd.read_file(shp_path)
        counties = (counties.set_crs(epsg=3824) if counties.crs is None else counties).to_crs(epsg=4326)
        counties.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=5)
    except Exception as e:
        print(f"[資訊] 略過縣市邊界（{e}）")

    pad = 0.05
    ax.set_xlim(lon.min() - pad, lon.max() + pad)
    ax.set_ylim(lat.min() - pad, lat.max() + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, extend='max')
    cbar.set_label(f'Predicted NO2 ({config.SCATTER_UNIT})')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=config.FIG_DPI)
    plt.close(fig)
    print(f"✅ NO2 濃度圖已儲存至: {out_path}")
    return out_path