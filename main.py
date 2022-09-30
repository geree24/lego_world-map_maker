import numpy as np
import xarray as xr

from geocube.api.core import make_geocube

import geopandas as gp
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = gp.read_file("ne_50m_admin_0_countries.zip")

# Check original projection
print(df.crs)
print(df.columns)

# Visualize
ax = df.plot()

ax.set_title("WGS84 (lat/lon)")

# Reproject to Mercator (after dropping Antartica)
# df = df[(df.name != "Antarctica") & (df.name != "Fr. S. Antarctic Lands")]

new_df = df.to_crs("EPSG:8857")  # (epsg=8857) would also work

new_ax = new_df.plot()

new_ax.set_title("Equal Earth")

plt.show()

# Rasterizing

df['value'] = 1
res = 0.1     # in degrees

out_grid = make_geocube(
    vector_data=df,
    measurements=["value"],
    resolution=(-res, res),
    fill=-9999
)

fig = plt.figure(figsize=(15, 13))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

df.plot(ax=ax, facecolor='None', edgecolor='r')
da_grib = xr.where(out_grid.value < - 1999.0, np.nan, out_grid.value)
da_grib.plot(ax=ax, add_colorbar=False)

# ax.set_extent([112.5, 154.0, -42.116943, -9.142176])
ax.set_title("Rasterized with "+str(res)+"$^o$ Grids", fontsize=24)
plt.show()


