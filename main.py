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

# Visualize original WGS84 (epsg:4326)
ax = df.plot()
ax.set_title("WGS84 (lat/lon)")

# Dropping Antartica
df = df[(df.NAME != "Antarctica") & (df.NAME != "Fr. S. Antarctic Lands")]

# Reproject Equal Earth
new_df = df.to_crs("EPSG:8857")  # (epsg=8857) would also work

# Visualizing new Projection (Equal Earth)
new_ax = new_df.plot(column='MAPCOLOR7', cmap='Dark2')
new_ax.set_title("Equal Earth")
plt.show()

# Rasterizing with given resolution
res = 0.5     # in degrees

out_grid = make_geocube(
    vector_data=df,
    measurements=["MAPCOLOR7"],
    resolution=(-res, res),
    fill=-9999
)

# Visualizing the final maps

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())


# Visualizing rasterized map
# df.plot(ax=ax, facecolor='None')          # Original non-rasterized borders
da_grib = xr.where(out_grid.MAPCOLOR7 < - 1999.0, np.nan, out_grid.MAPCOLOR7)
da_grib.plot(ax=ax, add_colorbar=False, cmap='Dark2')
ax.set_title("Rasterized with "+str(res)+"$^o$ Grids", fontsize=24)


# Visualizing LEGO bricks
lego_df = da_grib.to_dataframe().reset_index()
lego_gdf = gp.GeoDataFrame(
    lego_df.MAPCOLOR7, geometry=gp.points_from_xy(lego_df.x, lego_df.y))
lego_ax = lego_gdf.plot(column='MAPCOLOR7', cmap='Dark2')
lego_ax.set_title("LEGO Earth")
plt.show()

# ax.set_extent([112.5, 154.0, -42.116943, -9.142176])      # Zoom-in to a specific area
plt.show()


