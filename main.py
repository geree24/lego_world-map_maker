import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rio
from mpl_toolkits.basemap import Basemap

from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial

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

country_nrs = list(range(0, 242))
df.insert(0, "COUNTRY_NR", country_nrs)

# Dropping Antartica
df = df[(df.NAME != "Antarctica") & (df.NAME != "Fr. S. Antarctic Lands")]
# print(df.head())

# Reproject Equal Earth
new_df = df.to_crs("EPSG:8857")     # (epsg=8857) would also work
# new_df = new_df["geometry"].centroid()
# print(new_df.head())

# Saving the important columns to variables
country_names = new_df.NAME_EN
country_colors = new_df.MAPCOLOR7

# Visualizing new Projection (Equal Earth)
new_ax = new_df.plot(column='MAPCOLOR7', cmap='Dark2')
new_ax.set_title("Equal Earth")
#plt.show()

# Rasterizing with given resolution
res = 50000     # in degrees

out_grid = make_geocube(
    vector_data=df,
    resolution=(-res, res),
    output_crs="EPSG:8857",
    # rasterize_function=rasterize_points_griddata,
    # fill=-9999
)

test1 = out_grid.rio.to_raster("planet_scope.tif")

# Visualizing the final maps

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)   # , projection=ccrs.EqualEarth()
# Visualizing rasterized map
# df.plot(ax=ax, facecolor='None')          # Original non-rasterized borders
da_grib = xr.where(out_grid.COUNTRY_NR < - 1999.0, np.nan, out_grid.COUNTRY_NR)
da_grib.plot(ax=ax, add_colorbar=False, cmap='Dark2')
ax.set_title("Rasterized with "+str(res)+"$^o$ Grids", fontsize=24)


# Visualizing LEGO bricks
lego_df = da_grib.to_dataframe().reset_index()
lego_df.x = lego_df.x / res
lego_df.y = lego_df.y / res
lego_gdf = gp.GeoDataFrame(
    lego_df.COUNTRY_NR, geometry=gp.points_from_xy(lego_df.x, lego_df.y))

lego_gdf = lego_gdf.merge(country_names, left_on='COUNTRY_NR', right_on=country_names.index, how='left')
lego_gdf = lego_gdf.merge(country_colors, left_on='COUNTRY_NR', right_on=country_colors.index, how='left')

lego_ax = lego_gdf.plot(column='MAPCOLOR7', cmap='Dark2')
lego_ax.set_title("LEGO Earth")
#plt.show()

list_of_points = lego_gdf.convex_hull
hungary = lego_gdf.loc[lego_gdf['NAME_EN'] == 'Hungary']
print("done")
# ax.set_extent([112.5, 154.0, -42.116943, -9.142176])      # Zoom-in to a specific area

