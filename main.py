import numpy as np
import xarray as xr
import pandas as pd
from geocube.api.core import make_geocube
import geopandas as gp
import matplotlib.pyplot as plt

# Read the input data file
df = gp.read_file("ne_50m_admin_0_countries.zip")

# Check original crs
print(df.crs)

# Choosing the project's crs, raster resolution
out_crs = "EPSG:8857"
res = 50000  # in degrees by default but depends on the scale of the crs

# Visualizing the original WGS84 (epsg:4326)
ax = df.plot()
ax.set_title("Original map: " + str(df.crs))

# Assigning IDs to each country
df = df.reset_index(names=["COUNTRY_NR"])

# Dropping Antartica
df = df[(df.NAME != "Antarctica") & (df.NAME != "Fr. S. Antarctic Lands")]

# Reproject Equal Earth
projected_df = df.to_crs(out_crs)  # (epsg=8857) would also work

# Saving the important columns to dataframe
important_string_cols = projected_df[['COUNTRY_NR', 'NAME_EN', 'NAME_HU', 'TYPE', 'CONTINENT', 'MAPCOLOR7',
                                      'MAPCOLOR13', 'GDP_MD', 'GDP_YEAR', 'POP_EST', 'POP_YEAR']]

# Visualizing the new Projection (Equal Earth)
new_ax = projected_df.plot(column='MAPCOLOR7')
new_ax.set_title("Equal Earth")
# plt.show()

# Rasterizing with given resolution
out_grid = make_geocube(
    vector_data=projected_df,
    resolution=(-res, res),
    output_crs=out_crs
)

# Visualizing the rasterized map
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # , projection=ccrs.EqualEarth()
da_grib = xr.where(out_grid.COUNTRY_NR < - 1999.0, np.nan, out_grid.COUNTRY_NR)
da_grib.plot(ax=ax, add_colorbar=False, cmap='tab10')
ax.set_title("Rasterized with " + str(res) + "$^o$ Grids")

# Converting to point geometry, modifying the x and y coordinates, finalizing the lego dataframe
lego_df = da_grib.to_dataframe().reset_index()
lego_df.x = lego_df.x / res + 0.5
lego_df.y = lego_df.y / res + 0.5
lego_gdf = gp.GeoDataFrame(
    lego_df.COUNTRY_NR, geometry=gp.points_from_xy(lego_df.x, lego_df.y), crs=out_crs)
lego_gdf = lego_gdf.merge(important_string_cols, left_on='COUNTRY_NR', right_on='COUNTRY_NR', how='left')

# Visualizing the center point of 1x1 LEGO bricks as points
lego_ax = lego_gdf.plot(column='MAPCOLOR7')
lego_ax.set_title("LEGO Earth")
plt.show()


# Creating pivot and statistics dataframes about the geometry
pivot = lego_gdf.dissolve(by='NAME_HU')
pivot['Length'] = pivot.length
bound_names = ['min_x', 'min_y', 'max_x', 'max_y']
bound_data = lego_gdf.total_bounds
bound_df = pd.DataFrame([bound_data], columns=bound_names)

# Writing to excel
with pd.ExcelWriter('output.xlsx') as writer:
    pivot.to_excel(writer, sheet_name='Map Data')
    bound_df.to_excel(writer, sheet_name='Map Size', index=False)

plt.show()

print("done")
