import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rio
from mpl_toolkits.basemap import Basemap

from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata, rasterize_points_radial

import geopandas as gp
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from shapely.geometry import box

import shapely.speedups

shapely.speedups.enable()

import cartopy.crs as ccrs

# import cartopy.feature as cfeature

# Read the input data file
df = gp.read_file("ne_50m_admin_0_countries.zip")

# Check original crs
print(df.crs)

# Choosing the project's crs, raster resolution
out_crs = "EPSG:8857"
res = 50000  # in degrees

# Visualizing the original WGS84 (epsg:4326)
ax = df.plot()
ax.set_title("Original map: " + str(df.crs))

country_nrs = list(range(0, 242))
df.insert(0, "COUNTRY_NR", country_nrs)

# Dropping Antartica
df = df[(df.NAME != "Antarctica") & (df.NAME != "Fr. S. Antarctic Lands")]

# Reproject Equal Earth
projected_df = df.to_crs(out_crs)  # (epsg=8857) would also work
# new_df = new_df["geometry"].centroid()
# print(new_df.head())

# Saving the important columns to dataframe
important_string_cols = projected_df[['COUNTRY_NR', 'NAME_EN', 'NAME_HU', 'TYPE', 'CONTINENT', 'MAPCOLOR7',
                                      'MAPCOLOR13', 'GDP_MD', 'GDP_YEAR', 'POP_EST', 'POP_YEAR']]

# Visualizing the new Projection (Equal Earth)
new_ax = projected_df.plot(column='MAPCOLOR7', cmap='tab10')
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
    lego_df.COUNTRY_NR, geometry=gp.points_from_xy(lego_df.x, lego_df.y), crs="EPSG:8857")
lego_gdf = lego_gdf.merge(important_string_cols, left_on='COUNTRY_NR', right_on='COUNTRY_NR', how='left')

# Visualizing the center point of 1x1 LEGO bricks as points
lego_ax = lego_gdf.plot(column='MAPCOLOR7', cmap='Dark2')
lego_ax.set_title("LEGO Earth")
plt.show()

"""
list_of_points = lego_gdf.convex_hull
hungary = lego_gdf.loc[lego_gdf['NAME_EN'] == "Hungary"]
hungary_geometry = hungary.convex_hull
gdf = gp.GeoDataFrame(geometry=hungary_geometry)

# ax.set_extent([112.5, 154.0, -42.116943, -9.142176])      # Zoom-in to a specific area


def find_maximum_area_rectangle_with_parallel_sides(geo_df):
    # Find the convex hull of the points
    coords = list(geo_df.geometry.apply(lambda p: (p.x, p.y)))

    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)

    # Initialize variables to store the maximum area and the coordinates of the rectangle
    max_area = 0
    max_rect = None
    rect_covered_points = None

    # Loop through all possible pairs of x and y coordinates
    for x1 in np.arange(min_x, max_x):
        for y1 in np.arange(min_y, max_y):
            for x2 in np.arange(x1, max_x):
                for y2 in np.arange(y1, max_y):
                    # Find the width and height of the rectangle
                    width = x2 - x1
                    height = y2 - y1

                    # Calculate the area of the rectangle
                    area = width * height
                    rect_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    # Check if the rectangle contains all the points
                    rect = Polygon(rect_coords)
                    rect_border_points = polygon_border_points(x1, x2, y1, y2)
                    if all(item in coords for item in rect_border_points):
                        # Update the maximum area and the coordinates of the rectangle if necessary
                        if area > max_area:
                            max_area = area
                            max_rect = rect
                            rect_covered_points = gp.GeoSeries(gp.points_from_xy([i[0] for i in rect_border_points],
                                                                                 [i[1] for i in rect_border_points]),
                                                               crs=out_crs)
                            points_inside_rect = geo_df.loc[geo_df.within(max_rect)]
                            if not points_inside_rect.empty:
                                rect_covered_points = pd.concat([rect_covered_points.geometry,
                                                                points_inside_rect.geometry],
                                                                ignore_index=True)\
                                    .drop_duplicates().reset_index(drop=True)

    return max_rect, rect_covered_points


def polygon_border_points(x1, x2, y1, y2):
    points = []
    for x in range(int(x1), int(x2), 2):
        points.append((float(x), y1))
        points.append((float(x), y2))
    for y in range(int(y1), int(y2), 2):
        points.append((x1, float(y)))
        points.append((x2, float(y)))
    points.append((x1, y1))
    points.append((x2, y1))
    points.append((x2, y2))
    points.append((x1, y2))
    return points


max_rectangle, max_rect_points = find_maximum_area_rectangle_with_parallel_sides(gdf)

print(max_rectangle.area)
print(max_rect_points)
ax2 = gdf.plot(markersize=10, color='red')
gp.GeoSeries(max_rectangle, crs=out_crs).plot(ax=ax2, facecolor='none', edgecolor='blue')
"""
"""
Lego kockák helyett összesíthetné, hogy melyik országhoz mekkora lego kocka kerület és terület kell
"""

# Creating pivot and statistics dataframes about the geometry
pivot = lego_gdf.dissolve(by='NAME_HU')
bound_names = ['min_x', 'min_y', 'max_x', 'max_y']
bound_data = lego_gdf.total_bounds
bound_df = pd.DataFrame([bound_data], columns=bound_names)

# Writing to excel
with pd.ExcelWriter('output.xlsx') as writer:
    pivot.to_excel(writer, sheet_name='Map Data')
    bound_df.to_excel(writer, sheet_name='Map Size', index=False)

plt.show()

print("done")
