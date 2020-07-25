"""Test that a grid is created over polygons and boroughs."""

from __future__ import annotations
from typing import TYPE_CHECKING, Union
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely import wkb
from haversine import haversine
from cleanair.databases.tables import LondonBoundary


if TYPE_CHECKING:
    import pandas as pd


def fishnet_checks(
    fishnet_df: pd.DataFrame, geom: Union[Polygon, MultiPolygon], grid_resolution: int
) -> None:
    """Run checks that a fishnet has been cast correctly on the geometry."""
    assert "row" in fishnet_df.columns
    assert "col" in fishnet_df.columns
    assert "geom" in fishnet_df.columns
    assert len(fishnet_df) == grid_resolution ** 2
    assert 0 < fishnet_df["row"].all() < grid_resolution + 1
    assert 0 < fishnet_df["col"].all() < grid_resolution + 1

    # create a list of points and load them into a multi point object
    nodes = []
    fishnet_df["geom"].apply(
        lambda x: nodes.extend(
            [Point(y) for y in wkb.loads(x, hex=True).exterior.coords]
        )
    )
    multi_points = MultiPoint(nodes)

    # check there are 5 points for every grid cell
    assert len(nodes) == grid_resolution ** 2 * 5
    # check the convex hull of the points contains the geometry
    assert multi_points.bounds[0] - geom.bounds[0] < 10e-6  # minx
    assert multi_points.bounds[1] - geom.bounds[1] < 10e-6  # miny
    assert geom.bounds[2] - multi_points.bounds[2] < 10e-6  # maxx
    assert geom.bounds[3] - multi_points.bounds[3] < 10e-6  # maxy
    assert multi_points.convex_hull.buffer(1e-10).contains(geom)


def test_fishnet_over_square(grid, square: Polygon) -> None:
    """Test the fishnet is cast correctly over a square."""
    grid_res = 4
    srid = 4326

    # calcuate the length between the bottom left and top right corner
    bottom_left = square.bounds[0], square.bounds[1]
    top_right = square.bounds[2], square.bounds[3]
    length = haversine(bottom_left, top_right) * 1000

    # size of step
    grid_step = int(length / grid_res)
    print("GRID STEP:", grid_step)

    fishnet_df = grid.st_fishnet(
        from_shape(square, srid=srid),
        grid_resolution=grid_res,
        grid_step=grid_step,
        srid=srid,
        output_type="df",
    )
    print(fishnet_df)
    fishnet_checks(fishnet_df, square, grid_res)


def test_fishnet_over_borough(grid) -> None:
    """Test the fishnet is cast over the borough."""
    borough = "Westminster"
    grid_res = 8

    # get the borough geometry
    with grid.dbcnxn.open_session() as session:
        result = (
            session.query(LondonBoundary).filter(LondonBoundary.name == borough).one()
        )
        geom = to_shape(result.geom)
        assert isinstance(geom, MultiPolygon)

    print(grid.fishnet_over_borough(borough, grid_res, output_type="sql"))
    fishnet_df = grid.fishnet_over_borough(borough, grid_res, output_type="df")
    fishnet_checks(fishnet_df, geom, grid_res)
