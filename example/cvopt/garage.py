from src.cvopt.tilings import TemplateTile


def parking_simple(verbose=False):
    """
    road and parking tile 
    """
    road_tile = TemplateTile(1, 1, color=0, weight=0.1, name='road')
    lot_tile1 = TemplateTile(2, 1, color=1, weight=2, name='lot1')

    road_tile.add_color(-1, half_edges=[0, 1, 2], boundary=True)
    road_tile.add_color(+1, half_edges=[3], boundary=True)

    lot_tile1.add_color(+1, half_edges=[0, 3], boundary=True)

    tiles = [road_tile, lot_tile1]
    return tiles


def parking_tiling_nd(verbose=False):
    road_tile = TemplateTile(1, 1, color=0, weight=0.1, name='road')
    lot_tile1 = TemplateTile(2, 1, color=1, weight=2, name='lot1')

    road_tile.add_color(-1, half_edges=[0, 1, 2, 3], boundary=True)
    lot_tile1.add_color(+1, half_edges=[0, 3], boundary=True)

    tiles = [road_tile, lot_tile1]
    return tiles


def parking_tiling_2color(verbose=False):
    road_tile = TemplateTile(1, 1, color=0, weight=0.1, name='road')
    lot_tile1 = TemplateTile(2, 1, color=1, weight=2, name='lot1')

    road_tile.add_color(-1, half_edges=[0, 2], boundary=True)
    lot_tile1.add_color(+1, half_edges=[0, 3], boundary=True)

    road_tile.add_color(-2, half_edges=[1], boundary=True)
    road_tile.add_color(+2, half_edges=[3], boundary=True)
    tiles = [road_tile, lot_tile1]
    return tiles






