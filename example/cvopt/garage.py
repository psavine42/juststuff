from src.cvopt.tilings import TemplateTile


def parking_simple(verbose=False):
    road_tile = TemplateTile(1, 1, color=0, name='road')
    lot_tile1 = TemplateTile(2, 1, color=1, weight=5, name='lot1')

    road_tile.add_color(-1, half_edges=[0, 1, 2])
    road_tile.add_color(+1, half_edges=[3])

    lot_tile1.add_color(+1, half_edges=[0, 3])

    tiles = [road_tile, lot_tile1]
    return tiles

