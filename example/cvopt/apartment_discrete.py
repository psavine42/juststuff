import argparse

from src.cvopt.tilings import TemplateTile
from src.cvopt.floorplanexample import AdjPlanOO


def generate_units(verbose=False):
    """ """

    hall_tile = TemplateTile(1, 1, weight=0.5)
    hall_tile.add_color(-1, half_edges=[0, 1, 2, 3])

    unit_studio = TemplateTile(1, 1, weight=1)
    unit_1bed = TemplateTile(2, 1, weight=2)
    unit_2bed = TemplateTile(2, 2, weight=3, max_uses=5)
    unit_3bed = TemplateTile(2, 3, weight=4, max_uses=5)
    common_area = TemplateTile(3, 5, weight=4, max_uses=1)

    hall_tile.add_color(+1, half_edges=[0, 2])
    hall_tile.add_color(+1, half_edges=[0, 2])
    hall_tile.add_color(+1, half_edges=[0, 2])

    return [hall_tile, unit_studio, unit_1bed, unit_2bed, unit_3bed, common_area]


def generate_space(levels=2):
    return


if __name__ == '__main__':
    units = generate_units()
    space = generate_space()
    problem = AdjPlanOO(units, space)


    solution = problem.run()


