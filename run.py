import argparse
from src.experiment import *
from src.problem import Problem, PointsInBound
from src.problem.example import *
from src.algo.metroplis_hastings import MetropolisHastings
import src.objectives as objectives





def simple_exp():
    problem = setup_house()

    env = MetroLayoutEnv()

    costfn = objectives.ConstraintsHeur(problem)

    model = MetropolisHastings(env, costfn, num_iter=1000)

    expirement = SimpleMH(
        env,
        problem,
        model=model,
        cost_fn=costfn,
        initializer=PointsInBound(problem, env, seed=69)
    )
    return expirement


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    exp = simple_exp()


