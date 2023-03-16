import gym
import numpy as np
from deap import algorithms, base, creator, tools
from pymoo.visualization.scatter import Scatter

# Define the objectives
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

# Define the individual
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Define the toolbox
toolbox = base.Toolbox()

# Define the actions
actions = [0, 1, 2, 3, 4, 5]

# Define the maximum number of actions
max_actions = 50

# Define the evaluation function
def evaluate(individual):
    env = gym.make('Taxi-v3')
    env.reset()
    fitness1 = 0
    fitness2 = 0
    for action in individual:
        obs, reward, done, info, _ = env.step(action)
        fitness1 += reward
        fitness2 += 1
        if done:
            break
    env.close()
    return fitness1, fitness2

# Register the functions with the toolbox
toolbox.register("attr_action", np.random.choice, actions)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_action, n=max_actions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(actions)-1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# Define the main function
def main():
    # Set the random seed
    np.random.seed(0)

    # Create the initial population
    pop = toolbox.population(n=100)

    # Run the algorithm
    algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=50, cxpb=0.5, mutpb=0.2, ngen=100)

    # Print the Pareto-optimal solutions
    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    for ind in pareto_front:
        print(ind, ind.fitness.values)

if __name__ == "__main__":
    main()