# Shared Mobility Redistribution <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->
- [High Level Overview](#high-level-overview)
- [Dependencies](#dependencies)
- [Build Taxi Implementations](#build-taxi-implementations)
- [Build Redistribution Environment Implementations](#build-redistribution-environment-implementations)

## High Level Overview
Shared mobility has experienced significant growth in recent years and has the potential to reduce the climate impact of transportation through a more efficient use of resources in terms of vehicle utilisation, road space and energy.

On-demand services have been offered for a wide range of vehicles including cars, vans, bicycles, electric bicycles and electric scooters.
For these services to offer a compelling service, competitive with individual vehicle ownership, they need to offer reliable availability of their vehicles to customers.
Customer preferences however may result in asymmetric demand for journey origin and destination points which can lead to a deficit of vehicles in one location and a surplus at another.
Service providers can counter this by redistributing their vehicles but of course these redistributing efforts come with costs.

This project is to explore the application of Reinforcement Learning to vehicle rebalancing decisions. An openai gym environment will be developed to test the effectiveness of a range of Reinforcement Learning techniques to optimising redistributing efforts across a range of conditions. 

## Dependencies
<ul>
    <li>The latest version of Python</li>
    <li>OpenAI Gym installed on device</li>
    <li>IPython installed on device</li>
    <li>NumPy installed on device</li>
    <li>MatPlotLib installed on device</li>
</ul>

## Build Taxi Implementations
To Run the Q-Learning implementation of the Gym Taxi Game
```bash
make q-learning-taxi
```

To Run the State Action Reward State Action (SARSA) implemetation of Taxi Game
```bash
make sarsa-taxi
```

To Run the Expected SARSA implemetation of Taxi Game
```bash
make expected-sarsa-taxi
```

To Run the Double Q-Learning implemetation of Taxi Game
```bash
make double-q-taxi
```

## Build Redistribution Environment Implementations

To Run Q-Learning implemetation of the Redistribution Environment
```bash
make q-learning-model
```

To Run SARSA implemetation of the Redistribution Environment
```bash
make sarsa-model
```

To Run Expected-SARSA implemetation of the Redistribution Environment
```bash
make expected-sarsa-model
```

To Run Double Q-Learning implemetation of the Redistribution Environment
```bash
make double-q-model
```

To see a explanation of all Makefile Commands
```bash
make help
```