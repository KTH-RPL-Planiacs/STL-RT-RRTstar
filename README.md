# Real-time STL-RRT\* for Social Robot Navigation
Code for robot real-time path planning under STL specifications



## Introduction

In the context of social navigation with a mobile robot, we present an approach for real-time path-planning in a dynamic environment under spatio-temporal constraints expressed in Signal Temporal Logic (STL).
The STL specification encapsulates preferences on how robots should avoid humans in a shared space, where humans are represented as dynamic obstacles in the planning algorithm.
We derive a cost function from the spatial robustness of a trajectory, given an STL specification, that we use in a real-time planning approach based on RRT\*.
The proposed STL-RT-RRT\* driven by this cost guarantees that the motion plan (if found) asymptotically minimizes the cost function. 
Our results show that our approach outperforms the baseline real-time version of RT-RRT\* in terms of social appropriateness.


## Downloading sources

You can use this API by cloning this repository:
```
$ git clone https://github.com/KTH-RPL-Planiacs/STL-RT-RRTstar
```

Dependencies:
* Python 3.8
	* numpy
	* matplotlib
	* scipy
	* dill



## STLFormula

The module `STLFormula.py` implements the formalism of STL Formulae.
It supports boolean (Conjunction, Disjunction, Negation) and temporal operators (Always, Eventually).

```
from STLFormula import *
```

### True and False boolean constants

```
t = TrueF()
f = FalseF()
```


### Predicate

```
x_gt3 = Predicate(dimension,operator,mu,pi_index_signal)
```
is an STL Formula, where the constructor takes 4 arguments:
* `dimension`: string/name of the dimension (ex: `'x'`)
* `operator`: operator (`operatorclass.geq`, `operatorclass.lt`, `operatorclass.leq`, `operatorclass.gt`)
* `mu`: float mu (ex: `3`)
* `pi_index_signal`: in the signal, which index corresponds to the predicate's dimension (ex: `0`)


### STLPredicate2D

```
zone1 = STLPredicate2D(index_dimension_x,index_dimension_y,alpha,beta,gamma,delta)
```
is an STL Formula of the form (\alpha < x < \beta  \wedge \gamma < y < \delta), where the constructor takes 6 arguments:
* index_signal_dimension_x: dimension index for x-dimension (typically 0)
* index_signal_dimension_y: dimension index for y-dimension (typically 1)
* alpha: \alpha
* beta: \beta
* gamma: \gamma
* delta: \delta
		

### Conjunction and Disjunction

```
c = Conjunction(phi1,phi2)
d = Disjunction(phi1,phi2)
```
are STL Formulae respectively representing the Conjunction and Disjunction of 2 STL Formulae `phi1` and `phi2`.


### Negation

```
n = Negation(phi)
```
is an STL Formula representing the negation of an STL Formula `phi`.


### Always and Eventually

```
a = Always(phi,t1,t2)
e = Eventually(phi,t1,t2)
```
are STL Formulae respectively representing the Always and Eventually of an STL Formulae `phi`. They both takes 3 arguments:
* `phi`: an STL formula
* `t1`: lower time interval bound
* `t2`: upper time interval bound


### Untimed Always and Eventually

```
a = Untimed_Always(phi)
e = Untimed_Always(phi)
```
are STL Formulae respectively representing the Untimed Always and Untimed Eventually of an STL Formulae `phi`. They both takes 1 argument:
* `phi`: an STL formula


### Robustness

All STL Formulae contain 1 function to compute the robustness of a signal given the STL Formula.

```
x_gt3 = Predicate('x',operatorclass.gt,3,0)
a = Always(x_gt3,0,5)
a.robustness([[3.1],[3.3],[3.2],[3.0],[2.9],[3.1],[3.5],[3.1],[2.2]],0)
-0.1
```





## Real-Time RRTStar

The module `rt_rrt_star.py` implements the real-time version of the RRT star planning algorithm (RT-RRT star: A Real-Time Path Planning Algorithm Based On RRT star, Naderi et al., 2015).

```
rrt_star = RRTStar(
    start=[50, 50],
    goal=[470, 390],
    rand_area=[50, 470, 50, 390],
    obstacle_list=[],
    expand_dis=10,
    max_iter=10000,
    max_time=0.1,
    goal_sample_rate=5,
    path_resolution=10,
    grid_size=20,
    warm_start=False,
    warm_start_tree_size=1000,
    robot_radius=30)
```
The constructor of the RRTStar object takes several arguments into account, among others: 
* `start` and `goal`: the 2D start and goal points
* `rand_area` of the form `[x_min, x_max, y_min, y_max]`: the min/max values for the 2 dimensions to sample from
* `warm_start`: if starts building a tree offline. If yes, then set also `warm_start_tree_size`: the max size of growing the tree offline before using it online.

```
rrt_star.set_new_start_new_goal(new_start,new_goal)
```
sets new start (i.e. new tree root) and new goal to the tree, and rewires the tree (if existing) from the root

```
path, path_nodes = rrt_star.planning(current_pos=current_pos,updated_obstacle_list=updated_obstacle_list)
```
is what is used in real-time, where with a given frequency this is called to update the planning. `current_pos` is the current measured position of the agent, and `updated_obstacle_list` is the updated list of obstacles to avoid.
This returns `path`, a list of 2D positions from the current measured position of the agent and the goal, as well as `path_nodes`, the list of nodes in the RRT star tree from the current measured position of the agent and the goal.




## STL Real-Time RRTStar

The module `stl_rt_rrt_star.py` implements the real-time version of the RRT star planning algorithm, with STL constraints.

```
rrt_star = RRTStar(
    start=[50, 50],
    goal=[470, 390],
    rand_area=[50, 470, 50, 390],
    obstacle_list=[],
    expand_dis=10,
    max_iter=10000,
    max_time=0.1,
    goal_sample_rate=5,
    path_resolution=10,
    grid_size=20,
    warm_start=False,
    warm_start_tree_size=1000,
    robot_radius=30)
```
The constructor of the RRTStar object takes several arguments into account, among others: 
* `start` and `goal`: the 2D start and goal points
* `rand_area` of the form `[x_min, x_max, y_min, y_max]`: the min/max values for the 2 dimensions to sample from
* `warm_start`: if starts building a tree offline. If yes, then set also `warm_start_tree_size`: the max size of growing the tree offline before using it online.

```
rrt_star.set_new_start_new_goal(new_start,new_goal,specification)
```
sets new start (i.e. new tree root) and new goal to the tree, and rewires the tree (if existing) from the root. `specification` is an STL specification defined with the STL package of module `STLFormula.py`.

```
(path, path_nodes), _, _ = rrt_star.planning(current_pos=current_pos,previous_human_position=previous_human_position,updated_human_position=updated_human_position,stl_specification=specification)
```
is designed for the planning of human-robot encounters, under STL constraints. It is used in real-time, where with a given frequency this is called to update the planning. `current_pos` is the current measured position of the agent, `previous_human_position` is the previous human position, `updated_human_position` is the updated human position, and specification is an STL specification defined with the STL package of module `STLFormula.py`.
This returns `path`, a list of 2D positions from the current measured position of the agent and the goal, as well as `path_nodes`, the list of nodes in the RRT star tree from the current measured position of the agent and the goal.
