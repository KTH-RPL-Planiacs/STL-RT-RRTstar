"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import os
import sys
import time 
import random
import pickle, dill
from collections import deque
import itertools, more_itertools

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ellipsis import get_random_node_ellipsis_sampling

sys.setrecursionlimit(10000)

show_animation = False


class RRTStar():
    """
    Class for RRT Star planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None            
            self.children = {}
            self.cost = 0.0
            self.blocking_cost = 0.0
            self.total_cost = lambda: self.cost + self.blocking_cost
            self.trajectory_until_node = []
            
        def __str__(self):
            return '('+str(round(self.x))+','+str(round(self.y))+')'
            
        def __repr__(self):
            return '('+str(round(self.x))+','+str(round(self.y))+')'
            
        def __hash__(self):
            return hash((self.x, self.y))
        
        def __lt__(self, other):
            return (self.x, self.y) < (other.x, other.y)

        def __eq__(self, other):
            try:
                return (self.x, self.y) == (other.x, other.y)
            except AttributeError:
                return False
    
    
    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=8.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 grid_size=20,
                 max_time=0.1,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 warm_start=True,
                 warm_start_tree_size=6000,
                 robot_radius=0.0):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_xrand = rand_area[0]
        self.max_xrand = rand_area[1]
        self.min_yrand = rand_area[2]
        self.max_yrand = rand_area[3]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.max_time=max_time
        
        #The RT-RRT* queues (to rewire from tree root and recursively the children)
        self.Q_r = deque()
        self.Q_s = deque()
        
        #for plot purposes
        self.rewired_r = []
        self.rewired_s = []
        self.rewired_s_success = []
        
        #Grid for grid indexing of the nodes
        self.grid = {}
        self.cells = []
        self.grid_size=grid_size
        for x in range((rand_area[0]//grid_size)-2,(rand_area[1]//grid_size)+3):
            self.grid[x] = {}
            for y in range((rand_area[2]//grid_size)-2,(rand_area[3]//grid_size)+3):
                self.grid[x][y] = []
                self.cells.append((x,y))
                
        #If warm start required, perform a warm start, but without any consideration for obstacles
        if warm_start:
            start_time = time.time()
            obstacles = self.obstacle_list
            self.warm_start(warm_start_tree_size)
            self.obstacle_list = obstacles
            print("warm start calculation terminated in ",time.time() - start_time)
        
        #Blocked nodes by obstacles
        #we block the cell containing an obstacle + all adjacent cells
        self.blocked_cells = {}
        for obstacle in self.obstacle_list:
            self.blocked_cells[(obstacle[0]//self.grid_size, obstacle[1]//self.grid_size)] = None
            for x,y in itertools.product([(obstacle[0]//self.grid_size)-1,obstacle[0]//self.grid_size,(obstacle[0]//self.grid_size)+1],[(obstacle[1]//self.grid_size)-1,obstacle[1]//self.grid_size,(obstacle[1]//self.grid_size)+1]):
                self.blocked_cells[(x,y)] = None
            # self.blocked_cells[(obstacle[0]//self.grid_size, obstacle[1]//self.grid_size)] = None
    
    
    def draw_graph(self, rnd=None, room_area=None):
        plt.clf()
        if room_area is not None:
            plt.xlim(room_area[:2])
            plt.ylim(room_area[2:])            
        else:
            plt.axis("equal")
            plt.axis([-2, 15, -2, 15])
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        for node in self.node_list:
            if node.parent:
                if node.blocking_cost == float("inf") or node.parent.blocking_cost == float("inf"):
                    plt.plot([node.x,node.parent.x], [node.y,node.parent.y], "-y")
                else:
                    plt.plot([node.x,node.parent.x], [node.y,node.parent.y], "-g")
        # for node in self.rewired_r:
            # plt.plot(node.x, node.y, "1b")
        # for node in self.rewired_s:
            # # plt.plot(node.x, node.y, "2c")
        for node in self.rewired_s_success:
            plt.plot(node.x, node.y, "2m")
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)
            # self.plot_ellipsis(ox, oy, size)
        plt.plot(self.start.x, self.start.y, "or")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.grid(True)
        # plt.pause(0.01)
    
    
    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)
    
    
    def nodes_inside_ellipsis(self,obstaclex,obstacley):
        g_ell_center = (obstaclex-40, obstacley-40)
        g_ell_width = 100
        g_ell_height = 40
        
        angle = 45.
        cos_angle = np.cos(np.radians(180.-angle))
        sin_angle = np.sin(np.radians(180.-angle))

        x = np.array([node.x for node in self.node_list])
        y = np.array([node.y for node in self.node_list])
        xc = x - g_ell_center[0]
        yc = y - g_ell_center[1]

        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle 

        rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)
        
        nodes_inside_ellipsis = [self.node_list[index] for index in np.where(rad_cc <= 1.)[0]]
        # print(nodes_inside_ellipsis)
        return nodes_inside_ellipsis

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)
        
    @staticmethod
    def plot_ellipsis(x,y,size,color="-b"):
        g_ell_center = (x-40, y-40)
        g_ell_width = 100
        g_ell_height = 40
        angle = 45.
        g_ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=angle, fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(g_ellipse)
    
    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind
    
    
    @staticmethod
    def check_collision(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision

        return True  # safe
    
    
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
    
    
    
    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        # d, alpha = self.calc_distance_and_angle(new_node, to_node)
        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
        # if d <= self.path_resolution and abs(math.degrees(alpha))<=10:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        new_node.blocking_cost = from_node.blocking_cost
        
        return new_node
    
    
    
    def line_is_free(self, from_node, to_node):
        if from_node is None or to_node is None:
            return False
        
        path_x = [from_node.x]
        path_y = [from_node.y]
        nb_points = math.ceil(math.hypot(to_node.x - from_node.x, to_node.y - from_node.y)/self.path_resolution)
        x_spacing = (to_node.x - from_node.x) / (nb_points + 1)
        y_spacing = (to_node.y - from_node.y) / (nb_points + 1)
        for i in range(1, nb_points+1):
            path_x.append(from_node.x + i * x_spacing)
            path_y.append(from_node.y + i * y_spacing)
        path_x.append(to_node.x)
        path_y.append(to_node.y)

        for (ox, oy, size) in self.obstacle_list:
            dx_list = [ox - x for x in path_x]
            dy_list = [oy - y for y in path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+self.robot_radius)**2:
                return False  # collision

        return True  # safe
    
    
    
    def get_random_node(self):
        #random sampling
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_xrand, self.max_xrand),
                random.uniform(self.min_yrand, self.max_yrand))
        else:
            # goal point sampling
            if random.randint(0, 100) < self.goal_sample_rate:
                rnd = self.Node(self.end.x, self.end.y)
            #ellipsis sampling
            else:
                x,y = get_random_node_ellipsis_sampling([self.start.x,self.start.y],[self.end.x,self.end.y])
                rnd = self.Node(x,y)
        return rnd
        
    
    
    def generate_partial_course(self, goal_ind):
        node = self.node_list[goal_ind]
        path = [[round(node.x), round(node.y)]]
        path_nodes = [node]
        while node.parent is not None and node.parent != self.start:
            path.append([round(node.x), round(node.y)])
            path_nodes.append(node)
            node = node.parent
        path.append([round(node.x), round(node.y)])
        path_nodes.append(node)
        path.append([round(self.start.x), round(self.start.y)])
        path_nodes.append(self.start)
        return path, path_nodes
    
    
    
    #Finds all the nodes near to a given node (the nodes in the grid of the given node + adjacent grids)
    def find_nodes_near(self, node):
        X_near = []
        for x,y in itertools.product([(node.x//self.grid_size)-1,node.x//self.grid_size,(node.x//self.grid_size)+1],[(node.y//self.grid_size)-1,node.y//self.grid_size,(node.y//self.grid_size)+1]):
            X_near.extend(self.grid[x][y])
        try:
            X_near.remove(node)
        except ValueError:
            pass
        return X_near
    
    
    #Returns the nearest node of a given node (in the ones in the grid of the given node + adjacent grids)
    def find_nearest_node(self, node):
        X_near = self.find_nodes_near(node)
        s = sorted(X_near, key=lambda x_near: math.hypot(node.x - x_near.x, node.y - x_near.y))
        return s[0]    
    
    
    #Returns the nearest node in a path of a given node
    def find_nearest_node_path(self, measured_robot_position, path_nodes):
        s = sorted(path_nodes, key=lambda path_node: math.hypot(measured_robot_position[0] - path_node.x, measured_robot_position[1] - path_node.y))
        return s[0]
    
    
    def find_nodes_near_within_expanddis(self, node):
        X_near = []
        for x,y in itertools.product([(node.x//self.grid_size)-1,node.x//self.grid_size,(node.x//self.grid_size)+1],[(node.y//self.grid_size)-1,node.y//self.grid_size,(node.y//self.grid_size)+1]):
            X_near.extend(self.grid[x][y])
            
        #Remove itself
        try:
            X_near.remove(node)
        except ValueError:
            pass
        
        #Remove nodes with distance greater than expand_dis
        X_near = [x_near for x_near in X_near if math.hypot(node.x - x_near.x, node.y - x_near.y) <= self.expand_dis]

        return X_near    
    
    
    
    def find_nodes_near_within_expanddis_and_not_infcost(self, node, unblocked_cells, new_blocked_cells):
        X_near = []
        
        neighbouring_cells = [(x,y) for x,y in itertools.product([(node.x//self.grid_size)-1,node.x//self.grid_size,(node.x//self.grid_size)+1],[(node.y//self.grid_size)-1,node.y//self.grid_size,(node.y//self.grid_size)+1])]
        neighbouring_cells_not_blocked = list(set(neighbouring_cells)-set(new_blocked_cells)-set(unblocked_cells))
        
        for cell in neighbouring_cells_not_blocked:
            X_near.extend(self.grid[cell[0]][cell[1]])
        
        #Remove itself
        try:
            X_near.remove(node)
        except ValueError:
            pass
        
        #Remove nodes with distance greater than expand_dis
        X_near = [x_near for x_near in X_near if (math.hypot(node.x - x_near.x, node.y - x_near.y) <= self.expand_dis and x_near.blocking_cost != float("inf"))]
        s = sorted(X_near, key=lambda x_near: math.hypot(node.x - x_near.x, node.y - x_near.y))
        # for x_near in s:
            # print("\t\t",node,x_near,math.hypot(node.x - x_near.x, node.y - x_near.y))
        return s
    
    
    
    def recursively_swap_parent_child(parent,child):
        node = self.node_list[goal_ind]
        while node.parent is not None and node.parent != self.start:
            path.append([round(node.x), round(node.y)])
            path_nodes.append(node)
            node = node.parent


    def set_new_start_new_goal(self,new_start,new_goal):
        new_start_node = self.find_nearest_node(self.Node(new_start[0],new_start[1]))
        new_end_node = self.find_nearest_node(self.Node(new_goal[0],new_goal[1]))
        
        old_start = self.start
        self.start = new_start_node
        
        processing_node = self.start
        chain = []
        while processing_node != old_start:
            chain.append(processing_node)
            processing_node = processing_node.parent
        chain.append(old_start)        
        
        for elt in more_itertools.windowed(chain,n=2):
            child, par = elt[0], elt[1]
            child.children[par] = None
            try:
                del par.children[child]
            except KeyError:
                pass
            par.parent = child
        
        self.start.parent = None
        self.start.cost = 0
        self.update_cost_to_leaves(self.start)
        self.end = new_end_node
        
        #Rewire the entire tree from the root node
        print("updating tree with start and end goal")
        self.rewired_s = []
        self.rewired_s_success = []
        if not self.Q_s:
            self.Q_s.append(self.start)
        time_start = time.time()
        while self.Q_s:
            x_s = self.Q_s.popleft()
            for child in x_s.children:
                self.Q_s.append(child)
            X_near = self.find_nodes_near_within_expanddis(x_s)
            for x_near in X_near:
                c_old = x_near.total_cost()
                c_new = x_s.total_cost()+math.hypot(x_s.x-x_near.x,x_s.y-x_near.y)
                if c_new<c_old and self.line_is_free(x_s, x_near) and x_s not in self.successors(x_near):
                    try:
                        del x_near.parent.children[x_near]
                    except KeyError:
                        #TODO
                        pass
                    x_near.parent = x_s
                    x_s.children[x_near] = None
                    x_near.cost = x_s.cost+math.hypot(x_near.x - x_s.x, x_near.y - x_s.y)
                    x_near.blocking_cost = x_s.blocking_cost
                    self.update_cost_to_leaves(x_near)
                    self.rewired_s_success.append(x_near)
                    self.Q_s.appendleft(x_near)    
                self.rewired_s.append(x_near)
    
    
    def warm_start(self, number_nodes):
        """
        warm start rt-rrt* planning

        """
        self.node_list = [self.start]
        self.obstacle_list = []
        
        i = -1
        #Recursively add the desired number of nodes in the tree
        while len(self.node_list) < number_nodes:
            i+=1
            if len(self.node_list) % 100 == 0:
                print("warm start in progress: ", round((len(self.node_list)/number_nodes)*100,1), "% completed")
            
            #ADD NODE TO THE TREE
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            
            
            #rewire the end node
            if new_node == self.end:
                near_node = self.node_list[nearest_ind]
                self.end.cost = near_node.cost + math.hypot(self.end.x-near_node.x, self.end.y-near_node.y)
                self.end.blocking_cost = near_node.blocking_cost
                try:
                    del self.end.parent.children[self.end]
                except Exception:
                    pass
                self.end.parent = near_node
                self.end.parent.children[self.end] = None
                continue
            
            #Find the nearest node and connect to it. Rewire if needed.
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + \
                math.hypot(new_node.x-near_node.x,
                           new_node.y-near_node.y)
            new_node.blocking_cost = near_node.blocking_cost

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    node_with_updated_parent.parent.children[node_with_updated_parent] = None
                    self.node_list.append(node_with_updated_parent)
                    self.grid[node_with_updated_parent.x//self.grid_size][node_with_updated_parent.y//self.grid_size].append(node_with_updated_parent)
                    self.Q_r.append(node_with_updated_parent)
                else:
                    new_node.parent.children[new_node] = None
                    self.node_list.append(new_node)
                    self.grid[new_node.x//self.grid_size][new_node.y//self.grid_size].append(new_node)
                    self.Q_r.append(new_node)
    
        #Rewire the entire tree from the root node
        print("rewire from tree root")
        self.rewired_s = []
        self.rewired_s_success = []
        #REWIRE FROM TREE ROOT
        if not self.Q_s:
            self.Q_s.append(self.start)
        time_start = time.time()
        while self.Q_s:
            x_s = self.Q_s.popleft()
            for child in x_s.children:
                self.Q_s.append(child)
            X_near = self.find_nodes_near_within_expanddis(x_s)
            for x_near in X_near:
                c_old = x_near.total_cost()
                c_new = x_s.total_cost()+math.hypot(x_s.x-x_near.x,x_s.y-x_near.y)
                if c_new<c_old and self.line_is_free(x_s, x_near) and x_s not in self.successors(x_near):
                    try:
                        del x_near.parent.children[x_near]
                    except KeyError:
                        #TODO
                        pass
                    x_near.parent = x_s
                    x_s.children[x_near] = None
                    x_near.cost = x_s.cost+math.hypot(x_near.x - x_s.x, x_near.y - x_s.y)
                    x_near.blocking_cost = x_s.blocking_cost
                    self.update_cost_to_leaves(x_near)
                    self.rewired_s_success.append(x_near)
                    self.Q_s.appendleft(x_near)    
                self.rewired_s.append(x_near)
        print("rewire from tree root processed in ",time.time()-time_start,"s")
        print("end of warm start computation, tree size is ",len(self.node_list))
    
    
    
    def rewire_unblocked_nodes(self,unblocked_cells,new_blocked_cells):

        #a dict with key the unblocked node, and value the min dist to a no blocked node
        unblocked_nodes = {}
        for unblocked_cell in unblocked_cells:
            neighbouring_cells = [(x,y) for x,y in itertools.product([unblocked_cell[0]-1,unblocked_cell[0],unblocked_cell[0]+1],[unblocked_cell[1]-1,unblocked_cell[1],unblocked_cell[1]+1])]
            neighbouring_cells_not_blocked = list(set(neighbouring_cells)-set(new_blocked_cells)-set(unblocked_cells))
            
            #if there is no unblocked neighbouring cell, set the min dist to a no blocked node to inf
            if not neighbouring_cells_not_blocked:
                for node in self.grid[unblocked_cell[0]][unblocked_cell[1]]:
                    unblocked_nodes[node] = float("inf")
                continue
            
            for node in self.grid[unblocked_cell[0]][unblocked_cell[1]]:
                list_dist_neighbours = []
                for neighbour_cell in neighbouring_cells_not_blocked:
                    for neighbour in self.grid[neighbour_cell[0]][neighbour_cell[1]]:
                        list_dist_neighbours.append(math.hypot(neighbour.x - node.x, neighbour.y - node.y))
                try:
                    unblocked_nodes[node] = min(list_dist_neighbours)
                except ValueError:
                    unblocked_nodes[node] = float("inf")
                    
        unblocked_nodes = dict(sorted(unblocked_nodes.items(), key=lambda item: item[1]))
            
        #now rewire the nodes by acending distance to the nearest non-blocked node (that is not recently unblocked)
        for node in unblocked_nodes:
            X_near = self.find_nodes_near_within_expanddis_and_not_infcost(node,unblocked_cells,new_blocked_cells)
            if not X_near:
                continue
            costs  = [x_near.total_cost()+math.hypot(node.x - x_near.x, node.y - x_near.y) for x_near in X_near]
            new_parent = list(X_near)[costs.index(min(costs))]
            if self.line_is_free(new_parent, node) and new_parent not in self.successors(node):
                try:
                    del node.parent.children[node]
                except KeyError:
                    pass
                node.parent = new_parent
                new_parent.children[node] = None
                node.cost = new_parent.cost+math.hypot(node.x - new_parent.x, node.y - new_parent.y)
                node.blocking_cost = new_parent.blocking_cost
                self.update_cost_to_leaves(node)
                self.rewired_s_success.append(node)
    
    
    
    
    def planning(self, animation=True, current_pos=None, updated_obstacle_list=None):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """
        
        #Update costs and tree's root
        time_start = time.time()
        if current_pos is None:
            self.node_list = [self.start]
        elif self.start == current_pos:
            #do nothing
            pass
        else:
            # nearest_ind = self.get_nearest_node_index(self.node_list, self.Node(current_pos[0],current_pos[1]))
            # if self.node_list[nearest_ind] == self.start:
                # print("no move was done")
            # else:
            #for the new start node, set its parent as children, and set no parent. Also, set the cost of the old parent as "inf" UNLESS the goal position is changed!
            # old_start = self.start
            # # self.start = self.node_list[nearest_ind]
            # self.start = current_pos
            # self.start.parent = None
            # del old_start.children[self.start]
            # old_start.parent = self.start
            # self.start.children[old_start] = None
            # self.start.cost = 0
            # self.update_cost_to_leaves(self.start)
            
            
            old_start = self.start
            self.start = current_pos
            
            processing_node = self.start
            chain = []
            while processing_node != old_start:
                chain.append(processing_node)
                processing_node = processing_node.parent
            chain.append(old_start)        
            
            for elt in more_itertools.windowed(chain,n=2):
                child, par = elt[0], elt[1]
                child.children[par] = None
                try:
                    del par.children[child]
                except KeyError:
                    pass
                par.parent = child
            
            self.start.parent = None
            self.start.cost = 0
            self.update_cost_to_leaves(self.start)
            
        
        self.rewired_s = []
        self.rewired_s_success = []
        
        
        if updated_obstacle_list is not None:
            self.obstacle_list = updated_obstacle_list
            new_blocked_cells = {}
            for obstacle in self.obstacle_list:
                # new_blocked_cells[(obstacle[0]//self.grid_size, obstacle[1]//self.grid_size)] = None
                for x,y in itertools.product([(obstacle[0]//self.grid_size)-1,obstacle[0]//self.grid_size,(obstacle[0]//self.grid_size)+1],[(obstacle[1]//self.grid_size)-1,obstacle[1]//self.grid_size,(obstacle[1]//self.grid_size)+1]):
                    new_blocked_cells[(x,y)] = None
                        
            became_blocked   = list(set(list(new_blocked_cells))-set(list(self.blocked_cells)))
            became_unblocked = list(set(list(self.blocked_cells))-set(list(new_blocked_cells)))
                        
            #set unblocked nodes blocking cost to 0.0 and propagate to leaves
            for cell in became_unblocked:
                for node in self.grid[cell[0]][cell[1]]:
                    node.blocking_cost = 0.0
                    self.unblock_leaves(node)
                    
            #rewiring of unblocked nodes, since some of them might be rewired to parts of the tree that are not blocked!
            self.rewire_unblocked_nodes(became_unblocked,new_blocked_cells)
            
            #nodes in front of the obstacle (ellipsis) are set to a high blocking cost
            # nodes_inside_ellipsis = []
            # for obstacle in self.obstacle_list:
                # nodes_inside_ellipsis.extend(self.nodes_inside_ellipsis(obstacle[0],obstacle[1]))
            # for node in nodes_inside_ellipsis:
                # node.blocking_cost = float("inf")
                # self.block_leaves(node)
                
            #set newly blocked nodes blocking cost to inf and propagate to leaves
            for cell in new_blocked_cells:
                for node in self.grid[cell[0]][cell[1]]:
                    node.blocking_cost = float("inf")
                    self.block_leaves(node)
            self.blocked_cells = new_blocked_cells

        

        
        remaining_time = self.max_time - (time.time() - time_start)
        
        # print("rewire from tree root")
        #REWIRE FROM TREE ROOT
        time_start = time.time()
        # while time.time()-time_start < self.max_time * 0.9:
        while time.time()-time_start < remaining_time * 0.9:
            if not self.Q_s:
                self.Q_s.append(self.start)
            x_s = self.Q_s.popleft()
            for child in x_s.children:
                self.Q_s.append(child)
            X_near = self.find_nodes_near_within_expanddis(x_s)
            for x_near in X_near:
                c_old = x_near.total_cost()
                c_new = x_s.total_cost()+math.hypot(x_s.x-x_near.x,x_s.y-x_near.y)   
                if c_new<c_old and self.line_is_free(x_s, x_near) and x_s not in self.successors(x_near):
                    try:
                        del x_near.parent.children[x_near]
                    except KeyError:
                        #TODO fix this KeyError
                        pass
                    x_near.parent = x_s
                    x_s.children[x_near] = None
                    x_near.cost = x_s.cost+math.hypot(x_near.x - x_s.x, x_near.y - x_s.y)
                    x_near.blocking_cost = x_s.blocking_cost
                    self.update_cost_to_leaves(x_near)
                    self.rewired_s_success.append(x_near)
                    # self.Q_s.append(x_near)
                self.rewired_s.append(x_near)
                    # self.Q_s.append(x_near)
            # for child in x_s.children:
                # self.Q_s.append(child)

        #we reblock the cells in the area of the obstacles, becaume maybe because of rewiring some cells that are actually blocked became unblocked
        for cell in new_blocked_cells:
            for node in self.grid[cell[0]][cell[1]]:
                node.blocking_cost = float("inf")
                self.block_leaves(node)

        return self.generate_path()


    
    
    
    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        new_node.blocking_cost = self.node_list[min_ind].blocking_cost

        return new_node
    
    
    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                print("COST",i,self.node_list[i],self.node_list[i].cost)
                return i

        return None


    def search_best_temporary_goal_node(self):
        #In case no goal can be reached, find the node the closest to it, so we can already go towards it
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        return dist_to_goal_list.index(min(dist_to_goal_list))
    
    
    
    def generate_path(self):
        if self.end.parent is not None and self.end.total_cost() != float("inf"):
            path = [[round(self.end.x), round(self.end.y)]]
            path_nodes = [self.end]
            node = self.end
            while node.parent is not None and node.parent != self.start:
                path.append([round(node.x), round(node.y)])
                path_nodes.append(node)
                node = node.parent
            path.append([round(node.x), round(node.y)])
            path_nodes.append(node)
            path.append([round(self.start.x), round(self.start.y)])
            path_nodes.append(self.start)
            return path, path_nodes
        else:
            print(self.end,"is inf or has no parent",self.end.total_cost())
            dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) if n.total_cost()!=float("inf") else float("inf") for n in self.node_list]
            if min(dist_to_goal_list) == float("inf"):
                return [[round(self.start.x), round(self.start.y)]], [self.start]
            return self.generate_partial_course(dist_to_goal_list.index(min(dist_to_goal_list)))
            
    
    
    
    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds
    
    
    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(
                edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                try:
                    del near_node.parent.children[near_node]
                except KeyError:
                    print("\t",near_node,"was not in the children list of",near_node.parent)
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.blocking_cost = edge_node.blocking_cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)
    
    
    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d
    
    
    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)    
    
    
    def block_leaves(self, parent_node):
        for node in parent_node.children:
            node.blocking_cost = float("inf")
            self.block_leaves(node)    
    
    
    def unblock_leaves(self, parent_node):
        for node in parent_node.children:
            node.blocking_cost = 0.0
            self.unblock_leaves(node)
    
    def find_near_nodes_close_updated_obstacles(self):
        for node in self.node_list:
            try:
                for (ox, oy, size) in self.obstacle_list:
                    if math.hypot(ox-node.x, oy-node.y) <= (size+self.robot_radius):
                        node.cost = float("inf")
                        self.propagate_cost_to_leaves(node)
                        print("update",node.x,node.y)
            except ValueError:
                continue

    def successors(self,node):
        lst = []
        for child in node.children:
            lst.append(child)
            lst.extend(self.successors(child))
        return lst

    def update_cost_to_leaves(self, parent_node):
        for node in parent_node.children:
            node.cost = parent_node.cost+math.hypot(node.x - parent_node.x, node.y - parent_node.y)
            node.blocking_cost = parent_node.blocking_cost
            self.update_cost_to_leaves(node)  
    
    
def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (470,390,20)
        # (200, 165, 20),
        # (300, 248, 20)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    
    #expand_dis=8  Pepper max speed is 80cm.s-1 = 8cm.ds-1 (we have a refresh frequency of 10Hz or every 100ms)
    #aldebaran naoqi expand_dis=5.5  Pepper max speed is 55cm.s-1 = 5.5cm.ds-1 (we have a refresh frequency of 10Hz or every 100ms)
    
    
    
    
    ENVIRONMENT = {}
    radius_obs = 20
    # ENVIRONMENT[0.0]=[(470.0,390.0,radius_obs),(0,0,radius_obs)]
    ENVIRONMENT[0.0]=[(470,390,radius_obs)]
    ENVIRONMENT[0.1]=[(470,390,radius_obs)]
    ENVIRONMENT[0.2]=[(470,390,radius_obs)]
    ENVIRONMENT[0.3]=[(470,390,radius_obs)]
    ENVIRONMENT[0.4]=[(470,390,radius_obs)]
    ENVIRONMENT[0.5]=[(470,390,radius_obs)]
    ENVIRONMENT[0.6]=[(470,390,radius_obs)]
    ENVIRONMENT[0.7]=[(470,390,radius_obs)]
    ENVIRONMENT[0.8]=[(470,390,radius_obs)]
    ENVIRONMENT[0.9]=[(470,390,radius_obs)]
    ENVIRONMENT[1.0]=[(461,383,radius_obs)]
    ENVIRONMENT[1.1]=[(453,376,radius_obs)]
    ENVIRONMENT[1.2]=[(444,369,radius_obs)]
    ENVIRONMENT[1.3]=[(436,362,radius_obs)]
    ENVIRONMENT[1.4]=[(428,356,radius_obs)]
    ENVIRONMENT[1.5]=[(419,349,radius_obs)]
    ENVIRONMENT[1.6]=[(411,342,radius_obs)]
    ENVIRONMENT[1.7]=[(402,335,radius_obs)]
    ENVIRONMENT[1.8]=[(394,328,radius_obs)]
    ENVIRONMENT[1.9]=[(386,322,radius_obs)]
    ENVIRONMENT[2.0]=[(377,315,radius_obs)]
    ENVIRONMENT[2.1]=[(369,308,radius_obs)]
    ENVIRONMENT[2.2]=[(360,301,radius_obs)]
    ENVIRONMENT[2.3]=[(352,294,radius_obs)]
    ENVIRONMENT[2.4]=[(344,288,radius_obs)]
    ENVIRONMENT[2.5]=[(335,281,radius_obs)]
    ENVIRONMENT[2.6]=[(327,274,radius_obs)]
    ENVIRONMENT[2.7]=[(318,267,radius_obs)]
    ENVIRONMENT[2.8]=[(310,260,radius_obs)]
    ENVIRONMENT[2.9]=[(302,254,radius_obs)]
    ENVIRONMENT[3.0]=[(293,247,radius_obs)]
    ENVIRONMENT[3.1]=[(285,240,radius_obs)]
    ENVIRONMENT[3.2]=[(276,233,radius_obs)]
    ENVIRONMENT[3.3]=[(268,226,radius_obs)]
    ENVIRONMENT[3.4]=[(260,220,radius_obs)]
    ENVIRONMENT[3.5]=[(251,213,radius_obs)]
    ENVIRONMENT[3.6]=[(243,206,radius_obs)]
    ENVIRONMENT[3.7]=[(234,199,radius_obs)]
    ENVIRONMENT[3.8]=[(226,192,radius_obs)]
    ENVIRONMENT[3.9]=[(218,186,radius_obs)]
    ENVIRONMENT[4.0]=[(209,179,radius_obs)]
    ENVIRONMENT[4.1]=[(201,172,radius_obs)]
    ENVIRONMENT[4.2]=[(192,165,radius_obs)]
    ENVIRONMENT[4.3]=[(184,158,radius_obs)]
    ENVIRONMENT[4.4]=[(176,152,radius_obs)]
    ENVIRONMENT[4.5]=[(167,145,radius_obs)]
    ENVIRONMENT[4.6]=[(159,138,radius_obs)]
    ENVIRONMENT[4.7]=[(150,131,radius_obs)]
    ENVIRONMENT[4.8]=[(142,124,radius_obs)]
    ENVIRONMENT[4.9]=[(134,118,radius_obs)]
    ENVIRONMENT[5.0]=[(125,111,radius_obs)]
    ENVIRONMENT[5.1]=[(117,104,radius_obs)]
    ENVIRONMENT[5.2]=[(108,97,radius_obs)]
    ENVIRONMENT[5.3]=[(100,90,radius_obs)]
    ENVIRONMENT[5.4]=[(92,84,radius_obs)]
    ENVIRONMENT[5.5]=[(83,77,radius_obs)]
    ENVIRONMENT[5.6]=[(75,70,radius_obs)]
    ENVIRONMENT[5.7]=[(66,63,radius_obs)]
    ENVIRONMENT[5.8]=[(58,56,radius_obs)]
    ENVIRONMENT[5.9]=[(50,50,radius_obs)]
    ENVIRONMENT[6.0]=[(50,50,radius_obs)]
    ENVIRONMENT[6.1]=[(50,50,radius_obs)]
    ENVIRONMENT[6.2]=[(50,50,radius_obs)]
    ENVIRONMENT[6.3]=[(50,50,radius_obs)]
    ENVIRONMENT[6.4]=[(50,50,radius_obs)]
    ENVIRONMENT[6.5]=[(50,50,radius_obs)]
    ENVIRONMENT[6.6]=[(50,50,radius_obs)]
    ENVIRONMENT[6.7]=[(50,50,radius_obs)]
    ENVIRONMENT[6.8]=[(50,50,radius_obs)]
    ENVIRONMENT[6.9]=[(50,50,radius_obs)]
    ENVIRONMENT[7.0]=[(50,50,radius_obs)]
    ENVIRONMENT[7.1]=[(50,50,radius_obs)]
    ENVIRONMENT[7.2]=[(50,50,radius_obs)]
    ENVIRONMENT[7.3]=[(50,50,radius_obs)]
    ENVIRONMENT[7.4]=[(50,50,radius_obs)]
    ENVIRONMENT[7.5]=[(50,50,radius_obs)]
    ENVIRONMENT[7.6]=[(50,50,radius_obs)]
    ENVIRONMENT[7.7]=[(50,50,radius_obs)]
    ENVIRONMENT[7.8]=[(50,50,radius_obs)]
    ENVIRONMENT[7.9]=[(50,50,radius_obs)]
    ENVIRONMENT[8.0]=[(50,50,radius_obs)]
    # ENVIRONMENT[8.1]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.2]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.3]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.4]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.5]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.6]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.7]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.8]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[8.9]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.0]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.1]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.2]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.3]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.4]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.5]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.6]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.7]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.8]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[9.9]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.0]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.1]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.2]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.3]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.4]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.5]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.6]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.7]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.8]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[10.9]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.0]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.1]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.2]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.3]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.4]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.5]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.6]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.7]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.8]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[11.9]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.0]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.1]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.2]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.3]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.4]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.5]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.6]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.7]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.8]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[12.9]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.0]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.1]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.2]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.3]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.4]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.5]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.6]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.7]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.8]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[13.9]=[(50.0,50.0,radius_obs)]
    # ENVIRONMENT[14.0]=[(50.0,50.0,radius_obs)]

    
    
    """
    Instantiation
    """
    
    MAX_PEPPER_SPEED = 20
    TREE_SIZE = 1500
    
    # rrt_star = RRTStar(
        # start=[50, 50],
        # goal=[470, 390],
        # rand_area=[50, 470, 50, 390],
        # obstacle_list=[],
        # expand_dis=MAX_PEPPER_SPEED,
        # max_iter=10000,
        # max_time=0.1,
        # goal_sample_rate=5,
        # path_resolution=MAX_PEPPER_SPEED,
        # grid_size=20,
        # warm_start=True,
        # warm_start_tree_size=TREE_SIZE,
        # robot_radius=30)
    
    # #save a pre-computed tree as pickle
    # with open("tree_200_1500_20220804.dill", "wb") as dill_file:
        # dill.dump(rrt_star, dill_file)    
    # exit()
    
    # #load a pre-existing pickled tree
    rrt_star = dill.load(open("tree_200_1500_20220804.dill", "rb"))
    
    #July 5 - useful when restarting from not exact position, or with a custom goal
    new_start = [50,50]
    new_goal = [366,335]
    rrt_star.set_new_start_new_goal(new_start,new_goal)
    
    move_pepper = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8]

    
    for t in ENVIRONMENT:
    
        # ENVIRONMENT[t]=[(180,190,radius_obs)]

        #Update current position with the first goal of last calculated path
        #TODO REAL EXP PEPPER: replace path[1] by the measured current position of Pepper
        
        #Before July 5 - assuming that the agent would go to the next waypoint specified in the path
        # current_pos = path[1]
        
        #July 5 - finding the nearest node in the path to the current position 
        #fake this in simulation
        try:
            if t in move_pepper:
                measured_robot_position = [path[1][0]+random.random(),path[1][1]+random.random()]
                current_pos = rrt_star.find_nearest_node_path(measured_robot_position, path_nodes)
                print("measured position",measured_robot_position,current_pos)
            else:
                measured_robot_position = [path[0][0]+random.random(),path[0][1]+random.random()]
                current_pos = rrt_star.find_nearest_node_path(measured_robot_position, path_nodes)
                print("measured position",measured_robot_position,current_pos)
        except UnboundLocalError:
            current_pos = rrt_star.start
        except IndexError:
            measured_robot_position = [path[0][0]+random.random(),path[0][1]+random.random()]
            current_pos = rrt_star.find_nearest_node_path(measured_robot_position, path_nodes)
            print("measured position",measured_robot_position,current_pos)
        
        

        #Plan for the current iteration of given frequency
        #TODO REAL EXP PEPPER: replace "ENVIRONMENT[t]" in the "updated_obstacle_list=ENVIRONMENT[t]" by the measured positions of the human/obstacle
        start_time = time.time()
        path, path_nodes = rrt_star.planning(animation=show_animation,current_pos=current_pos,updated_obstacle_list=ENVIRONMENT[t])
        elapsed = (time.time() - start_time)
        
        path.reverse()
        path_nodes.reverse()
        #!! path[0] is the recorded position of the agent, and path[1] is the next waypoint !!
        
        print("t=",t,", found path in",elapsed,"seconds")
        print(path)
        #TODO REAL EXP PEPPER: use path[1] as the next goal to reach, then path[2], path[3] etc...
        #RESOLUTION 10Hz = 1 point per 100ms
        
        try:
            if path[1] == [rrt_star.end.x,rrt_star.end.y]:
                break
        except IndexError:
            if path[0] == [rrt_star.end.x,rrt_star.end.y]:
                break
        
        rrt_star.draw_graph(room_area=[0,520,0,440])
        plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
        plt.grid(True)
        # plt.show()
        plt.savefig('img/'+str(t)+'.png')
        
        print("\n")

    
    
    exit()
    
    




# x = [50,470]
# y = [50,390]
# i = 0
# for x, y in zip([470-((470-50)/50)*i for i in range(0,51)], list(np.interp([470-((470-50)/50)*i for i in range(0,51)], x, y))):
    # print("ENVIRONMENT[",i,"] = [(",x,",",y,",radius_obs),(0,0,radius_obs)]")
    # i += 0.1

# k = 5.1
# for i in range(0,40):
    # print("ENVIRONMENT[",round(k,1),"] = [(50.0,50.0,radius_obs),(0,0,radius_obs)]")
    # k += 0.1



if __name__ == '__main__':
    main()
