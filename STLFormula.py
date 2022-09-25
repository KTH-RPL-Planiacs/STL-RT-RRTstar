class STLFormula:
    """
    Class for representing an STL Formula.
    """
    def __init__(self):
        pass
    
    def rho_bar_bar(self,node,t):
        return min(0,self.rho_bar(node,t))
    
    def rho_bar_bar_humanref(self,node,t):
        return min(0,self.rho_bar_humanref(node,t))
    
    def trapz_discrete(val_a,val_b):
        return (val_a+val_b)/2
    
    def cost_node(parent_cost,parent_rho_bar,node_rho_bar):
        return parent_cost + (-STLFormula.trapz_discrete(parent_rho_bar,node_rho_bar))
    
    def stl_rrt_cost_function(self,node):
        node.rho_bar = self.rho_bar_bar(node, len(node.trajectory_until_node)-1)
        node.stl_cost = STLFormula.cost_node(node.parent.stl_cost,node.parent.rho_bar,node.rho_bar)
    
    def stl_rrt_cost_function_humanref(self, node):
        node.rho_bar = self.rho_bar_bar_humanref(node, len(node.trajectory_until_node_humanreferential)-1)
        node.stl_cost = STLFormula.cost_node(node.parent.stl_cost,node.parent.rho_bar,node.rho_bar)
    
    def stl_rrt_cost_function_root_node(self, node, old_parent):
        if not old_parent:
            node.rho_bar = self.rho_bar_bar(node, len(node.trajectory_until_node)-1)
            node.stl_cost = 0.0
        else:
            node.rho_bar = self.rho_bar_bar(node, len(node.trajectory_until_node)-1)
            node.stl_cost = STLFormula.cost_node(old_parent.stl_cost,old_parent.rho_bar,node.rho_bar)
    
    def stl_rrt_cost_function_root_node_humanref(self, node, old_parent):
        if not old_parent:
            node.rho_bar = self.rho_bar_bar_humanref(node, len(node.trajectory_until_node_humanreferential)-1)
            node.stl_cost = 0.0
        else:
            node.rho_bar = self.rho_bar_bar_humanref(node, len(node.trajectory_until_node_humanreferential)-1)
            print(old_parent)
            node.stl_cost = STLFormula.cost_node(old_parent.stl_cost,old_parent.rho_bar,node.rho_bar)
    
    def stl_test_new_cost(self, candidate_parent, node):
        #save current values
        node_old_parent = node.parent
        node_old_trajectory_until_node = node.trajectory_until_node
        
        #set temporary values (given the candidate parent)
        node.parent = candidate_parent
        node.trajectory_until_node = candidate_parent.trajectory_until_node + [node]
        
        #calculate value of rho_bar for the node with the candidate parent
        node_rho_bar_val = self.rho_bar_bar(node, len(candidate_parent.trajectory_until_node)-1)
        
        #rollback to current values
        node.parent = node_old_parent
        node.trajectory_until_node = node_old_trajectory_until_node
        
        return STLFormula.cost_node(candidate_parent.stl_cost,candidate_parent.rho_bar,node_rho_bar_val)
    
    def stl_test_new_cost_humanref(self, candidate_parent, node):
        #save current values
        node_old_parent = node.parent
        node_old_trajectory_until_node_humanreferential = node.trajectory_until_node_humanreferential
        
        #set temporary values (given the candidate parent)
        node.parent = candidate_parent
        node.trajectory_until_node_humanreferential = candidate_parent.trajectory_until_node_humanreferential + [[node.x_humanreferential,node.y_humanreferential]]
        
        #calculate value of rho_bar for the node with the candidate parent
        node_rho_bar_val = self.rho_bar_bar_humanref(node, len(node.trajectory_until_node_humanreferential)-1)
        
        #rollback to current values
        node.parent = node_old_parent
        node.trajectory_until_node_humanreferential = node_old_trajectory_until_node_humanreferential
        
        return STLFormula.cost_node(candidate_parent.stl_cost,candidate_parent.rho_bar,node_rho_bar_val)
        
        

class TrueF(STLFormula):
    """
    Class representing the True boolean constant
    """
    def __init__(self):
        self.robustness = lambda s, t : float('inf')
        self.rho_bar = lambda node, t : float('inf')
        self.rho_bar_humanref = lambda node, t : float('inf')
        self.sat = True
        self.horizon = 0
        
    def __str__(self):
        return "\\top"


class FalseF(STLFormula):
    """
    Class representing the False boolean constant
    """
    def __init__(self):
        self.robustness = lambda s, t : float('-inf')
        self.rho_bar = lambda node, t : float('-inf')
        self.rho_bar_humanref = lambda node, t : float('-inf')
        self.sat = False
        self.horizon = 0
        
    def __str__(self):
        return "\\bot"


class Predicate(STLFormula):
    """
    Class representing a Predicate, s.t. f(s) \sim \mu
    The constructor takes 4 arguments:
        * dimension: string/name of the dimension
        * operator: operator (geq, lt...)
        * mu: \mu
        * pi_index_signal: in the signal, which index corresponds to the predicate's dimension
    The class contains 2 additional attributes:
        * robustness: a function \rho(s,(f(s) \sim \mu),t) & = \begin{cases} \mu-f(s_t) & \sim=\le \\ f(s_t)-\mu & \sim=\ge \end{cases}
        * sat: a function returning whether \rho(s,(f(s) \sim \mu),t) > 0
        * horizon: 0
    """
    def __init__(self,dimension,operator,mu,pi_index_signal):
        self.pi_index_signal = pi_index_signal
        self.dimension = dimension
        self.operator = operator
        self.mu = mu
        if operator == operatorclass.gt or operator == operatorclass.ge:
            self.robustness = lambda s, t : s[t][pi_index_signal] - mu
            self.rho_bar = lambda node, t : node.trajectory_until_node[t].x - mu if pi_index_signal==0 else node.trajectory_until_node[t].y - mu
            self.rho_bar_humanref = lambda node, t : node.trajectory_until_node_humanreferential[t][pi_index_signal] - mu
            self.sat = lambda s, t : s[t][pi_index_signal] - mu > 0
        else:
            self.robustness = lambda s, t : -s[t][pi_index_signal] + mu
            self.rho_bar = lambda node, t : -node.trajectory_until_node[t].x + mu if pi_index_signal==0 else -node.trajectory_until_node[t].y + mu
            self.rho_bar_humanref = lambda node, t : -node.trajectory_until_node_humanreferential[t][pi_index_signal] + mu
            self.sat = lambda s, t : -s[t][pi_index_signal] + mu > 0
        
        self.horizon = 0
    
    def __str__(self):
        return self.dimension+operators_iv[self.operator]+str(self.mu)


class STLPredicate2D(STLFormula):
    """
    Class representing a Spatio-Temporal 2D Predicate of the form (\alpha < x < \beta  \wedge \gamma < y < \delta)
    The constructor takes 6 arguments:
        * index_signal_dimension_x: dimension index for x-dimension (typically 0)
        * index_signal_dimension_y: dimension index for y-dimension (typically 1)
        * alpha: \alpha
        * beta: \beta
        * gamma: \gamma
        * delta: \delta
    The class contains 2 additional attributes:
        * robustness: a function \rho(s,(f(s) \sim \mu),t) & = \begin{cases} \mu-f(s_t) & \sim=\le \\ f(s_t)-\mu & \sim=\ge \end{cases}
        * sat: a function returning whether \rho > 0
        * horizon: 0
    """
    
    def __init__(self,index_signal_dimension_x,index_signal_dimension_y,alpha,beta,gamma,delta):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        
        #encoding \alpha < x
        alpha_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_x] - alpha
        alpha_lt_x_rho_bar = lambda node, t : node.trajectory_until_node[t].x - alpha
        alpha_lt_x_rho_bar_humanref = lambda node, t : node.trajectory_until_node_humanreferential[t][index_signal_dimension_x] - alpha
        alpha_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_x] - alpha > 0
        #encoding x < \beta
        beta_gt_x_robustness  = lambda s, t : -s[t][index_signal_dimension_x] + beta
        beta_gt_x_rho_bar  = lambda node, t : -node.trajectory_until_node[t].x + beta
        beta_gt_x_rho_bar_humanref  = lambda node, t : -node.trajectory_until_node_humanreferential[t][index_signal_dimension_x] + beta
        beta_gt_x_sat         = lambda s, t : -s[t][index_signal_dimension_x] + beta > 0
        #encoding \gamma < y
        gamma_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_y] - gamma
        gamma_lt_x_rho_bar = lambda node, t : node.trajectory_until_node[t].y - gamma
        gamma_lt_x_rho_bar_humanref = lambda node, t : node.trajectory_until_node_humanreferential[t][index_signal_dimension_y] - gamma
        gamma_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_y] - gamma > 0
        #encoding y < \delta
        delta_gt_x_robustness = lambda s, t : -s[t][index_signal_dimension_y] + delta
        delta_gt_x_rho_bar = lambda node, t : -node.trajectory_until_node[t].y + delta
        delta_gt_x_rho_bar_humanref = lambda node, t : -node.trajectory_until_node_humanreferential[t][index_signal_dimension_y] + delta
        delta_gt_x_sat        = lambda s, t : -s[t][index_signal_dimension_y] + delta > 0
        
        self.horizon = 0
        
        self.robustness = lambda s, t : min([alpha_lt_x_robustness(s,t),beta_gt_x_robustness(s,t),gamma_lt_x_robustness(s,t),delta_gt_x_robustness(s,t)])
        
        # def rho_bar(node, t):
            # try:
                # return min([alpha_lt_x_rho_bar(node,t),beta_gt_x_rho_bar(node,t),gamma_lt_x_rho_bar(node,t),delta_gt_x_rho_bar(node,t)])
            # except IndexError:
                # return float('nan')
        # self.rho_bar = rho_bar
        self.rho_bar = lambda node, t : min([alpha_lt_x_rho_bar(node,t),beta_gt_x_rho_bar(node,t),gamma_lt_x_rho_bar(node,t),delta_gt_x_rho_bar(node,t)])
        
        # def rho_bar_humanref(node, t):
            # try:
                # return min([alpha_lt_x_rho_bar_humanref(node,t),beta_gt_x_rho_bar_humanref(node,t),gamma_lt_x_rho_bar_humanref(node,t),delta_gt_x_rho_bar_humanref(node,t)])
            # except IndexError:
                # return float('nan')
        # self.rho_bar_humanref = rho_bar_humanref
        self.rho_bar_humanref = lambda node, t : min([alpha_lt_x_rho_bar_humanref(node,t),beta_gt_x_rho_bar_humanref(node,t),gamma_lt_x_rho_bar_humanref(node,t),delta_gt_x_rho_bar_humanref(node,t)])
        
        self.sat        = lambda s, t : all([alpha_lt_x_sat(s,t),beta_gt_x_sat(s,t),gamma_lt_x_sat(s,t),delta_gt_x_sat(s,t)])
    
    
    def __str__(self):
        return "("+str(round(self.alpha,3))+" < x < "+str(round(self.beta,3))+" \wedge "+str(round(self.gamma,3))+" < y < "+str(round(self.delta,3))+")"


class Conjunction(STLFormula): 
    """
    Class representing the Conjunction operator, s.t. \phi_1 \wedge \phi_2 \wedge \ldots \wedge \phi_n.
    The constructor takes 1 arguments:
        * lst_conj: a list of STL formulae in the conjunction
    The class contains 1 additional attributes:
        * sat: a function \sigma(t_i) \models \phi_1 \land \phi_2 \land  \ldots \land \phi_n \Leftrightarrow (\sigma(t_i) \models \phi_1 ) \land (\sigma(t_i) \models \phi_2) \land \ldots \land (\sigma(t_i) \models \phi_n )
    """
    def __init__(self,lst_conj):
        self.lst_conj   = lst_conj
        self.sat        = lambda s, t : all([formula.sat(s,t) for formula in self.lst_conj])
        self.robustness = lambda s, t : min([formula.robustness(s,t) for formula in self.lst_conj])
        self.rho_bar = lambda node, t : min([formula.rho_bar(node,t) for formula in self.lst_conj])
        self.rho_bar_humanref = lambda node, t : min([formula.rho_bar_humanref(node,t) for formula in self.lst_conj])
        self.horizon    = max([formula.horizon for formula in self.lst_conj])
    
    def __str__(self):
        s = "("
        for conj in self.lst_conj:
            s += str(conj) + " \wedge "
        return s[:-8]+")"


class Negation(STLFormula): 
    """
    Class representing the Negation operator, s.t. \neg \phi.
    The constructor takes 1 argument:
        * formula 1: \phi
    The class contains 2 additional attributes:
        * robustness: a function \rho(s,\neg \phi,t) = - \rho(s,\phi,t)
        * horizon: \left\|\phi\right\|=\left\|\neg \phi\right\|
    """
    def __init__(self,formula):
        self.formula = formula
        self.robustness = lambda s, t : -formula.robustness(s,t)
        self.rho_bar = lambda node, t : -formula.rho_bar(node,t)
        self.rho_bar_humanref = lambda node, t : -formula.rho_bar_humanref(node,t)
        self.sat = lambda s, t : not formula.sat(s,t)
        self.horizon = formula.horizon
    
    def __str__(self):
        return "\lnot ("+str(self.formula)+")"


class Disjunction(STLFormula): 
    """
    Class representing the Disjunction operator, s.t. \phi_1 \vee \phi_2.
    The constructor takes 2 arguments:
        * formula 1: \phi_1
        * formula 2: \phi_2
    The class contains 2 additional attributes:
        * robustness: a function \rho(s,\phi_1 \lor \phi_2,t) = \max(\rho(s,\phi_1,t),\rho(s,\phi_2,t) )
        * horizon: \left\|\phi_1 \lor \phi_2\right\|= \max\{\left\|\phi_1\right\|, \left\|\phi_2\right\|\}
    """
    # def __init__(self,lst_disj,list_probas):
    def __init__(self,lst_disj):
        self.lst_disj    = lst_disj
        # self.list_probas = list_probas
        self.sat         = lambda s, t : any([formula.sat(s,t) for formula in self.lst_disj])
        self.robustness  = lambda s, t : max([formula.robustness(s,t) for formula in self.lst_disj])
        self.rho_bar  = lambda node, t : max([formula.rho_bar(node,t) for formula in self.lst_disj])
        self.rho_bar_humanref  = lambda node, t : max([formula.rho_bar_humanref(node,t) for formula in self.lst_disj])
        self.horizon    = max([formula.horizon for formula in self.lst_disj])


class Always(STLFormula): 
    """
    Class representing the Always operator, s.t. \mathcal{G}_{[t1,t2]} \phi.
    The constructor takes 3 arguments:
        * formula: a formula \phi
        * t1: lower time interval bound
        * t2: upper time interval bound
    The class contains 2 additional attributes:
        * robustness: a function \rho(s,\mathcal{G}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\min~  \rho(s,\phi,t').
        * horizon: \left\|\mathcal{G}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
    """
    def __init__(self,formula,t1,t2):
        self.formula = formula
        self.t1 = t1
        self.t2 = t2
        self.robustness = lambda s, t : min([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
        self.sat        = lambda s, t : all([ formula.sat(s,k) for k in range(t+t1, t+t2+1)])
        self.horizon = t2 + formula.horizon
        
        self.rho_bar_nodes = {}
        def rho_bar(node,t):
            if t<self.t1:
                self.rho_bar_nodes[node] = float('nan')
                return float('nan')
            elif t>self.t2:
                self.rho_bar_nodes[node] = float('nan')
                return float('nan')
            elif t==self.t1:
                r = self.formula.rho_bar(node,t)
                self.rho_bar_nodes[node] = r
                return r
            else:
                try:
                    r = min(self.formula.rho_bar(node,t),self.rho_bar_nodes[node.parent])
                except KeyError:
                    r = self.formula.rho_bar(node,t)
                self.rho_bar_nodes[node] = r
                return r
        self.rho_bar = rho_bar
        
        self.rho_bar_nodes_humanref = {}
        def rho_bar_humanref(node,t):
            if t<self.t1:
                self.rho_bar_nodes_humanref[node] = float('nan')
                return float('nan')
            elif t>self.t2:
                self.rho_bar_nodes_humanref[node] = float('nan')
                return float('nan')
            elif t==self.t1:
                r = self.formula.rho_bar_humanref(node,t)
                self.rho_bar_nodes_humanref[node] = r
                return r
            else:
                try:
                    r = min(self.formula.rho_bar_humanref(node,t),self.rho_bar_nodes_humanref[node.parent])
                except KeyError:
                    r = self.formula.rho_bar_humanref(node,t)
                self.rho_bar_nodes_humanref[node] = r
                return r
        self.rho_bar_humanref = rho_bar_humanref
    
    def __str__(self):
        return "\mathcal{G}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"


class Eventually(STLFormula): 
    """
    Class representing the Eventually operator, s.t. \mathcal{F}_{[t1,t2]} \phi.
    The constructor takes 3 arguments:
        * formula: a formula \phi
        * t1: lower time interval bound
        * t2: upper time interval bound
    The class contains 2 additional attributes:
        * robustness: a function \rho(s,\mathcal{F}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\max~  \rho(s,\phi,t').
        * horizon: \left\|\mathcal{F}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
    """
    def __init__(self,formula,t1,t2):
        self.formula = formula
        self.t1 = t1
        self.t2 = t2
        self.robustness = lambda s, t :  max([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
        self.sat        = lambda s, t :  any([ formula.sat(s,k) for k in range(t+t1, t+t2+1)])
        self.horizon = t2 + formula.horizon
        
        self.rho_bar_nodes = {}
        def rho_bar(node,t):
            if t<self.t1:
                self.rho_bar_nodes[node] = float('nan')
                return float('nan')
            elif t>self.t2:
                self.rho_bar_nodes[node] = float('nan')
                return float('nan')
            elif t==self.t1:
                r = self.formula.rho_bar(node,t)
                self.rho_bar_nodes[node] = r
                return r
            else:
                try:
                    r = max(self.formula.rho_bar(node,t),self.rho_bar_nodes[node.parent])
                except KeyError:
                    r = self.formula.rho_bar(node,t)
                self.rho_bar_nodes[node] = r
                return r
        self.rho_bar = rho_bar
        
        self.rho_bar_nodes_humanref = {}
        def rho_bar_humanref(node,t):
            if t<self.t1:
                self.rho_bar_nodes_humanref[node] = float('nan')
                return float('nan')
            elif t>self.t2:
                self.rho_bar_nodes_humanref[node] = float('nan')
                return float('nan')
            elif t==self.t1:
                r = self.formula.rho_bar_humanref(node,t)
                self.rho_bar_nodes_humanref[node] = r
                return r
            else:
                try:
                    r = max(self.formula.rho_bar_humanref(node,t),self.rho_bar_nodes_humanref[node.parent])
                except KeyError:
                    r = self.formula.rho_bar_humanref(node,t)
                self.rho_bar_nodes_humanref[node] = r
                return r
        self.rho_bar_humanref = rho_bar_humanref
    
    def __str__(self):
        return "\mathcal{F}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"


class Untimed_Always(STLFormula): 
    """
    Class representing the Untimed Always operator, s.t. \mathcal{G} \phi.
    The constructor takes 1 argument:
        * formula: a formula \phi
    The class contains 2 additional attributes:
        * robustness
    """
    def __init__(self,formula):
        self.formula = formula
        self.robustness = lambda s : min([ formula.robustness(s,t) for t in range(0, len(s))])
        self.sat        = lambda s : all([ formula.sat(s,t) for t in range(0, len(s))])
        
        self.rho_bar_nodes = {}
        def rho_bar(node,t):
            try:
                r = min(self.formula.rho_bar(node,t),self.rho_bar_nodes[node.parent])
            except KeyError:
                r = self.formula.rho_bar(node,t)
            self.rho_bar_nodes[node] = r
            return r
        self.rho_bar = rho_bar
        
        self.rho_bar_nodes_humanref = {}
        def rho_bar_humanref(node,t):
            try:
                r = min(self.formula.rho_bar_humanref(node,t),self.rho_bar_nodes_humanref[node.parent])
            except KeyError:
                r = self.formula.rho_bar_humanref(node,t)
            self.rho_bar_nodes_humanref[node] = r
            return r
        self.rho_bar_humanref = rho_bar_humanref
    
    def __str__(self):
        return "\mathcal{G}("+str(self.formula)+")"


class Untimed_Eventually(STLFormula): 
    """
    Class representing the Untimed Eventually operator, s.t. \mathcal{F} \phi.
    The constructor takes 1 argument:
        * formula: a formula \phi
    The class contains 2 additional attributes:
        * robustness
    """
    def __init__(self,formula):
        self.formula = formula
        self.robustness = lambda s : max([ formula.robustness(s,t) for t in range(0, len(s))])
        self.sat        = lambda s : any([ formula.sat(s,t) for t in range(0, len(s))])
        
        self.rho_bar_nodes = {}
        def rho_bar(node,t):
            try:
                r = max(self.formula.rho_bar(node,t),self.rho_bar_nodes[node.parent])
            except KeyError:
                r = self.formula.rho_bar(node,t),
            self.rho_bar_nodes[node] = r
            return r
        self.rho_bar = rho_bar
        
        self.rho_bar_nodes_humanref = {}
        def rho_bar_humanref(node,t):
            try:
                r = max(self.formula.rho_bar_humanref(node,t),self.rho_bar_nodes_humanref[node.parent])
            except KeyError:
                r = self.formula.rho_bar_humanref(node,t)
            self.rho_bar_nodes_humanref[node] = r
            return r
        self.rho_bar_humanref = rho_bar_humanref
    
    def __str__(self):
        return "\mathcal{F}("+str(self.formula)+")"


