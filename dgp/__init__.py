import random
import numpy as np
import polars as pl

def uniform_sample():
    v = np.random.uniform(-0.8,1)
    v = v if v > 0.1 else v - 0.2
    return v

def get_quantiles(num_cats, min_perc_per_cat):
    num_objects = int((1.0-num_cats*min_perc_per_cat)*100)

    containers = [[] for _ in range(num_cats)]
    objects = (1 for _ in range(num_objects))
    # we don't need a list of these, just have to iterate over it, so this is a genexp

    for object in objects:
        random.choice(containers).append(object)
        
    containers = [sum(c)/100 for c in containers]
    containers


    containers = [c+min_perc_per_cat for c in containers]
    #assert sum(containers) == 1

    quantiles = [containers[0]]
    for i in range(1,len(containers)-1):
        quantiles.append(containers[i] + quantiles[-1])
        
    return quantiles

def to_categorical(c, quantiles):
    return c.qcut(quantiles, labels=[str(i) for i in range(1,len(quantiles)+2)])

def toposort(variables):
    sorted_list = []
    no_inc_edge_nodes = {v for v in variables if len(v.parents) == 0}
    graph = {v:set(v.parents) for v in variables if len(v.parents) > 0}
    
    while len(no_inc_edge_nodes) > 0:
        node = no_inc_edge_nodes.pop()
        sorted_list.append(node)
        for node_i in [v for v,ps in graph.items() if node in ps]:
            graph[node_i].remove(node)
            if len(graph[node_i]) == 0:
                no_inc_edge_nodes.add(node_i)
                
    return sorted_list


class Node:
    def __init__(self, name, parents=[], **kwargs):
        self.name = name
        self.parents = parents.copy()
        self.coefficients = [uniform_sample() for _ in self.parents]
        self.intercept = uniform_sample()
        self.value = None
        
    def get(self, num_samples=100):
        if self.value is None:
            self.value = self._calc(num_samples)
        return self.value
    
    def reset(self):
        self.coefficients = [uniform_sample() for _ in self.parents]
        self.intercept = uniform_sample()
        self.value = None
        #for p in self.parents:
        #    p.()
        
    def _calc(self, num_samples):
        val = pl.Series(name=self.name, values=np.random.normal(0,1, num_samples))
        if len(self.parents) > 0:
            val += self.intercept 
            for parent, coeff in zip(self.parents, self.coefficients):
                val += parent.get(num_samples).cast(pl.Float64)*coeff
            
        return val
    
    def __repr__(self):
        return f'Node {self.name} - descends from {", ".join([p.name for p in self.parents]) if len(self.parents) > 0 else "none"}'
    
class CategoricalNode(Node):
    def __init__(self, name, parents=[], min_categories=2, max_categories=4, min_percent_per_category=0.15):
        assert max_categories > 2, 'Categorical Nodes should only be used for variables with more than two expression levels'
        
        super().__init__(name, parents)
        
        self.num_categories = np.random.randint(min_categories, max_categories+1)
        self.min_percent_per_category = min_percent_per_category
        
    def _calc(self, num_samples):
       quantiles = get_quantiles(self.num_categories, self.min_percent_per_category)
       return to_categorical(super()._calc(num_samples), quantiles).cast(pl.Utf8)
   
class OrdinalNode(Node):
    def __init__(self, name, parents=[], min_categories=2, max_categories=4, min_percent_per_category=0.15):
        super().__init__(name, parents)
        
        self.num_categories = np.random.randint(min_categories, max_categories+1)
        self.min_percent_per_category = min_percent_per_category
        
    def _calc(self, num_samples):
        quantiles = get_quantiles(self.num_categories, self.min_percent_per_category)
        return to_categorical(super()._calc(num_samples), quantiles).cast(pl.Int32)
    
class BinaryNode(Node):
    def __init__(self, name, parents=[], min_percent_per_category=0.15):
        super().__init__(name, parents)
        
        self.num_categories = 2
        self.min_percent_per_category = min_percent_per_category
        
    def _calc(self, num_samples):
        quantiles = get_quantiles(self.num_categories, self.min_percent_per_category)
        return to_categorical(super()._calc(num_samples), quantiles).cast(pl.Int32) == 1

class GenericNode(Node):
    def __init__(self, name, parents=[], node_restrictions=None, **kwargs):
        self.name = name
        self.parents = parents.copy()
        
        self.kwargs = kwargs
        
        if node_restrictions is None:
            self.node_restrictions = [
                Node,
                CategoricalNode,
                OrdinalNode,
                BinaryNode
            ]
        else:
            self.node_restrictions = node_restrictions
        
        self._init_node()
      
    def _init_node(self):  
        self.node = random.choice([n(self.name, self.parents, **self.kwargs) for n in self.node_restrictions])
        self.coefficients = self.node.coefficients
        self.intercept = uniform_sample()
        
    def reset(self):
        self.node.reset()
        self._init_node()
        
    def get(self, num_samples):
        return self.node.get(num_samples)
   
class NodeCollection():
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = toposort(nodes)
        
    def get(self, num_samples):
        data = [n.get(num_samples) for n in self.nodes[::-1]]
        data = pl.DataFrame(data)
        return data
    
    def reset(self):
        for node in self.nodes[::-1]:
            node.reset()
        #self.nodes[-1].reset()