import random
import numpy as np
import polars as pl
import polars.selectors as cs

def uniform_sample():
    v = np.random.uniform(-0.8,1)
    v = v if v > 0.1 else v - 0.2
    return v

def get_quantiles(num_cats, min_perc_per_cat):
    assert min_perc_per_cat >= 0 and min_perc_per_cat <= 1
    assert num_cats*min_perc_per_cat <= 1.0

    remaining_perc = 1-num_cats*min_perc_per_cat
    quantiles = np.ones(num_cats-1) * min_perc_per_cat
    quantiles = np.cumsum(quantiles)
    randomness_vector = np.random.uniform(0,1,num_cats-1)
    quantiles = quantiles + remaining_perc*(np.cumsum(randomness_vector)/np.sum(np.cumsum(randomness_vector)))
    return quantiles.tolist()

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

    def get(self, num_samples):
        if self.value is None:
            self.value = self._calc(num_samples)
        return self.value

    def reset(self):
        self.value = None
        return self

    def _calc(self, num_samples):
        val = pl.Series(name=self.name, values=np.random.normal(0,1,num_samples))
        if len(self.parents) == 0:
            return val

        val += self.intercept
        for parent in self.parents:
            coefficients = uniform_sample()
            data = parent.get(num_samples)
            if data.dtype == pl.Utf8:
                coeff_map = {cat: uniform_sample() for cat in sorted(data.unique().to_list())}
                coefficients = data.replace_strict(coeff_map).rename('coeffs')
            val += data.cast(pl.Float64)*coefficients

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
    def __init__(self, name, parents=[]):
        super().__init__(name, parents)

    def _calc(self, num_samples):
        quantiles = get_quantiles(2, 0.3)
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
        self.node = random.choice(self.node_restrictions)(self.name, self.parents, **self.kwargs)
        self.coefficients = self.node.coefficients
        self.intercept = self.node.intercept

    def reset(self):
        self.node.reset()
        self._init_node()
        return self

    def get(self, num_samples):
        return self.node.get(num_samples)

class NodeCollection():
    def __init__(self, name, nodes, drop_vars=[]):
        self.name = name
        self.nodes = toposort(nodes)
        self.drop_vars = drop_vars

    def get(self, num_samples):
        data = [n.get(num_samples) for n in self.nodes[::-1]]
        data = pl.DataFrame(data)
        if len(self.drop_vars) > 0:
            data = data.drop(self.drop_vars)
        return data

    def reset(self):
        for node in self.nodes[::-1]:
            node.reset()
        return self
