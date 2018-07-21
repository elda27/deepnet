import sys
import os.path

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../../'
    ))

from deepnet.utils import network
import pytest
import networkx as nx

@pytest.mark.parametrize( ("input_list", "edge_node_name", "network_nodes"),
    [
        (# Begin of definition of the 1st case
            ["i1"],
            "5",
            [
                network.NetworkNode("1", ["i1"], ["v1"], None),
                network.NetworkNode("2", ["v1"], ["v2_1", "v2_2"], None),
                network.NetworkNode("3", ["i1", "v2_1"], ["v3"], None),
                network.NetworkNode("4", ["v3", "v1"], ["v4"], None),
                network.NetworkNode("5", ["v3", "v4"], ["o1"], None),
            ]
        ), # End of definition of the 1st case
        (# Begin of definition of the 2nd case
            ["i1"],
            "5",
            [
                network.NetworkNode("1", ["i1"], ["v1"], None),
                network.NetworkNode("2", ["v1"], ["v2_1", "v2_2"], None),
                network.NetworkNode("3", ["i1", "v2_1"], ["v3"], None),
                network.NetworkNode("4", ["v2_1", "v2_2", "v1"], ["v4"], None),
                network.NetworkNode("5", ["v3", "v4"], ["o1"], None),
            ]
        ), # End of definition of the 2nd case
        (# Begin of definition of the 3rd case
            ["i1"],
            "6",
            [
                network.NetworkNode("1", ["i1"], ["v1"], None),
                network.NetworkNode("2", ["v1"], ["v2_1", "v2_2"], lambda x: zip(range(10), range(10)), iterate_from="5"),
                network.NetworkNode("3", ["i1", "v2_1"], ["v3"], None),
                network.NetworkNode("4", ["v2_1", "v2_2", "v1"], ["v4"], None),
                network.NetworkNode("5", ["v3", "v4"], ["o1"], None),
                network.NetworkNode("6", ["o1", "v4"], ["o2"], None),
            ]
        ), # End of definition of the 3rd case
    ]
)
def test_build_network(input_list, edge_node_name, network_nodes):
    graph = nx.DiGraph()
    
    for node in network_nodes:
        graph.add_node(node.name, node=node)

    network.NetworkBuilder(graph).build(input_list)
    for source_name in graph:
        if source_name == edge_node_name:
            continue
        path = nx.all_simple_paths(graph, source=source_name, target=edge_node_name)
        assert len(list(path)) != 0, "Source:{}, Destination:{}, Edges:{}".format(source_name, edge_node_name, graph.edges)


def dummy_process(*x):
    return [ x[0] ] # Always return value is list object.

class DummyProcess(network.IterableProcessor):
    def __init__(self):
        self.x = []
        self.generator = iter(range(10))
        self.next()

    def __call__(self, *x):
        return self.current_value

    def next(self):
        self.current_value = next(self.generator)

    def insert(self, x):
        self.x.append(x)

    def get_output(self):
        return [ self.x ] # Always output is list object.

@pytest.mark.parametrize( ("input_list", "edge_node_name", "network_nodes", "comp"),
    [
        (# Begin of definition of the 1st case
            {"i1": "i1" },
            "6",
            [
                network.NetworkNode("1", ["i1"], ["v1"], dummy_process),
                network.NetworkNode("2", ["v1"], ["v2"], DummyProcess(), iterate_from="4"),
                network.NetworkNode("3-1", ["v2"], ["v3-1"], dummy_process),
                network.NetworkNode("3-2", ["v1"], ["v3-2"], dummy_process),
                network.NetworkNode("4", ["v3-1", "v3-2"], ["v4"], dummy_process),
                network.NetworkNode("5", ["v4"], ["o1"], dummy_process),
            ],
            lambda x: any([ i == j for i, j in enumerate(x) ])
        ), # End of definition of the 1st case
    ]
)
def test_iterable_node(input_list, edge_node_name, network_nodes, comp):
    manager = network.NetworkManager(input_list.keys())
    
    for node in network_nodes:
        manager.add(node)
    
    manager(**input_list)
    assert comp(manager.variables['o1'])
    
    


if __name__ == '__main__':
    pytest.main()
