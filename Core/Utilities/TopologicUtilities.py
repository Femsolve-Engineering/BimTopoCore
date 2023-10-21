
from typing import List

from Core.Topology import Topology
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.TopologyConstants import TopologyTypes

class VertexUtility:

    @staticmethod
    def adjacent_edges(
        vertex: 'Vertex', 
        parent_topology: 'Topology') -> List['Edge']:
        """

        """
        core_adjacent_edges: List[Edge] = []
        core_adjacent_topologies: List[Topology] = vertex.upward_navigation(
            parent_topology.get_occt_shape(), 
            TopologyTypes.EDGE) 
        
        for adjacent_topology in core_adjacent_topologies:
            # ToDo: Check this if this is correct
            core_adjacent_edges.append(Edge(adjacent_topology.get_occt_shape()))

        return core_adjacent_edges
