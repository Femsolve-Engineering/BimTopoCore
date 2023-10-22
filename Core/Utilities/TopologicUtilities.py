
from typing import List

from Core.Topology import Topology
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Face import Face
from Core.Cell import Cell
from Core.TopologyConstants import TopologyTypes

class VertexUtility:

    @staticmethod
    def adjacent_edges(
        vertex: 'Vertex', 
        parent_topology: 'Topology') -> List['Edge']:
        """
        TODO
        """
        core_adjacent_edges: List[Edge] = []
        core_adjacent_topologies: List[Topology] = vertex.upward_navigation(
            parent_topology.get_occt_shape(), 
            TopologyTypes.EDGE) 
        
        for adjacent_topology in core_adjacent_topologies:
            # ToDo: Check this if this is correct
            core_adjacent_edges.append(Edge(adjacent_topology.get_occt_shape()))

        return core_adjacent_edges
    
class FaceUtility:

    @staticmethod
    def adjacent_cells(face: 'Face', parent_topology: 'Topology') -> List['Cell']:
        """
        TODO
        """
        ret_cells: List['Cell'] = []
        adjacent_topologies: List['Topology'] = face.upward_navigation(
            parent_topology.get_occt_shape(),
            TopologyTypes.CELL)
        
        for adj_top in adjacent_topologies:
            # Here we should downcast to Cell
            ret_cells.append(Cell(adj_top.get_occt_shape()))
