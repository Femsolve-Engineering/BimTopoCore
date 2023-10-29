
from typing import List

# OCC
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

# BimTopoCore
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
    
    @staticmethod
    def distance(
        vertex: 'Vertex',
        topology: 'Topology') -> float:
        """
        Measures the distance from a vertex to any topology.
        ToDo?: We are using BRepExtrema here, in the legacy code this had 
        a specific implementation for any two different types.
        """

        # ToDo: Need to consider multiple distances.
        brep_extrema = BRepExtrema_DistShapeShape(vertex.get_occt_shape(),topology.get_occt_shape())
        return brep_extrema.Value()
    

class EdgeUtility:

    @staticmethod
    def length(edge: 'Edge') -> float:
        """
        Returns the length of an edge.
        """
        occt_shape_properties = GProp_GProps()
        brepgprop.LinearProperties(edge.get_occt_shape(), occt_shape_properties)
        return occt_shape_properties.Mass()
    
class FaceUtility:

    @staticmethod
    def area(face: TopoDS_Face) -> float:
        """
        Calculates and returns the area of a face.
        """
        occt_shape_properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, occt_shape_properties)
        return occt_shape_properties.Mass()

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
