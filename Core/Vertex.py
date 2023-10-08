# Base
from typing import List

# OCC
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Shape
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_MapIteratorOfMapOfShape
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.Geom import Geom_CartesianPoint, Geom_Point
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.gp import gp_Pnt

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes

class Vertex(Topology):
    """
    Represents a 1D vertex object. Serves as a wrapper around 
    TopoDS_VERTEX entity of OCC.
    """
    def __init__(self, occt_vertex: TopoDS_Vertex):
        """Constructor saves shape and processes GUID.

        Args:
            occt_vertex (TopoDS_Vertex): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """
        super().__init__(occt_vertex, TopologyTypes.VERTEX)
        self.base_shape_vertex = occt_vertex

    def adjacent_vertices(self, host_topology: Topology) ->  'List[Vertex]':
        """
        Returns:
            List[Vertex]: the Vertices adjacent to the Vertex.
        """

        # Find the constituent edges
        occt_adjacent_vertices = TopTools_MapOfShape()
        edges = []

        if host_topology:
            explorer = TopExp_Explorer(host_topology.GetOcctShape(), TopAbs_EDGE)
            while explorer.More():
                edge_shape = explorer.Current()
                edges.append(TopoDS_Shape(edge_shape))
                explorer.Next()
        else:
            raise RuntimeError("Host Topology cannot be None when searching for ancestors.")

        for edge in edges:
            vertices = []
            kpEdge = TopoDS_Shape(edge)
            explorer = TopExp_Explorer(kpEdge, TopAbs_VERTEX)
            while explorer.More():
                vertex_shape = explorer.Current()
                vertices.append(TopoDS_Vertex(vertex_shape))
                explorer.Next()

            for vertex in vertices:
                if not self.is_same(vertex):  # Assuming IsSame is defined elsewhere
                    occt_adjacent_vertices.Add(vertex)

        r_adjacent_vertices = []
        occt_adjacent_vertex_iterator = TopTools_MapIteratorOfMapOfShape(occt_adjacent_vertices)
        while occt_adjacent_vertex_iterator.More():
            vertex_shape = occt_adjacent_vertex_iterator.Value()
            r_adjacent_vertices.append(TopoDS_Vertex(vertex_shape))
            occt_adjacent_vertex_iterator.Next()

        return r_adjacent_vertices

    def x(self) -> float:
        """Getter of X-coordinate for the vertex.

        Returns:
            float: X-coordinate
        """
        geom_cartesian_point = self.__get_point()
        return geom_cartesian_point.X()
    
    def y(self) -> float:
        """Getter of Y-coordinate for the vertex.

        Returns:
            float: Y-coordinate
        """
        geom_cartesian_point = self.__get_point()
        return geom_cartesian_point.Y()

    def z(self) -> float:
        """Getter of Z-coordinate for the vertex.

        Returns:
            float: Z-coordinate
        """
        geom_cartesian_point = self.__get_point()
        return geom_cartesian_point.Z()
    
    def is_manifold(self, host_topology: Topology):
        """
        For now, in the context of an Edge, a Vertex is considered always manifold.
        TODO: Vertex is considered manifold if it is the start or end vertex.
        """

        if host_topology.get_shape_type() == TopologyTypes.EDGE:
            return True

        # In the context of a Wire, a Vertex is non-manifold if it connects more than two edges.
        if host_topology.get_shape_type() == TopologyTypes.WIRE:
            edges = []
            self.Edges(host_topology, edges)
            if len(edges) > 2:
                return False

        # In the context of a Face, a Vertex is non-manifold if it connects more than two edges.
        if host_topology.get_shape_type() == TopologyTypes.FACE:
            edges = []
            self.Edges(host_topology, edges)
            if len(edges) > 2:
                return False

        # In the context of a Shell, a Vertex is non-manifold if it connects more than two Faces.
        if host_topology.get_shape_type() == TopologyTypes.SHELL:
            faces = []
            self.Faces(host_topology, faces)
            if len(faces) > 2:
                return False

        # In the context of a Cell, a Vertex is non-manifold if it connects more than two Faces.
        if host_topology.get_shape_type() == TopologyTypes.CELL:
            faces = []
            self.Faces(host_topology, faces)
            if len(faces) > 2:
                return False

        # In the context of a CellComplex, a Vertex is non-manifold if it connects more than one Cell.
        if host_topology.get_shape_type() == TopologyTypes.CELLCOMPLEX:
            cells = []
            self.Cells(host_topology, cells)
            if len(cells) > 1:
                return False

        # In the context of a Cluster, Check all the SubTopologies
        if host_topology.get_shape_type() == TopologyTypes.TOPOLOGY_CLUSTER:
            cellComplexes = []
            host_topology.CellComplexes(None, cellComplexes)
            for kpCellComplex in cellComplexes:
                if not self.IsManifold(kpCellComplex):
                    return False

            cells = []
            for kpCell in cells:
                if not self.IsManifold(kpCell):
                    return False

            shells = []
            for kpShell in shells:
                if not self.IsManifold(kpShell):
                    return False

            faces = []
            for kpFace in faces:
                if not self.IsManifold(kpFace):
                    return False

            wires = []
            for kpWire in wires:
                if not self.IsManifold(kpWire):
                    return False

        return True


    def __get_point(self) -> Geom_CartesianPoint:
        """Gets OCC's Geom_Point object.

        Returns:
            Geom_CartesianPoint: OCC Cartesian point object
        """
        geometric_point = BRep_Tool.Pnt(self.base_shape_vertex)
        return Geom_CartesianPoint(geometric_point)

    @staticmethod
    def by_coordinates(kx: float, ky: float, kz: float) -> "Vertex":
        """
        Static factory method to create a new Vertex by coordinates.

        Args:
            kx (float): X-coordinate of the constructable vertex
            ky (float): Y-coordinate of the constructable vertex
            kz (float): Z-coordinate of the constructable vertex

        Returns:
            Vertex: new Vertex topology object
        """
        occt_pnt = gp_Pnt(kx,ky,kz)
        occt_vertex_builder = BRepBuilderAPI_MakeVertex(occt_pnt)
        occt_vertex = occt_vertex_builder.Vertex()
        occt_fixed_vertex = Topology.fix_shape(occt_vertex)
        new_vertex = Vertex(occt_fixed_vertex)
        return new_vertex
    
    @staticmethod
    def by_point(occt_geom_point: Geom_Point) -> 'Vertex':
        """
        Returns:
            A vertex constructed using an Occt geom point.
        """
        occt_vertex = BRepBuilderAPI_MakeVertex(occt_geom_point.Pnt()).Vertex()
        occt_fixed_vertex = Topology.fix_shape(occt_vertex)
        new_vertex = Vertex(occt_fixed_vertex)
        return new_vertex

        




