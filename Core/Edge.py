
from typing import Tuple

# OCC
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import EdgeEnd, TopologyTypes
from Core.Vertex import Vertex

class Edge(Topology):
    """
    Represents a 2D edge object. 
    Serves as a wrapper around TopoDS_EDGE entity of OCC.
    """
    def __init__(self, occt_edge: TopoDS_Edge, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_vertex (TopoDS_Vertex): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        super().__init__(occt_edge, TopologyTypes.EDGE)
        self.base_shape_edge = occt_edge

    @staticmethod
    def by_start_vertex_end_vertex(start_vertex: Vertex, end_vertex: Vertex) -> "Edge":
        """Construct a new edge by start and end vertex.

        Args:
            start_vertex (Vertex): Start vertex
            end_vertex (Vertex): End vertex

        Returns:
            Edge: Constructed edge
        """

        is_start_vertex_ok = start_vertex != None and not start_vertex.is_null_shape()
        is_end_vertex_ok = end_vertex != None and not end_vertex.is_null_shape()

        if is_start_vertex_ok and is_end_vertex_ok:
            occt_edge = BRepBuilderAPI_MakeEdge(
                start_vertex.base_shape_vertex,
                end_vertex.base_shape_vertex)
            
            new_edge = Edge(occt_edge.Shape())
            return new_edge
        else:
            print('One of the vertices is not valid or is null!')
            return None
        
    def vertices(self) -> Tuple[Vertex, Vertex]:
        """Returns a tuple of vertices.

        Returns:
            Tuple[Vertex, Vertex]: (start_vertex, end_vertex)
        """
        return (self.start_vertex(), self.end_vertex())


    def start_vertex(self) -> Vertex:
        """Getter for start vertex.

        Returns:
            Vertex: new Vertex object
        """
        occt_vertex = self.__get_vertex_at_end(self.base_shape_edge, EdgeEnd.START)
        return Vertex(occt_vertex)
    
    def end_vertex(self) -> Vertex:
        """Getter for end vertex.

        Returns:
            Vertex: new Vertex object
        """
        occt_vertex = self.__get_vertex_at_end(self.base_shape_edge, EdgeEnd.END)
        return Vertex(occt_vertex)
    
    @staticmethod
    def __get_vertex_at_end(occt_edge: TopoDS_Edge, requested_edge_end: EdgeEnd) -> TopoDS_Vertex:
        """
        Using ShapeAnalysis class of OCC we determine the start or end vertex.
        It is important to use the ShapeAnalysis class so 'start' and 'end'
        are consistent throughout.

        Args:
            occt_edge (TopoDS_Edge): subject edge
            requested_edge_end (EdgeEnd): which end's vertex do we want?

        Returns:
            TopoDS_Vertex: found vertex at requested edge end
        """

        edge_analyser = ShapeAnalysis_Edge()
        if requested_edge_end == EdgeEnd.START:
            return edge_analyser.FirstVertex(occt_edge)
        elif requested_edge_end == EdgeEnd.END:
            return edge_analyser.LastVertex(occt_edge)
        else:
            raise Exception('Wrong edge end input requested!')