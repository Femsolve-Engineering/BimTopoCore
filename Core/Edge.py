
from typing import Tuple
from typing import List

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Vertex, TopoDS_Edge
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopTools import TopTools_MapOfShape
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Curve, Geom_Geometry
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_EdgeDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_PointProjectionFailed
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_ParameterOutOfRange
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_DifferentPointsOnClosedCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_PointWithInfiniteParameter
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_DifferentsPointAndParameter
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import EdgeEnd, TopologyTypes
from Core.Factories.AllFactories import EdgeFactory

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
        self.register_factory(self.get_class_guid(), EdgeFactory())

    def adjacent_edges(self, host_topology: Topology) -> 'List[Edge]':
        """Calculates and returns the edges adjacent to this edge.

        Args:
            host_topology (Topology): _description_

        Returns:
            List[Edge]: List of adjacent edges
        """
        from Core.Vertex import Vertex
        collected_adjacent_edges: List[Edge] = []

        # Returns start and end vertex only
        vertices: 'List[Vertex]' = self.vertices()
        for vertex in vertices:
            edges: 'List[Edge]' = vertex.edges(host_topology)
            
            for edge in edges:
                if not self.is_same(edge) and edge not in collected_adjacent_edges:
                    new_constructed_edge = topods.Edge(edge.get_occt_shape())
                    collected_adjacent_edges.append(new_constructed_edge)

    @staticmethod
    def by_curve(
        occt_poles: 'List[gp_Pnt]', 
        occt_weights: 'List[float]', 
        occt_knots: 'List[float]', 
        occt_multiplicities: 'List[int]', 
        degree: int, 
        is_periodic: bool, 
        is_rational: bool) -> 'Edge':
        """Creates an Edge by NURBS curve parameters

        Args:
            occt_poles (List[gp_Pnt]): The OCCT poles
            occt_weights (List[float]): The weights
            occt_knots (List[float]): The knots
            occt_multiplicities (List[int]): The knots' multiplicities
            degree (int): degree
            is_periodic (bool): The curve's periodic status
            is_rational (bool): The curve's rational status

        Raises:
            RuntimeError: If spline curve creation failed
            RuntimeError: Setting spline curve to periodic failed.

        Returns:
            Edge: _description_
        """
        try:
            occt_bspline_curve = Geom_BSplineCurve(
                occt_poles, 
                occt_weights, 
                occt_knots, 
                occt_multiplicities, 
                degree, 
                False, 
                is_rational)
        except Exception as e:
            raise RuntimeError(str(e))

        if is_periodic:
            try:
                occt_bspline_curve.SetPeriodic()
            except Exception as e:
                raise RuntimeError(str(e))
            
        return Edge.by_curve(occt_bspline_curve)
    
    def wires(self, host_topology: Topology) -> List[Topology]:
        """
        Returns the Wires incident to the edge.
        """
        if not host_topology.is_null_shape():
            return self.upward_navigation(host_topology.get_occt_shape())
        else:
            raise RuntimeError("Host Topology cannot be NULL when searching for ancestors.")

    def faces(self, host_topology: Topology) -> List[Topology]:
        """
        Returns the Faces incident to the edge.
        """
        if not host_topology.is_null_shape():
            return self.upward_navigation(host_topology.get_occt_shape())
        else:
            raise RuntimeError("Host Topology cannot be NULL when searching for ancestors.")
        
    @staticmethod
    def by_curve(occt_curve: Geom_Curve, first_parameter: float = 0.0, last_parameter: float = 1.0):
        """Creates an Edge by an OCCT Curve and the minimum and maximum parameters.

        Args:
            occt_curve (Geom_Curve): The underlying Curve
            first_parameter (float): The first parameter, ranging between 0 and 1.
            last_parameter (float): The second parameter, ranging between 0 and 1.
                Must be larger than first_parameter, otherwise they will be swapped

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        occt_first_parameter = occt_curve.FirstParameter()
        occt_last_parameter = occt_curve.LastParameter()
        occt_delta_parameter = occt_last_parameter - occt_first_parameter

        occt_parameter1 = occt_first_parameter + first_parameter * occt_delta_parameter
        occt_parameter2 = occt_first_parameter + last_parameter * occt_delta_parameter

        occt_make_edge = BRepBuilderAPI_MakeEdge(occt_curve, occt_parameter1, occt_parameter2)
        if occt_make_edge.Error() != BRepBuilderAPI_MakeEdge.IsDone():
            Edge.throw(occt_make_edge.Error())

        occt_fixed_edge = Topology.fix_shape(occt_make_edge) 
        edge_instance = Edge(occt_fixed_edge) 
        
        # ToDo?: Replace with appropriate global cluster addition mechanism
        # GlobalCluster.get_instance().add_topology(edge_instance.get_occt_edge())

        return edge_instance


    @staticmethod
    def by_start_vertex_end_vertex(start_vertex: 'Vertex', end_vertex: 'Vertex') -> "Edge":
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
            
            if occt_edge.Error() != BRepBuilderAPI_EdgeDone:
                Edge.throw(occt_edge.Error())
            
            new_edge = Edge(occt_edge.Shape())
            return new_edge
        else:
            print('One of the vertices is not valid or is null!')
            return None
        
    def shared_vertices(self, another_edge: 'Edge') -> 'List[Vertex]':
        """
        Find shared vertices between this edge and another edge.

        Parameters:
            another_edge (Edge): The other edge to check against.

        Returns:
            list: List of shared Vertex objects.
        """

        from Core.Vertex import Vertex
    
        occt_vertices1: TopTools_MapOfShape = Topology.static_downward_navigation(self.get_occt_shape(), TopAbs_VERTEX)
        occt_vertices2: TopTools_MapOfShape = Topology.static_downward_navigation(another_edge.get_occt_shape(), TopAbs_VERTEX)

        shared_vertices: 'List[Vertex]' = []
        
        occt_vertex_iterator1 = occt_vertices1.cbegin()
        while occt_vertex_iterator1.More():

            occt_vertex_iterator2 = occt_vertices2.cbegin()
            while occt_vertex_iterator2.More():

                if occt_vertex_iterator1.Value().IsSame(occt_vertex_iterator2.Value()):
                    shared_vertex = Vertex(topods(occt_vertex_iterator1.Value()))
                    shared_vertices.append(shared_vertex)

                occt_vertex_iterator2.Next()
            occt_vertex_iterator1.Next()

        return shared_vertices
    
    def is_manifold(self, host_topology: Topology) -> bool:
        """
        Returns true if the edge is manifold, otherwise false.
        """
        # In the context of a Wire, an Edge is non-manifold if it connects more than two edges.
        if host_topology.get_shape_type() == TopologyTypes.WIRE:
            edges = self.adjacent_edges(host_topology)
            if len(edges) > 2:
                return False

        # In the context of a Face, an Edge is non-manifold if it connects more than two edges.
        if host_topology.get_shape_type() == TopologyTypes.FACE:
            edges = self.adjacent_edges(host_topology)
            if len(edges) > 2:
                return False
            
        # In the context of a Shell, an Edge is non-manifold if it connects more than one Face.
        if host_topology.get_shape_type() == TopologyTypes.SHELL:
            faces = self.faces(host_topology)
            if len(faces) > 1:
                return False
            
        # In the context of a Cell, an Edge is non-manifold if it connects more than two Faces.
        if host_topology.get_shape_type() == TopologyTypes.CELL:
            faces = self.faces(host_topology)
            if len(faces) > 2:
                return False
            
        # In the context of a CellComplex, an Edge is non-manifold if it connects more than one Cell.
        if host_topology.get_shape_type() == TopologyTypes.CELLCOMPLEX:
            cells = self.cells(host_topology)
            if len(cells) > 1:
                return False

        # In the context of a Cluster, Check all the SubTopologies
        if host_topology.get_shape_type() == TopologyTypes.CLUSTER:

            cell_complexes = self.cell_complexes(host_topology)
            for cell_complex in cell_complexes:
                if self.is_manifold(cell_complex):
                    return False

        return True
        
    def vertices(self) -> list:
        """Returns a tuple of vertices.

        Returns:
            list: [start_vertex, end_vertex]
        """
        return [self.start_vertex(), self.end_vertex()]

    def start_vertex(self) -> 'Vertex':
        """Getter for start vertex.

        Returns:
            Vertex: new Vertex object
        """
        from Core.Vertex import Vertex
        occt_vertex = self.__get_vertex_at_end(self.base_shape_edge, EdgeEnd.START)
        return Vertex(occt_vertex)
    
    def end_vertex(self) -> 'Vertex':
        """Getter for end vertex.

        Returns:
            Vertex: new Vertex object
        """
        from Core.Vertex import Vertex
        occt_vertex = self.__get_vertex_at_end(self.base_shape_edge, EdgeEnd.END)
        return Vertex(occt_vertex)
    
    def curve(self) -> Tuple[Geom_Curve, float, float]:
        """Quick getter for the geometry of the curve.

        Returns:
            Geom_Curve: The geometric curve representing the edge.
        """
        return BRep_Tool.Curve(self.get_occt_edge())
    
    def geometry(self) -> 'List[Geom_Geometry]':
        """
        Returns:
            List[Geom_Geometry]: List of the single 'Geom_Curve' entity
        """
        return [self.curve()]
    
    def set_occt_shape(self, occt_shape: TopoDS_Shape) -> None:
        """
        Generic setter for underlying OCCT shape.
        """
        try:
            self.set_occt_edge(topods.Edge(occt_shape))
        except Exception as ex:
            raise RuntimeError(str(ex.args))

    def get_occt_edge(self) -> TopoDS_Edge:
        """
        Getter for the underlying OCCT edge.
        """
        if self.base_shape_edge.IsNull():
            raise RuntimeError("A null Edge is encountered!")
        
        return self.base_shape_edge
    
    def set_occt_edge(self, occt_edge: TopoDS_Edge) -> None:
        """
        Setter for underlying OCCT edge.
        """
        self.base_shape_edge = occt_edge
        self.base_shape = occt_edge

    @staticmethod
    def normalize_parameter(
        occt_first_parameter: float, 
        occt_last_parameter: float, 
        non_normalized_parameter: float) -> float:
        """Normalize the parameters. 
        (OCCT uses non-normalized parameters, while Topologic uses normalized parameters)

        Args:
            occt_first_parameter (float): The first OCCT parameter
            occt_last_parameter (float): The last OCCT parameter
            non_normalized_parameter (float): A non-normalized parameter

        Raises:
            ValueError: If the last parameter is less than the first parameter

        Returns:
            float: A normalized parameter
        """
        occt_d_parameter = occt_last_parameter - occt_first_parameter
        if occt_d_parameter <= 0.0:
            raise ValueError("Negative range")
        
        return (non_normalized_parameter - occt_first_parameter) / occt_d_parameter

    @staticmethod
    def non_normalize_parameter(
        occt_first_parameter: float, 
        occt_last_parameter: float, 
        normalized_parameter: float) -> float:
        """Non-normalize the parameters. 
        (OCCT uses non-normalized parameters, while Topologic uses normalized parameters)

        Args:
            occt_first_parameter (float):  The first OCCT parameter
            occt_last_parameter (float): The last OCCT parameter
            normalized_parameter (float): A non-normalized parameter

        Returns:
            float: A non-normalized parameter
        """
        occt_d_parameter = occt_last_parameter - occt_first_parameter
        return occt_first_parameter + normalized_parameter * occt_d_parameter
    
    @staticmethod
    def occt_shape_fix(input_edge: TopoDS_Edge) -> TopoDS_Edge:
        """
        Fixes the input OCCT edge, returns the fixed edge.
        """
        edge_fix = ShapeFix_Shape(input_edge)
        edge_fix.Perform()
        return topods.Edge(edge_fix.Shape())
        
    def center_of_mass(self):
        """
        Returns the center of mass of this Edge.
        """
        occt_center_of_mass = self.__center_of_mass(self.get_occt_edge())
        return Topology.by_occt_shape(occt_center_of_mass)

    @staticmethod
    def __center_of_mass(occt_edge: TopoDS_Edge):
        """
        Returns the center of mass of this an OCCT Edge.
        """
        occt_shape_properties = GProp_GProps()
        brepgprop_LinearProperties(occt_edge, occt_shape_properties)
        return BRepBuilderAPI_MakeVertex(occt_shape_properties.CentreOfMass()).Vertex()
    
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
        
    @staticmethod
    def throw(occt_edge_error: int) -> None:
        """
        Adds context to OCC thrown errors.
        """
        if occt_edge_error == BRepBuilderAPI_PointProjectionFailed:
            raise ValueError("No parameters were given but the projection of the 3D points on the curve failed. This happens when the point distance to the curve is greater than the precision value.")
        
        elif occt_edge_error == BRepBuilderAPI_ParameterOutOfRange:
            raise ValueError("The given parameters are not in the parametric range.")
        
        elif occt_edge_error == BRepBuilderAPI_DifferentPointsOnClosedCurve:
            raise ValueError("The two vertices or points are the extremities of a closed curve but have different locations.")
        
        elif occt_edge_error == BRepBuilderAPI_PointWithInfiniteParameter:
            raise ValueError("A finite coordinate point was associated with an infinite parameter.")
        
        elif occt_edge_error == BRepBuilderAPI_DifferentsPointAndParameter:
            raise ValueError("The distance between the 3D point and the point evaluated on the curve with the parameter is greater than the precision.")
        
        else: # BRepBuilderAPI.BRepBuilderAPI_LineThroughIdenticPoints:
            raise ValueError("Two identical points were given to define a line (construction of an edge without curve).")
        
    def is_container_type(self) -> bool:
        """
        Determines if this topology is container type.
        Container type = stores multiple subshapes.
        """
        return False
    
    def get_type(self) -> TopologyTypes:
        """
        Returns:
            TopologyTypes: Internal definition for types.
        """
        return TopologyTypes.EDGE
    
    def get_type_as_string(self) -> str:
        """
        Returns the name of the type.
        """
        return 'Edge'