
from typing import Tuple
from typing import List

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Vertex, TopoDS_Edge, TopoDS_Wire, topods
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Curve, Geom_Geometry
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_EdgeDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_EmptyWire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_DisconnectedWire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NonManifoldWire
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import EdgeEnd, TopologyTypes
from Core.Factories.AllFactories import WireFactory
from Core.Utilities.TopologicUtilities import VertexUtility

class Wire(Topology):
    """
    Represents a collection of edges in the 3D space. 
    Serves as a wrapper around TopoDS_WIRE entity of OCC.
    """
    def __init__(self, occt_wire: TopoDS_Wire, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_wire (TopoDS_Wire): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        super().__init__(occt_wire, TopologyTypes.WIRE)
        self.base_shape_wire = occt_wire
        self.register_factory(self.get_class_guid(), WireFactory())

    def edges(self, host_topology: 'Topology') -> List['Edge']:
        """
        Retrieves the edges of the current wire.
        """
        from Core.Edge import Edge

        ret_edges: List[Edge] = []
        if not self.is_manifold():
            # Gives in any order
            ret_edges = self.downward_navigation()
        else:
            # This only works for manifold wire with a flow
            vertices = self.downward_navigation()  # Assuming Vertex is a class

            if not vertices:
                return
            
            is_closed = self.is_closed()
            starting_vertex = None

            if is_closed:
                # any vertex
                starting_vertex = vertices[0]
            else:
                for vertex in vertices:
                    adjacent_edges: List['Edge'] = VertexUtility.adjacent_edges(vertex, self, adjacent_edges)

                    if len(adjacent_edges) == 1:
                        edge_start_vertex = adjacent_edges[0].start_vertex()

                        if edge_start_vertex.is_same(vertex):
                            # This vertex needs to be the start vertex of this edge
                            starting_vertex = vertex
                            break

                if starting_vertex is None:
                    raise RuntimeError("This Wire is closed, but is identified as an open Wire.")
            
            # Get an adjacent edge
            current_vertex = starting_vertex
            previous_edge = None

            while True:
                adjacent_edges = VertexUtility.adjacent_edges(current_vertex, self, adjacent_edges)
                current_edge = None

                for adjacent_edge in adjacent_edges:
                    if previous_edge is None:
                        if adjacent_edge.start_vertex().is_same(current_vertex):
                            current_edge = adjacent_edge
                    elif not previous_edge.is_same(adjacent_edge):
                        current_edge = adjacent_edge
                
                # Still null? break. This happens in the open wire when the last Vertex is visited.
                if current_edge is None:
                    break
                
                ret_edges.append(current_edge)
                previous_edge = current_edge

                # Get the other vertex
                edge_vertices = []
                current_edge.vertices(None, edge_vertices)

                for vertex in edge_vertices:
                    if vertex.is_same(current_vertex):
                        continue

                    current_vertex = vertex
                    break
                
                # If the starting vertex is revisited, break. This happens in the open wire.
                if current_vertex.is_same(starting_vertex):
                    break
        
        return ret_edges

    def faces(self, host_topology: Topology) -> List[Topology]:
        """
        Returns the Faces incident to the edge.
        """
        if not host_topology.is_null_shape():
            return self.upward_navigation(host_topology.get_occt_shape())
        else:
            raise RuntimeError("Host Topology cannot be NULL when searching for ancestors.")
        
    def is_closed(self) -> bool:
        """
        Determines if the wire is closed.
        """
        check_wire = BRepCheck_Wire(topods.Wire(self.base_shape_wire))
        status = check_wire.Closed()
        return status == BRepCheck_NoError
    
    def edges(self) -> List['Edge']:
        """
        Returns a list of Edges that comprise the wire.
        """
        from Core.Vertex import Vertex
        from Core.Edge import Edge

        if not self.is_manifold():
            # Gives in any order
            return self.downward_navigation()
        else:
            # This only works for manifold wire with a flow
            ret_edges: List[Edge] = []
            vertices: List[Vertex] = Topology.downward_navigation(self.get_occt_shape(), TopAbs_VERTEX)
            if len(vertices):
                return []
            
            is_closed = self.is_closed()
            starting_vertex = None
            if is_closed:
                starting_vertex = vertices[0]
            else:
                for vertex in vertices:
                    adjacent_edges = VertexUtility.adjacent_edges(vertex, self)

                    if len(adjacent_edges) == 1:

                        edge_start_vertex: Vertex = adjacent_edges[0].start_vertex()

                        if edge_start_vertex.is_same(vertex):
                            starting_vertex = vertex
                            break

                if starting_vertex == None:
                    raise RuntimeError("This Wire is closed, but is identified as an open Wire.")
    
        # Get an adjacent edge
        current_vertex = starting_vertex
        previous_edge: Edge = None

        while True:

            adjacent_edges = VertexUtility.adjacent_edges(current_vertex, self)
            current_edge: Edge = None
            for adjacent_edge in adjacent_edges:
                if previous_edge == None:
                    tmp_start_vertex: Vertex = adjacent_edge.start_vertex()
                    if tmp_start_vertex.is_same(current_vertex):
                        current_edge = adjacent_edge
                elif not previous_edge.is_same(adjacent_edge):
                    current_edge = adjacent_edge
            
            if current_edge == None:
                break

            ret_edges.append(current_edge)
            previous_edge = current_edge

            (tmp_s_vertex, tmp_e_vertex) = current_edge.vertices()
            vertices.append([tmp_s_vertex, tmp_e_vertex])

            for vertex in vertices:
                vertex: Vertex = vertex # For intellisense
                if vertex.is_same(current_vertex):
                    continue

                current_vertex = vertex
                break

            if current_vertex.is_same(starting_vertex):
                break

    def vertices(self) -> List['Vertex']:
        """
        Returns the list of vertices that comprise the wire
        """
        from Core.Edge import Edge
        from Core.Vertex import Vertex

        occt_vertices: TopTools_MapOfShape = TopTools_MapOfShape()
        edges: List[Edge] = self.edges()

        result_vertices: List[Vertex] = []

        for edge in edges:
            edge_vertices: List[Vertex] = edge.vertices()

            # Special case when handling the second edge
            if len(result_vertices) == 2:
                for vertex in edge_vertices:
                    if vertex.is_same(result_vertices[0]):
                        first_vertex = result_vertices[0]
                        result_vertices.remove(first_vertex)
                        result_vertices.append(first_vertex)
                        break

            for vertex in edge_vertices:
                if not occt_vertices.Contains(vertex.get_occt_shape()):
                    occt_vertices.Add(vertex.get_occt_shape())
                    result_vertices.append(vertex)

        return result_vertices
    
    @staticmethod
    def by_edges(edges: List['Edge'], copy_attributes: bool = False) -> 'Wire':
        """
        TODO!!!! Will need to implement deep copy.
        """
        from Core.Edge import Edge

        if not edges:
            return None
        
        edges: List[Edge] = edges

        occt_edges = TopTools_ListOfShape()
        for edge in edges:
            occt_edges.Append(edge.get_occt_shape())

        occt_wire = Wire.by_occt_edges(occt_edges)
        wire = Wire(occt_wire)
        # ToDo: Will need Topology deep copy implementation
        copy_wire = Wire(occt_wire) # wire.deep_copy()

        if copy_attributes:
            for edge in edges:
                pass # ToDo?: Placeholder for AttributeManager)

        # Assuming there's a similar global cluster setup in Python
        # GlobalCluster.get_instance().add_topology(copy_wire.occt_wire)
        
        return copy_wire
    
    @staticmethod
    def by_occt_edges(occt_edges: TopTools_ListOfShape) -> TopoDS_Wire:
        """
        Construcs a new wire based on the list of edges.

        Args:
            occt_edges (TopTools_ListOfShape): _description_

        Returns:
            TopoDS_Wire: _description_
        """
        occt_make_wire = BRepBuilderAPI_MakeWire()
        occt_make_wire.Add(occt_edges)

        try:
            Topology.transfer_make_shape_contents(occt_make_wire, occt_edges)
            occt_wire = topods.Wire(Topology.fix_shape(occt_make_wire.Shape()))
            return occt_make_wire.Shape()
        except Exception as ex:
            print(f"Failed to construct Wire from edges: {ex.args}")
            Wire.throw(occt_make_wire)
            return TopoDS_Wire()
        
    def is_manifold(self) -> bool:
        """
        Is this wire manifold?
        """
        vertices: List['Topology'] = self.downward_navigation()

        for vertex in vertices:
            edges = vertex.upward_navigation(self.get_occt_wire())
            if len(edges) > 2:
                return False

        return True

    def number_of_branches(self) -> int:
        """
        Returns the number of branches.
        """
        vertices = self.downward_navigation()

        num_of_branches = 0
        for vertex in vertices:
            edges = vertex.upward_navigation(self.get_occt_wire())
            if len(edges) > 2:
                num_of_branches += 1

        return num_of_branches
    
    def geometry(self) -> List[Geom_Geometry]:
        """
        Returns the list of OCC geometric entities that compries the Wire.
        """
        from Core.Edge import Edge
        ret_geometries: List[Geom_Geometry] = []
        edges: List['Edge'] = self.edges()
        for edge in edges:
            ret_geometries.append(edge.curve())

        return ret_geometries

    def set_occt_shape(self, occt_shape: TopoDS_Shape) -> None:
        """
        Generic setter for underlying OCCT shape.
        """
        try:
            self.set_occt_wire(topods.Wire(occt_shape))
        except Exception as ex:
            raise RuntimeError(str(ex.args))

    def get_occt_wire(self) -> TopoDS_Wire:
        """
        Getter for the underlying OCCT wire.
        """
        if self.base_shape_wire.IsNull():
            raise RuntimeError("A null Wire is encountered!")
        
        return self.base_shape_wire
    
    def set_occt_wire(self, occt_wire: TopoDS_Wire) -> None:
        """
        Setter for underlying OCCT wire.
        """
        self.base_shape_wire = occt_wire
        self.base_shape = self.base_shape_wire

    def throw(self, occt_make_wire: BRepBuilderAPI_MakeWire) -> None:
        """
        # The error messages are based on those in the OCCT documentation.
        # https://www.opencascade.com/doc/occt-7.2.0/refman/html/_b_rep_builder_a_p_i___wire_error_8hxx.html
        """
        # The error messages are based on those in the OCCT documentation.
        # https://www.opencascade.com/doc/occt-7.2.0/refman/html/_b_rep_builder_a_p_i___wire_error_8hxx.html

        if occt_make_wire.Error() == BRepBuilderAPI_EmptyWire:
            raise RuntimeError("No initialization of the algorithm. Only an empty constructor was used.")
        
        elif occt_make_wire.Error() == BRepBuilderAPI_DisconnectedWire:
            raise RuntimeError("The last edge which you attempted to add was not connected to the wire.")
        
        elif occt_make_wire.Error() == BRepBuilderAPI_NonManifoldWire:
            raise RuntimeError("The wire has some singularity.")

        # In the case of BRepBuilderAPI_WireDone or any other state, do nothing

    def center_of_mass(self) -> Topology:
        """
        Computes the center of mass for the wire.
        """
        occt_center_of_mass = self.center_of_mass(self.get_occt_wire())
        return Topology.by_occt_shape(occt_center_of_mass)

    @staticmethod
    def center_of_mass(occt_wire: TopoDS_Wire):
        """
        Static method to compute the center of mass for a wire.
        """
        occt_shape_properties = GProp_GProps()
        brepgprop_LinearProperties(occt_wire, occt_shape_properties)
        occt_center_of_mass = BRepBuilderAPI_MakeVertex(occt_shape_properties.CentreOfMass()).Vertex()
        occt_fixed_center_of_mass = Topology.fix_shape(occt_center_of_mass)
        return occt_fixed_center_of_mass

    def is_container_type(self) -> bool:
        """
        Determines if this topology is container type.
        """
        return True
    
    def get_type(self) -> TopologyTypes:
        """
        Returns:
            TopologyTypes: Internal definition for types.
        """
        return TopologyTypes.WIRE

    def get_type_as_string(self) -> str:
        """
        Returns the name of the type.
        """
        return 'Wire'