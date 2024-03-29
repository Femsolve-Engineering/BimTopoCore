# Base
from typing import List, Tuple

# OCC
from OCC.Core.TopoDS import topods, TopoDS_Vertex, TopoDS_Shape
from OCC.Core.TopTools import TopTools_MapOfShape
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.Geom import Geom_CartesianPoint, Geom_Point, Geom_Geometry
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_ListOfShape
from OCC.Core.gp import gp_Pnt


# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.Factories.AllFactories import VertexFactory

from Core.Edge import Edge

class Vertex(Topology):

    """
    Represents a 1D vertex object. Serves as a wrapper around 
    TopoDS_VERTEX entity of OCC.
    """
    def __init__(self, occt_vertex: TopoDS_Vertex, guid: str = ""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_vertex (TopoDS_Vertex): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """
        # ctor_guid = self.get_class_guid() if guid == "" else guid
        instance_topology = super().__init__(occt_vertex, TopologyTypes.VERTEX)
        self.base_shape_vertex = occt_vertex
        self.register_factory(self.get_class_guid(), VertexFactory())

        # Register the instances

        occt_vertex.__setattr__('id', Topology.topologic_entity_count)

        # Topology.shape_to_id[occt_vertex] = Topology.topologic_entity_count

        Topology.shapeID_to_topology[occt_vertex.id] = instance_topology
        Topology.topology_to_shapeID[instance_topology] = occt_vertex.id

        # Topology.topology_to_subshape[instance_topology] = occt_vertex.HashCode(10**14)
        # Topology.subshape_to_topology[occt_vertex.HashCode(10**14)] = instance_topology

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
    
    def edges(self, host_topology: Topology) -> List[Edge]:
        """
        Looks up what edges contain this vertex.

        Args:
            host_topology (Topology): 

        Returns:
            List[Edge]: The edges containing this vertex as a constituent member.
        """
        if not host_topology.is_null_shape():
            print('REMINDER!!! Vertex.edges() -> Base Topology method of upward_navigation() is not yet correctly implemented.')
            return self.upward_navigation(host_topology.get_occt_shape(), TopologyTypes.EDGE)
        else:
            raise RuntimeError("Host Topology cannot be NULL when searching for ancestors.")
    
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
            edges.append(self.edges(host_topology))
            if len(edges) > 2:
                return False

        # In the context of a Face, a Vertex is non-manifold if it connects more than two edges.
        if host_topology.get_shape_type() == TopologyTypes.FACE:
            edges = []
            edges.append(self.edges(host_topology))
            if len(edges) > 2:
                return False

        # In the context of a Shell, a Vertex is non-manifold if it connects more than two Faces.
        if host_topology.get_shape_type() == TopologyTypes.SHELL:
            faces = []
            self.faces(host_topology, faces)
            if len(faces) > 2:
                return False

        # In the context of a Cell, a Vertex is non-manifold if it connects more than two Faces.
        if host_topology.get_shape_type() == TopologyTypes.CELL:
            faces = []
            self.faces(host_topology, faces)
            if len(faces) > 2:
                return False

        # In the context of a CellComplex, a Vertex is non-manifold if it connects more than one Cell.
        if host_topology.get_shape_type() == TopologyTypes.CELLCOMPLEX:
            cells = []
            self.cells(host_topology, cells)
            if len(cells) > 1:
                return False

        # In the context of a Cluster, Check all the SubTopologies
        if host_topology.get_shape_type() == TopologyTypes.CLUSTER:
            cellComplexes = []
            host_topology.cell_complexes(None, cellComplexes)
            for kpCellComplex in cellComplexes:
                if not self.is_manifold(kpCellComplex):
                    return False

            cells = []
            for kpCell in cells:
                if not self.is_manifold(kpCell):
                    return False

            shells = []
            for kpShell in shells:
                if not self.is_manifold(kpShell):
                    return False

            faces = []
            for kpFace in faces:
                if not self.is_manifold(kpFace):
                    return False

            wires = []
            for kpWire in wires:
                if not self.is_manifold(kpWire):
                    return False

        return True

    def adjacent_vertices(self, host_topology: Topology) ->  'List[Vertex]':
        """
        Returns:
            List[Vertex]: the Vertices adjacent to the Vertex.
        """

        # Find the constituent edges
        occt_adjacent_vertices = TopTools_MapOfShape()
        edges = []

        if host_topology:
            explorer = TopExp_Explorer(host_topology.get_occt_shape(), TopAbs_EDGE)
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
                vertices.append(topods.Vertex(vertex_shape))
                explorer.Next()

            for vertex in vertices:
                if not self.is_same(vertex):  # Assuming IsSame is defined elsewhere
                    occt_adjacent_vertices.Add(vertex)

        adjacent_vertices = []
        current_iterator = occt_adjacent_vertices.cbegin()
        end_iterator = occt_adjacent_vertices.cend()
        while current_iterator != end_iterator:
            vertex_shape = current_iterator.Value()
            adjacent_vertices.append(topods.Vertex(vertex_shape))
            current_iterator.Next()

        return adjacent_vertices
    
    def center_of_mass(self) -> 'Vertex':
        """
        Returns:
            Fixed vertex point center of mass
        """
        occt_vertex = Vertex.center_of_mass(self.get_occt_vertex())
        new_vertex = Topology.by_occt_shape(occt_vertex)
        return new_vertex
    
    def geometry(self) -> List[Geom_Geometry]:
        """
        Virtual method.
        Returns:
            List of geometric entities.
        """
        occt_geometries = [self.get_point()]
        return occt_geometries
    
    def get_occt_shape(self) -> TopoDS_Shape:
        """
        Virtual method.
        Returns:
            TopoDS_Shape of vertex.
        """
        return self.get_occt_vertex()
    
    def set_occt_shape(self, new_shape: TopoDS_Shape) -> None:
        """
        Sets the TopoDS_Shape representing the vertex.
        """
        try:
            self.set_occt_vertex(topods.Vertex(new_shape))
        except Exception as ex:
            raise RuntimeError(ex.args)
        
    def get_occt_vertex(self) -> TopoDS_Vertex:
        """
        Returns:
            Underlying OCCT vertex object.
        """

        if self.base_shape_vertex.IsNull():
            raise RuntimeError('A null Vertex is encountered!')
        else: return self.base_shape_vertex

    def set_occt_vertex(self, occt_vertex: TopoDS_Vertex) -> None:
        """
        Sets the underlying TopoDS_Vertex.
        """
        self.base_shape_vertex = occt_vertex
        self.base_shape = self.base_shape_vertex

    def x(self) -> float:
        """Getter of X-coordinate for the vertex.

        Returns:
            float: X-coordinate
        """
        geom_cartesian_point = self.get_point()
        return geom_cartesian_point.X()
    
    def y(self) -> float:
        """Getter of Y-coordinate for the vertex.

        Returns:
            float: Y-coordinate
        """
        geom_cartesian_point = self.get_point()
        return geom_cartesian_point.Y()

    def z(self) -> float:
        """Getter of Z-coordinate for the vertex.

        Returns:
            float: Z-coordinate
        """
        geom_cartesian_point = self.get_point()
        return geom_cartesian_point.Z()
    
    def coordinates(self) -> Tuple[float, float, float]:
        """
        Returns:
            Tuple (triplet) of x,y,z coordinates.
        """
        geom_point = self.get_point()
        return (geom_point.X(), geom_point.Y(), geom_point.Z())

    def get_point(self) -> Geom_CartesianPoint:
        """Gets OCC's Geom_Point object.

        Returns:
            Geom_CartesianPoint: OCC Cartesian point object
        """
        geometric_point = BRep_Tool.Pnt(self.base_shape_vertex)
        return Geom_CartesianPoint(geometric_point)
        
    @staticmethod
    def center_of_mass(rkOcctVertex: TopoDS_Vertex) -> TopoDS_Vertex:
        """
        Static method to get center of mass for OCCT point.
        Returns:
            Fixed vertex point center of mass.
        """
        pnt = BRep_Tool.Pnt(rkOcctVertex)
        occt_center_of_mass = BRepBuilderAPI_MakeVertex(pnt).Vertex()
        occt_fixed_center_of_mass = topods.Vertex(Topology.fix_shape(occt_center_of_mass))
        return occt_fixed_center_of_mass
    
    def is_container_type(self) -> bool:
        """
        Determines if this topology is container type.
        """
        return False
    
    def get_type(self) -> TopologyTypes:
        """
        Returns:
            TopologyTypes: Internal definition for types.
        """
        return TopologyTypes.VERTEX
    
    def get_type_as_string(self) -> str:
        """
        Returns:
            Type name as string
        """
        return 'Vertex'