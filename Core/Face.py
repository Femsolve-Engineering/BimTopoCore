
from typing import Tuple
from typing import List

# OCC
from OCC.Core.Precision import precision
# from OCC.Core.StdFail import Standard_Failure
from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge, TopoDS_Iterator, TopoDS_Face, TopoDS_Vertex
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE, TopAbs_REVERSED
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_CartesianPoint, Geom_Surface, Geom_Geometry
from OCC.Core.ShapeAnalysis import shapeanalysis
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_FaceDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NoFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NotPlanar
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_CurveProjectionFailed
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_ParametersOutOfRange
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop, brepgprop_SurfaceProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Copy
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndUniqueAncestors
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_IndexedDataMapOfShapeListOfShape

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.Factories.AllFactories import FaceFactory

class Face(Topology):
    """
    Represents a 2D face object. 
    Serves as a wrapper around TopoDS_Face entity of OCC.
    """
    def __init__(self, occt_face: TopoDS_Face, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_face (TopoDS_Face): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        super().__init__(occt_face, TopologyTypes.FACE)
        self.base_shape_face = occt_face
        self.register_factory(self.get_class_guid(), FaceFactory())

    def adjacent_faces(self, host_topology: Topology):
        """
        Iterates through the edges and finds the incident faces which are not this face.
        """
        # Create an empty list to store the adjacent faces.
        adjacent_faces = []

        # Create a map to store edges and their incident faces.
        occt_edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp_MapShapesAndUniqueAncestors(
            host_topology.get_occt_shape(), 
            TopAbs_EDGE, 
            TopAbs_FACE, 
            occt_edge_face_map)

        # Create a map to store edges.
        occt_edges = TopTools_MapOfShape()

        # Iterate through edges of the current face and add them to the map.
        occt_explorer = TopExp_Explorer(self.get_occt_shape(), TopAbs_EDGE)
        while occt_explorer.More():
            occt_edges.Add(occt_explorer.Current())
            occt_explorer.Next()

        occt_face = self.get_occt_face()
        occt_adjacent_faces = TopTools_MapOfShape()

        # For each edge, find the incident faces and add them 
        # to the map if they're not the same as the current face.
        for occt_edge in occt_edges:
            for incident_face in occt_edge_face_map.FindFromKey(occt_edge):
                if not occt_face.IsSame(incident_face):
                    occt_adjacent_faces.Add(incident_face)

        # Convert the map to a list of faces.
        for face in occt_adjacent_faces:
            adjacent_faces.append(Face(topods.Face(face)))

        return adjacent_faces

    def cells(self, host_topology: Topology) -> List['Cell']:
        """
        Returns the list of cells that contain this face.
        """
        if host_topology != None and not host_topology.is_null_shape():
            return self.upward_navigation(host_topology, TopAbs_SOLID)
        else:
            raise RuntimeError("Host Topology cannot be NULL when searching for ancestors.")
        
    def shells(self, host_topology: Topology) -> List['Cell']:
        """
        Returns the list of shells that contain this face.
        """
        if host_topology != None and not host_topology.is_null_shape():
            return self.upward_navigation(host_topology, TopAbs_SHELL)
        else:
            raise RuntimeError("Host Topology cannot be NULL when searching for ancestors.")

    def wires(self, host_topology: Topology) -> List['Wire']:
        """
        Returns the list of wires associated with this face.
        """
        return self.downward_navigation(TopAbs_WIRE)
    
    def edges(self, host_topology: Topology) -> List['Edge']:
        """
        Returns the list of edges associated with this face.
        """
        return self.downward_navigation(TopAbs_EDGE)

    def vertices(self) -> List['Vertex']:
        """
        Returns the list of vertices associated with this face.
        """
        return self.downward_navigation(TopAbs_VERTEX)
    
    def center_of_mass(self):
        """
        Calculates the center of mass of this face.
        """
        occt_center_of_mass = Face.center_of_mass_static(self.get_occt_face())
        return Topology.by_occt_shape(occt_center_of_mass)

    @staticmethod
    def center_of_mass_static(occt_face: TopoDS_Face) -> TopoDS_Vertex:
        """
        Calculates the center of mass of any OCCT face.
        """
        occt_shape_properties = GProp_GProps()
        brepgprop.SurfaceProperties(occt_face, occt_shape_properties)
        return BRepBuilderAPI_MakeVertex(occt_shape_properties.CentreOfMass()).Vertex()
    
    @staticmethod
    def by_external_boundary(
        external_boundary: 'Wire', 
        copy_attributes=False):
        """
        Constructs a face using external boundary only.
        """
        internal_boundaries = []
        face = Face.by_external_internal_boundaries(external_boundary, internal_boundaries)

        if copy_attributes:
            pass
            # ToDo: AttributeManager
            # AttributeManager.get_instance().deep_copy_attributes(
            #     external_boundary.get_occt_wire(), 
            #     face.get_occt_face())

        return face

    @staticmethod 
    def by_external_internal_boundaries(
        external_boundary: 'Wire', 
        internal_boundaries: List['Wire'], 
        copy_attributes: bool = False) -> 'Face':
        """
        Using an external and internal boundary wire, a face is constructed.
        """
        from Core.Wire import Wire
        from Core.Utilities.TopologicUtilities import FaceUtility
        # Just type checking specs
        external_boundary: Wire = external_boundary
        internal_boundaries: List[Wire] = internal_boundaries

        if not external_boundary.is_closed():
            raise ValueError("The input Wire is open.")

        occt_external_boundary = external_boundary.get_occt_wire()
        occt_make_face = BRepBuilderAPI_MakeFace(occt_external_boundary)
        if occt_make_face.Error() != BRepBuilderAPI_FaceDone:
            Face.throw(occt_make_face)

        occt_face = occt_make_face.Face()
        area = FaceUtility.area(occt_face)
        if area <= 0.0:
            occt_external_boundary.Reverse()
            occt_reverse_make_face = BRepBuilderAPI_MakeFace(occt_external_boundary)
            if occt_reverse_make_face.Error() != BRepBuilderAPI_FaceDone:
                Face.throw(occt_reverse_make_face)
            occt_face = occt_reverse_make_face.Face()

        for internal_boundary in internal_boundaries:
            occt_copier = BRepBuilderAPI_Copy(occt_face)
            occt_copy_face = topods.Face(occt_copier.Shape())

            occt_copy_make_face = BRepBuilderAPI_MakeFace(occt_copy_face)
            occt_internal_wire = internal_boundary.get_occt_wire()
            occt_copy_make_face.Add(occt_internal_wire)

            new_copy_face = occt_copy_make_face.Face()
            new_area = FaceUtility.area(new_copy_face)
            if new_area > area:
                occt_internal_wire.Reverse()

            occt_make_face.Add(occt_internal_wire)
            area = FaceUtility.area(occt_make_face.Face())

        occt_fixed_face = Face.occt_shape_fix(occt_make_face.Face())
        face = Face(occt_fixed_face)
        copy_face: Face = Face(face.deep_copy_shape())

        wires_as_topologies = []
        if copy_attributes:
            pass
            # ToDo: AttributeManager
            # AttributeManager.get_instance().deep_copy_attributes(
            #     external_boundary.get_occt_wire(), 
            #     copy_face.get_occt_face())
        wires_as_topologies.append(external_boundary)

        for internal_boundary in internal_boundaries:
            wires_as_topologies.append(internal_boundary)
            if copy_attributes:
                pass
                # ToDo: AttributeManager
                # AttributeManager.get_instance().deep_copy_attributes(
                #     internal_boundary.get_occt_wire(), 
                #     copy_face.get_occt_face())

        # ToDo: AttributeManager        
        # if copy_attributes:
        #     copy_face.deep_copy_attributes_from(wires_as_topologies)

        return copy_face

    @staticmethod
    def by_edges(edges: List['Edge'], copy_attributes: bool=False) -> 'Face':
        """
        Constructs a new face using edges.
        """
        from Core.Wire import Wire

        if len(edges) < 3:
            raise ValueError("Fewer than 3 edges are passed.")

        wire = Wire.by_edges(edges)
        face = Face.by_external_boundary(wire)
        edges_as_topologies: List[Topology] = []

        for edge in edges:
            edges_as_topologies.append(edge)
            if copy_attributes:
                # ToDo?: AttributeManager implementation
                pass

        # ToDo?: AttributesManager implementation
        # face.deep_copy_attributes_from(edges_as_topologies)

        return face

    @staticmethod
    def by_surface(occt_surface: Geom_Surface) -> 'Face':
        """
        Construct a Face from occt surface.
        """
        make_face = None
        try:
            make_face = BRepBuilderAPI_MakeFace(occt_surface, precision.Confusion())
        except Exception as ex:
            print(f'Exception occured when creating face.\n{ex}')
            Face.throw(make_face)

        shape_fix = ShapeFix_Face(make_face)
        shape_fix.Perform()
        face_instance = Face(topods.Face(shape_fix.Result()))

        return face_instance


    def shared_edges(self, another_face: 'Face') -> List['Edge']:
        """
        Returns the shared edges with another face.
        """
        from Core.Edge import Edge

        # Collect this face edges
        occt_shape_1 = self.get_occt_shape()
        occt_edges_1: List[TopoDS_Edge] = []
        occt_exp1 = TopExp_Explorer(occt_shape_1, TopAbs_EDGE)
        while occt_exp1.More():
            current_edge = occt_exp1.Current()
            if current_edge not in occt_edges_1:
                occt_edges_1.append(current_edge)
            occt_exp1.Next()   

        # Collect other face edges
        occt_shape_2 = another_face.get_occt_shape()
        occt_edges_2: List[TopoDS_Edge] = []
        occt_exp2 = TopExp_Explorer(occt_shape_2, TopAbs_EDGE)
        while occt_exp2.More():
            current_edge = occt_exp2.Current()
            if current_edge not in occt_edges_2:
                occt_edges_2.append(current_edge)
            occt_exp2.Next()   

        # Collect the shared edges
        shared_edges = []
        for occt_edge_1 in occt_edges_1:
            for occt_edge_2 in occt_edges_2:
                if occt_edge_1.IsSame(occt_edge_2):
                    edge = Edge(topods.Edge(occt_edge_1))
                    shared_edges.append(edge)

        return shared_edges

    def shared_vertices(self, another_face: 'Face') -> List['Vertex']:
        """
        Returns the shared vertices with another face.
        """
        from Core.Vertex import Vertex

        occt_shape_1 = self.get_occt_shape()
        occt_vertices_1: List[TopoDS_Vertex] = self.downward_navigation(occt_shape_1, TopAbs_VERTEX)

        occt_shape_2 = another_face.get_occt_shape()
        occt_vertices_2: List[TopoDS_Vertex] = self.downward_navigation(occt_shape_2, TopAbs_VERTEX)

        shared_vertices = []

        for occt_vertex_1 in occt_vertices_1:
            for occt_vertex_2 in occt_vertices_2:
                if occt_vertex_1.IsSame(occt_vertex_2):
                    vertex = Vertex(topods.Vertex(occt_vertex_1))
                    shared_vertices.append(vertex)

        return shared_vertices

    def external_boundary(self) -> 'Wire':
        """
        Instance bound method to get the external boundary wire.
        """
        from Core.Wire import Wire
        occt_wire = Face.static_external_boundary(self.get_occt_face())
        return Wire(occt_wire)

    @staticmethod
    def static_external_boundary(rk_occt_face: 'TopoDS_Face') -> TopoDS_Wire:
        """
        Static method to get an external boundary of a face.
        """
        occt_outer_wire = breptools.OuterWire(rk_occt_face)
        if occt_outer_wire.IsNull():
            occt_outer_wire = shapeanalysis.OuterWire(rk_occt_face)

        if rk_occt_face.Orientation() == TopAbs_REVERSED:
            occt_reversed_outer_wire = topods.Wire(occt_outer_wire.Reversed())
            return occt_reversed_outer_wire

        return occt_outer_wire

    def internal_boundaries(self) -> List['Wire']:
        """
        Returns the internal boundary wires.
        """
        from Core.Wire import Wire
        rk_face = self.get_occt_face()
        occt_outer_wire = Face.static_external_boundary(rk_face)
        internal_boundaries = []

        occt_explorer = TopoDS_Iterator(rk_face, False)
        while occt_explorer.More():
            if occt_explorer.Value().ShapeType() == TopAbs_WIRE:
                rk_wire = topods.Wire(occt_explorer.Value())

                if not rk_wire.IsSame(occt_outer_wire):
                    internal_boundaries.append(Wire(rk_wire))

            occt_explorer.Next()

        return internal_boundaries

    def add_internal_boundary(self, wire: 'Wire') -> None:
        """
        TODO: Description
        """
        wires = [wire]
        self.add_internal_boundaries(wires)

    def add_internal_boundaries(self, wires: List['Wire']) -> None:
        """
        TODO: Description
        """
        from Core.Wire import Wire
        if len(wires) == 0:
            return
        
        wires: List[Wire] = wires
        
        occt_make_face = BRepBuilderAPI_MakeFace(self.get_occt_face())
        for wire in wires:
            occt_make_face.Add(topods.Wire(wire.get_occt_wire().Reversed()))

        # ToDo?: Why are we passing in builderApi 
        # instead of builderApi.Shape()???
        self.set_instance_guid(occt_make_face, self.get_instance_guid())
        self.base_shape_face = occt_make_face#.Shape()


    def triangulate(
            self, 
            deflection: float, 
            angular_deflection: float) -> List['Face']:
        """
        Meshes the face, we save the mesh edges and vertices.
        """
        from Core.Vertex import Vertex
        from Core.Edge import Edge

        rk_occt_face = self.get_occt_face()

        occt_fix_face = ShapeFix_Face(rk_occt_face)
        occt_fix_face.Perform()
        occt_incremental_mesh = BRepMesh_IncrementalMesh(
            occt_fix_face.Result(), deflection)
        
        occt_location = TopLoc_Location()
        p_occt_triangulation: Poly_Triangulation = BRep_Tool.Triangulation(
            topods.Face(occt_fix_face.Result()), occt_location)

        if p_occt_triangulation == None:
            raise RuntimeError("No triangulation was produced.")

        ret_triangles = []
        num_of_triangles = p_occt_triangulation.NbTriangles()

        for i in range(1, num_of_triangles + 1):
            index1, index2, index3 = 0, 0, 0
            p_occt_triangulation.Triangle(i).Get(index1, index2, index3)

            point1 = p_occt_triangulation.Node(index1)
            point2 = p_occt_triangulation.Node(index2)
            point3 = p_occt_triangulation.Node(index3)

            vertex1 = Vertex.by_point(Geom_CartesianPoint(point1))
            vertex2 = Vertex.by_point(Geom_CartesianPoint(point2))
            vertex3 = Vertex.by_point(Geom_CartesianPoint(point3))

            edge1 = Edge.by_start_vertex_end_vertex(vertex1, vertex2)
            edge2 = Edge.by_start_vertex_end_vertex(vertex2, vertex3)
            edge3 = Edge.by_start_vertex_end_vertex(vertex3, vertex1)
            
            edges = [edge1, edge2, edge3]
            
            face = Face.by_edges(edges)
            ret_triangles.append(face)

        return ret_triangles


    @staticmethod
    def occt_shape_fix(occt_face: TopoDS_Shape) -> TopoDS_Face:
        """
        Performs OCC shape fix.
        """
        occt_face_fix = ShapeFix_Face(occt_face)
        _ = occt_face_fix.Perform()
        return topods.Face(occt_face_fix.Result())

    def is_manifold(self, host_topology: Topology) -> bool:
        """
        A manifold face has 0 or 1 cell.
        """
        from Core.Cell import Cell
        cells: List['Cell'] = FaceUtility.adjacent_cells(self, host_topology)

        if len(cells) < 2:
            return True
        else:
            return False

    def is_manifold_to_topology(self, host_topology: Topology) -> bool:
        """
        TODO
        """
        from Core.Cell import Cell
        cells: List['Cell'] = []
        if host_topology == None:
            cells = self.cells(host_topology)
        else:
            cells = FaceUtility.adjacent_cells(self, host_topology)

        if len(cells) < 2:
            return True
        else:
            return False

    def geometry(self) -> List[Geom_Geometry]:
        """
        Returns a list of comprising OCCT Geom_Geometries.
        """
        ret_geometries = []
        ret_geometries.append(self.surface())
        return ret_geometries

    def set_occt_shape(self, occt_shape: TopoDS_Shape) -> None:
        """
        Generic setter for underlying OCCT shape.
        """
        try:
            self.set_occt_face(topods.Face(occt_shape))
        except Exception as ex:
            raise RuntimeError(str(ex.args))

    def get_occt_face(self) -> TopoDS_Face:
        """
        Getter for the underlying OCCT face.
        """
        if self.base_shape_face.IsNull():
            raise RuntimeError("A null Face is encountered!")
        
        return self.base_shape_face
    
    def set_occt_face(self, occt_face: TopoDS_Face) -> None:
        """
        Setter for underlying OCCT face.
        """
        self.base_shape_face = occt_face
        self.base_shape = self.base_shape_face
    
    def surface(self) -> Geom_Surface:
        """
        Getter for the the OCC surface.
        """
        occt_face = self.get_occt_face()
        return BRep_Tool.Surface(occt_face)

    @staticmethod
    def throw(occt_make_face: BRepBuilderAPI_MakeFace) -> None:
        """
		The error messages are based on those in the OCCT documentation.
		https://www.opencascade.com/doc/occt-7.2.0/refman/html/_b_rep_builder_a_p_i___face_error_8hxx.html#ac7a498a52546f7535a3f73f6bab1599a
        """
        if occt_make_face.Error() == BRepBuilderAPI_NoFace:
            raise RuntimeError("No initialization of the algorithm; only an empty constructor was used.")
        elif occt_make_face.Error() == BRepBuilderAPI_NotPlanar:
            raise RuntimeError("No surface was given and the wire was not planar.")
        elif occt_make_face.Error() == BRepBuilderAPI_CurveProjectionFailed:
            raise RuntimeError("Curve projection failed.")
        elif occt_make_face.Error() == BRepBuilderAPI_ParametersOutOfRange:
            raise RuntimeError("The parameters given to limit the surface are out of its bounds.")

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
        return TopologyTypes.FACE

    def get_type_as_string(self) -> str:
        """
        Returns the stringified type.
        """
        return 'Face'