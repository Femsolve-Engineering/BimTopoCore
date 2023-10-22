
from typing import Tuple
from typing import List

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Wire, TopoDS_Iterator, TopoDS_Face, topods
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_WIRE, TopAbs_REVERSED
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_CartesianPoint, Geom_Surface, Geom_Geometry
from OCC.Core.ShapeAnalysis import shapeanalysis
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_EdgeDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NoFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NotPlanar
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_CurveProjectionFailed
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_ParametersOutOfRange
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import EdgeEnd, TopologyTypes
from Core.Factories.AllFactories import FaceFactory
from Core.Utilities.TopologicUtilities import FaceUtility

class Face(Topology):
    """
    Represents a face in the 3D space. 
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

    def external_boundary(self) -> 'Wire':
        """
        Instance bound method to get the external boundary wire.
        """
        from Core.Wire import Wire
        return Wire(self.__external_boundary(self.get_occt_face()))

    @staticmethod
    def __external_boundary(rk_occt_face: 'TopoDS_Face') -> TopoDS_Wire:
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
        occt_outer_wire = self.__external_boundary(rk_face)
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
    def occt_shape_fix(occt_face) -> TopoDS_Face:
        """
        Performs OCC shape fix.
        """
        occt_face_fix = ShapeFix_Face(occt_face)
        _ = occt_face_fix.Perform()
        return topods.Face(occt_face_fix)

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

    def throw(self, occt_make_face: BRepBuilderAPI_MakeFace) -> None:
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

    def get_type_as_string(self) -> str:
        """
        Returns the stringified type.
        """
        return 'Face'