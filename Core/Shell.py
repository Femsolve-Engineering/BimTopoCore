

from typing import Tuple
from typing import List
from xmlrpc.client import boolean

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Builder, topods
from OCC.Core.TopAbs import TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Surface, Geom_Geometry
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_EdgeDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NoFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NotPlanar
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_CurveProjectionFailed
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_ParametersOutOfRange
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, SurfaceProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_Shell, BRepCheck_NoError

# BimTopoCore
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Wire import Wire
from Core.Face import Face
from Core.Cell import Cell

from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.Factories.AllFactories import ShellFactory

# Not implemented yet
# import Core.AttributeManager

class Shell(Topology):
    
    def __init__(self, occt_shell: TopoDS_Shell, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_solid (TopoDS_Compound): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        super().__init__(occt_shell, TopologyTypes.SHELL)
        self.base_shape_shell = occt_shell
        self.register_factory(self.get_class_guid(), ShellFactory())

#--------------------------------------------------------------------------------------------------
    def is_container_type(self) -> bool:
        """
        Determines if this topology is container type.
        """
        return True

#--------------------------------------------------------------------------------------------------
    def get_type(self) -> TopologyTypes:
        """
        Returns:
            TopologyTypes: Internal definition for types.
        """
        return TopologyTypes.SHELL

#--------------------------------------------------------------------------------------------------
    def cells(self, host_topology: Topology, cells: List[Cell]):
        """

        """
        print('REMINDER!!! Base Topology method of upward_navigation() is not yet correctly implemented.')
        if not host_topology.is_null_shape():
            return self.upward_navigation(host_topology.get_occt_shape(), cells)
        else:
            raise RuntimeError("Host Topology cannot be None when searching for ancestors.")

#--------------------------------------------------------------------------------------------------
    # def edges(self, host_topology: Topology, edges: List[Edge]) -> List[Edge]:
    def edges(self) -> List[Edge]:
        """

        """
        shape_type = TopologyTypes.EDGE
        return self.downward_navigation(shape_type)

#--------------------------------------------------------------------------------------------------
    # def wires(self, host_topology: Topology, wires: List[Wire]) -> List[Wire]:
    def wires(self) -> List[Wire]:
        """
        Creates a collection of all wires that belong to current shape.
        """
        shape_type = TopologyTypes.WIRE
        return self.downward_navigation(shape_type)

#--------------------------------------------------------------------------------------------------
    # def faces(self, host_topology: Topology, faces: List[Face]) -> List[Face]:
    def faces(self) -> List[Face]:
        """

        """
        shape_type = TopologyTypes.FACE
        return self.downward_navigation(shape_type)

#--------------------------------------------------------------------------------------------------
    def is_closed(self):
        occt_brep_check_shell = BRepCheck_Shell(TopoDS_Shell(self.get_occt_shape()))
        return occt_brep_check_shell.Closed() == BRepCheck_NoError

#--------------------------------------------------------------------------------------------------
    # def vertices(self, host_topology: Topology, vertices: List[Vertex]) -> List[Vertex]:
    def vertices(self) -> List[Vertex]:
        """

        """
        shape_type = TopologyTypes.VERTEX
        return self.downward_navigation(shape_type)

#--------------------------------------------------------------------------------------------------
    def by_faces(self, faces: List[Face], tolerance: float, copy_attributes: boolean) -> 'Shell':
        
        if not faces:

            # raise RuntimeError("No face is passed.")
            return None

        occt_shapes: List[Face] = TopTools_ListOfShape()

        for face in faces:
            occt_shapes.Append(face.get_occt_shape())

        if occt_shapes.Size() == 1:
            occt_shell = TopoDS_Shell()
            occt_builder = TopoDS_Builder()
            occt_builder.MakeShell(occt_shell)

            for occt_face in occt_shapes:
                occt_builder.Add(occt_shell, TopoDS_Face(occt_face))
                if copy_attributes:

                    # AttributeManager not implemented yet!
                    # instance = AttributeManager.get_instance()
                    # instance.deep_copy_attributes(occt_face, occt_shell)

                    pass

            p_shell = Shell(occt_shell)
            # GlobalCluster.get_instance().add_topology(p_shell) # not needed
            return p_shell

        # Topology.occt_sew_faces not implemented yet!
        occt_shape = Topology.occt_sew_faces(occt_shapes, tolerance)
        try:
            occt_shell = TopoDS_Shell(occt_shape)
            p_shell = Shell(occt_shell)

            faces_as_topologies: List[Topology]

            for face in faces:
                faces_as_topologies.append(face) 

            if copy_attributes:
                # Topology.deep_copy_attributes_from() not implemented yet!
                p_copy_shell = p_shell.deep_copy_attributes_from(faces_as_topologies)
                return p_copy_shell
            else:
                return p_shell

        except:
            raise Exception("Error: The set of faces does not create a valid shell.")
            # return None

#--------------------------------------------------------------------------------------------------
    def is_manifold(self, host_topology: Topology) -> boolean:

            # A Shell is non-manifold if at least one of its edges is shared by more than two faces
            edges = self.edges()

            for edge in edges:

                topology_type = TopologyTypes.FACE

                print('REMINDER!!! Base Topology method of upward_navigation() is not yet correctly implemented.')
                faces = edge.upward_navigation(self.get_occt_shape(), topology_type)

                if len(faces) > 2:
                    return False

            return True

#--------------------------------------------------------------------------------------------------
    def get_occt_shape(self) -> TopoDS_Shell:
        
        return self.get_occt_shell()

#--------------------------------------------------------------------------------------------------
    def get_occt_shell(self) -> TopoDS_Shell:
        
        if self.base_shape_shell.IsNull():
            raise RuntimeError("A null Shell is encountered.")

        return self.base_shape_shell

#--------------------------------------------------------------------------------------------------
    def set_occt_shape(self, occt_shape: TopoDS_Shape):
        occt_shell = TopoDS_Shell(occt_shape)
        self.set_occt_shell(occt_shell)

#--------------------------------------------------------------------------------------------------
    def set_occt_shell(self, occt_shell: TopoDS_Shell):
        self.base_shape_shell = occt_shell

#--------------------------------------------------------------------------------------------------
    def center_of_mass(self) -> 'Vertex':

        # Topology.center_of_mass not implemented yet.
        return super().center_of_mass(self.get_occt_shell())

#--------------------------------------------------------------------------------------------------
    def make_pnt_at_center_of_mass(self, occt_shell: TopoDS_Shell) -> 'Vertex':
        
        occt_shape_properties = GProp_GProps()
        SurfaceProperties(occt_shell, occt_shape_properties)

        center_of_mass_point = occt_shape_properties.center_of_mass_point()
        return BRepBuilderAPI_MakeVertex(center_of_mass_point).Vertex()

#--------------------------------------------------------------------------------------------------
    def get_type_as_string(self):

        return "Shell"

#--------------------------------------------------------------------------------------------------
    def geometry(self, occt_geometries):

        # Returns a list of faces
        faces = self.faces()

        for face in faces:
            occt_geometries.append(face.Surface())


#--------------------------------------------------------------------------------------------------