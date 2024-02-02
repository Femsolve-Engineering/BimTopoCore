

from typing import Tuple
from typing import List

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Vertex, TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Builder, topods
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE
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
from OCC.Core.BRepGProp import brepgprop, brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_Shell, BRepCheck_NoError

# BimTopoCore

from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.Factories.AllFactories import ShellFactory

# Not implemented yet
from Core.AttributeManager import AttributeManager

class Shell(Topology):
    
    def __init__(self, occt_shell: TopoDS_Shell, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_solid (TopoDS_Compound): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        instance_topology = super().__init__(occt_shell, TopologyTypes.SHELL)
        self.base_shape_shell = occt_shell
        self.register_factory(self.get_class_guid(), ShellFactory())

        # Register the instances
        Topology.topology_to_subshape[instance_topology] = self
        Topology.subshape_to_topology[self] = instance_topology

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
    def cells(self, host_topology: Topology) -> List[Topology]:
        """
        Returns the Cells which contain this Shell.
        """
        print('REMINDER!!! Base Topology method of upward_navigation() is not yet correctly implemented.')
        if not host_topology.is_null_shape():
            return self.upward_navigation(host_topology.get_occt_shape())
        else:
            raise RuntimeError("Host Topology cannot be None when searching for ancestors.")

#--------------------------------------------------------------------------------------------------
    # def edges(self, host_topology: Topology, edges: List[Edge]) -> List[Edge]:
    def edges(self) -> List['Edge']:
        """
        Returns the Edge contituents to this Shell
        """
        shape_enum = TopAbs_EDGE
        return self.downward_navigation(shape_enum)

#--------------------------------------------------------------------------------------------------
    # def wires(self, host_topology: Topology, wires: List[Wire]) -> List[Wire]:
    def wires(self) -> List['Wire']:
        """
        Returns the Wire contituents to this Shell
        """
        shape_type = TopologyTypes.WIRE
        shape_enum = TopAbs_WIRE
        
        occt_shape = self.get_occt_shape()
        return Topology.static_downward_navigation(occt_shape, shape_enum)

#--------------------------------------------------------------------------------------------------
    # def faces(self, host_topology: Topology, faces: List[Face]) -> List[Face]:
    def faces(self) -> List['Face']:
        """
        Returns the Face contituents to this Shell
        """
        # shape_type = TopologyTypes.FACE
        shape_enum = TopAbs_FACE

        occt_shape = self.get_occt_shape()
        return Topology.static_downward_navigation(occt_shape, shape_enum)

#--------------------------------------------------------------------------------------------------
    def is_closed(self):
        """
        Checks if the shell is fully enclosed
        """
        occt_brep_check_shell = BRepCheck_Shell(topods.Shell(self.get_occt_shape()))
        return occt_brep_check_shell.Closed() == BRepCheck_NoError

#--------------------------------------------------------------------------------------------------
    # def vertices(self, host_topology: Topology, vertices: List[Vertex]) -> List[Vertex]:
    def vertices(self) -> List['Vertex']:
        """
        Returns the Vertex contituents to this Shell
        """
        # shape_enum = TopologyTypes.VERTEX
        shape_enum = TopAbs_VERTEX

        occt_shape = self.get_occt_shape()
        return Topology.static_downward_navigation(occt_shape, shape_enum)
        # return self.downward_navigation(shape_type)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_faces(faces: List['Face'], tolerance: float, copy_attributes: bool) -> 'Shell':
        """
        Creates a Shell by a set of faces.
        """
        from Core.Face import Face
        from Core.Shell import Shell
        
        if not faces:

            # raise RuntimeError("No face is passed.")
            return None

        occt_shapes: List[Face] = TopTools_ListOfShape()

        for face in faces:
            occt_shapes.Append(face.get_occt_shape())

        if occt_shapes.Size() == 1:
            occt_shell = topods.Shell()
            occt_builder = TopoDS_Builder()
            occt_builder.MakeShell(occt_shell)

            for occt_face in occt_shapes:
                occt_builder.Add(occt_shell, topods.Face(occt_face))
                if copy_attributes:

                    # AttributeManager not implemented yet!
                    instance = AttributeManager.get_instance()
                    instance.deep_copy_attributes(occt_face, occt_shell)

                    pass

            p_shell = Shell(occt_shell)
            # GlobalCluster.get_instance().add_topology(p_shell) # not needed
            return p_shell

        occt_shape = Topology.occt_sew_faces(occt_shapes, tolerance)
        # try:
        occt_shell = topods.Shell(occt_shape)
        p_shell = Shell(occt_shell)

        faces_as_topologies: List[Topology] = []

        for face in faces:
            faces_as_topologies.append(face) 

        if False:#copy_attributes: 
            p_copy_shell = p_shell.deep_copy_attributes_from(faces_as_topologies)
            return p_copy_shell
        else:
            return p_shell

        # except:
        #     raise Exception("Error: The set of faces does not create a valid shell.")


#--------------------------------------------------------------------------------------------------
    def is_manifold(self, host_topology: Topology) -> bool:
        """
        Returns True, if this Shell is a manifold, otherwise a False.
        """

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
        """
        Returns the underlying OCCT shape.
        """
        
        return self.get_occt_shell()

#--------------------------------------------------------------------------------------------------
    def get_occt_shell(self) -> TopoDS_Shell:
        """
        Returns the underlying OCCT base shell.
        """
        
        if self.base_shape_shell.IsNull():
            raise RuntimeError("A null Shell is encountered.")

        return self.base_shape_shell

#--------------------------------------------------------------------------------------------------
    def set_occt_shape(self, occt_shape: TopoDS_Shape):
        """
        Sets the underlying OCCT shape.
        """

        occt_shell = topods.Shell(occt_shape)
        self.set_occt_shell(occt_shell)

#--------------------------------------------------------------------------------------------------
    def set_occt_shell(self, occt_shell: TopoDS_Shell):
        """
        Sets the underlying OCCT shell.
        """

        self.base_shape_shell = occt_shell

#--------------------------------------------------------------------------------------------------
    def center_of_mass(self) -> 'Vertex':
        """
        Returns the Vertex at the center of mass of this OCCT shell.
        """

        from Core.Vertex import Vertex
        from Core.Shell import Shell

        occt_vertex = Shell.make_pnt_at_center_of_mass(self.get_occt_shell())
        vertex = Vertex(occt_vertex)

        return vertex

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def make_pnt_at_center_of_mass(self, occt_shell: TopoDS_Shell) -> TopoDS_Vertex:
        """
        Returns the OCCT vertex at the center of mass of this OCCT shell.
        """
        
        occt_shape_properties = GProp_GProps()
        brepgprop.SurfaceProperties(occt_shell, occt_shape_properties)

        center_of_mass_point = occt_shape_properties.CenterOfMass()
        return BRepBuilderAPI_MakeVertex(center_of_mass_point).Vertex()

#--------------------------------------------------------------------------------------------------
    def get_type_as_string(self):
        """
        Returns the type of this Shell as a string.
        """

        return "Shell"

#--------------------------------------------------------------------------------------------------
    def geometry(self) -> List[Geom_Geometry]:
        """
        Creates a geometry from this Shell.
        """

        occt_geometries = []
        
        # Returns a list of faces
        faces = self.faces()
        
        # Get Geom_Surface for the OCC surface.
        for face in faces:
            occt_geometries.append(face.surface())

        return occt_geometries

#--------------------------------------------------------------------------------------------------