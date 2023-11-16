
from typing import Tuple
from typing import List

# OCC
from OCC.Core.Precision import precision
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
from Core.Factories.AllFactories import ApertureFactory
from Core.Utilities.TopologicUtilities import FaceUtility
from Core.Context import Context

class Aperture(Topology):
    """
    Represents an Aperture. 
    Serves as a wrapper around TopoDS_Face entity of OCC.
    """
    def __init__(self, 
                 topology: Topology,
                 context: Context, 
                 guid: str=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_shape (TopoDS_Face): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        super().__init__(topology.get_occt_shape(), TopologyTypes.FACE)
        self.register_factory(self.get_class_guid(), ApertureFactory())

        if topology == None:
            raise RuntimeError("A null topology was passed in!")
        
        self.base_topology = topology

        if context != None:
            self.add_context(context)

    @staticmethod
    def by_topology_context(
        topology: Topology,
        context_topology: Topology) -> 'Aperture':
        """
        Creates an aperture by a topology and a context topology.
        """
        default_parameter = 0.0

        # Identify the closest simplest subshape
        closest_simple_subshape = context_topology.closest_simplest_subshape(
            topology.center_of_mass())
        
        # Create a context context_topology
        context = Context(
            closest_simple_subshape, 
            default_parameter,
            default_parameter,
            default_parameter)
        
        # Create the Aperture
        return Aperture(topology, context)
    
    def center_of_mass(self) -> TopoDS_Vertex:
        """
        Implemented override method for center of mass.
        """
        return self.base_topology.center_of_mass()
    
    def is_manifold(self) -> bool:
        """
        Determines if Aperture is manifold.
        """
        return self.base_topology.is_manifold()
    
    def geometry(self) -> Geom_Geometry:
        """
        Returns:
            Geom_Geometry: Underlying OCC geometry
        """
        return self.base_topology.geometry()
    
    def set_occt_shape(self, shape: TopoDS_Shape) -> None:
        """
        Sets the underlying OCC shape.
        """
        self.base_topology.base_shape = shape
    
    def get_occt_shape(self) -> TopoDS_Shape:
        """
        Returns underlying OCC shape.
        """
        return self.base_topology.get_occt_shape()
    
    def get_type_as_string(self) -> str:
        """
        Returns stringified name.
        """
        return 'Aperture'
    
    def get_type(self) -> TopologyTypes:
        """
        Returns:
            TopologyTypes: Internal definition for types.
        """
        return TopologyTypes.APERTURE
    
    def is_container_type(self) -> bool:
        """
        Returns if base topology is container type.
        """
        return self.base_topology.is_container_type()
    
    def occt_shape_fix(self, occt_input_shape: TopoDS_Shape) -> TopoDS_Shape:
        """
        No shape fix method attached to Aperture.
        """
        return occt_input_shape
    
    def topology(self) -> Topology:
        """
        Underlying topology getter with validation.
        """
        if self.base_topology == None:
            raise RuntimeError("The underlying topology is Null!")
        else: return self.base_topology


    
    

