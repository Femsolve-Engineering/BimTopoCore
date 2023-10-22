

from typing import Tuple
from typing import List

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Edge, TopoDS_Face, topods
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
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import EdgeEnd, TopologyTypes
from Core.Factories.AllFactories import CellFactory
from Core.Utilities.TopologicUtilities import VertexUtility

class Cell(Topology):
    
    def __init__(self, occt_solid: TopoDS_Solid, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_solid (TopoDS_Solid): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        super().__init__(occt_solid, TopologyTypes.CELL)
        self.base_shape_solid = occt_solid
        self.register_factory(self.get_class_guid(), CellFactory())