
from enum import Enum

# OCC
from OCC.Core import TopAbs
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopoDS import TopoDS_TShape

class TopologyType(Enum):
    TOPOLOGY_VERTEX = TopAbs.TopAbs_VERTEX
    TOPOLOGY_EDGE = TopAbs.TopAbs_EDGE
    TOPOLOGY_WIRE = TopAbs.TopAbs_WIRE
    TOPOLOGY_FACE = TopAbs.TopAbs_FACE
    TOPOLOGY_SHELL = TopAbs.TopAbs_SHELL
    TOPOLOGY_CELL = TopAbs.TopAbs_SOLID
    TOPOLOGY_CELLCOMPLEX = TopAbs.TopAbs_COMPSOLID
    TOPOLOGY_CLUSTER = TopAbs.TopAbs_COMPOUND
    TOPOLOGY_APERTURE = TopAbs.TopAbs_SHAPE
    TOPOLOGY_ALL = TopAbs.TopAbs_SHAPE

class OcctShapeComparator:
    def __call__(self, 
                 occt_shape1: TopoDS_Shape, 
                 occt_shape2: TopoDS_Shape):
        t_shape_1: TopoDS_TShape = occt_shape1.TShape()
        t_shape_2: TopoDS_TShape = occt_shape2.TShape()
        
        # ToDo: Check if this comparison is correct!
        value1 = id(t_shape_1)
        value2 = id(t_shape_2)
        return value1 < value2


def OcctTypeFromTopologicType(kTopologyType):
    return kTopologyType