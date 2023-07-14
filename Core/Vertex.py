# OCC
from OCC.Core.TopoDS import TopoDS_Vertex, topods_Vertex
from OCC.Core.Standard import Standard_True
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.gp import gp_Pnt

# BimTopoCore
from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes

class Vertex(Topology):
    """
    Represents a 1D vertex object. Serves as a wrapper around 
    TopoDS_VERTEX entity of OCC.
    """
    def __init__(self, occt_vertex: TopoDS_Vertex):
        """Constructor saves shape and processes GUID.

        Args:
            occt_vertex (TopoDS_Vertex): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """
        super().__init__(self, occt_vertex, TopologyTypes.VERTEX)
        self.base_shape_vertex = occt_vertex

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
        occt_vertex = BRepBuilderAPI_MakeVertex(occt_pnt)
        occt_fixed_vertex = Vertex(Topology.fix_shape(occt_vertex))
        new_vertex = Vertex(occt_fixed_vertex)
        return new_vertex
        




