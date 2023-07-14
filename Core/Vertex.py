# OCC
from OCC.Core.TopoDS import TopoDS_Vertex, topods_Vertex
from OCC.Core.Standard import Standard_True
from OCC.Core.Geom import Geom_CartesianPoint
from OCC.Core.BRepTools import BRep_Tool
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

    def x(self) -> float:
        """Getter of X-coordinate for the vertex.

        Returns:
            float: X-coordinate
        """
        geom_cartesian_point = self.__get_point()
        return geom_cartesian_point.X()
    
    def y(self) -> float:
        """Getter of Y-coordinate for the vertex.

        Returns:
            float: Y-coordinate
        """
        geom_cartesian_point = self.__get_point()
        return geom_cartesian_point.Y()

    def z(self) -> float:
        """Getter of Z-coordinate for the vertex.

        Returns:
            float: Z-coordinate
        """
        geom_cartesian_point = self.__get_point()
        return geom_cartesian_point.Z()

    def __get_point(self) -> Geom_CartesianPoint:
        """Gets OCC's Geom_Point object.

        Returns:
            Geom_CartesianPoint: OCC Cartesian point object
        """
        geometric_point = BRep_Tool.Pnt(self.base_shape_vertex)
        return Geom_CartesianPoint(geometric_point)

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
        




