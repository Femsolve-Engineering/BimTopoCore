
from OCC.Core.TopoDS import topods, TopoDS_Vertex, TopoDS_Edge
from OCC.Core.TopoDS import TopoDS_Shape

from Core.Factories.TopologyFactory import TopologyFactory

class VertexFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Vertex import Vertex
        return Vertex(topods.Vertex(occt_shape))
    
class EdgeFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Edge import Edge
        return Edge(topods.Edge(occt_shape))

class WireFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Wire import Wire
        return Wire(topods.Wire(occt_shape))
    
class FaceFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Face import Face
        return Face(topods.Face(occt_shape))

class ShellFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Shell import Shell
        return Shell(topods.Shell(occt_shape))
    
class CellFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Cell import Cell
        return Cell(topods.Solid(occt_shape))
    
class CellComplexFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from CellComplex import CellComplex
        return CellComplex(topods.CompSolid(occt_shape))
    
class ClusterFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Cluster import Cluster
        return Cluster(topods.Compound(occt_shape))
    
class ApertureFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        from Core.Aperture import Aperture
        raise NotImplementedError("Need to write ApertureFactory in AllFactories.py!")
