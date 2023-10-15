
from OCC.Core.TopoDS import TopoDS_Vertex, TopoDS_Edge
from OCC.Core.TopoDS import TopoDS, TopoDS_Shape

from Core.Factories import TopologyFactory
from Core.Topology import Topology
from Core.Vertex import Vertex
from Core.Edge import Edge

class VertexFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        return Vertex(TopoDS_Vertex(occt_shape))
    
class EdgeFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        return Edge(TopoDS_Edge(occt_shape))

# ToDo
class WireFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        raise NotImplementedError("WireFactory create")
    
class FaceFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        raise NotImplementedError("FaceFactory create")

class ShellFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        raise NotImplementedError("ShellFactory create")
    
class CellFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        raise NotImplementedError("CellFactory create")
    
class CellComplexFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        raise NotImplementedError("CellComplexFactory create")
    
class ClusterFactory(TopologyFactory):
    def create(self, occt_shape: TopoDS_Shape):
        raise NotImplementedError("ClusterFactory create")
