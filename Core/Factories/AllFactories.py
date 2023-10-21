
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
