
from OCC.Core.TopoDS import TopoDS_Builder, TopoDS_Compound

from Topology import Topology


class GlobalCluster(Topology):

    def __init__(self) -> None:
        
        self.base_occt_compound = TopoDS_Compound()

        occt_builder = TopoDS_Builder()
        occt_builder.MakeCompound(self.base_occt_compound)

    @staticmethod
    def get_instance():
        instance = GlobalCluster()
        return instance

    def get_occt_compound(self) -> TopoDS_Compound:
        return self.base_occt_compound