from typing import Dict
from typing import Tuple

# Occ
from OCC.Core.TopAbs import *

# BimTopoCore
from Core.Factories.TopologyFactory import TopologyFactory
from Core.Factories.AllFactories import VertexFactory
from Core.Factories.AllFactories import EdgeFactory
from Core.Factories.AllFactories import WireFactory
from Core.Factories.AllFactories import FaceFactory
from Core.Factories.AllFactories import ShellFactory
from Core.Factories.AllFactories import CellFactory
from Core.Factories.AllFactories import CellComplexFactory
from Core.Factories.AllFactories import ClusterFactory

class TopologyFactoryManager:

    _instance = None

    def __init__(self):
        """
        Singleton instance constructor.
        """

        if TopologyFactoryManager._instance != None:
            raise RuntimeError("Cannot new up a new TopologyFactoryManager! It is a singleton!")

        self.topology_factory_map: Dict[str, TopologyFactory] = {}

    def add(self, guid: str, rkTopologyFactory: TopologyFactory) -> None:
        if guid not in self.topology_factory_map:
            self.topology_factory_map[guid] = rkTopologyFactory

    def find(self, rkGuid: str) -> TopologyFactory:
        topology_factory = self.topology_factory_map.get(rkGuid)
        return topology_factory

    def get_default_factory(self, occt_type: TopAbs_ShapeEnum) -> TopologyFactory:
        if occt_type == TopAbs_COMPOUND:
            return ClusterFactory()
        elif occt_type == TopAbs_COMPSOLID:
            return CellComplexFactory()
        elif occt_type == TopAbs_SOLID:
            return CellFactory()
        elif occt_type == TopAbs_SHELL:
            return ShellFactory()
        elif occt_type == TopAbs_FACE:
            return FaceFactory()
        elif occt_type == TopAbs_WIRE:
            return WireFactory()
        elif occt_type == TopAbs_EDGE:
            return EdgeFactory()
        elif occt_type == TopAbs_VERTEX:
            return VertexFactory()
        else:
            raise RuntimeError("Topology::ByOcctShape: unknown topology.")
        
    @staticmethod
    def get_instance() -> 'TopologyFactoryManager':
        """
        Getter for singleton instance.
        """

        if TopologyFactoryManager._instance == None:
            TopologyFactoryManager._instance = TopologyFactoryManager()

        return TopologyFactoryManager._instance
