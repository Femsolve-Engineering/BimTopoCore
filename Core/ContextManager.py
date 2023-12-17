
from typing import Tuple
from typing import List
from xmlrpc.client import boolean

# OCC
from OCC.Core.TopoDS import TopoDS_Shape

# BimTopoCore
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Wire import Wire
from Core.Face import Face
from Core.Cell import Cell
from Core.Shell import Shell
from Core.Cluster import Cluster

from Core.Context import Context
from Topology import Topology
from Core.TopologyConstants import TopologyTypes


class ContextManager():

    def __init__(self) -> None:
        
        self.occt_shape_to_contexts_map = {}

#--------------------------------------------------------------------------------------------------
    def add(self, occt_shape: TopoDS_Shape, context: Context) -> None:
        
        # If the OCCT shape does not have a content, initialise it in the map
        if occt_shape not in self.occt_shape_to_contexts_map:
            self.occt_shape_to_contexts_map[occt_shape] = []

        self.occt_shape_to_contexts_map[occt_shape].append(context)

#--------------------------------------------------------------------------------------------------
    def remove(self, occt_shape: TopoDS_Shape, occt_context_shape: TopoDS_Shape) -> None:

        new_list: List[TopoDS_Shape] = []
        
        # If the OCCT shape does not have a context, initialise it in the map
        if occt_shape in self.occt_shape_to_contexts_map:

            for content in self.occt_shape_to_contexts_map[occt_shape]:
                if not content.get_occt_shape().IsSame(occt_context_shape):
                    new_list.append(content)

            self.occt_shape_to_contexts_map[occt_shape] = new_list

#--------------------------------------------------------------------------------------------------
    def find(self, occt_shape: TopoDS_Shape, contents: List[Context]) -> boolean:
        
        if occt_shape in self.occt_shape_to_contexts_map:

            contents.extend(self.occt_shape_to_contexts_map[occt_shape])
            return True

        return False

#--------------------------------------------------------------------------------------------------
    def clear_one(self, occt_shape: TopoDS_Shape) -> None:
        
        if occt_shape in self.occt_shape_to_contexts_map:

            del self.occt_shape_to_contexts_map[occt_shape]

#--------------------------------------------------------------------------------------------------
    def clear_all(self):
        
        self.occt_shape_to_contexts_map = {}

#--------------------------------------------------------------------------------------------------