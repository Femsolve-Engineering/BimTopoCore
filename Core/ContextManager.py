
from typing import Tuple
from typing import List
from xmlrpc.client import boolean

# OCC
from OCC.Core.TopoDS import TopoDS_Shape

# BimTopoCore
from Core.Context import Context
from Topology import Topology
from Core.TopologyConstants import TopologyTypes


class ContextManager():

    def __init__(self) -> None:
        
        self.occt_shape_to_contexts_map = {}

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def get_instance():

        instance = ContextManager()
        return instance

#--------------------------------------------------------------------------------------------------
    def add(self, occt_shape: TopoDS_Shape, context: Context) -> None:
        """
        Add context to dictionary
        """
        
        # If the OCCT shape does not have a content, initialise it in the map
        if occt_shape not in self.occt_shape_to_contexts_map:
            self.occt_shape_to_contexts_map[occt_shape] = []

        self.occt_shape_to_contexts_map[occt_shape].append(context)

#--------------------------------------------------------------------------------------------------
    def remove(self, occt_shape: TopoDS_Shape, occt_context_shape: TopoDS_Shape) -> None:
        """
        Remove context from dictionary
        """

        new_list: List[TopoDS_Shape] = []
        
        # If the OCCT shape does not have a context, initialise it in the map
        if occt_shape in self.occt_shape_to_contexts_map:

            for context in self.occt_shape_to_contexts_map[occt_shape]:
                if not context.get_occt_shape().IsSame(occt_context_shape):
                    new_list.append(context)

            self.occt_shape_to_contexts_map[occt_shape] = new_list

#--------------------------------------------------------------------------------------------------
    def find(self, occt_shape: TopoDS_Shape, contexts: List[Context]) -> boolean:
        """
        Find shape in dictionary and extend list of contexts
        """
        
        if occt_shape in self.occt_shape_to_contexts_map:

            contexts.extend(self.occt_shape_to_contexts_map[occt_shape])
            return True

        return False

#--------------------------------------------------------------------------------------------------
    def clear_one(self, occt_shape: TopoDS_Shape) -> None:
        """
        Remove shape from dictionary
        """
        
        if occt_shape in self.occt_shape_to_contexts_map:

            del self.occt_shape_to_contexts_map[occt_shape]

#--------------------------------------------------------------------------------------------------
    def clear_all(self) -> None:
        """
        Clear dictionary
        """
        
        self.occt_shape_to_contexts_map = {}

#--------------------------------------------------------------------------------------------------