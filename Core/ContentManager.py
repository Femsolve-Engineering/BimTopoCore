
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

from Topology import Topology
from Core.TopologyConstants import TopologyTypes


class ContentManager():

    def __init__(self) -> None:
        
        self.occt_shape_to_contents_map = {}

#--------------------------------------------------------------------------------------------------
    def add(self, occt_shape: TopoDS_Shape, content_topology: Topology) -> None:
        """
        Add content to dictionary
        """
        
        # If the OCCT shape does not have a content, initialise it in the map
        if occt_shape not in self.occt_shape_to_contents_map:
            self.occt_shape_to_contents_map[occt_shape] = []

        self.occt_shape_to_contents_map[occt_shape].append(content_topology)

#--------------------------------------------------------------------------------------------------
    def remove(self, occt_shape: TopoDS_Shape, occt_content_topology: TopoDS_Shape) -> None:
        """
        Remove content from dictionary
        """

        new_list: List[TopoDS_Shape] = []
        
        # If the OCCT shape does not have a content, initialise it in the map
        if occt_shape in self.occt_shape_to_contents_map:

            for content in self.occt_shape_to_contents_map[occt_shape]:
                if not content.get_occt_shape().IsSame(occt_content_topology):
                    new_list.append(content)

            self.occt_shape_to_contents_map[occt_shape] = new_list

#--------------------------------------------------------------------------------------------------
    def find(self, occt_shape: TopoDS_Shape, contents: List[Topology]) -> boolean:
        """
        Find shape in dictionary and extend list of contents
        """
        
        if occt_shape in self.occt_shape_to_contents_map:

            contents.extend(self.occt_shape_to_contents_map[occt_shape])
            return True

        return False

#--------------------------------------------------------------------------------------------------
    def has_content(self, occt_shape: TopoDS_Shape, occt_content_topology: TopoDS_Shape) -> boolean:
        """
        Returns True if the OCCT shape contains the content Topology, otherwise False
        """
        
        contents = list[Topology] = []

        has_contents = self.find(occt_shape, contents)

        if not has_contents:
            return False

        for content in contents:

            if content.get_occt_shape().IsSame(occt_content_topology):
                return True

        return False

#--------------------------------------------------------------------------------------------------
    def clear_one(self, occt_shape: TopoDS_Shape) -> None:
        """
        Remove shape from dictionary
        """
        
        if occt_shape in self.occt_shape_to_contents_map:

            del self.occt_shape_to_contents_map[occt_shape]

#--------------------------------------------------------------------------------------------------
    def clear_all(self):
        """
        Clear dictionary
        """
        
        self.occt_shape_to_contents_map = {}

#--------------------------------------------------------------------------------------------------