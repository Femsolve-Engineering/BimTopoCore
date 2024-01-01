
import sys
from datetime import datetime

import math

from queue import Queue
from turtle import distance
from typing import Dict
from typing import Tuple
from typing import List

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_CompSolid, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, topods
from OCC.Core.TopAbs import TopAbs_ShapeEnum, TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPOUND, TopAbs_SHAPE
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape, TopTools_DataMapOfShapeInteger
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Surface, Geom_Geometry, Geom_CartesianPoint
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_EdgeDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NoFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NotPlanar
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_CurveProjectionFailed
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_ParametersOutOfRange
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_VolumeProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeFace
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError
from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder
from OCC.Core.ShapeAnalysis import ShapeAnalysis_ShapeContents
from OCC.Core.IntTools import IntTools_Context
from OCC.Core.BOPTools import BOPTools_AlgoTools

# BimTopoCore
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Wire import Wire
from Core.Face import Face
from Core.Cell import Cell
from Core.Shell import Shell
from Core.Cluster import Cluster

from Attribute import Attribute
from Topology import Topology
from Utilities.TopologicUtilities import CellUtility

class AttributeManager():

    def __init__(self):
        self.occt_shape_to_attributes_map: Dict[TopoDS_Shape, Attribute] = {}
        # self.occt_shape_to_attributes_map: Dict[TopoDS_Shape, Dict[str, Attribute]] = {}

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def get_instance():
        instance = AttributeManager()
        return instance

#--------------------------------------------------------------------------------------------------
    def add(self, topology: Topology, attribute_name: str, attribute: Attribute) -> None:
        
        self.add(topology.get_occt_shape(), attribute_name, attribute)

#--------------------------------------------------------------------------------------------------
    def add(self, occt_shape: TopoDS_Shape, attribute_name: str, attribute: Attribute) -> None:
        
        occt_shapes = self.occt_shape_to_attributes_map.keys()

        if occt_shape not in occt_shapes:

            attribute_map: Dict[str, Attribute] = {}

            self.occt_shape_to_attributes_map[occt_shape] = attribute_map

        self.occt_shape_to_attributes_map[occt_shape][attribute_name] = attribute

#--------------------------------------------------------------------------------------------------
    def remove(self, topology: Topology, attribute_name: str) -> None:
        
        self.remove(topology.get_occt_shape(), attribute_name)

#--------------------------------------------------------------------------------------------------
    def remove(self, occt_shape: TopoDS_Shape, attribute_name: str) -> None:
        
        occt_shapes = self.occt_shape_to_attributes_map.keys()

        if occt_shape in occt_shapes:

            del self.occt_shape_to_attributes_map[occt_shape][attribute_name]

#--------------------------------------------------------------------------------------------------
    def find(self, occt_shape: TopoDS_Shape, attribute_name: str) -> Attribute:
        
        occt_shapes = self.occt_shape_to_attributes_map.keys()

        if occt_shape in occt_shapes:

            attribute_map: Dict[str, Attribute] = self.occt_shape_to_attributes_map[occt_shape]

            names = attribute_map.keys()

            if attribute_name in names:

                return attribute_map[attribute_name]

            return None

        return None

#--------------------------------------------------------------------------------------------------
    def find_all(self, occt_shape: TopoDS_Shape, attributes: Dict[str, Attribute]) -> bool:
        
        occt_shapes = self.occt_shape_to_attributes_map.keys()

        if occt_shape in occt_shapes:

            attributes = self.occt_shape_to_attributes_map[occt_shape]
            return True

        return False

#--------------------------------------------------------------------------------------------------
    def clear_one(self, occt_shape: TopoDS_Shape) -> None:
        
        occt_shapes = self.occt_shape_to_attributes_map.keys()

        if occt_shape in occt_shapes:

            del self.occt_shape_to_attributes_map[occt_shape]

#--------------------------------------------------------------------------------------------------
    def clear_all(self) -> None:
        
        self.occt_shape_to_attributes_map = {}

#--------------------------------------------------------------------------------------------------
    def copy_attributes(self, occt_origin_shape: TopoDS_Shape, occt_destination_shape: TopoDS_Shape, add_duplicate_entries: bool) -> None:
        
        origin_attributes: Dict[str, Attribute] = {}
        does_origin_have_dictionary: bool = self.find_all(occt_origin_shape, origin_attributes)

        if not does_origin_have_dictionary:
            return

        destination_attributes: Dict[str: Attribute] = {}
        does_origin_have_dictionary: bool = self.find_all(occt_origin_shape, destination_attributes)

        if does_origin_have_dictionary:

            origin_names = origin_attributes.keys()

            for origin_name in origin_names:

                # This mode will add values of the same keys into a list
                if add_duplicate_entries:

                    # Does the key already exist in the destination's Dictionary?

                    destination_names = destination_attributes.keys()

                    # If yes (there is already an attribe), create a list
                    if origin_name in destination_names:

                        old_destination_attribute = destination_attributes[origin_name]

                        attributes: List[Attribute] = []

                        # If a list, get the old list
                        if isinstance(old_destination_attribute, list) :
                            attributes = old_destination_attribute
                        else:
                            attributes.append(old_destination_attribute)

                        attributes.append(origin_attributes[origin_name])
                        destination_attributes[origin_name] = attributes

                    # If not, assign the value from the origin
                    else:
                        destination_attributes[origin_name] = origin_attributes[origin_name]

                #This mode will overwrite an old value with the same key
                else:
                    destination_attributes[origin_name] = origin_attributes[origin_name]

            self.occt_shape_to_attributes_map[occt_destination_shape] = destination_attributes

        else:
            self.occt_shape_to_attributes_map[occt_destination_shape] = origin_attributes

#--------------------------------------------------------------------------------------------------
    def deep_copy_attributes(self, occt_shape_1: TopoDS_Shape, occt_shape_2: TopoDS_Shape) -> None:
        
        # For parent topology
        attributes: Dict[str, Attribute] = {}
        is_found_1 = self.find_all(occt_shape_1, attributes)

        if is_found_1:

            if occt_shape_1.ShapeType() == TopAbs_SOLID:
                occt_vertex = CellUtility.internal_vertex(topods.Solid(occt_shape_1), 0.0001).get_occt_vertex()
            
            else:
                occt_vertex = Topology.center_of_mass(occt_shape_1)

            topology_type = Topology.get_topology_type(occt_shape_1.ShapeType())

            occt_selected_sub_topology: TopoDS_Shape = Topology.select_sub_topology(occt_shape_2, occt_vertex, topology_type)

            if not occt_selected_sub_topology.IsNull():
                self.copy_attributes(occt_shape_1, occt_selected_sub_topology)

        # Get all subtopologies
        for occt_shape_type_int in range(int(occt_shape_1.ShapeType()) + 1, int(TopAbs_SHAPE)):

            occt_shape_type: TopAbs_ShapeEnum = TopAbs_ShapeEnum(occt_shape_type_int)

            occt_explorer = TopExp_Explorer(occt_shape_1, occt_shape_type)

            while occt_explorer.More():

                occt_sub_shape_1 = occt_explorer.Current()

                attributes: Dict[str, Attribute] = {}

                is_found_2: bool = self.find_all(occt_sub_shape_1, attributes)

                if not is_found_2:
                    continue

                # WARNING: very costly. Only do this if necessary.
                if occt_shape_1.ShapeType() == TopAbs_SOLID:
                    occt_vertex = CellUtility.internal_vertex(topods.Solid(occt_sub_shape_1), 0.0001).get_occt_vertex()
            
                else:
                    occt_vertex = Topology.center_of_mass(occt_sub_shape_1)

                topology_type = Topology.get_topology_type(occt_sub_shape_1.ShapeType())

                occt_selected_sub_topology: TopoDS_Shape = Topology.select_sub_topology(occt_shape_2, occt_vertex, topology_type)

                if not occt_selected_sub_topology.IsNull():
                    self.copy_attributes(occt_sub_shape_1, occt_selected_sub_topology)

                occt_explorer.Next()

#--------------------------------------------------------------------------------------------------
    def get_attributes_in_sub_shapes(self, occt_shape: TopoDS_Shape, shapes_to_attributes_map: Dict[TopoDS_Shape, Dict[str, Attribute]]) -> None:
        
        # For parent topology
        parent_attribute_map: Dict[str, Attribute] = {}
        is_found: bool = self.find_all(occt_shape, parent_attribute_map)

        if len(list(parent_attribute_map.keys())) != 0:

            shapes_to_attributes_map[occt_shape] = parent_attribute_map

        # Get all subtopologies
            for occt_shape_type_int in range(int(occt_shape.ShapeType() + 1), int(TopAbs_SHAPE)):

                occt_shape_type = TopAbs_ShapeEnum(occt_shape_type_int)

                occt_explorer = TopExp_Explorer(occt_shape, occt_shape_type)

                while occt_explorer.More():

                    occt_sub_shape = occt_explorer.Current()

                    child_attribute_map: Dict[str, Attribute] = {}

                    is_found: bool = self.find_all(occt_sub_shape, child_attribute_map)

                    if len(list(child_attribute_map.keys())) != 0:
                        shapes_to_attributes_map[occt_sub_shape] = child_attribute_map

                    occt_explorer.Next()