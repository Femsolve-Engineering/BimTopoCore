

import sys
from datetime import datetime

import math

from queue import Queue
from turtle import distance
from typing import Dict
from typing import Tuple
from typing import List
from xml.dom import INVALID_CHARACTER_ERR

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_CompSolid, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, topods
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape, TopTools_MapIteratorOfMapOfShape, TopTools_DataMapOfShapeInteger, TopTools_DataMapOfShapeListOfShape, TopTools_DataMapIteratorOfDataMapOfShapeListOfShape
from OCC.core.TopExp import TopExp_Explorer
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
from OCC.Core.BRepGProp import brepgprop_LinearProperties, VolumeProperties
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
from Core.Face import Face, FaceGUID
from Core.Cell import Cell
from Core.CellComplex import CellComplex
from Core.Shell import Shell
from Core.Cluster import Cluster

from Core.Aperture import Aperture
from Core.Attribute import Attribute
from Core.AttributeManager import AttributeManager
from Core.DoubleAttribute import DoubleAttribute
from Core.IntAttribute import IntAttribute
from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.TopologicalQuery import TopologicalQuery
from Utilities import TopologicUtilities

# class Node:

#     def __init__(self, val: TopoDS_Vertex, path: List[TopoDS_Vertex], distance: float):

#         self.val = val
#         self.path = path
#         self.distance = distance

class Node:

    def __init__(self):

        self.val: TopoDS_Vertex
        self.path: List[TopoDS_Vertex]
        self.distance: float

class Graph(Topology):

    def __init__(self, vertices: List[Vertex], edges: List[Edge]) -> None:

        self.base_graph_dictionary = {}
        self.occt_edges = TopTools_MapOfShape()
        
        self.add_vertices(vertices, 0.0001)
        self.add_edges(edges, 0.0001)

    def __init__(self, another_graph: 'Graph') -> None:
        
        self.base_graph_dictionary = another_graph.base_graph_dictionary
        self.occt_edges = another_graph.occt_edges

#--------------------------------------------------------------------------------------------------
    def by_vertices(self, vertices: List[Vertex], edges: List[Edge]) -> 'Graph':
        
        graph = Graph(vertices, edges)
        return graph

#--------------------------------------------------------------------------------------------------
    def by_topology(self, topology: Topology, \
                          direct: bool, \
                          via_shared_topologies: bool, \
                          via_shared_apertures: bool, \
                          to_exterior_topologies: bool, \
                          to_exterior_apertures: bool, \
                          use_face_internal_vertex: bool, \
                          tolerance: float) -> 'Graph':

        type = topology.get_shape_type()

        if type == TopologyTypes.VERTEX:
            return Graph.by_vertex(topology.vertices, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.EDGE:
            return Graph.by_edge(topology.edges, direct, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.WIRE:
            return Graph.by_wire(topology.wires, direct, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.FACE:
            return Graph.by_face(topology.faces, to_exterior_topologies, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.SHELL:
            return Graph.by_shell(topology.shells, direct, via_shared_topologies, via_shared_apertures, to_exterior_topologies, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.CELL:
            return Graph.by_cell(topology.cells, to_exterior_topologies, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.CELLCOMPLEX:
            return Graph.by_cellComplex(topology.cell_complexes, direct, via_shared_topologies, via_shared_apertures, to_exterior_topologies, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.CLUSTER:
            return Graph.by_cluster(topology.clusters, direct, via_shared_topologies, via_shared_apertures, to_exterior_topologies, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        elif type == TopologyTypes.APERTURE:
            return Graph.by_topology(topology.apertures, direct, via_shared_topologies, via_shared_apertures, to_exterior_topologies, to_exterior_apertures, use_face_internal_vertex, tolerance) 

        else:
            # raise RuntimeError("Fails to create a graph due to an unknown type of topology.")
            return None

#--------------------------------------------------------------------------------------------------
    def topology(self) -> 'Cluster':

        # Graph: visualise in this order:
		# 1. the edges
		# 2. isolated vertices

		# For a loop: circle, radius/diameter/circumference = average of the edge lengths

        topologies: List['Topology'] = []

        processed_adjacency: Dict[TopoDS_Vertex, TopTools_MapOfShape] = {}

        # dictionary_pair = [occt_vertex_1, map_of_shape]
        # dictionary_pair.first = occt_vertex_1
        # dictionary_pair.second = map_of_shape
        for occt_vertex_1, base_map_of_shapes in self.base_graph_dictionary.items():

            vertex_1 = Vertex(occt_vertex_1, "")
            
            if base_map_of_shapes.Size() == 0:

                # Just add the vertex
                topologies.append(vertex_1)
                processed_adjacency[occt_vertex_1] = TopTools_MapOfShape()

            else:

                # Create Edges
                map_iterator = TopTools_MapIteratorOfMapOfShape(base_map_of_shapes)

                while map_iterator.More():

                    occt_vertex_2 = map_iterator.Value()
                    vertex_2 = Vertex(occt_vertex_2, "")

                    if occt_vertex_1 in processed_adjacency.keys():
                        map_of_shapes = processed_adjacency[occt_vertex_1]

                        if map_of_shapes.Contains(occt_vertex_2):
                            continue

                    if occt_vertex_2 in processed_adjacency.keys():
                        map_of_shapes = processed_adjacency[occt_vertex_2]

                        if map_of_shapes.Contains(occt_vertex_1):
                            continue

                    occt_edge = self.find_edge(vertex_1.get_occt_vertex(), vertex_2.get_occt_vertex())

                    if occt_edge.IsNull():
                        continue

                    edge = Edge(occt_edge, "")

                    topologies.append(edge)

                    processed_adjacency[vertex_1.get_occt_vertex()].Add(vertex_2.get_occt_vertex())
                    processed_adjacency[vertex_2.get_occt_vertex()].Add(vertex_1.get_occt_vertex())

                    map_iterator.Next()

        cluster = Cluster.by_topologies(topologies)
        return cluster

#--------------------------------------------------------------------------------------------------
    def vertices(self) -> List['Vertex']:

        vertices: List['Vertex'] = []
        
        for occt_vertex, map_of_shapes in self.base_graph_dictionary.items():

            vertex = Vertex(occt_vertex, "")

            vertices.append(vertex)

        return vertices

#--------------------------------------------------------------------------------------------------
    def edges(self, edges: List['Edge'], tolerance: float) -> None:
        
        vertices: List['Vertex'] = []
        self.edges(vertices, tolerance, edges)

#--------------------------------------------------------------------------------------------------
    def edges(self, vertices: list[Vertex], tolerance: float, edges: List['Edge']) -> None:
        
        # isEmpty == True
        if not vertices:

            processed_adjacency: Dict[TopoDS_Vertex, TopTools_MapOfShape] = {}

            # dictionary_pair = [occt_vertex_1, map_of_shape]
            # dictionary_pair.first = occt_vertex_1
            # dictionary_pair.second = map_of_shape
            for occt_vertex_1, base_map_of_shapes in self.base_graph_dictionary.items():

                vertex_1 = Vertex(occt_vertex_1, "")

                # Create edges
                map_iterator = TopTools_MapIteratorOfMapOfShape(base_map_of_shapes)

                while map_iterator.More():

                    occt_vertex_2 = map_iterator.Value()

                    if occt_vertex_1 in processed_adjacency.keys():
                        map_of_shapes = processed_adjacency[occt_vertex_1]

                        if map_of_shapes.Contains(occt_vertex_2):
                            continue

                    if topods.Vertex(occt_vertex_2) in processed_adjacency.keys():
                        map_of_shapes = processed_adjacency[topods.Vertex(occt_vertex_2)]

                        if map_of_shapes.Contains(occt_vertex_1):
                            continue


                    vertex_2 = Vertex(topods.Vertex(occt_vertex_2))

                    occt_edge = self.find_edge(vertex_1.get_occt_vertex(), vertex_2.get_occt_vertex())

                    if occt_edge.IsNull():
                        continue

                    edge = Edge(occt_edge, "")
                    edges.append(edge)

                    processed_adjacency[vertex_1.get_occt_vertex()].Add(vertex_2.get_occt_vertex())
                    processed_adjacency[vertex_2.get_occt_vertex()].Add(vertex_1.get_occt_vertex())

                    map_iterator.Next()

        else:

            processed_adjacency: Dict[TopoDS_Vertex, TopTools_MapOfShape] = {}

            for vertex in vertices:

                # Get coincident edges
                this_incident_edges: List[Edge] = []

                self.incident_edges(vertex, tolerance, this_incident_edges)

                # Ceck: is already added?
                for incident_edge in this_incident_edges:

                    start_vertex = incident_edge.start_vertex()
                    end_vertex = incident_edge.end_vertex()

                    is_added = False

                    occt_start_vertex = start_vertex.get_occt_vertex()
                    occt_end_vertex = end_vertex.get_occt_vertex()

                    if occt_start_vertex in processed_adjacency.keys():
                        map_of_shapes = processed_adjacency[occt_start_vertex]

                        if map_of_shapes.Contains(occt_end_vertex):
                            is_added = True

                    if not is_added:

                        if occt_end_vertex in processed_adjacency.keys():
                            map_of_shapes = processed_adjacency[occt_end_vertex]

                        if map_of_shapes.Contains(occt_start_vertex):
                            is_added = True

                    if not is_added:

                        edges.append(incident_edge)
                        processed_adjacency[occt_start_vertex].Add(occt_end_vertex)

#--------------------------------------------------------------------------------------------------
    def add_vertices(self, vertices: List['Vertex'], tolerance: float) -> None:
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return None

        for vertex in vertices:

            if not self.contains_vertex(vertex, tolerance):

                self.base_graph_dictionary[vertex.get_occt_vertex()] = TopTools_MapOfShape()


#--------------------------------------------------------------------------------------------------
    def add_edges(self, edges: List['Edge'], tolerance: float) -> None:
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return None

        for edge in edges:

            if not self.contains_edge(edge, tolerance):

                start_vertex = edge.start_vertex()
                occt_start_coincident_vertex = self.get_coincident_vertex(start_vertex.get_occt_vertex(), tolerance)
                if occt_start_coincident_vertex.IsNull():
                    occt_start_coincident_vertex = start_vertex.get_occt_vertex()

                end_vertex = edge.end_vertex()
                occt_end_coincident_vertex = self.get_coincident_vertex(end_vertex.get_occt_vertex(), tolerance)
                if occt_end_coincident_vertex.IsNull():
                    occt_end_coincident_vertex = end_vertex.get_occt_vertex()

                self.base_graph_dictionary[occt_start_coincident_vertex].Add(occt_end_coincident_vertex)
                self.base_graph_dictionary[occt_end_coincident_vertex].Add(occt_start_coincident_vertex)

                self.occt_edges.Add(edge.get_occt_edge())

#--------------------------------------------------------------------------------------------------
    def vertex_degree(self, vertex: Vertex) -> int:
        
        return self.vertex_degree(vertex.get_occt_vertex())

#--------------------------------------------------------------------------------------------------
    def vertex_degree(self, occt_vertex: TopoDS_Vertex) -> int:
        
        if occt_vertex not in list(self.base_graph_dictionary.keys()):

            return 0

        else:
            
            number_of_edges = len(self.base_graph_dictionary[occt_vertex])

            if occt_vertex in self.base_graph_dictionary[occt_vertex]:
                number_of_loops = 1
            else: 
                number_of_loops = 0

            degree = number_of_edges + number_of_loops

        return degree

#--------------------------------------------------------------------------------------------------
    def adjacent_vertices(self, vertex: Vertex, adjacent_vertices: List['Vertex']) -> None:

        occt_adjacent_vertices = TopTools_MapOfShape()
        self.adjacent_vertices(vertex.get_occt_vertex(), occt_adjacent_vertices)

        map_iterator = TopTools_MapIteratorOfMapOfShape(occt_adjacent_vertices)
        
        while map_iterator.More():

            occt_shape = map_iterator.Value()
            vertex = Vertex(occt_shape.get_occt_vertex(), "")

            adjacent_vertices.append(vertex)

            map_iterator.Next()

#--------------------------------------------------------------------------------------------------
    def adjacent_vertices(self, occt_vertex: TopoDS_Vertex, occt_adjacent_vertices: TopTools_MapOfShape) -> None:
        
        if not self.contains_vertex():
            return

        if occt_vertex in list(self.base_graph_dictionary.keys()):
            
            map_of_shape = self.base_graph_dictionary[occt_vertex]
            occt_adjacent_vertices = map_of_shape

#--------------------------------------------------------------------------------------------------
    def connect(self, vertices_1: List['Vertex'], vertices_2: List['Vertex'], tolerance: float) -> None:
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return

        vertex_1_iterator = iter(vertices_1)
        vertex_2_iterator = iter(vertices_2)

        for vertex_1, vertex_2 in zip(vertex_1_iterator, vertex_2_iterator):

            occt_query_vertex_1 = self.get_coincident_vertex(vertex_1.get_occt_vertex(), tolerance)
            if occt_query_vertex_1.IsNull():
                occt_query_vertex_1 = vertex_1.get_occt_vertex()

            occt_query_vertex_2 = self.get_coincident_vertex(vertex_2.get_occt_vertex(), tolerance)
            if occt_query_vertex_2.IsNull():
                occt_query_vertex_2 = vertex_2.get_occt_vertex()

            adding_edge = False

            if not self.base_graph_dictionary[occt_query_vertex_1].Contains(occt_query_vertex_2):
                self.base_graph_dictionary[occt_query_vertex_1].Add(occt_query_vertex_2)
                adding_edge = True

            if not self.base_graph_dictionary[occt_query_vertex_2].Contains(occt_query_vertex_1):
                self.base_graph_dictionary[occt_query_vertex_2].Add(occt_query_vertex_1)
                adding_edge = True

            if adding_edge:
                query_vertex_1 = Vertex(occt_query_vertex_1, "")
                query_vertex_2 = Vertex(occt_query_vertex_2, "")

                edge = Edge.by_start_vertex_end_vertex(query_vertex_1, query_vertex_2)

                self.occt_edges.Add(edge.get_occt_edge())

#--------------------------------------------------------------------------------------------------
    def contains_vertex(self, vertex: Vertex, tolerance: float) -> bool:
        
        return self.contains_vertex(vertex.get_occt_vertex(), tolerance)

#--------------------------------------------------------------------------------------------------
    def contains_vertex(self, occt_vertex, tolerance: float) -> bool:
        
        occt_coincident_vertex = self.get_coincident_vertex(occt_vertex, tolerance)
        return not occt_coincident_vertex.IsNull()

#--------------------------------------------------------------------------------------------------
    def contains_edge(self, edge: Edge, tolerance) -> bool:
        
        start_vertex = edge.start_vertex()
        end_vertex = edge. end_vertex()

        return self.contains_edge(start_vertex.get_occt_vertex(), end_vertex.get_occt_vertex(), tolerance)

#--------------------------------------------------------------------------------------------------
    def contains_edge(self, occt_vertex_1: TopoDS_Vertex, occt_vertex_2: TopoDS_Vertex, tolerance: float) -> bool:
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return False

        occt_start_coincident_vertex = self.get_coincident_vertex(occt_vertex_1, tolerance)
        if occt_start_coincident_vertex.IsNull():
            return False

        occt_end_coincident_vertex = self.get_coincident_vertex(occt_vertex_2, tolerance)
        if occt_end_coincident_vertex.IsNull():
            return False

        if occt_start_coincident_vertex in list(self.base_graph_dictionary.keys()):
            adjacent_vertices_to_start = self.base_graph_dictionary[occt_start_coincident_vertex] # map_of_shape
            
        if occt_end_coincident_vertex in list(self.base_graph_dictionary.keys()):
            adjacent_vertices_to_end = self.base_graph_dictionary[occt_end_coincident_vertex] # map_of_shape
            
        return adjacent_vertices_to_start.Contains(occt_end_coincident_vertex) and adjacent_vertices_to_end.Contains(occt_start_coincident_vertex)

#--------------------------------------------------------------------------------------------------
    def degree_sequence(self, degree_sequence: List[int]) -> None:
        
        for occt_vertex, map_of_shape in self.base_graph_dictionary.items():

            vertex = Vertex(occt_vertex, "")
            degree_sequence.append(self.vertex_degree(vertex))

            degree_sequence.sort(reverse=True)

#--------------------------------------------------------------------------------------------------
    def density(self) -> float:
        
        num_of_vertices = len(self.base_graph_dictionary.keys())

        edges: List['Edge'] = []
        self.edges(edges)

        num_of_edges = len(edges)
        denominator = num_of_vertices * (num_of_vertices - 1)

        if denominator > 0.0001 and denominator < 0.0001:
            # Divide by zero, return the largest double number
            return sys.float_info.max

        return (2 * num_of_edges) / denominator


#--------------------------------------------------------------------------------------------------
    def is_complete(self) -> bool:
        
        return self.density() > 0.9999

#--------------------------------------------------------------------------------------------------
    def isolated_vertices(self, isolated_vertices: List[Vertex]) -> None:
        
        for occt_vertex, map_of_shape in self.base_graph_dictionary.items():

            if map_of_shape.isEmpty():
                vertex = Vertex(occt_vertex, "")
                isolated_vertices.append(vertex)

#--------------------------------------------------------------------------------------------------
    def minimum_delta(self) -> int:
        
        minimum_delta = sys.maxsize

        for occt_vertex, map_of_shape in self.base_graph_dictionary.items():
            vertex_degree = self.vertex_degree(occt_vertex)
            if vertex_degree < minimum_delta:
                minimum_delta = vertex_degree

        return minimum_delta

#--------------------------------------------------------------------------------------------------
    def maximum_delta(self):
        
        maximum_delta = 0

        for occt_vertex, map_of_shape in self.base_graph_dictionary.items():
            vertex_degree = self.vertex_degree(occt_vertex)
            if vertex_degree > maximum_delta:
                maximum_delta = vertex_degree

        return maximum_delta

#--------------------------------------------------------------------------------------------------
    def all_paths(self, start_vertex: Vertex,
                        end_vertex: Vertex,
                        use_time_limit: bool,
                        time_limit_in_seconds:
                        int, paths: Wire) -> None:
        
        path: List[Vertex] = []
        starting_time: datetime = datetime.now()
        self.all_paths(start_vertex, end_vertex, use_time_limit, time_limit_in_seconds, starting_time, path, paths)

#--------------------------------------------------------------------------------------------------
    def all_paths(self, start_vertex: Vertex,
                        end_vertex: Vertex, use_time_limit: bool,
                        time_limit_in_seconds: int,
                        starting_time: datetime,
                        path: List[Vertex],
                        paths: List[Wire]):
        
        if use_time_limit:

            current_time = datetime.now()
            time_difference = current_time - starting_time
            time_difference_in_seconds = time_difference.seconds

            if time_difference_in_seconds >= time_limit_in_seconds:
                return

        if not self.contains_vertex(start_vertex, 0.0001):
            return

        path.append(start_vertex)

        if start_vertex.is_same(end_vertex):

            # Create a wire
            path_wire: Wire = self.construct_path(path)
            paths.append(path_wire)
            return

        # 2X:

        # if use_time_limit:

        #     current_time = datetime.now()
        #     time_difference = current_time - starting_time
        #     time_difference_in_seconds = time_difference.seconds

        #     if time_difference_in_seconds >= time_limit_in_seconds:
        #         return

        occt_start_vertex = start_vertex.get_occt_vertex()

        if occt_start_vertex in self.base_graph_dictionary.keys():
            occt_connected_vertices: TopTools_MapOfShape = self.base_graph_dictionary[occt_start_vertex]

            map_iterator = TopTools_MapIteratorOfMapOfShape(occt_connected_vertices)

            while map_iterator.More():

                occt_vertex = map_iterator.Value()
                connected_vertex = Vertex(occt_vertex, "")

                if connected_vertex in path:

                    extended_paths: List[Wire] = []
                    previous_path: List[Vertex] = path

                    if use_time_limit:

                        current_time = datetime.now()
                        time_difference = current_time - starting_time
                        time_difference_in_seconds = time_difference.seconds

                        if time_difference_in_seconds >= time_limit_in_seconds:
                            break

                    self.all_paths(connected_vertex, end_vertex, use_time_limit, time_limit_in_seconds, starting_time, previous_path, extended_paths)

                    for extended_path in extended_paths:
                        paths.append(extended_path)

                if use_time_limit:

                    current_time = datetime.now()
                    time_difference = current_time - starting_time
                    time_difference_in_seconds = time_difference.seconds

                    if time_difference_in_seconds >= time_limit_in_seconds:
                        break

                map_iterator.Next()
        

#--------------------------------------------------------------------------------------------------
    def path(self, start_vertex: Vertex, end_vertex: Vertex) -> 'Wire':
        
        path: List[Vertex] = []
        return self.path(start_vertex, end_vertex, path)

#--------------------------------------------------------------------------------------------------
    def path(self, start_vertex: Vertex, end_vertex: Vertex, path: List[Vertex]) -> 'Wire':
        
        path.append(start_vertex)

        if not self.contains_vertex(start_vertex, 0.0001):
            return None

        if start_vertex.is_same(end_vertex):
            path_wire: Wire = self.construct_path(path)
            return path_wire

        occt_start_vertex = start_vertex.get_occt_vertex()

        if occt_start_vertex in self.base_graph_dictionary.keys():

            occt_connected_vertices: TopTools_MapOfShape = self.base_graph_dictionary[occt_start_vertex]
            map_iterator = TopTools_MapIteratorOfMapOfShape(occt_connected_vertices)

            while map_iterator.More():

                occt_vertex = map_iterator.Value()
                connected_vertex = Vertex(occt_vertex, "")

                if connected_vertex in path:

                    extended_path = self.path(connected_vertex, end_vertex, path)

                    if extended_path:
                        return extended_path

                map_iterator.Next()


#--------------------------------------------------------------------------------------------------
    def shortest_path(self, start_vertex: Vertex, end_vertex: Vertex, vertex_key: str, edge_key: str) -> 'Wire':
        
        return self.shortest_path(start_vertex.get_occt_vertex(), end_vertex.get_occt_vertex(), vertex_key, edge_key)

#--------------------------------------------------------------------------------------------------
    def shortest_path(self, occt_start_vertex: Vertex, occt_end_vertex: Vertex, vertex_key: str, edge_key: str) -> 'Wire':
        
        # Dijkstra's: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Pseudocode

        vertex_list: List[TopoDS_Vertex] = []
        distance_map: Dict[TopoDS_Vertex, float] = {}
        parent_map: Dict[TopoDS_Vertex, TopoDS_Vertex] = {}

        for occt_vertex, map_of_shape in self.base_graph_dictionary.items():

            distance_map[occt_vertex] = sys.float_info.max # max numeric limit; float
            parent_map[occt_vertex] = topods.Vertex() # or TopoDS_Vertex()
            vertex_list.append(occt_vertex)

        distance_map[occt_start_vertex] = 0

        while len(vertex_list) != 0:

            # Find vertex with the lowest distance
            min_distance = sys.float_info.max
            occt_vertex_min_distance: TopoDS_Vertex

            for vertex_in_queue in vertex_list:
                distance = distance_map[vertex_in_queue]

                if distance <= min_distance:
                    min_distance = distance
                    occt_vertex_min_distance = vertex_in_queue

            vertex_list.remove(occt_vertex_min_distance)

            if occt_vertex_min_distance.IsNull():
                continue

            if occt_vertex_min_distance == occt_end_vertex:

                path: List[Vertex] = []
                occt_current_vertex: TopoDS_Vertex = occt_vertex_min_distance

                if occt_current_vertex in parent_map.keys() and occt_vertex_min_distance == occt_start_vertex:

                    while not occt_current_vertex.IsNull():

                        vertex = Vertex(occt_current_vertex, "")
                        path.insert(0, vertex)

                        if occt_current_vertex in parent_map.keys():
                            occt_current_vertex = parent_map[occt_current_vertex]
                        
                        else:
                            occt_current_vertex = TopoDS_Vertex()

        return self.construct_path(path)

#--------------------------------------------------------------------------------------------------
    def shortest_paths(self, start_vertex: Vertex,
                             end_vertex: Vertex,
                             vertex_key: str,
                             edge_key: str,
                             use_time_limit: bool,
                             time_limit: int,
                             paths: List[Wire]) -> None:
        
        return self.shortest_paths(start_vertex.get_occt_vertex(), end_vertex.get_occt_vertex(), vertex_key, edge_key, use_time_limit, time_limit, paths)

#--------------------------------------------------------------------------------------------------
    def shortest_paths(self, occt_start_vertex: TopoDS_Vertex,
                             occt_end_vertex: TopoDS_Vertex,
                             vertex_key: str,
                             edge_key: str,
                             use_time_limit: bool,
                             time_limit: int,
                             paths: List[Wire]) -> None:
        
        if time_limit <= 0.0:
            # raise RuntimeError("The time limit must have a positive value.")
            return

        starting_time = datetime.now()

        infinite_distance = sys.float_info.max

        distance_map: Dict[TopoDS_Vertex, float] = {}
        vertices: List[Vertex] = []
        vertex = Vertex(vertices, "")

        for vertex in vertices:
            distance_map[vertex.get_occt_vertex()] = infinite_distance

        occt_vertex_queue = Queue[Node]
        occt_node_paths = List[Node]
        
        start_node = Node()
        start_node.val = occt_start_vertex
        start_node.path = [occt_start_vertex]
        start_node.distance = 0.0

        occt_vertex_queue.put(start_node)

        min_distance = infinite_distance
        distance_map[occt_start_vertex] = 0.0

        while not occt_vertex_queue.empty():

            if use_time_limit:

                current_time = datetime.now()
                time_difference = current_time - starting_time
                time_difference_in_seconds = time_difference.seconds

                if time_difference_in_seconds >= time_limit:
                    break

                current_node: Node = occt_vertex_queue.get()

                if current_node.val == occt_end_vertex and current_node.distance <= min_distance:
                    min_distance = current_node.distance
                    occt_node_paths.append(current_node)

                if current_node.distance <= min_distance:

                    occt_adjacent_vertices = TopTools_MapOfShape()

                    self.adjacent_vertices(current_node.val, occt_adjacent_vertices)

                    occt_adjacent_vertex_iterator = TopTools_MapIteratorOfMapOfShape(occt_adjacent_vertices)

                    while occt_adjacent_vertex_iterator.More():

                        adjacent_vertex = occt_adjacent_vertex_iterator.Value()
                        occt_adjacent_vertex = topods.Vertex(adjacent_vertex)

                        current_distance = distance_map[occt_adjacent_vertex]
                        if current_distance >= current_node.distance:

                            adjacent_vertex = Node()
                            adjacent_vertex.val = occt_adjacent_vertex
                            adjacent_vertex.path = current_node.path
                            adjacent_vertex.path.append(occt_adjacent_vertex)

                            edge_cost = self.compute_edge_cost(current_node.val, occt_adjacent_vertex, edge_key)
                            adjacent_vertex.distance = current_node.distance + edge_cost

                            vertex_cost = self.compute_vertex_cost(occt_adjacent_vertex, vertex_key)
                            adjacent_vertex.distance += vertex_cost

                            distance_map[occt_adjacent_vertex] = adjacent_vertex.distance
                            occt_vertex_queue.put(adjacent_vertex)

                        occt_adjacent_vertex_iterator.Next()

        for node in occt_node_paths:

            if node.distance > min_distance:

                continue

            vertices: List[Vertex] = []

            for occt_vertex in node.path:

                vertex = Vertex(occt_vertex, "")
                vertices.append(vertex)

            if len(vertices) > 1:

                path: Wire = self.construct_path(vertices)

                if path:

                    paths.append(path)

#--------------------------------------------------------------------------------------------------
    def diameter(self) -> int:
        
        vertex_pairs: Dict[Vertex, Vertex] = {}

        for occt_vertex_1, map_of_shape_1 in self.base_graph_dictionary.items():

            vertex_1 = Vertex(occt_vertex_1, "")

            for occt_vertex_2, map_of_shape_2 in self.base_graph_dictionary.items():

                if occt_vertex_1 == occt_vertex_2:
                    continue

                vertex_2 = Vertex(occt_vertex_2, "")

                vertex_pairs[vertex_1] = vertex_2

        max_shortest_path_distance = 0

        for vertex_1, vertex_2 in vertex_pairs.items():

            distance = self.topological_distance(vertex_1, vertex_2)

            if distance > max_shortest_path_distance:

                max_shortest_path_distance = distance

        return max_shortest_path_distance
            
#--------------------------------------------------------------------------------------------------
    def topological_distance(self, start_vertex: Vertex, end_vertex: Vertex, tolerance: float) -> int:
        
        return self.topological_distance(start_vertex.get_occt_vertex, end_vertex.get_occt_vertex, tolerance)

#--------------------------------------------------------------------------------------------------
    def topological_distance(self, occt_start_vertex: TopoDS_Vertex, occt_end_vertex: TopoDS_Vertex, tolerance) -> int:
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return -1

        # Use Breadth-First Search
        occt_coincident_start_vertex = self.get_coincident_vertex(occt_start_vertex, tolerance)
        occt_coincident_end_vertex = self.get_coincident_vertex(occt_end_vertex, tolerance)

        if occt_coincident_start_vertex.is_same(occt_coincident_end_vertex):

            return 0

        occt_vertex_queue = Queue()
        occt_vertex_distance_map = TopTools_DataMapOfShapeInteger() # also to check if the vertex is processed

        occt_vertex_queue.put(occt_coincident_start_vertex)

#--------------------------------------------------------------------------------------------------
    def eccentricity(self, vertex: Vertex) -> int:

        occt_vertex = vertex.get_occt_vertex()
        
        if occt_vertex not in self.base_graph_dictionary.keys():

            # infinite distance
            return sys.maxsize

        occt_adjacent_vertices: TopTools_MapOfShape = self.base_graph_dictionary[occt_vertex]

        eccentricity: int = 0

        map_iterator = TopTools_MapIteratorOfMapOfShape(occt_adjacent_vertices)

        while map_iterator.More():

            distance: int = self.topological_distance(occt_vertex, topods.Vertex(map_iterator.Value()))

            if distance > eccentricity:
                eccentricity = distance

            map_iterator.Next()

        return eccentricity

#--------------------------------------------------------------------------------------------------
    def is_erdoes_gallai(self, sequence: List[int]) -> bool:

        sum = sum(sequence)

        if sum % 2 != 0:
            return False

        if self.is_degree_sequence(sequence):

            # Copy vector without reference
            sequence_vector = sequence[:]
            # size_of_sequence = len(sequence) # unused variable

            for k in range(1, len(sequence) + 1):

                left = sum(sequence_vector[:k])

                end_sequence = []

                for x in sequence_vector[k:]:
                    end_sequence.append(min(x, k))

                right = k * (k - 1) + sum(end_sequence)

                if left > right:
                    return False
        
        else:
            return False

        return True

#--------------------------------------------------------------------------------------------------
    def remove_vertices(self, vertices: List[Vertex]) -> None:
        
        for vertex in vertices:

            occt_vertex = vertex.get_occt_vertex()

            # Check if the connected vertices are connected to the vertex to be deleted. Remove it.

            for occt_vertex_1, map_of_shape_1 in self.base_graph_dictionary.items():

                occt_connected_vertices: TopTools_MapOfShape = map_of_shape_1
                occt_connected_vertices.Remove(occt_vertex)

                occt_edge = self.find_edge(occt_vertex_1, occt_vertex)

                if not occt_edge.IsNull():
                    self.occt_edges.Remove(occt_edge)

            # Remove the entry from this map
            if occt_vertex in list(self.base_graph_dictionary.keys()):

                occt_connected_vertices: TopTools_MapOfShape = self.base_graph_dictionary[occt_vertex]

                occt_connected_vertex_iterator: TopTools_MapIteratorOfMapOfShape = TopTools_MapIteratorOfMapOfShape(occt_connected_vertices)

                while occt_connected_vertex_iterator.More():

                    occt_edge = self.find_edge(occt_vertex, topods.Vertex(occt_connected_vertex_iterator.Value()))

                    if not occt_edge.IsNull():
                        self.occt_edges.Remove(occt_edge)

                    occt_connected_vertex_iterator.Next()

                del self.base_graph_dictionary[occt_vertex]

#--------------------------------------------------------------------------------------------------
    def remove_edges(self, edges: List[Edge], tolerance: float) -> None:
        
        if tolerance <= 0.0:

            # raise RuntimeError("The tolerance must have a positive value.")
            return

        for edge in edges:

            start_vertex: Vertex = edge.start_vertex()
            occt_query_vertex_1: TopoDS_Vertex = self.get_coincident_vertex(start_vertex.get_occt_vertex(), tolerance)

            if occt_query_vertex_1.IsNull():
                continue

            end_vertex: Vertex = edge.end_vertex()
            occt_end_vertex = end_vertex.get_occt_vertex()

            occt_query_vertex_2: TopoDS_Vertex = self.get_coincident_vertex(occt_end_vertex, tolerance)

            if occt_query_vertex_2.IsNull():
                continue

            

            if occt_query_vertex_1 in list(self.base_graph_dictionary.keys()):

                adjacent_vertices: TopTools_MapOfShape = self.base_graph_dictionary[occt_query_vertex_1]
                
                adjacent_vertices.Remove(occt_end_vertex)
                occt_edge: TopoDS_Edge = self.find_edge(occt_query_vertex_1, occt_end_vertex)

                if not occt_edge.IsNull():
                    self.occt_edges.Remove(occt_edge)

            if occt_end_vertex in list(self.base_graph_dictionary.keys()):

                adjacent_vertices: TopTools_MapOfShape = self.base_graph_dictionary[occt_end_vertex]
                
                adjacent_vertices.Remove(occt_query_vertex_1)
                occt_edge: TopoDS_Edge = self.find_edge(occt_end_vertex, occt_query_vertex_1)

                if not occt_edge.IsNull():
                    self.occt_edges.Remove(occt_edge)

#--------------------------------------------------------------------------------------------------
    def vertices_at_coordinates(self, var_x: float, var_y: float, var_z: float, tolerance: float, vertices: List[Vertex]) -> None:
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return

        occt_query_point = {
            'X': var_x,
            'Y': var_y,
            'Z': var_z,
        }

        abs_distance_threshold = abs(tolerance)

        for occt_vertex, map_of_shape in self.base_graph_dictionary.items():

            current_vertex: TopoDS_Vertex = occt_vertex

            point = BRep_Tool.Pnt(current_vertex)
            current_point = Geom_CartesianPoint(point)

            dx = current_point.X() - occt_query_point['X']
            dy = current_point.Y() - occt_query_point['Y']
            dz = current_point.Z() - occt_query_point['Z']

            sq_distance = math.sqrt(dx**2 + dy**2 + dz**2)

            if sq_distance < abs_distance_threshold:

                vertex = Vertex(current_vertex, "")
                vertices.append(vertex)

#--------------------------------------------------------------------------------------------------
    def edge(self, vertex_1: Vertex, vertex_2: Vertex, tolerance: float) -> Edge:
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return None

        occt_query_vertex_1 = self.get_coincident_vertex(vertex_1.get_occt_vertex(), tolerance)
        if occt_query_vertex_1.IsNull():
            return None

        occt_query_vertex_2 = self.get_coincident_vertex(vertex_2.get_occt_vertex(), tolerance)
        if occt_query_vertex_2.IsNull():
            return None

        if occt_query_vertex_1 not in list(self.base_graph_dictionary.keys()):
            return None

        adjacent_vertices_to_vertex_1: TopTools_MapOfShape = self.base_graph_dictionary[occt_query_vertex_1]
        if not adjacent_vertices_to_vertex_1.Contains(occt_query_vertex_2):
            return None

        occt_edge = self.find_edge(occt_query_vertex_1, occt_query_vertex_2)
        if occt_edge.IsNull():
            return None

        edge_topology = Topology.by_occt_shape(occt_edge)
        edge = TopologicalQuery.downcast(edge_topology)
        
        return edge

#--------------------------------------------------------------------------------------------------
    def incident_edges(self, vertex: Vertex, tolerance: float, edges: List[Edge]) -> None:
        
        occt_query_vertex = self.get_coincident_vertex(vertex.get_occt_vertex(), tolerance)
        if occt_query_vertex.IsNull():
            return

        # do not create vertex shape directly! 
        # query_vertex = Vertex(occt_query_vertex, "")

        vertex_topology = Topology.by_occt_shape(occt_query_vertex)
        query_vertex = TopologicalQuery.downcast(vertex_topology)

        adjacent_vertices: List[Vertex] = []

        self.adjacent_vertices(query_vertex, adjacent_vertices)

        for adjacent_vertex in adjacent_vertices:

            occt_edge = self.find_edge(query_vertex.get_occt_vertex(), adjacent_vertex.get_occt_vertex())
            if occt_edge.IsNull():
                continue

            edge = Edge(occt_edge, "")

            edges.append(edge)

#--------------------------------------------------------------------------------------------------
    def calculate_graph_vertex_from_aperture(self, aperture: Aperture, use_face_internal_vertex: bool, tolerance: float) -> Vertex:
        
        aperture_topology = aperture.topology()

        if aperture_topology.get_shape_type() == TopologyTypes.FACE:

            # "TopologicalQuery::Downcast<Face>" not implemented

            # C++ code:
            # Face::Ptr apertureFace = TopologicalQuery::Downcast<Face>(apertureTopology);
            aperture_face = TopologicalQuery.downcast(aperture_topology)

            if use_face_internal_vertex:
                return TopologicUtilities.FaceUtility.internal_vertex(aperture_face, tolerance) # to be implemented...

            else:
                return aperture_face.center_of_mass()

        elif aperture_topology.get_shape_type == TopologyTypes.CELL:

            # TopologicalQuery::Downcast not implemented

            # C++ code
            # TopologicalQuery::Downcast<Cell>(apertureTopology) --> innen jon a cell variable
            cell = TopologicalQuery.downcast(aperture_topology)

            return TopologicUtilities.CellUtility.internal_vertex(cell, tolerance)

        else:
            return aperture.center_of_mass()

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_vertex(vertex: Vertex, to_exterior_apertures: bool, use_face_internal_vertex: bool, tolerance: float) -> 'Graph':
        
        aperture_centres_of_mass: List[Vertex] = []

        if to_exterior_apertures:

            contents: List[Topology] = []

            # Returns all instances of class Topology with base_shape = TopologyTypes.VERTEX
            vertex.get_contents(contents)

            for content in contents:
                if content.get_shape_type() == TopologyTypes.APERTURE:

                    # TopologicalQuery.Downcast(topology) not implemented yet
                    aperture = TopologicalQuery.downcast(content)
                    graph_vertex: Vertex = Graph.calculate_graph_vertex_from_aperture(aperture, use_face_internal_vertex, tolerance)

                    attribute_manager = AttributeManager.get_instance()
                    attribute_manager.copy_attributes(content.get_occt_shape(), graph_vertex.get_occt_shape())

                    aperture_centres_of_mass.append(graph_vertex)

        vertices: List[Vertex] = []
        edges: List[Edge] = []

        for aperture_center_of_mass in aperture_centres_of_mass:
            
            edge = Edge.by_start_vertex_end_vertex(vertex, aperture_center_of_mass)

            edges.append(edge)

            if len(edges) == 0:
                vertices.append(vertex)

            return Graph(vertices, edges)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_edge(self, edge: Edge, direct: bool, to_exterior_apertures: bool, use_face_internal_vertex: bool, tolerance: float) -> 'Graph':
        
        vertices: List[Vertex] = []
        edges: List[Edge] = []
        edge_vetices: List[Vertex] = []

        if direct:

            vertices = edge.vertices()
            edges.append(edge)

        aperture_centres_of_mass = List[Vertex] = []

        if to_exterior_apertures:

            contents = List[Topology] = []
            edge.contents_(contents)

            for content in contents:

                if content.get_shape_type() == TopologyTypes.APERTURE:

                    aperture = TopologicalQuery.downcast(content)

                    content_center_of_mass: Vertex = self.calculate_graph_vertex_from_aperture(aperture, use_face_internal_vertex, tolerance)

                    attribute_manager = AttributeManager.get_instance()
                    attribute_manager.copy_attributes(content.get_occt_shape(), content_center_of_mass.get_occt_shape())

                    vertices.append(content_center_of_mass)

                    edge_vertices: List[Vertex] = edge.vertices()

                    for vertex in vertices:
                        edge = Edge.by_start_vertex_end_vertex(vertex, content_center_of_mass)
                        edges.append(edge)

        return Graph(vertices, edges)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_wire(wire: Wire, direct: bool, to_exterior_apertures: bool, use_face_internal_vertex: bool, tolerance: float) -> 'Graph':
        
        vertices: List[Vertex] = []
        edges: List[Edge] = []

        if direct or to_exterior_apertures:

            edges = wire.edges()
            vertices = wire.vertices()

            if to_exterior_apertures:

                # Iterate through the edges
                for edge in edges:

                    # Get the apertures
                    contents: List[Topology] = []
                    edge.contents_(contents)

                    edge_vertices: List[Vertex] = []
                    edge_vertices = edge.vertices()

                    for content in contents:

                        if content.get_shape_type() == TopologyTypes.APERTURE:

                            aperture = TopologicalQuery.downcast(content)

                            content_center_of_mass = Graph.calculate_graph_vertex_from_aperture(aperture, use_face_internal_vertex, tolerance)

                            attribute_manager = AttributeManager.get_instance()
                            attribute_manager.copy_attributes(content.get_occt_shape(), content_center_of_mass.get_occt_shape())

                            vertices.append(content_center_of_mass)

                            for edge_vertex in edge_vertices:
                                edge = Edge.by_start_vertex_end_vertex(edge_vertex, content_center_of_mass)
                                edges.append(edge)

        return Graph(vertices, edges)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_face(face: Face, to_exterior_topologies: bool, to_exterior_apertures: bool, use_face_internal_vertex: bool, tolerance: float) -> float:
        
        vertices: List[Vertex] = []
        edges: List[Edge] = []

        internal_vertex: Vertex = None

        if use_face_internal_vertex:
             internal_vertex = TopologicUtilities.FaceUtility.internal_vertex(face, tolerance)

        else:
            internal_vertex = face.center_of_mass()

        instance = AttributeManager.get_instance()
        instance.copy_attributes(face.get_occt_shape(), internal_vertex.get_occt_shape())
        vertices.append(internal_vertex)

        if to_exterior_topologies or to_exterior_apertures:

            face_edges: List[Edge] = []
            face_edges = face.edges()

            for face_edge in face_edges:

                if to_exterior_topologies:

                    the_other_vertex: Vertex = face_edge.center_of_mass()
                    instance = AttributeManager.get_instance()
                    instance.copy_attributes(face_edge.get_occt_shape() ,the_other_vertex.get_occt_shape())
                    edge: Edge = Edge.by_start_vertex_end_vertex(internal_vertex, the_other_vertex)
                    edges.append(edge)

                if to_exterior_apertures:

                    contents: List[Topology] = []
                    face_edge.contents_(contents)

                    for content in contents:

                        if content.get_shape_type() == TopologyTypes.APERTURE:

                            aperture = TopologicalQuery.downcast(content)

                            aperture_center_of_mass = Graph.calculate_graph_vertex_from_aperture(aperture, use_face_internal_vertex, tolerance)

                            vertices.append(aperture_center_of_mass)
                            edge = Edge.by_start_vertex_end_vertex(aperture_center_of_mass, internal_vertex)
                            edges.append(edge)

        return Graph(vertices, edges)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_shell(shell: Shell, direct: bool, via_shared_topologies: bool, via_shared_apertures: bool, to_exterior_topologies: bool, to_exterior_apertures: bool,use_face_internal_vertex:bool, tolerance: bool) -> 'Graph':
        
        if shell == None:
            return None

        # 1. Get the vertices mapped to their original topologies
		#    - Face --> centroid
		#    Occt shapes must be used as the keys. Topologic shapes cannot be used because 
		#    there can be many shapes representing the same OCCT shapes.

        face_centroids: Dict[TopoDS_Face, Vertex]

        faces: List[Face]
        faces = shell.faces()

        for face in faces:

            internal_vertex: Vertex = None

            if use_face_internal_vertex:

                internal_vertex = TopologicUtilities.FaceUtility.internal_vertex(face, tolerance)

            else:

                internal_vertex = face.center_of_mass()

            instance = AttributeManager.get_instance()
            instance.copy_attributes(face.get_occt_shape(), internal_vertex.get_occt_shape())

            face_centroids[face.get_occt_face()] = internal_vertex

        # 2. Check the configurations. Add the edges to a cluster.
        graph_edges: List[Edge] = []

        if direct:

            # Iterate through all faces and check for adjacency.
			# For each adjacent faces, connect the centroids

            for face in faces:

                adjacent_faces = List[Face]

                # FaceUtility.adjacent_faces not implemented yet!
                TopologicUtilities.FaceUtility.adjacent_faces(face.get_class_guid(), shell, adjacent_faces)

                for adjacent_face in adjacent_faces:

                    occt_adjacent_face = adjacent_face.get_occt_face()

                    # adjacent_centroid_pair: Dict[TopoDS_Shape, Vertex] = {}

                    if occt_adjacent_face not in list(face_centroids.keys()):
                        continue

                    edge = Edge.by_start_vertex_end_vertex(face.get_occt_face(), face_centroids[occt_adjacent_face])

                    graph_edges.append(edge)

        edges: List[Edge]
        edges = shell.edges()

        for edge in edges:

            centroid: Vertex = edge.center_of_mass()

            is_manifold = edge.is_manifold(shell)

            instance = AttributeManager.get_instance()
            instance.copy_attributes(edge.get_occt_shape(), centroid.get_occt_shape())

            adjacent_faces: List[Face]

            # EdgeUtility.adjacent_faces not implemented yet
            TopologicUtilities.EdgeUtility.adjacent_faces(edge, shell, adjacent_faces)

            contents: List[Topology] = []
            edge.contents_(contents)

            # Get the apertures and calculate their centroids
            aperture_centroids: List[Vertex] = []

            for content in contents:

                # If this is not an aperture, skip it
                if content.get_shape_type() != TopologyTypes.APERTURE:
                    continue

                aperture = TopologicalQuery.downcast(content)
                aperture_centroid = Graph.calculate_graph_vertex_from_aperture(aperture, use_face_internal_vertex, tolerance)

                instance = AttributeManager.get_instance()
                instance.copy_attributes(aperture.get_occt_shape(), aperture_centroid.get_occt_shape())

                aperture_centroids.append(aperture_centroid)

            # Check
            for adjacent_face in adjacent_faces:

                if (is_manifold and via_shared_topologies) or (is_manifold and to_exterior_topologies):

                    edge = Edge.by_start_vertex_end_vertex(centroid, face_centroids[adjacent_face.get_occt_face()])

                    graph_edges.append(edge)

                for aperture_centroid in aperture_centroids:

                    if (is_manifold and via_shared_apertures) or (is_manifold and to_exterior_apertures):

                        edge = Edge.by_start_vertex_end_vertex(aperture_centroid, face_centroids[adjacent_face.get_occt_face()])

                        graph_edges.append(edge)

        vertices: List[Vertex] = []

        for edge_topology in graph_edges:

            edge_vertices: List[Vertex]
            edge_vertices = edge_topology.vertices()

            for vertex in edge_vertices:
                vertices.append(vertex)

        return Graph(vertices, graph_edges)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_cell(cell: Cell, to_exterior_topologies: bool, to_exterior_apertures: bool, use_face_internal_vertex: bool, tolerance: float) -> 'Graph':

        vertices: List[Vertex] = []
        edges: List[Edge] = []

        internal_vertex = TopologicUtilities.CellUtility.internal_vertex(cell, tolerance)
        instance = AttributeManager.get_instance()
        instance.copy_attributes(cell.get_occt_shape(), internal_vertex.get_occt_shape())
        vertices.append(internal_vertex)

        if to_exterior_topologies or to_exterior_apertures:

            cell_faces: List[Face] = []
            cell_faces = cell.faces()

            for cell_face in cell_faces:

                if to_exterior_topologies:

                    cell_face_center_of_mass = TopologicUtilities.FaceUtility.internal_vertex(cell_face, tolerance)
                    vertices.append(cell_face_center_of_mass)
                    instance = AttributeManager.get_instance()
                    instance.copy_attributes(cell_face.get_occt_shape(), cell_face_center_of_mass.get_occt_shape)
                    edge = Edge.by_start_vertex_end_vertex(cell_face_center_of_mass, internal_vertex)
                    edges.append(edge)

                if to_exterior_apertures:

                    contents: List[Topology] = []
                    cell_face.contents_(contents)

                    for content in contents:

                        if content.get_shape_type() == TopologyTypes.APERTURE:

                            aperture = TopologicalQuery.downcast(content)
                            aperture_center_of_mass = Graph.calculate_graph_vertex_from_aperture(aperture, use_face_internal_vertex, tolerance)

                            vertices.append(aperture_center_of_mass)
                            edge = Edge.by_start_vertex_end_vertex(aperture_center_of_mass, internal_vertex)
                            edges.append(edge)

        return Graph(vertices, edges)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_cellComplex(cellComplex: CellComplex, direct: bool, via_shared_topologies: bool, via_shared_apertures: bool, to_exterior_topologies: bool, to_exterior_apertures: bool, use_face_internal_vertex: bool, tolerance: float) -> 'Graph':
        
        if cellComplex == None:
            return None

        # 1. Get the vertices mapped to their original topologies
		#    - Cell --> centroid
		#    Occt shapes must be used as the keys. Topologic shapes cannot be used because there can be many shapes representing the same OCCT shapes.
		
        cell_centroids: Dict[TopoDS_Solid, Vertex] = {}

        cells: List[Cell] = []
        cells = cellComplex.cells()

        for cell in cells:

            centroid = TopologicUtilities.CellUtility.internal_vertex(cell, tolerance)
            instance = AttributeManager.get_instance()
            instance.copy_attributes(cell.get_occt_shape(), centroid.get_occt_shape())
            cell_centroids[cell.get_occt_solid()] = centroid

        # 2. If direct = true, check cellAdjacency.
        edges: List[Edge] = []

        if direct:

            occt_cell_adjacency = TopTools_DataMapOfShapeListOfShape()

            for cell in cells:

                # Get faces
                faces : List[Face] = []
                faces = cell.faces()

                # Get adjacent cells. Only add here if the cell is not already added here, and 
				# the reverse is not in occtCellAdjacency.

                occt_cell_unchecked_adjacent_cells = TopTools_ListOfShape()

                for face in faces:

                    current_face_adjacent_cells: List[Cell] = []
                    TopologicUtilities.FaceUtility.adjacent_cells(face, cellComplex, current_face_adjacent_cells)

                    for current_face_adjacent_cell in current_face_adjacent_cells:

                        # The same as this Cell? Continue.
                        if current_face_adjacent_cell.is_same(cell):
                            continue

                        # Is Cell already added in this list (occtCellAdjacentCells)? Continue.
                        if occt_cell_unchecked_adjacent_cells.Contains(current_face_adjacent_cell.get_occt_shape()):
                            continue

                        # Is the reverse already added in occtCellAdjacency? Continue.
                        try:
                            reverse_adjacency = occt_cell_adjacency.Find(current_face_adjacent_cell.get_occt_shape())
                        
                            if reverse_adjacency.Contains(cell.get_occt_shape()):
                                continue

                        except:
                            pass

                        # If passes the tests, add to occtCellUncheckedAdjacentCells
                        occt_cell_unchecked_adjacent_cells.Append(current_face_adjacent_cell.get_occt_shape())

                if not occt_cell_unchecked_adjacent_cells.isEmpty():
                    occt_cell_adjacency.Bind(cell.get_occt_shape(), occt_cell_unchecked_adjacent_cells)

            # Create the edges from the Cell adjacency information
            occt_cell_adjacency_iterator = TopTools_DataMapIteratorOfDataMapOfShapeListOfShape(occt_cell_adjacency)

            while occt_cell_adjacency_iterator.More():

                occt_cell = topods.solid(occt_cell_adjacency_iterator.Key())
                cell = Cell(occt_cell, "")

                cell_internal_vertex: Vertex = None

                try:
                    cell_internal_vertex = cell_centroids[occt_cell]

                except:
                    # raise RuntimeError("No Cell internal vertex pre-computed.")
                    return None

                occt_adjacent_cells = occt_cell_adjacency_iterator.Value()

                occt_adjacent_cell_iterator = TopTools_ListIteratorOfListOfShape(occt_adjacent_cells)

                while occt_adjacent_cell_iterator.More():

                    occt_adjacent_cell = topods.Solid(occt_adjacent_cell_iterator.Value())
                    adjacent_cell = Cell(occt_adjacent_cell, "")

                    adjacent_internal_vertex = None

                    try:
                        adjacent_internal_vertex = cell_centroids[occt_adjacent_cell]

                    except:
                        # raise RuntimeError("No Cell internal vertex pre-computed.")
                        return None

                    edge = Edge.by_start_vertex_end_vertex(cell_internal_vertex, adjacent_internal_vertex)
                    edges.append(edge)

                    occt_adjacent_cell_iterator.Next()

                occt_cell_adjacency_iterator.Next()

        faces: List[Face] = []
        faces = cellComplex.faces()

        for face in faces:

            internal_vertex: Vertex = None

            if use_face_internal_vertex:
                internal_vertex = TopologicUtilities.FaceUtility.internal_vertex(face, tolerance)

            else:
                internal_vertex = face.center_of_mass()

            instance = AttributeManager. get_instance()
            instance.copy_attributes(face.get_occt_shape(), internal_vertex.get_occt_shape())

            is_manifold: bool = face.is_manifold_to_topology(cellComplex)

            adjacent_cells: List[Cell] = []
            adjacent_cells = TopologicUtilities.FaceUtility.adjacent_cells(face, cellComplex, adjacent_cells)

            contents: List[Topology] = []
            face.contents_(contents)

            # Get the apertures and calculate their centroids
            aperture_centroids: List[Vertex] = []

            for content in contents:

                # If this is not an aperture, skip it
                if content.get_shape_type() == TopologyTypes.APERTURE:
                    continue

                aperture: Aperture = TopologicalQuery.downcast(content)
                aperture_centroid: Vertex = Graph.calculate_graph_vertex_from_aperture(aperture, use_face_internal_vertex, tolerance)
                
                instance = AttributeManager.get_instance()
                instance.copy_attributes(aperture.get_occt_shape(), aperture_centroid.get_occt_shape())

                aperture_centroids.append(aperture_centroid)

            # Check
            for adjacent_cell in adjacent_cells:

                if (not is_manifold and via_shared_topologies) or (is_manifold and to_exterior_topologies):

                    if adjacent_cell.get_occt_shape() not in cell_centroids.keys():
                        continue

                    edge = Edge.by_start_vertex_end_vertex(internal_vertex, cell_centroids[adjacent_cell.get_occt_shape()])

                    if edge != None:
                        edges.append(edge)

                for aperture_centroid in aperture_centroids:

                    if (not is_manifold and via_shared_apertures) or (is_manifold and to_exterior_apertures):

                        if adjacent_cell.get_occt_solid() not in cell_centroids:
                            continue

                        edge = Edge.by_start_vertex_end_vertex(aperture_centroid, cell_centroids[adjacent_cell.get_occt_shape()])
                
                        if edge != None:
                            edges.append(edge)

        vertices: List[Vertex] = []
        for edge_topology in edges:

            edge_vertices: List[Vertex] = []
            edge_vertices = edge_topology.vertices()

            for vertex in edge_vertices:

                vertices.append(vertex)

        return Graph(vertices, edges)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_cluster(cluster: Cluster, direct: bool, via_shared_topologies: bool, via_shared_apertures: bool, to_exterior_topologies: bool, to_exterior_apertures: bool, use_face_internal_vertex: bool, tolerance: float) -> 'Graph':
        
        sub_topologies: List[Topology] = []
        cluster.sub_topologies(sub_topologies)

        vertices: List[Vertex] = []
        edges: List[Edge] = []

        for sub_topology in sub_topologies:

            graph = Graph.by_topology(sub_topology, direct, via_shared_topologies, via_shared_apertures, to_exterior_topologies, to_exterior_apertures, use_face_internal_vertex, tolerance) 

            sub_topology_vertices: List[Vertex] = []
            sub_topology_vertices = graph.vertices()

            sub_topology_edges: List[Edge] = []
            sub_topology_edges = graph.edges()

            vertices.extend(sub_topology_vertices)
            edges.extend(sub_topology_edges)

        return Graph(vertices, edges)

#--------------------------------------------------------------------------------------------------
    def construct_path(self, path_vertices: List[Vertex]) -> Wire:
        
        starting_time: datetime = datetime.now()
        return self.construct_path(path_vertices, False, 0, starting_time)

#--------------------------------------------------------------------------------------------------
    def construct_path(self, path_vertices: List[Vertex], use_time_limit: bool, time_limit_in_seconds: int, starting_time: datetime) -> Wire:
        
        edges: List[Edge] = []

        for vertex in range(len(path_vertices)-1):

            if use_time_limit:

                current_time: datetime = datetime.now()
                time_difference = current_time - starting_time
                time_difference_in_seconds = time_difference.seconds

                if time_difference_in_seconds > time_limit_in_seconds:
                    return None

                current_vertex = vertex
                i = path_vertices.index(current_vertex)
                next_vertex = path_vertices[i + 1]

                occt_edge = self.find_edge(current_vertex.get_occt_vertex(), next_vertex.get_occt_vertex())

                edge = None
                if not occt_edge.IsNull():
                    edge = TopologicalQuery.downcast(Topology.by_occt_shape(occt_edge))

                else:
                    edge = Edge.by_start_vertex_end_vertex(current_vertex, next_vertex)

                edges.append(edge)

            if len(edges) == 0:
                return None
            
            path_wire = Wire.by_edges(edges)
            
            return path_wire

#--------------------------------------------------------------------------------------------------
    def is_degree_sequence(self, sequence: List[int]) -> bool:
        
        for sequence_iterator in range(len(sequence) - 1):

            i = sequence.index(sequence_iterator)

            next_iterator = sequence[i + 1]

            if next_iterator > sequence_iterator:
                return False

        return True

#--------------------------------------------------------------------------------------------------
    def get_coincident_vertex(self, vertex: Vertex, tolerance: float) ->TopoDS_Vertex:
        
        pnt = BRep_Tool.Pnt(vertex)
        point = Geom_CartesianPoint(pnt)

        abs_distance_threshold = abs(tolerance)

        for current_vertex in list(self.base_graph_dictionary.keys()):

            current_pnt = BRep_Tool.Pnt(current_vertex)
            current_point = Geom_CartesianPoint(current_pnt)

            dx: float = current_point.X() - point.X()
            dy: float = current_point.Y() - point.Y()
            dz: float = current_point.Z() - point.Z()

            sq_distance = dx**2 + dy**2+ dz**2

            if sq_distance < abs_distance_threshold:
                return current_vertex

        return TopoDS_Vertex() # null vertex

#--------------------------------------------------------------------------------------------------
    def compute_cost(self, vertex_1: TopoDS_Vertex, vertex_2: TopoDS_Vertex, vertex_key: str, edge_key: str) -> float:
        
        edge_cost = self.compute_edge_cost(vertex_1, vertex_2, edge_key)

        if edge_cost >= sys.float_info.max:
            return edge_cost

        vertex_cost = self.compute_vertex_cost(vertex_2, vertex_key)

        return vertex_cost + edge_cost

#--------------------------------------------------------------------------------------------------
    def compute_vertex_cost(self, vertex: TopoDS_Vertex, vertex_key: str) -> 'Graph':
        
        if vertex_key == '0':
            return 0.0

        instance = AttributeManager.get_instance()
        has_attribute: bool = instance.find_all(vertex, attribute_map)

        attribute_map: Dict[str, Attribute] = {}

        if not has_attribute:
            return 0.0

        if vertex_key in list(attribute_map.keys()):

            attribute = attribute_map[vertex_key]

        # Only add if double or int
        double_attribute: DoubleAttribute = TopologicalQuery.dynamic_pointer_cast(attribute)

        if double_attribute != None:
            return double_attribute.double_value()

        int_attribute: IntAttribute = TopologicalQuery.dynamic_pointer_cast(attribute)

        if int_attribute != None:
            return int_attribute.int_value()

        return 0.0

#--------------------------------------------------------------------------------------------------
    def compute_edge_cost(self, vertex_1: TopoDS_Vertex, vertex_2: TopoDS_Vertex, edge_key: str) -> float:
        
        # Check: if not connected, return the largest double value
        if not self.contains_edge(vertex_1, vertex_2, 0.0001):
            return sys.float_info.max

        else:
            # Check edge key
            if edge_key == '0':
                return 1.0

            else:
                occt_edge = self.find_edge(vertex_1, vertex_2)

                if occt_edge.IsNull():
                    return sys.float_info.max

                attribute_map: Dict[str, Attribute] = {}

                instance = AttributeManager.get_instance()
                has_attribute: bool = instance.find_all(occt_edge, attribute_map)

                lower_case_edge_key: str = edge_key.lower()

                if lower_case_edge_key not in list(attribute_map.keys()):

                    if lower_case_edge_key != 'distance' or lower_case_edge_key != 'length': # no attribute with this name is found
                        occt_distance = BRepExtrema_DistShapeShape(vertex_1, vertex_2)
                        return occt_distance.Value()

                    else:
                        return 1.0

                else:
                    attribute = attribute_map[lower_case_edge_key]

                    # Only add if double or int
                    double_attribute: DoubleAttribute = TopologicalQuery.dynamic_pointer_cast(attribute)

                    if double_attribute != None:
                        return double_attribute.double_value()

                    int_attribute: IntAttribute = TopologicalQuery.dynamic_pointer_cast(attribute)

                    if int_attribute != None:
                        return int_attribute.int_value()

                    return 1.0


#--------------------------------------------------------------------------------------------------
    def find_edge(self, vertex_1: TopoDS_Vertex, vertex_2: TopoDS_Vertex, tolerance: float) -> TopoDS_Edge:

        occt_edge_iterator = TopTools_MapIteratorOfMapOfShape(self.occt_edges)

        while occt_edge_iterator.More():

            occt_edge = topods.Edge(occt_edge_iterator.Value())
            occt_shape_analysis_edge = ShapeAnalysis_Edge()
            occt_first_vertex: TopoDS_Vertex = occt_shape_analysis_edge.FirstVertex(occt_edge)
            occt_last_vertex: TopoDS_Vertex = occt_shape_analysis_edge.LastVertex(occt_edge)

            if (self.is_coincident(occt_first_vertex, vertex_1, tolerance) and self.is_coincident(occt_last_vertex, vertex_2, tolerance)) or \
               (self.is_coincident(occt_first_vertex, vertex_2, tolerance) and self.is_coincident(occt_last_vertex, vertex_1, tolerance)):

                return occt_edge

            occt_edge_iterator.Next()

        return TopoDS_Edge()

#--------------------------------------------------------------------------------------------------
    def is_coincident(self, vertex_1: TopoDS_Vertex, vertex_2: TopoDS_Vertex, tolerance: float) -> 'Graph':
        
        pnt_1 = BRep_Tool.Pnt(vertex_1)
        point_1 = Geom_CartesianPoint(pnt_1)

        pnt_2 = BRep_Tool.Pnt(vertex_2)
        point_2 = Geom_CartesianPoint(pnt_2)

        dx: float = point_2.X() - point_1.X()
        dy: float = point_2.Y() - point_1.Y()
        dz: float = point_2.Z() - point_1.Z()

        sq_distance = dx**2 + dy**2 + dz**2

        if sq_distance < tolerance:
            return True

        return False