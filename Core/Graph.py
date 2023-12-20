
from pkgutil import extend_path
import sys
from datetime import datetime

from queue import Queue
from typing import Dict
from typing import Tuple
from typing import List
from xmlrpc.client import boolean

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_CompSolid, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, topods
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape, TopTools_MapIteratorOfMapOfShape, TopTools_DataMapOfShapeInteger
from OCC.core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Surface, Geom_Geometry
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
from Core.Face import Face
from Core.Cell import Cell
from Core.Shell import Shell
from Core.Cluster import Cluster

from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes

# class Node:

#     def __init__(self, val: TopoDS_Vertex, path: List[TopoDS_Vertex], distance: float):

#         self.val = val
#         self.path = path
#         self.distance = distance

class Node:

    def __init__(self, val: TopoDS_Vertex, path: List[TopoDS_Vertex], distance: float):

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
                          direct: boolean, \
                          via_shared_topologies: boolean, \
                          via_shared_apertures: boolean, \
                          to_exterior_topologies: boolean, \
                          to_exterior_apertures: boolean, \
                          use_face_internal_vertex: boolean, \
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
    def contains_vertex(self, vertex: Vertex, tolerance: float) -> boolean:
        
        return self.contains_vertex(vertex.get_occt_vertex(), tolerance)

#--------------------------------------------------------------------------------------------------
    def contains_vertex(self, occt_vertex, tolerance: float) -> boolean:
        
        occt_coincident_vertex = self.get_coincident_vertex(occt_vertex, tolerance)
        return not occt_coincident_vertex.IsNull()

#--------------------------------------------------------------------------------------------------
    def contains_edge(self, edge: Edge, tolerance) -> boolean:
        
        start_vertex = edge.start_vertex()
        end_vertex = edge. end_vertex()

        return self.contains_edge(start_vertex.get_occt_vertex(), end_vertex.get_occt_vertex(), tolerance)

#--------------------------------------------------------------------------------------------------
    def contains_edge(self, occt_vertex_1: TopoDS_Vertex, occt_vertex_2: TopoDS_Vertex, tolerance: float) -> boolean:
        
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
    def is_complete(self) -> boolean:
        
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
                        use_time_limit: boolean,
                        time_limit_in_seconds:
                        int, paths: Wire) -> None:
        
        path: List[Vertex] = []
        starting_time: datetime = datetime.now()
        self.all_paths(start_vertex, end_vertex, use_time_limit, time_limit_in_seconds, starting_time, path, paths)

#--------------------------------------------------------------------------------------------------
    def all_paths(self, start_vertex: Vertex,
                        end_vertex: Vertex, use_time_limit: boolean,
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

                    if extend_path:
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
                             use_time_limit: boolean,
                             time_limit: int,
                             paths: List[Wire]) -> None:
        
        return self.shortest_paths(start_vertex.get_occt_vertex(), end_vertex.get_occt_vertex(), vertex_key, edge_key, use_time_limit, time_limit, paths)

#--------------------------------------------------------------------------------------------------
    def shortest_paths(self, occt_start_vertex: TopoDS_Vertex,
                             occt_end_vertex: TopoDS_Vertex,
                             vertex_key: str,
                             edge_key: str,
                             use_time_limit: boolean,
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
    def eccentricity(self):
        pass

#--------------------------------------------------------------------------------------------------
    def is_erdoes_gallai(self):
        pass

#--------------------------------------------------------------------------------------------------
    def remove_vertices(self):
        pass

#--------------------------------------------------------------------------------------------------
    def remove_edges(self):
        pass

#--------------------------------------------------------------------------------------------------
    def vertices_at_coordinates(self):
        pass

#--------------------------------------------------------------------------------------------------
    def edge(self):
        pass

#--------------------------------------------------------------------------------------------------
    def incident_edges(self):
        pass

#--------------------------------------------------------------------------------------------------
    def calculate_graph_vertex_from_aperture(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_vertex(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_edge(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_wire(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_face(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_shell(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_cell(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_cellComplex(self):
        pass

#--------------------------------------------------------------------------------------------------
    def by_cluster(self):
        pass

#--------------------------------------------------------------------------------------------------
    def construct_path(self):
        pass

#--------------------------------------------------------------------------------------------------
    def construct_path(self):
        pass

#--------------------------------------------------------------------------------------------------
    def is_degree_sequence(self):
        pass

#--------------------------------------------------------------------------------------------------
    def get_coincident_vertex(self):
        pass

#--------------------------------------------------------------------------------------------------
    def compute_cost(self):
        pass

#--------------------------------------------------------------------------------------------------
    def compute_vertex_cost(self):
        pass

#--------------------------------------------------------------------------------------------------
    def compute_edge_cost(self):
        pass

#--------------------------------------------------------------------------------------------------
    def find_edge(self):
        pass

#--------------------------------------------------------------------------------------------------
    def is_coincident(self):
        pass