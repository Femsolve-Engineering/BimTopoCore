
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
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape, TopTools_MapIteratorOfMapOfShape
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

class Graph(Topology):

    def __init__(self, vertices: List[Vertex], edges: List[Edge]) -> None:
        
        self.add_vertices(vertices, 0.0001)
        self.add_edges(edges, 0.0001)

    # def __init__(self, another_graph: 'Graph') -> None:
        
    #     self.base_graph_dictionary = another_graph.base_graph_dictionary
    #     self.occt_edges = another_graph.occt_edges

#--------------------------------------------------------------------------------------------------
    def by_vertices(self, vertices: List[Vertex], edges: List[Edge]):
        
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
                          tolerance: float):

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
    def edges(self, edges: List['Edge'], tolerance: float):
        
        vertices: List['Vertex'] = []
        self.edges(vertices, tolerance, edges)

#--------------------------------------------------------------------------------------------------
    def edges(self, vertices: list[Vertex], tolerance: float, edges: List['Edge']):
        
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
    def add_vertices(self, vertices: List['Vertex'], tolerance: float):
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return None

        for vertex in vertices:

            if not self.contains_vertex(vertex, tolerance):

                self.base_graph_dictionary[vertex.get_occt_vertex()] = TopTools_MapOfShape()


#--------------------------------------------------------------------------------------------------
    def add_edges(self, edges: List['Edge'], tolerance: float):
        
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
    def adjacent_vertices(self):
        pass

#--------------------------------------------------------------------------------------------------
    def connect(self):
        pass

#--------------------------------------------------------------------------------------------------
    def contains_vertex(self):
        pass

#--------------------------------------------------------------------------------------------------
    def contains_vertex(self):
        pass

#--------------------------------------------------------------------------------------------------
    def contains_edge(self):
        pass

#--------------------------------------------------------------------------------------------------
    def contains_edge(self):
        pass

#--------------------------------------------------------------------------------------------------
    def degree_sequence(self):
        pass

#--------------------------------------------------------------------------------------------------
    def density(self):
        pass

#--------------------------------------------------------------------------------------------------
    def is_complete(self):
        pass

#--------------------------------------------------------------------------------------------------
    def minimum_delta(self):
        pass

#--------------------------------------------------------------------------------------------------
    def maximum_delta(self):
        pass

#--------------------------------------------------------------------------------------------------
    def all_paths(self):
        pass

#--------------------------------------------------------------------------------------------------
    def all_paths(self):
        pass

#--------------------------------------------------------------------------------------------------
    def path(self):
        pass

#--------------------------------------------------------------------------------------------------
    def path(self):
        pass

#--------------------------------------------------------------------------------------------------
    def shortest_path(self):
        pass

#--------------------------------------------------------------------------------------------------
    def shortest_path(self):
        pass

#--------------------------------------------------------------------------------------------------
    def shortest_paths(self):
        pass

#--------------------------------------------------------------------------------------------------
    def shortest_paths(self):
        pass

#--------------------------------------------------------------------------------------------------
    def diameter(self):
        pass

#--------------------------------------------------------------------------------------------------
    def topological_distance(self):
        pass

#--------------------------------------------------------------------------------------------------
    def topological_distance(self):
        pass

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