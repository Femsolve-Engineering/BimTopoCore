
from typing import List
from typing import Tuple

# OCC
from OCC.Core import Precision
from OCC.Core.gp import gp_Dir, gp_Vec, gp_Trsf
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.Geom import Geom_RectangularTrimmedSurface, Geom_CartesianPoint
from OCC.Core.ShapeAnalysis import shapeanalysis, ShapeAnalysis_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform, BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeEdge
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

# BimTopoCore
from Core.Topology import Topology
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Face import Face
from Core.Cell import Cell
from Core.TopologyConstants import TopologyTypes

class TopologyUtility:

    @staticmethod
    def translate(topology: 'Topology',
                  x: float, y: float, z: float) -> 'Topology':
        """
        Translates the topological entity in the (x,y,z) direction.
        """
        # Create the translation transformation
        transformation = gp_Trsf()
        transformation.SetTranslation(gp_Vec(x, y, z))

        # Apply the transformation
        occt_shape = topology.get_occt_shape()  # Assuming a method to get the OCCT shape
        transform_api = BRepBuilderAPI_Transform(occt_shape, transformation, True)
        transformed_shape = transform_api.Shape()
        core_transformed_topology = Topology.by_occt_shape(
            transformed_shape, topology.get_class_guid())

        # ToDo: AttributeManager
        # AttributeManager.deep_copy_attributes(
        #   occt_shape, core_transformed_topology.get_occt_shape())

        sub_contents: List[Topology] = topology.sub_contents()

        for sub_content in sub_contents:
            # Recursively transform sub-contents
            transformed_subcontent = TopologyUtility.translate(sub_content, x, y, z)

            # ToDo: Context, ContextManager
            # contexts = sub_content.contexts()
            # context_type = 0

            # for context in contexts:
            #     context_topology = context.topology()
            #     context_topology_type = context_topology.get_type()
            #     context_type |= context_topology_type

            # core_transformed_topology.add_contents([transformed_subcontent], context_type)

        # ToDo: GlobalCluster
        # GlobalCluster.add_topology(core_transformed_topology)

        return core_transformed_topology

class VertexUtility:

    @staticmethod
    def adjacent_edges(
        vertex: 'Vertex', 
        parent_topology: 'Topology') -> List['Edge']:
        """
        TODO
        """
        core_adjacent_edges: List[Edge] = []
        core_adjacent_topologies: List[Topology] = vertex.upward_navigation(
            parent_topology.get_occt_shape(), 
            TopologyTypes.EDGE) 
        
        for adjacent_topology in core_adjacent_topologies:
            # ToDo: Check this if this is correct
            core_adjacent_edges.append(Edge(adjacent_topology.get_occt_shape()))

        return core_adjacent_edges
    
    @staticmethod
    def distance(
        vertex: 'Vertex',
        topology: 'Topology') -> float:
        """
        Measures the distance from a vertex to any topology.
        ToDo?: We are using BRepExtrema here, in the legacy code this had 
        a specific implementation for any two different types.
        """

        # ToDo: Need to consider multiple distances.
        brep_extrema = BRepExtrema_DistShapeShape(vertex.get_occt_shape(),topology.get_occt_shape())
        return brep_extrema.Value()
    

class EdgeUtility:

    @staticmethod
    def length(edge: 'Edge') -> float:
        """
        Returns the length of an edge.
        """
        occt_shape_properties = GProp_GProps()
        brepgprop.LinearProperties(edge.get_occt_shape(), occt_shape_properties)
        return occt_shape_properties.Mass()
    
class FaceUtility:

    @staticmethod
    def is_inside(face: 'Face', vertex: 'Vertex', tolerance: float) -> bool:
        """
        https://www.opencascade.com/content/how-find-if-point-belongs-face
	    https://www.opencascade.com/doc/occt-7.2.0/refman/html/class_b_rep_class___face_classifier.html
        """
        # ParametersAtVertex is assumed to be a method that retrieves the (u, v) parameters for the vertex on the face
        u, v = FaceUtility.parameters_at_vertex(face, vertex)
        
        # NormalAtParameters is assumed to be a method that calculates the normal at the given (u, v) parameters on the face
        normal = FaceUtility.normal_at_parameters(face, u, v)
        
        # Retrieve the gp_Pnt from the vertex
        occt_input_point = vertex.get_point().Pnt()
        
        # Translate the point in both directions along the normal
        point_a = occt_input_point.Translated(gp_Vec(normal).Multiplied(tolerance))
        point_b = occt_input_point.Translated(gp_Vec(normal.Reversed()).Multiplied(tolerance))
        
        # Create the vertices at the translated points
        vertex_a = Vertex.by_point(Geom_CartesianPoint(point_a))
        vertex_b = Vertex.by_point(Geom_CartesianPoint(point_b))
        edge = Edge.by_start_vertex_end_vertex(vertex_a, vertex_b)
        
        # ToDo: Topology.Slice
        slice_result = edge.slice(face)
        
        # If there's no result, the point is outside
        if slice_result is None:
            return False
        
        # Retrieve vertices from the slice result
        vertices = slice_result.vertices()
        
        # If there are 2 or fewer vertices, the point is outside
        if len(vertices) <= 2:
            return False
        
        # Retrieve vertices and edges from the face
        face_vertices = face.vertices()
        face_edges = face.edges()
        
        # Check each vertex against the original face's vertices and edges
        rejected_vertices = []
        for kp_vertex in vertices:
            if vertex.distance_to(vertex_a) < tolerance and vertex.distance_to(vertex_b) < tolerance:
                continue
            
            for kp_face_vertex in face_vertices:
                if vertex.distance_to(kp_face_vertex) < tolerance:
                    rejected_vertices.append(kp_vertex)
                    break
            
            for kp_face_edge in face_edges:
                if vertex.distance_to(kp_face_edge) < tolerance:
                    rejected_vertices.append(kp_vertex)
                    break
        
        # If the number of accepted vertices is 2 or fewer, the point is outside
        return len(vertices) - len(rejected_vertices) > 2

    @staticmethod
    def area(face: TopoDS_Face) -> float:
        """
        Calculates and returns the area of a face.
        """
        occt_shape_properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, occt_shape_properties)
        return occt_shape_properties.Mass()
    
    @staticmethod
    def vertex_at_parameters(face: 'Face', u: float, v: float) -> float:
        """
        Places a point at 'u' and 'v' normalized positions.
        """
        (occt_u, occt_v) = FaceUtility.non_normalize_uv(face, u, v) 
        
        (occt_min_u, occt_max_u, occt_min_v, occt_max_v) = shapeanalysis.GetFaceUVBounds(
            face.get_occt_face())
        surface_analysis = ShapeAnalysis_Surface(face.surface())
        r_surface = surface_analysis.Surface()
        
        trimmed_surface = Geom_RectangularTrimmedSurface(
            r_surface, 
            occt_min_u + 0.0001, occt_max_u - 0.0001, 
            occt_min_v + 0.0001, occt_max_v - 0.0001)
        bspline_surface = geomconvert.SurfaceToBSplineSurface(trimmed_surface)     
        occt_point = bspline_surface.Value(occt_u, occt_v)

        vertex = Vertex.by_point(Geom_CartesianPoint(occt_point))
        return vertex
    
    @staticmethod
    def parameters_at_vertex(face: 'Face', vertex: 'Vertex') -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: Normalized face U and V parameters
        """

    
    @staticmethod
    def normal_at_parameters(face: 'Face', u: float, v: float) -> gp_Dir:
        """
        Places a point at 'u' and 'v' normalized positions (between 0 and 1).
        """
        occt_u, occt_v = FaceUtility.non_normalize_uv(face, u, v)
        
        # Create the local properties object with the desired level of derivation (1 in this case)
        occt_properties = GeomLProp_SLProps(
            face.surface(), occt_u, occt_v, 1, Precision.precision_Confusion())

        occt_normal = occt_properties.Normal()
        
        # Check if the face is reversed and if so, reverse the normal
        if face.is_reversed():
            occt_normal.Reverse()
        
        return occt_normal

    @staticmethod
    def non_normalize_uv(face: 'Face', normalized_u: 'float', normalized_v: 'float') -> Tuple[float, float]:
        """
        Converts the normalized U and V parameters to non-normalized ones (required by OCCT).
        """
        (occt_u_min, occt_u_max, occt_v_min, occt_v_max) = shapeanalysis.GetFaceUVBounds(
            face.get_occt_face())
        occt_du = occt_u_max - occt_u_min
        occt_dv = occt_v_max - occt_v_min
        ret_normalized_u = occt_u_min + normalized_u * occt_du
        ret_normalized_v = occt_v_min + normalized_v * occt_dv
        return (ret_normalized_u, ret_normalized_v)
    
    @staticmethod
    def normalize_uv(face: 'Face', norm_u: float, norm_v: float) -> Tuple[float, float]:
        """
        Normalizes U and V parameters (from OCC to BimTopoCore)
        """
        (occt_u_min, occt_u_max, occt_v_min, occt_v_max) = shapeanalysis.GetFaceUVBounds(
            face.get_occt_face())
        
        occt_du = occt_u_max - occt_u_min
        occt_dv = occt_v_max - occt_v_min

        if occt_du <= 0.0 or occt_dv <= 0.0:
            raise RuntimeError('Negative range for normalize UV!')
        
        return ((norm_u - occt_u_min)/occt_du, (norm_v - occt_v_min)/occt_dv)

    @staticmethod
    def adjacent_cells(face: 'Face', parent_topology: 'Topology') -> List['Cell']:
        """
        TODO
        """
        ret_cells: List['Cell'] = []
        adjacent_topologies: List['Topology'] = face.upward_navigation(
            parent_topology.get_occt_shape(),
            TopologyTypes.CELL)
        
        for adj_top in adjacent_topologies:
            # Here we should downcast to Cell
            ret_cells.append(Cell(adj_top.get_occt_shape()))

