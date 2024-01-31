
from typing import List
from typing import Tuple
from math import pi
from enum import Enum

import sys

# OCC
from OCC.Core import Precision
from OCC.Core.gp import gp_Pnt, gp_Mat, gp_GTrsf, gp_XYZ, gp_Dir, gp_Vec, gp_Trsf, gp_Ax1
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.Geom import ( 
    Geom_Line,
    Geom_RectangularTrimmedSurface, 
    Geom_CartesianPoint
)

from OCC.Core.ShapeBuild import ShapeBuild_ReShape
from OCC.Core.ShapeFix import shapefix, ShapeFix_Wire, ShapeFix_Shape, ShapeFix_Edge, ShapeFix_Face

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHAPE,
    TopAbs_WIRE,
    TopAbs_VERTEX,
    TopAbs_IN,
    TopAbs_OUT,
    TopAbs_ON,
    TopAbs_State
)

from OCC.Core.TopoDS import (
    TopoDS_Face,
    TopoDS_Wire,
    TopoDS_Solid,
)

from OCC.Core.TopTools import toptools
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape, TopTools_IndexedDataMapOfShapeListOfShape, TopTools_MapOfShape

from OCC.Core.TopExp import topexp, TopExp_Explorer

from OCC.Core.GeomLib import GeomLib_Tool
from OCC.Core.ShapeAnalysis import shapeanalysis, ShapeAnalysis_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBuilderAPI import (
    brepbuilderapi,
    BRepBuilderAPI_FaceDone,
    BRepBuilderAPI_GTransform,
    BRepBuilderAPI_Transform, 
    BRepBuilderAPI_MakeVertex, 
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeEdge
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopoDS import topods, TopoDS_Face
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

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

    @staticmethod
    def rotate(topology: 'Topology', 
               origin_vertex: 'Vertex', 
               direction_x: float, direction_y: float, direction_z: float, 
               degree: float):
        """
        Rotates a Topological entity about a defined origin and axis.
        """
        radian = TopologyUtility.degrees_to_radians(degree)
        
        # Set up the rotation transformation
        transformation = gp_Trsf()
        occt_origin_point = origin_vertex.get_point()
        transformation.SetRotation(
            gp_Ax1(
                gp_Pnt(occt_origin_point.X(), occt_origin_point.Y(), occt_origin_point.Z()),
                gp_Dir(direction_x, direction_y, direction_z)), 
            radian)
        
        # Apply the transformation
        transform = BRepBuilderAPI_Transform(topology.get_occt_shape(), transformation, True)
        core_transformed_topology = Topology.by_occt_shape(
            transform.Shape(), topology.get_class_guid())
        
        # ToDo: AttributeManager
        # AttributeManager.get_instance().deep_copy_attributes(
        #   topology.get_occt_shape(), core_transformed_topology.get_occt_shape())
        
        # Recursive rotation for subcontents
        sub_contents = topology.sub_contents()
        for sub_content in sub_contents:
            transformed_subcontent = TopologyUtility.rotate(
                sub_content, origin_vertex, direction_x, direction_y, direction_z, degree)
            
            # ToDo: ContextManager
            # contexts = sub_content.contexts()
            # context_type = 0
            # for context in contexts:
            #     context_topology = context.topology()
            #     context_topology_type = context_topology.get_type()
            #     context_type |= context_topology_type
            # core_transformed_topology = core_transformed_topology.add_contents([transformed_subcontent], context_type)
        
        # ToDo: GlobalCluster
        # GlobalCluster.get_instance().add_topology(core_transformed_topology)
        
        return core_transformed_topology
    
    @staticmethod
    def scale(topology: 'Topology', origin_vertex: 'Vertex', 
              x_factor: float, y_factor: float, z_factor: float):
        """
        Scales topological entities about a vertex. 
        """

        scale_origin = origin_vertex.get_point()
        
        # Translate the topology to the origin
        trsf_to_origin = gp_Trsf()
        trsf_to_origin.SetTranslation(gp_Vec(-scale_origin.X(), -scale_origin.Y(), -scale_origin.Z()))
        transform_to_origin = BRepBuilderAPI_Transform(
            topology.get_occt_shape(), trsf_to_origin, True)
        
        # Create the scaling transformation
        scaling_mat = gp_Mat(
            x_factor, 0, 0, 
            0, y_factor, 0, 
            0, 0, z_factor)
        scaling_transformation = gp_GTrsf(scaling_mat, gp_XYZ(0, 0, 0))
        scaling_transform = BRepBuilderAPI_GTransform(
            transform_to_origin.Shape(), scaling_transformation, True)
        
        # Translate the topology back to its original position
        trsf_back = gp_Trsf()
        trsf_back.SetTranslation(gp_Vec(scale_origin.X(), scale_origin.Y(), scale_origin.Z()))
        transform_back = BRepBuilderAPI_Transform(scaling_transform.Shape(), trsf_back, True)
        
        # Create the new transformed topology
        transformed_shape = transform_back.Shape()
        core_transformed_topology = Topology.by_occt_shape(
            transformed_shape, topology.get_class_guid())
        
        # ToDo: AttributeManager
        # AttributeManager.get_instance().deep_copy_attributes(
        #     topology.get_occt_shape(), core_transformed_topology.get_occt_shape())
        
        # Scale subcontents recursively
        sub_contents = topology.sub_contents()
        for sub_content in sub_contents:
            transformed_subcontent = TopologyUtility.scale(
                sub_content, origin_vertex, x_factor, y_factor, z_factor)
            
            # ToDo: ContextManager
            # # Determine context types
            # contexts = sub_content.contexts()
            # context_type = 0
            # for context in contexts:
            #     context_topology = context.topology()
            #     context_topology_type = context_topology.get_type()
            #     context_type |= context_topology_type
            
            # # Add scaled subcontents to the transformed topology
            # core_transformed_topology.add_contents([transformed_subcontent], context_type)
        
        # ToDo: Global Cluster
        # GlobalCluster.get_instance().add_topology(core_transformed_topology)
        
        return core_transformed_topology
    
    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """
        Converts degrees to radians.
        """
        return degrees * pi / 180.0
    
    @staticmethod
    def radians_to_degress(radians: float) -> float:
        """
        Converts radians to degrees.
        """
        return radians * 180.0 / pi

class VertexUtility:

    @staticmethod
    def adjacent_edges(
        vertex: 'Vertex', 
        parent_topology: 'Topology') -> List['Edge']:
        """
        TODO
        """
        core_adjacent_edges: List[Edge] = []
        core_adjacent_topologies: List[Topology] = vertex.upward_navigation_(
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
    
    @staticmethod
    def parameter_at_point(edge: 'Edge', vertex: 'Vertex') -> float:
        """
        Computes at which location the vertex is located for the edge.
        """
        (occt_geom_curve, u0, u1) = edge.curve()
        occt_point = vertex.get_point()
        (is_on_curve, occt_parameter) = GeomLib_Tool.Parameter(
            occt_geom_curve,
            occt_point.Pnt(),
            Precision.precision_Confusion()) 
        
        if not is_on_curve:
            raise RuntimeError("Point not on Curve!")
        
        # Parameter may be non-normalized, so normalize it
        return edge.normalize_parameter(u0, u1, occt_parameter)
    
    @staticmethod
    def point_at_parameter(edge: 'Edge', parameter: float) -> 'Vertex':
        """
        Constructs a point along the length of the edge. The position
        is defined by the parameter passed in.
        """
        u0 = 0.0
        u1 = 0.0
        (occt_geom_curve, u0, u1) = edge.curve()
        # is_instance = isinstance(occt_geom_curve, Geom_Line)
        occt_line: Geom_Line = occt_geom_curve # Downcasting to 'geom_line'
        if not occt_line is None:
            u0 = 0.0
            u1 = EdgeUtility.length(edge)

        # Parameter is normalized, so non-normalize it
        occt_parameter = edge.non_normalize_parameter(u0, u1, parameter)
        occt_point = occt_geom_curve.Value(occt_parameter)

        return Vertex.by_point(Geom_CartesianPoint(occt_point))
    
class FaceUtility:

    @staticmethod
    def trim_by_wire(face: 'Face', wire: 'Wire', reverse_wire: bool) -> 'Face':
        """
        Public method to create a new face that was trimmed by the passed in wire.
        """
        output_face = FaceUtility.__trim_by_wire_impl(face, wire, reverse_wire)
        # ToDo: GlobalCluster
        # GlobalCluster.get_instance().add_topology(output_face.get_occt_face())
        return output_face

    @staticmethod
    def __trim_by_wire_impl(face: 'Face', occt_wire: TopoDS_Wire, reverse_wire: bool) -> 'Face':
        """
        Implementation of trimming with a wire.
        """
        pOcctSurface = face.surface() 

        wireFix = ShapeFix_Wire()
        wireFix.Load(occt_wire)
        wireFix.Perform()

        if reverse_wire:
            trimmingWire = topods.Wire(wireFix.Wire().Reversed())
        else:
            trimmingWire = topods.Wire(wireFix.Wire())
        
        occtTrimMakeFace = BRepBuilderAPI_MakeFace(pOcctSurface, trimmingWire)
        if occtTrimMakeFace.Error() != BRepBuilderAPI_FaceDone:
            raise Exception("Error in making face")

        core_resulting_face = occtTrimMakeFace.Face()

        # Perform general shape fix
        occtFixShape = ShapeFix_Shape(core_resulting_face)
        occtFixShape.Perform()

        # Fix edges
        edge_explorer = TopExp_Explorer(core_resulting_face, TopAbs_EDGE)
        while edge_explorer.More():
            occtEdge = topods.Edge(edge_explorer.Current())
            occtFixEdge = ShapeFix_Edge()
            occtFixEdge.FixAddCurve3d(occtEdge)
            occtFixEdge.FixVertexTolerance(occtEdge)
            edge_explorer.Next()

        # Fix wires
        wire_explorer = TopExp_Explorer(core_resulting_face, TopAbs_WIRE)
        while wire_explorer.More():
            occtWire = topods.Wire(wire_explorer.Current())
            occtFixWire = ShapeFix_Wire(occtWire, core_resulting_face, 0.0001)
            occtFixWire.Perform()
            wire_explorer.Next()

        faceFix = ShapeFix_Face(core_resulting_face)
        faceFix.Perform()

        occtContext = ShapeBuild_ReShape()
        occtContext.Apply(faceFix.Face())

        occtFinalFace = topods.Face(
            shapefix.RemoveSmallEdges(
                core_resulting_face, 0.0001, occtContext))

        # Debugging checks
        # Uncomment below for debugging purposes
        # occtAnalyzer = BRepCheck_Analyzer(occtFinalFace)
        # isValid = occtAnalyzer.IsValid()
        # occtFaceCheck = BRepCheck_Face(topoDS.Face(occtFinalFace))
        # isUnorientable = occtFaceCheck.IsUnorientable()
        # orientationStatus = occtFaceCheck.OrientationOfWires()
        # intersectionStatus = occtFaceCheck.IntersectWires()
        # classificationStatus = occtFaceCheck.ClassifyWires()

        return Face(topods.Face(occtFinalFace)) 

    @staticmethod
    def triangulate(face: 'Face', deflection: float) -> List['Face']:
        """
        List of triangles that are converted to faces.
        """
        occt_face = face.get_occt_face()
        occt_incremental_mesh = BRepMesh_IncrementalMesh(occt_face, deflection)
        occt_location = TopLoc_Location()
        occt_triangulation = BRep_Tool.Triangulation(occt_face, occt_location)
        
        if occt_triangulation == None:
            raise RuntimeError("No triangulation was produced.")
        
        rTriangles = []  # List to store the resulting triangles
        numOfTriangles = occt_triangulation.NbTriangles()
        
        for i in range(1, numOfTriangles + 1):
            index1, index2, index3 = occt_triangulation.Triangle(i).Get()
            point1 = occt_triangulation.Node(index1)
            point2 = occt_triangulation.Node(index2)
            point3 = occt_triangulation.Node(index3)
            
            vertex1 = Vertex.by_point(Geom_CartesianPoint(point1))
            vertex2 = Vertex.by_point(Geom_CartesianPoint(point2))
            vertex3 = Vertex.by_point(Geom_CartesianPoint(point3))
            
            edge1 = Edge.by_start_vertex_end_vertex(vertex1, vertex2)
            edge2 = Edge.by_start_vertex_end_vertex(vertex2, vertex3)
            edge3 = Edge.by_start_vertex_end_vertex(vertex3, vertex1)
            edges = [edge1, edge2, edge3]
            
            face = Face.by_edges(edges)
            rTriangles.append(face)
        
        return rTriangles


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
    def internal_vertex(face: Face, tolerance: float) -> Vertex:
        pass

    @staticmethod
    def area(face: 'TopoDS_Face') -> float:
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

        surface = face.surface()
        occ_surface_analysis = ShapeAnalysis_Surface(surface)
        occt_uv = occ_surface_analysis.ValueOfUV(
            vertex.get_point().Pnt(), 
            Precision.precision_Confusion())
        
        (norm_u, norm_v) = FaceUtility.normalize_uv(face, occt_uv.X(), occt_uv.Y())
        return (norm_u, norm_v)

    
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

    @staticmethod
    def adjacent_faces(face: Face, parent_topology: Topology, core_adjacent_faces: List[Face]) -> None:
        
        # Iterate through the edges and find the incident faces which are not this face.
        occt_edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndUniqueAncestors(parent_topology.get_occt_shape(), TopAbs_EDGE, TopAbs_FACE, occt_edge_face_map)

        # Find the constituent faces
        occt_edges = TopTools_MapOfShape()

        occt_explorer = TopExp_Explorer(face.get_occt_shape(), TopAbs_EDGE)

        while occt_explorer.More():

            occt_current = occt_explorer.Current()

            if not occt_edges.Contains(occt_current):
                occt_edges.Add(occt_current)

            occt_explorer.Next()

        occt_face: TopoDS_Face = face.get_occt_face()

        occt_adjacent_faces = TopTools_MapOfShape()
        occt_edges_iterator = toptools.TopTools_MapIteratorOfMapOfShape(occt_edges)

        while occt_edges_iterator.More():

            occt_edge = occt_edges_iterator.Value()

            incident_faces: TopTools_ListOfShape = occt_edge_face_map.FindFromKey(occt_edge)

            incident_faces_iterator = TopTools_ListIteratorOfListOfShape(incident_faces)

            while incident_faces_iterator.More():

                incident_face = incident_faces_iterator.Value()

                if not occt_face.IsSame(incident_face):
                    occt_adjacent_faces.Add(incident_face)

                incident_faces_iterator.Next()

            occt_edges_iterator.Next()

        occt_adjacent_face_iterator = toptools.TopTools_MapIteratorOfMapOfShape(occt_adjacent_faces)

        while occt_adjacent_face_iterator.More(topods.Face(occt_adjacent_face_iterator.Value())):

            core_adjacent_faces.append()

            occt_adjacent_face_iterator.Next()

#--------------------------------------------------------------------------------------------------
class CellcontainmentState(Enum):
    INSIDE = 0
    ON_BOUNDARY = 1
    OUTSIDE = 2
    UNKNOWN = 3

#--------------------------------------------------------------------------------------------------
class CellUtility:

    @staticmethod
    def internal_vertex(occt_solid: TopoDS_Solid, tolerance: float) -> Vertex:
        return CellUtility.internal_vertex(Cell(occt_solid, ""), tolerance)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def internal_vertex(cell: Cell, tolerance: float) -> Vertex:
        
        # Check the centroid first
        center_of_mass: Vertex = cell.center_of_mass()

        if CellUtility.contains(cell, center_of_mass, tolerance) == CellcontainmentState.INSIDE:
            return center_of_mass

        # This methods accepts as input a Cell and outputs a Vertex guaranteed to be inside the Cell. 
        # This method relies on #3 to first choose a Vertex on a random Face of the Cell, 
        # then it shoots a Ray in the opposite direction of its normal and finds the closest intersection. 
        # The midpoint is returned as a Vertex guaranteed to be inside the Cell.

        # Pick random Face
        faces: List[Face] = []
        faces = cell.faces()

        for face in faces:

            vertex_in_face = FaceUtility.internal_vertex(face, tolerance)

            # Get the bounding box diagonal length
            min_x = 0.0
            max_x = 0.0
            min_y = 0.0
            max_y = 0.0
            min_z = 0.0
            max_z = 0.0

            CellUtility.get_min_max(min_x, max_x, min_y, max_y, min_z, max_z)

            corner_1: Vertex = Vertex.by_coordinates(min_x, min_y, min_z)
            corner_2: Vertex = Vertex.by_coordinates(max_x, max_y, max_z)

            diagonal_length = VertexUtility.distance(corner_1, corner_2)

            # Get the normal at vertexInFace
            u = 0.0
            v = 0.0

            FaceUtility.parameters_at_vertex(face, vertex_in_face, u, v)
            occt_normal: gp_Dir = FaceUtility.normal_at_parameters(face, u, v)
            occt_reversed_normal: gp_Dir = occt_normal.Reversed()

            occt_ray_end: gp_Pnt = vertex_in_face.Point().Pnt().Translated(diagonal_length * occt_reversed_normal)
            ray_end: Vertex = Vertex.by_point(Geom_CartesianPoint(occt_ray_end))

            # Shoot the ray = edge vs cell intersection
            ray: Edge = Edge.by_start_vertex_end_vertex(vertex_in_face, ray_end)
            ray_shot: Topology = cell.intersect(ray)

            # Get the vertices of the ray
            ray_vertices: List[Vertex] = []
            ray_vertices = ray_shot.vertices()

            # Get the closest intersection (but not the original vertex in face)
            min_distance = sys.float_info.max
            closest_vertex: Vertex = None

            for ray_vertex in ray_vertices:

                distance = VertexUtility.distance(ray_vertex, vertex_in_face)

                if distance < tolerance: # the same vertex
                    continue

                if distance < min_distance:
                    min_distance = distance
                    closest_vertex = ray_vertex

            if distance == None:
                raise RuntimeError("Ray casting fails to identify the closest vertex from a random point.")

            # Create a line between the closest vertex and vertex in Face, then get the midpoint (center of mass)
            shortest_edge: Edge = Edge.by_start_vertex_end_vertex(vertex_in_face, closest_vertex)
            shortest_edge_center_of_mass: Vertex = shortest_edge.center_of_mass()

            return shortest_edge_center_of_mass

        return None

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def contains(cell: Cell, vertex: Vertex, tolerance: float) -> CellcontainmentState:
        
        occt_solid_classifier = BRepClass3d_SolidClassifier(cell.get_occt_solid(), vertex.Point().Pnt(), Precision.precision_Confusion())

        occt_state: TopAbs_State = occt_solid_classifier.State()

        if occt_state == TopAbs_IN:
            return CellcontainmentState.INSIDE

        elif occt_state == TopAbs_ON:
            return CellcontainmentState.ON_BOUNDARY

        elif occt_state == TopAbs_OUT:
            return CellcontainmentState.OUTSIDE
        
        else:
            return CellcontainmentState.UNKNOWN

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def get_min_max(cell: Cell, min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float) -> None:
        
        occt_bounding_box = Bnd_Box()

        brepbndlib.Add(cell.get_occt_shape(), occt_bounding_box)
        occt_bounding_box.Get(min_x, min_y, min_z, max_x, max_y, max_z)