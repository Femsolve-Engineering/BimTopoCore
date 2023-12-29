
from email import contentmanager
from graphlib import TopologicalSorter
import sys
from io import StringIO
from typing import List
from typing import Dict
from typing_extensions import runtime
from AttributeManager import AttributeManager

# OCC
from ContentManager import ContentManager
from ContextManager import ContextManager
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_CompSolid, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, TopoDS_Compound, topods
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_MapIteratorOfMapOfShape
from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_WireDone, BRepBuilderAPI_MakeFace, BRepBuilderAPI_FaceDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShape, BRepBuilderAPI_Copy
from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Vertex
from OCC.Core.TopAbs import (
    TopAbs_REVERSED,
    TopAbs_SHAPE,
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_WIRE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_IN, TopAbs_ON, TopAbs_State
)
from OCC.Core.Geom import Geom_Geometry
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.GeomAbs import precision_Confusion

# BimTopoCore

from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Wire import Wire
from Core.Face import Face, FaceGUID
from Core.Cell import Cell
from Core.CellComplex import CellComplex
from Core.Shell import Shell
from Core.Cluster import Cluster

from Core.TopologyConstants import TopologyTypes
from Core.InstanceGUIDManager import InstanceGUIDManager
from Core.Factories.TopologyFactory import TopologyFactory
from Core.Factories.TopologyFactoryManager import TopologyFactoryManager
from Core.Context import Context
from Utilities.TopologicUtilities import FaceUtility, TopologyUtility, VertexUtility
from Vertex import Vertex
from Attribute import Attribute
from Dictionary import Dictionary
from Aperture import Aperture

from TopologicalQuery import TopologicalQuery

class Topology:
    """Placeholder class for all logic that is shared between
    all topological entities.
    """

    topologic_entity_count: int = 0

    def __init__(self,
                 base_shape: TopoDS_Shape, 
                 shape_type: TopologyTypes=TopologyTypes.UNDEFINED):
        """
        Constructor. 

        Args:
            dimensionality (int): referring to the geometric dimension of the object
            base_shape (TopoDS_Shape): underlying OCC shape
            guid (str): served GUID
        """

        self.base_shape = base_shape

        # Initialize to undefined, unless constructed from derived classes
        self.shape_type = shape_type
        self.dimensionality = self.get_dimensionality()

        # ToDo?: There shall be a singleton GUID manager taking care of the GUID-to-shape assignment
        self.guid = self.get_class_guid()

        Topology.topologic_entity_count += 1

#--------------------------------------------------------------------------------------------------
    def closest_simplest_subshape(self, topology: 'Topology') -> 'Topology':
        """
        Returns the closest simplest subshape that is part of the passed in topology.
        """
        closest_subshape = TopoDS_Shape()
        min_distance = sys.float_info.max
        this_shape = self.get_occt_shape()
        query_shape = topology.get_occt_shape()
        
        shape_types = [TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID]
        for shape_type in shape_types:
            topexp_explorer = TopExp_Explorer(this_shape, shape_type)
            while topexp_explorer.More():
                current_child_shape = topexp_explorer.Current()

                check_distance_shape = current_child_shape
                # ... (the commented part about solid fixing) ...

                distance_calculation = BRepExtrema_DistShapeShape(
                    check_distance_shape, 
                    query_shape)
                is_done = distance_calculation.Perform()
                
                if is_done:
                    distance = distance_calculation.Value()
                    if distance < min_distance:
                        min_distance = distance
                        closest_subshape = current_child_shape
                    # larger value = lower dimension
                    elif min_distance <= distance <= min_distance + precision_Confusion() \
                        and current_child_shape.ShapeType() > closest_subshape.ShapeType():
                        min_distance = distance
                        closest_subshape = current_child_shape

                topexp_explorer.Next()

        if closest_subshape.IsNull():
            return None

        return Topology.by_occt_shape(closest_subshape)

#--------------------------------------------------------------------------------------------------
    def select_sub_topology(self, selector: Vertex, type_filter: int) -> 'Topology':
        
        occt_closest_sub_shape: TopoDS_Shape
        min_distance: sys.float_info.max
        occt_this_shape: TopoDS_Shape = self.get_occt_shape()
        occt_selector_shape: TopoDS_Shape = selector.get_occt_shape()

        occt_shape_types = [TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID]
        shape_types = [TopologyTypes.VERTEX, TopologyTypes.EDGE, TopologyTypes.FACE, TopologyTypes.CELL]

        for i in range(4):

            # If this is not the requested topology type, skip.
            if type_filter != shape_types[i]:
                continue

            occt_shape_type: int = shape_types[i]
            occt_cells: TopTools_MapOfShape

            occt_expolorer = TopExp_Explorer(occt_this_shape, occt_shape_type)

            while occt_expolorer.More():

                current_child_shape = occt_expolorer.Value()
                check_distance_shape = current_child_shape
                check_distance_topology = Topology.by_occt_shape(check_distance_shape, "")

                distance = VertexUtility.distance(selector, check_distance_topology)

                if distance < min_distance:
                    min_distance = distance
                    occt_closest_sub_shape = current_child_shape

                elif (min_distance <= distance) and \
                     (distance <= min_distance + precision_Confusion()) and \
                     (current_child_shape.ShapeType() > occt_closest_sub_shape.ShapeType()):

                     closest_shape_type = occt_closest_sub_shape.ShapeType()
                     current_shape_type = current_child_shape.ShapeType()

                     min_distance = distance
                     occt_closest_sub_shape = current_child_shape

                occt_expolorer.Next()

        if occt_closest_sub_shape.IsNull():
            return None

        return Topology.by_occt_shape(occt_closest_sub_shape, "")

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def select_sub_topology(self, occt_shape: TopoDS_Shape, occt_selector_shape: TopoDS_Shape, type_filter: int, distance_threshold: float) -> TopoDS_Shape:
        
        min_distance = 0.0
        self.select_sub_topology(occt_shape, occt_selector_shape, min_distance, type_filter, distance_threshold)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def select_sub_topology(self, occt_shape: TopoDS_Shape, occt_selector_shape: TopoDS_Shape, min_distance: float, type_filter: int, distance_threshold: float):
        
        occt_closest_sub_shape: TopoDS_Shape

        topology = Topology.by_occt_shape(occt_selector_shape)
        selector = TopologicalQuery.downcast(topology)

        min_distance = sys.float_info.max

        for i in range(TopAbs_SHAPE):

            occt_shape_type = TopAbs_ShapeEnum(i)
            shape_type = Topology.get_topology_type(occt_shape_type)

            if shape_type != type_filter:
                continue

            occt_cells: TopTools_MapOfShape()
            occt_explorer = TopExp_Explorer(occt_shape, occt_shape_type)

            while occt_explorer.More():

                current_child_shape: TopoDS_Shape = occt_explorer.Current()
                check_distance_shape: TopoDS_Shape = current_child_shape

                check_distance_topology = Topology.by_occt_shape(check_distance_shape, "")

                distance = VertexUtility.distance(selector, check_distance_topology)

                if distance < min_distance:
                    min_distance = distance
                    occt_closest_sub_shape = current_child_shape

                elif (min_distance <= distance) and \
                     (distance <= min_distance + precision_Confusion) and \
                     (current_child_shape.ShapeType() > occt_closest_sub_shape.ShapeType()):

                    closest_shape_type = occt_closest_sub_shape.ShapeType()
                    current_shape_type = current_child_shape.ShapeType()

                    min_distance = distance
                    occt_closest_sub_shape = current_child_shape

                occt_explorer.Next()

            if min_distance < distance_threshold:
                return occt_closest_sub_shape

            else:
                return TopoDS_Shape()

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def center_of_mass(occt_shape: TopoDS_Shape) -> TopoDS_Vertex:
        """
        TODO: Implement for all types.
        """

        shape_type = occt_shape.ShapeType()

        if shape_type == TopAbs_VERTEX:
            return Vertex.center_of_mass(topods.Vertex(occt_shape))
        elif shape_type == TopAbs_EDGE:
            return Edge.center_of_mass(topods.Edge(occt_shape))
        elif shape_type == TopAbs_WIRE:
            return Wire.center_of_mass(topods.Wire(occt_shape))
        elif shape_type == TopAbs_FACE:
            return Face.center_of_mass(topods.Face(occt_shape))

        elif shape_type == TopAbs_SHELL:
            return Shell.center_of_mass(topods.Shell(occt_shape))
        elif shape_type == TopAbs_SOLID:
            return Cell.center_of_mass(topods.Solid(occt_shape))
        elif shape_type == TopAbs_COMPSOLID:
            return CellComplex.center_of_mass(topods.CompSolid(occt_shape))
        elif shape_type == TopAbs_COMPOUND:
            return Cluster.center_of_mass(topods.Compound(occt_shape))

        else:
            raise NotImplementedError(f"Missing implementation for shape type {shape_type}!")

#--------------------------------------------------------------------------------------------------
    # def center_of_mass(self) -> 'Vertex':
    #     """
    #     Pure virtual method for center of mass getting.
    #     """
    #     raise NotImplementedError("Topology is missing center of mass getter!")

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def get_topology_type(occt_Type: TopAbs_ShapeEnum) -> TopologyTypes:
        
        if occt_Type == TopAbs_VERTEX:
            return TopologyTypes.VERTEX
        elif occt_Type == TopAbs_EDGE:
            return TopologyTypes.EDGE
        elif occt_Type == TopAbs_WIRE:
            return TopologyTypes.WIRE
        elif occt_Type == TopAbs_FACE:
            return TopologyTypes.FACE

        elif occt_Type == TopAbs_SHELL:
            return TopologyTypes.SHELL
        elif occt_Type == TopAbs_SOLID:
            return TopologyTypes.CELL
        elif occt_Type == TopAbs_COMPSOLID:
            return TopologyTypes.CELLCOMPLEX
        elif occt_Type == TopAbs_COMPOUND:
            return TopologyTypes.CLUSTER

        else:
            raise RuntimeError("Unrecognised topology")

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def get_occt_topology_type(type: TopologyTypes) -> TopAbs_ShapeEnum:

        if type == TopologyTypes.VERTEX:
            return TopAbs_VERTEX
        elif type == TopologyTypes.EDGE:
            return TopAbs_EDGE
        elif type == TopologyTypes.WIRE:
            return TopAbs_WIRE
        elif type == TopologyTypes.FACE:
            return TopAbs_FACE
        elif type == TopologyTypes.SHELL:
            return TopAbs_SHELL
        elif type == TopologyTypes.CELL:
            return TopAbs_SOLID
        elif type == TopologyTypes.CELLCOMPLEX:
            return TopAbs_COMPSOLID
        elif type == TopologyTypes.CLUSTER:
            return TopAbs_COMPOUND

        else:
            raise RuntimeError("Unrecognised topology")

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def fix_shape(init_shape: TopoDS_Shape) -> TopoDS_Shape:
        """Uses OCC's ShapeFix_Shape to perform a shape fix.

        Returns:
            TopoDS_Shape: Fixed shape
        """
        shape_fixes = ShapeFix_Shape(init_shape)
        try: 
            shape_fixes.Perform()
            return shape_fixes.Shape()
        except Exception as ex:
            print(f'WARN: Shape fix failed! (Exception: {ex}) Returning initial shape!')
            return init_shape

#--------------------------------------------------------------------------------------------------
    def register_factory(self, guid: str, topology_factory: TopologyFactory) -> None:
        """
        Registers topological factory if it does not exist yet.
        """
        TopologyFactoryManager.get_instance().add(guid, topology_factory)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_occt_shape(occt_shape: TopoDS_Shape, instance_guid: str="") -> 'Topology':
        """
        Searches for the topology inside the TopologyFactor to find the instac
        """

        if occt_shape.IsNull():
            return None
        
        topology_factory_manager = TopologyFactoryManager.get_instance()
        p_topology_factory = None

        if instance_guid == "":
            p_topology_factory = topology_factory_manager.get_default_factory(occt_shape.ShapeType())
        else:
            p_topology_factory = topology_factory_manager.find(instance_guid)

        assert p_topology_factory is not None
        p_topology = p_topology_factory.create(occt_shape)

        return p_topology

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def make_comp_solid(occt_shape: TopoDS_Shape) -> TopoDS_CompSolid:
        
        occt_comp_solid: TopoDS_CompSolid()
        occt_builder: BRep_Builder()
        occt_builder.MakeCompSolid(occt_comp_solid)

        occt_cells = TopTools_MapOfShape()

        occt_explorer = TopExp_Explorer(occt_shape, TopAbs_SOLID)

        while occt_explorer.More():
            occt_current: TopoDS_Shape = occt_explorer.Curernt()

            if not occt_cells.Contains(occt_current):

                occt_cells.Add(occt_current)

                try:
                    occt_builder.Add(occt_comp_solid, occt_current)
                
                except:
                    raise RuntimeError("The Cell and Face are not compatible.")

                    # or
                    # raise RuntimeError("The Cell is not free and cannot be modified.")

        return occt_comp_solid

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_geometry(geometry: Geom_Geometry) -> 'Topology':
        return None

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_context(context: Context) -> 'Topology':
        return None

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_vertex_index(vertices: List[Vertex], vertex_indices: List[int], topologies: List['Topology']) -> None:
        
        if len(vertices) == 0 or len(vertex_indices) == 0:
            return

        occt_vertices: List[TopoDS_Vertex] = []

        for vertex_index in vertex_indices:
            occt_vertices.append(vertices[vertex_index].get_occt_vertex())

        if len(occt_vertices) > 2:

            # Create edges
            occt_edges: TopTools_ListOfShape

            num_of_vertices = len(occt_vertices)

            for i in range(num_of_vertices - 1):

                current_occt_vertex = occt_vertices[i]
                next_occt_vertex    = occt_vertices[i + 1]

                occt_make_edge = BRepBuilderAPI_MakeEdge(current_occt_vertex, next_occt_vertex)

                occt_edges.append(occt_make_edge)

            # No need to connect the first and last vertices

            # Creating a face
            occt_make_wire = BRepBuilderAPI_MakeWire()

            occt_make_wire.Add(occt_edges)

            if occt_make_wire.Error() == BRepBuilderAPI_WireDone:
                occt_wire = occt_make_wire.Wire()

                if BRepCheck_Wire(occt_wire).Closed() == BRepCheck_NoError:
                    occt_make_face = BRepBuilderAPI_MakeFace(occt_wire)

                    if occt_make_face.Error() == BRepBuilderAPI_FaceDone:
                        topologies.append(Face(occt_make_face).deep_copy())

                    else:
                        # Add the closed wire
                        topologies.append(Wire(occt_wire).deep_copy())

                else:
                    # Add the poen wire
                    topologies.append(Wire(occt_wire).deep_copy())

            else:
                # Add the edge
                occt_edge_iterator = TopTools_ListIteratorOfListOfShape(occt_edges)

                while occt_edge_iterator.More():

                    topologies.append(topods.Edge(occt_edge_iterator.Value()).deep_copy())

                    occt_edge_iterator.Next()

        elif len(occt_vertices) == 2:
            # Try creating an edge
            occt_make_edge = BRepBuilderAPI_MakeEdge(occt_vertices[0], occt_vertices[-1])
            topologies.append(Edge(occt_make_edge).deep_copy())

        elif len(occt_vertices) == 1:
            # Insert the vertices
            topologies.append(Vertex(occt_vertices[0]).deep_copy())

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_faces(faces: List[Face], tolerance: float) -> 'Topology':
        
        if len(faces) == 0:
            return None

        occt_shapes: TopTools_ListOfShape()

        for face in faces:
            occt_shapes.append(face.get_occt_shape())

        occt_shape = Topology.occt_sew_faces(occt_shapes, tolerance)
        topology = Topology.by_occt_shape(occt_shape, "")
        copy_topology = TopologicalQuery.dynamic_pointer_cast(topology.deep_copy())

        faces_as_topologies: List['Topology'] = []

        for face in faces:
            faces_as_topologies.append(face)

        copy_topology.deep_copy_attributes_from(faces_as_topologies)

        return topology

#--------------------------------------------------------------------------------------------------
    def add_content(self, topology: 'Topology') -> None:
        
        has_content = ContentManager.get_instance().has_content(self.get_occt_shape, topology.get_occt_shape())

        if has_content:
            return

        default_parameter = 0.0
        copy_topology = TopologicalQuery.dynamic_pointer_cast(self.deep_copy())
        copy_content_topology = TopologicalQuery.dynamic_pointer_cast(topology.deep_copy())

        ContentManager.get_instance().add(self.get_occt_shape(), topology)
        ContextManager.get_instance().add(topology.get_occt_shape(), \
                                          Context.by_topology_parameters(topology.by_occt_shape(self.get_occt_shape()), \
                                                                         default_parameter, \
                                                                         default_parameter, \
                                                                         default_parameter) \
                                          )

#--------------------------------------------------------------------------------------------------
    def add_content(self, topology: 'Topology', type_filter: int) -> 'Topology':
        
        contents = List['Topology']
        contents.append(topology)

        return self.add_contents(contents, type_filter)


#--------------------------------------------------------------------------------------------------
    def add_contents(self, content_topologies: List['Topology'], type_filter: int) -> 'Topology':
        
        # Deep copy this topology
        copy_topology = TopologicalQuery.dynamic_pointer_cast(self.deep_copy())

        # For all contents
        for content_topology in content_topologies:

            select_sub_topology: 'Topology'

            if (type_filter == 0) or (type_filter == self.get_shape_type):
                has_content = ContentManager.get_instance().has_content(self.get_occt_shape(), content_topology.get_occt_shape())

                if has_content:
                    continue

                select_sub_topology = copy_topology

            else:
                center_of_mass: Vertex = content_topology.center_of_mass()

                if type_filter == Cell.get_type():

                    # Iterate over all Cells of the original Topology. If any Cell contains this content, continue
                    has_content = False

                    cells: List[Cell] = []
                    cells = self.cells(None)

                    for cell in cells:
                        
                        cell_contents: List[Topology]
                        cell.contents_(cell_contents)

                        for cell_content in cell_contents:
                            
                            if cell_content.IsSame(content_topology):

                                has_content = True
                                break

                        if has_content:
                            break

                    if has_content:
                        continue

                    # Select the closest Face from the copy Topology
                    face = TopologicalQuery.downcast(copy_topology.select_sub_topology(center_of_mass, Face.get_type()))

                    adjacent_cells: List[Cell] = []
                    adjacent_cells = FaceUtility.adjacent_cells(face, copy_topology)

                    for cell in adjacent_cells:

                        occt_solid_classifier = BRepClass3d_SolidClassifier(cell.get_occt_solid(), center_of_mass.get_point().Pnt(), 0.1)
                        occt_state: TopAbs_State = occt_solid_classifier.State()

                        if occt_state == TopAbs_IN or occt_state == TopAbs_ON:

                            selected_sub_topology = cell
                            break

                    # If selectedSubtopology is still null, try with the rest of the Cells.
                    if selected_sub_topology == None:

                        cells: List[Cell] = []
                        cells = copy_topology.cells(None)

                        for cell in cells:

                            occt_solid_classifier = BRepClass3d_SolidClassifier(cell.get_occt_solid(), center_of_mass.get_point().Pnt(), 0.1)
                            occt_state: TopAbs_State = occt_solid_classifier.State()

                            if occt_state == TopAbs_IN:

                                selected_sub_topology = cell
                                break

                else:

                    selected_sub_topology = copy_topology.select_sub_topology(center_of_mass, type_filter)

            if selected_sub_topology != None:

                copy_content_topology: Topology = TopologicalQuery.dynamic_pointer_cast(content_topology.deep_copy())

                ContentManager.get_instance().add(selected_sub_topology, copy_content_topology)

                default_parameter = 0.0 # TODO: calculate the parameters

                ContextManager.get_instance().add(copy_content_topology.get_occt_shape(), \
                                    Context.by_topology_parameters(selected_sub_topology, \
                                                                    default_parameter, \
                                                                    default_parameter, \
                                                                    default_parameter) \
                                    )

        return copy_topology

#--------------------------------------------------------------------------------------------------
    def remove_content(self, topology: 'Topology') -> None:
        
        ContentManager.get_instance().remove(self.get_occt_shape(), topology.get_occt_shape())
        ContentManager.get_instance().remove(topology.get_occt_shape(), self.get_occt_shape())

#--------------------------------------------------------------------------------------------------
    def centroid(self) -> 'Vertex':
        """
        Averages all the existing vertices of the topological object and constructs
        a new vertex that is the arithmetic average. (Note, this is not the same as 
        center of mass). ToDo?: Should this really be called centroid?
        """
        from Core.Vertex import Vertex
        vertices: List[Vertex] = self.vertices()
        if len(vertices) == None:
            return None
        
        sumX = 0.0
        sumY = 0.0
        sumZ = 0.0
        for vertex in vertices:
            (xcoor, ycoor, zcoor) = vertex.coordinates()
            sumX += xcoor
            sumY += ycoor
            sumZ += zcoor

        num_of_vertices = len(vertices)
        avgX = sumX/num_of_vertices
        avgY = sumY/num_of_vertices
        avgZ = sumZ/num_of_vertices

        centroid = Vertex.by_coordinates(avgX, avgY, avgZ)
        return centroid        
    
#--------------------------------------------------------------------------------------------------
    def remove_contents(self, topologies: List['Topology']) -> 'Topology':
        
        contents: List['Topology']
        self.contents_(contents)

        added_contents: List['Topology']

        for content in contents:

            is_removed = False

            for removed_content in topologies:

                if content.IsSame(removed_content):

                    is_removed = True
                    break

            if not is_removed:

                copy_content: Topology = content.deep_copy()
                added_contents.append(copy_content)

        copy_topology: Topology = self.shallow_copy().add_contents(added_contents, 0)

        return copy_topology

#--------------------------------------------------------------------------------------------------
    def add_context(self, context: Context) -> None:
        """
        TODO
        """

        # 1. Register to ContextManager
        ContextManager.get_instance().add(self.get_occt_shape(), context)

        # 2. Register to ContentManager
        ContentManager.get_instance().add(context.topology().get_occt_shape(), Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid()))

#--------------------------------------------------------------------------------------------------
    def add_contexts(self, contexts: List[Context]) -> 'Topology':
        
        copy_topology: Topology = TopologicalQuery.dynamic_pointer_cast(self.deep_copy())

        content_instance_guid: str

        for context in contexts:

            has_content = ContentManager.get_instance().has_content(context.topology().get_occt_shape(), self.get_occt_shape())

            if has_content:
                continue

            occt_copy_content_shape = copy_topology.get_occt_shape()
            content_instance_guid = copy_topology.get_instance_guid()

            copy_context_topology = TopologicalQuery.dynamic_pointer_cast(context.topology().deep_copy())

            ContentManager.get_instance().add(copy_context_topology.get_occt_shape(), copy_topology)

            default_parameter = 0.0 # TODO: calculate the parameters

            ContextManager.get_instance().add(copy_topology.get_occt_shape(), \
                                              Context.by_topology_parameters(copy_context_topology, \
                                                                             default_parameter, \
                                                                             default_parameter, \
                                                                             default_parameter) \
                                             )

        return copy_topology


#--------------------------------------------------------------------------------------------------
    def remove_context(self, context: Context) -> 'Topology':
        
        # 1. Remove from ContextManager
        ContextManager.get_instance().remove(self.get_occt_shape(), context.topology().get_occt_shape())

        # 2. Remove from ContentManager
        ContentManager.get_instance().remove(context.topology().get_occt_shape(), self.get_occt_shape())

#--------------------------------------------------------------------------------------------------
    def remove_contexts(self, contexts: List[Context]) -> 'Topology':
        
        copy_topology = self.shallow_copy()

        for context in contexts:
            is_removed = False

            for removed_context in contexts:
                if context.topology().IsSame(removed_context.topology()):
                    is_removed = True
                    break

            if not is_removed:
                copy_context_topology: Topology = context.topology().deep_copy()
                copy_context: Context = Context.topology_by_parameters(copy_context_topology, context.u, context.v, context.w)
                copy_topology.add_context(copy_context)

        return copy_topology

#--------------------------------------------------------------------------------------------------
    def shared_topologies(self, topology: 'Topology', filter_type: int, shared_topologies: List['Topology']) -> None:
        
        occt_shape_1 = self.get_occt_shape()
        occt_shape_2 = topology.get_occt_shape()

        # Bitwise shift
        for i in range(9):

            int_topology_type = 1 << i

            if filter_type != int_topology_type:
                continue

            occt_sub_topology_type: TopAbs_ShapeEnum = Topology.get_occt_topology_type(int_topology_type)
            occt_sub_topologies_1: TopTools_MapOfShape

            occt_sub_topologies_1 = Topology.static_downward_navigation(occt_shape_1, occt_sub_topology_type)

            occt_sub_topologies_2 = TopTools_MapOfShape
            occt_sub_topologies_2 = Topology.static_downward_navigation(occt_shape_2, occt_sub_topology_type)

            occt_sub_topology_iterator_1 = TopTools_MapIteratorOfMapOfShape(occt_sub_topologies_1)
            occt_sub_topology_iterator_2 = TopTools_MapIteratorOfMapOfShape(occt_sub_topologies_2)

            while occt_sub_topology_iterator_1.More():

                while occt_sub_topology_iterator_2.More():

                    if occt_sub_topology_iterator_1.Value().IsSame(occt_sub_topology_iterator_2.Value()):
                        
                        topology = Topology.by_occt_shape(occt_sub_topology_iterator_1.Value(), "")
                        shared_topologies.append(topology)

#--------------------------------------------------------------------------------------------------
    def set_dictionaries(self, selectors: List[Vertex], dictionaries: List[Dict[str, Attribute]], type_filter: int) -> 'Topology':
        
        selector_size: int = len(selectors)

        type_filters: List[int] = [selector_size] * type_filter

        return self.set_dictionaries(selectors, dictionaries, type_filters)

#--------------------------------------------------------------------------------------------------
    def set_dictionaries(self, selectors: List[Vertex], dictionaries: List[Dict[str, Attribute]], type_filters: List[int], expect_duplicate_topologies: bool) -> 'Topology':
        
        if len(selectors) != len(dictionaries):
            raise RuntimeError("The lists of selectors and dictionaries do not have the same length.")

        if len(selectors) != len(type_filters):
            raise RuntimeError("The lists of selectors and type filters do not have the same length.")

        copy_topology: Topology = TopologicalQuery.dynamic_pointer_cast(self.deep_copy())
        context_instance_guid: str

        selected_sub_topologies: List[Topology]

        for selector, type_filter in zip(selectors, type_filters):

            if type_filter == 0:
                raise RuntimeError("No type filter specified.")

            selected_sub_topology: Topology = None

            if type_filter == Cell.get_type():

                closest_face_as_topology: Topology = copy_topology.select_sub_topology(selector, Face.get_type())

                # Select the closest Face. Note: if there is no Face, there is no Cell.
                closest_face: Face = None

                try:
                    closest_face = TopologicalQuery.downcast(closest_face_as_topology)
                except:
                    raise RuntimeError('Some error occured.')

                # Only continue if there is a closestFace.
                if closest_face != None:

                    adjacent_cells: List[Cell]
                    adjacent_cells = FaceUtility.adjacent_cells(closest_face, copy_topology)

                    for cell in adjacent_cells:
                        occt_solid_classifier = BRepClass3d_SolidClassifier(cell.get_occt_solid, selector.Point().Pnt(), 0.1)
                        occt_state = occt_solid_classifier.State()

                        if occt_state == TopAbs_IN:
                            selected_sub_topology = cell
                            break

                    # If selectedSubtopology is still null, try with the rest of the Cells.
                    if selected_sub_topology == None:

                        cells: List[Cell] = []
                        cells = self.cells(None)

                        for cell in cells:

                            occt_solid_classifier = BRepClass3d_SolidClassifier(cell.get_occt_solid(), selector.Point().Pnt(), 0.1)
                            occt_state: TopAbs_State = occt_solid_classifier.State()

                            if occt_state == TopAbs_IN:

                                selected_sub_topology = cell
                                break

            else:
                selected_sub_topology = copy_topology.select_sub_topology(selector, type_filter)

            if selected_sub_topology != None and not expect_duplicate_topologies:

                for kp_selected_sub_topology in selected_sub_topologies:

                    if kp_selected_sub_topology != None and kp_selected_sub_topology.is_same(selected_sub_topology):

                        raise RuntimeError("Another selector has selected the same member of the input Topology.")

            selected_sub_topologies.append(selected_sub_topology)

        for kp_selected_sub_topology in selected_sub_topologies:

            if kp_selected_sub_topology != None:

                AttributeManager.get_instance().clear_one(kp_selected_sub_topology.get_occt_shape())

                for attribute_pair in dictionaries:

                    for key, value in attribute_pair.items():

                        AttributeManager.get_instance().add(selected_sub_topology.get_occt_shape(), key, value)

        return copy_topology

#--------------------------------------------------------------------------------------------------
    def set_dictionaries(self, selectors: List[Vertex], dictionaries: List[Dictionary], type_filter: int) -> 'Topology':
        
        new_dictionaries: List[Dict[str, Attribute]]
        for i in dictionaries:
            new_dictionaries.append(i)

        return self.set_dictionaries(selectors, new_dictionaries, type_filter)

#--------------------------------------------------------------------------------------------------
    def set_dictionaries(self, selectors: List[Vertex], dictionaries: List[Dictionary], type_filters: List[int], expect_duplicate_topologies: bool) -> 'Topology':
        
        new_dictionaries: List[Dict[str, Attribute]]
        for i in dictionaries:
            new_dictionaries.append(i)

        return self.set_dictionaries(selectors, new_dictionaries, type_filters, expect_duplicate_topologies)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def occt_sew_faces(occt_faces: TopTools_ListOfShape, tolerance: float) -> TopoDS_Shape:
        
        occt_sewing = BRepBuilderAPI_Sewing(tolerance, True, True, True, True)

        occt_edge_iterator = TopTools_ListIteratorOfListOfShape(occt_faces)

        while occt_edge_iterator.More():

            occt_sewing.Add(occt_edge_iterator.Value())

            occt_edge_iterator.Next()

        occt_sewing.Perform()

        if occt_sewing.SewedShape().IsNull():

            raise RuntimeError("A null shape is created.")

        type: TopAbs_ShapeEnum = occt_sewing.SewedShape().ShapeType()

        while occt_edge_iterator.More():

            modified_shape = occt_sewing.Modified(occt_edge_iterator.Value())
            child_topology = Topology.by_occt_shape(modified_shape, "")

            # Map the aperture to the modified shell faces.
            contents: List[Topology] = []
            ContentManager.get_instance().find(occt_edge_iterator.Value(), contents)

            for content in contents:

                if content.get_shape_type() != TopologyTypes.APERTURE:
                    continue

                aperture: Aperture = TopologicalQuery.downcast(content)

                if aperture.topology().get_shape_type() != TopologyTypes.FACE:
                    continue

                aperture_face: Face = TopologicalQuery.downcast(aperture.topology())
                upcast_aperture_face: Topology = TopologicalQuery.upcast(aperture_face)

            occt_edge_iterator.Next()

        return occt_sewing.SewedShape()

#--------------------------------------------------------------------------------------------------
    def boolean_transfer_dictionary(self, origin_topology_1: 'Topology', origin_topology_2: 'Topology', destination_topology: 'Topology', init_clear_dictionary: bool) -> None:
        
        occt_origin_shape_1 = origin_topology_1.get_occt_shape()
        occt_origin_shape_2 = origin_topology_2.get_occt_shape()
        occt_destination_shape = destination_topology.get_occt_shape()

        # Get vertices, edges, faces, cells, cellComplexes from kpkDestinationTopology, and map them to the originTopology
        occt_origin_shape: TopoDS_Shape

        if origin_topology_1 == None and origin_topology_2 == None:
            raise RuntimeError("Fails to transfer dictionari in a Boolean operation because the original Topologies are null.")

        elif origin_topology_1 == None and origin_topology_2 != None:
            occt_origin_shape = origin_topology_2.get_occt_shape()

        elif origin_topology_1 != None and origin_topology_2 == None:
            occt_origin_shape = origin_topology_1.get_occt_shape()

        else:
            occt_origin_shapes = TopTools_MapOfShape()

            occt_origin_shapes.Add(origin_topology_1.get_occt_shape())
            occt_origin_shapes.Add(origin_topology_2.get_occt_shape())
            occt_origin_shape = Cluster.by_occt_topologies(occt_origin_shapes)

        # Get vertices, edges, faces, cells, cellComplexes from kpkDestinationTopology, and map them to the originTopology
        topology_types = [TopologyTypes.VERTEX, TopologyTypes.EDGE, TopologyTypes.FACE, TopologyTypes.CELL, TopologyTypes.CELLCOMPLEX]
        occt_topology_types = [TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPSOLID]

        for i in range(5):

            occt_destination_members: TopTools_MapOfShape
            occt_destination_members = destination_topology.static_downward_navigation(occt_destination_shape, occt_topology_types[i])

            occt_destination_member_iterator = TopTools_MapIteratorOfMapOfShape(occt_destination_members)

            while occt_destination_member_iterator.More():

                occt_destination_member = occt_destination_member_iterator.Value()

                if init_clear_dictionary:
                    AttributeManager.get_instance().clear_one(occt_destination_member)

                # Find the member in originTopology
                occt_destination_member_center_of_mass = Topology.center_of_mass(occt_destination_member)
                min_distance_1 = 0.0
                occt_origin_member_1 = Topology.select_sub_topology(occt_origin_shape_1, occt_destination_member_center_of_mass, min_distance_1, topology_types[i], 0.0001)

                min_distance_2 = 0.0
                occt_origin_member_2 = Topology.select_sub_topology(occt_origin_shape_2, occt_destination_member_center_of_mass, min_distance_2, topology_types[i], 0.0001)

                if not occt_destination_member_center_of_mass.IsNull() and not occt_origin_member_1.IsNull():
                    AttributeManager.get_instance().copy_attributes(occt_origin_member_1, occt_destination_member, True)

                if not occt_destination_member_center_of_mass.IsNull() and not occt_origin_member_2.IsNull():
                    AttributeManager.get_instance().copy_attributes(occt_origin_member_2, occt_destination_member, True)

                occt_destination_member_iterator.Next()

#--------------------------------------------------------------------------------------------------
    def difference(self, other_topology: 'Topology', transfer_dictionary: bool) -> 'Topology':
        
        if other_topology == None:
            return Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())

        occt_arguments_A = TopTools_ListOfShape()
        occt_arguments_B = TopTools_ListOfShape()

        self.add_boolean_operands(other_topology, occt_arguments_A, occt_arguments_B)

        occt_cells_builder = BOPAlgo_CellsBuilder()
        Topology.non_regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_cells_builder)

        # 2. Select the parts to be included in the final result.
        occt_list_to_take = TopTools_ListOfShape
        occt_list_to_avoid = TopTools_ListOfShape

        occt_shape_iterator_A = TopTools_ListIteratorOfListOfShape(occt_arguments_A)
        occt_shape_iterator_B = TopTools_ListIteratorOfListOfShape(occt_arguments_B)

        while occt_shape_iterator_A.More():

            occt_list_to_take.Clear()
            occt_list_to_avoid.Clear()
            occt_list_to_take.Append(occt_shape_iterator_A.Value())

            while occt_shape_iterator_B.More():

                occt_list_to_avoid.Append(occt_shape_iterator_B.Value())

                occt_shape_iterator_B.Next()

            occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid)

            occt_shape_iterator_A.Next()

        occt_cells_builder.MakeContainers()

        occt_result_shape = occt_cells_builder.Shape()
        if occt_result_shape.IsNull():
            occt_post_processed_shape = occt_result_shape
        else:
            occt_post_processed_shape = self.post_process_boolean_operation(occt_result_shape)

        post_processed_shape = Topology.by_occt_shape(occt_post_processed_shape, "")

        if post_processed_shape == None:
            return None

        Topology.transfer_contents(self.get_occt_shape(), post_processed_shape)
        Topology.transfer_contents(other_topology.get_occt_shape(), post_processed_shape)

        copy_post_processed_shape = post_processed_shape.deep_copy()

        if transfer_dictionary:
            self.boolean_transfer_dictionary(self, other_topology, True)

        return copy_post_processed_shape
        
#--------------------------------------------------------------------------------------------------
    def contents_(self, contents: List['Topology']) -> None:
        Topology.contents(self.get_occt_shape(), contents)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def contents(occt_shape: TopoDS_Shape, contents: List['Topology']) -> None:
        instance = ContentManager.get_instance()
        instance.find(occt_shape, contents)

#--------------------------------------------------------------------------------------------------
    def apertures(self, apertures: List[Aperture]) -> None:
        """
        TODO - M3
        """
        Topology.apertures(self.get_occt_shape(), apertures)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def apertures(occt_shape: TopoDS_Shape, apertures: List[Aperture]):
        
        contents: List[Topology]
        ContentManager.get_instance().find(occt_shape, contents)

        for content in contents:

            if content.get_shape_type() == TopologyTypes.APERTURE:

                aperture: Aperture = TopologicalQuery.downcast(content)
                apertures.append(aperture)

#--------------------------------------------------------------------------------------------------
    def sub_contents(self, sub_contents: List['Topology']) -> None:
        """
        Returns:
            All topologies that are stored under this topology.
        """
        self.static_sub_contents(self.get_occt_shape(), sub_contents)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def static_sub_contents(occt_shape: TopoDS_Shape, sub_contents: List['Topology']) -> None:
        """
        Finds all the topologies that are of lower type.
        """

        Topology.contents(occt_shape, sub_contents)

        occt_type: TopAbs_ShapeEnum = occt_shape.ShapeType()

        occt_type_int = occt_type + 1 # +1 for the next lower type

        for  i in range(occt_type_int, int(TopAbs_SHAPE)):

            # Get members in each level
            occt_type_iteration: TopAbs_ShapeEnum = i
            occt_members = TopTools_MapOfShape()
            Topology.static_downward_navigation(occt_shape, occt_type_iteration, occt_members)

            # For each member, get the contents
            occt_member_iterator = TopTools_MapIteratorOfMapOfShape()

            while occt_member_iterator.More():

                ContentManager.get_instance().find(occt_member_iterator.Value(), sub_contents)

                occt_member_iterator.Next()

#--------------------------------------------------------------------------------------------------
    def contexts(self, contexts: List[Context]) -> bool:
        
        return Topology.Context(self.get_occt_shape(), contexts)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def contexts(occt_shape: TopoDS_Shape, contexts: List[Context]) -> bool:
        
        return ContextManager.get_instance().find(occt_shape, contexts)

#--------------------------------------------------------------------------------------------------
    def export_to_brep(self, file_path: str, version: int = 3) -> bool:
        
        if version == 1:
            return BRepTools.Write(self.get_occt_shape(), file_path, False, True, TopTools_FormanVersion_1)

        elif version == 2:
            return BRepTools.Write(self.get_occt_shape(), file_path, False, True, TopTools_FormanVersion_2)

        elif version == 3:
            return BRepTools.Write(self.get_occt_shape(), file_path, False, True, TopTools_FormanVersion_3)

        return BRepTools.Write(self.get_occt_shape(), file_path, False, True, TopTools_FormanVersion_CURRENT)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_imported_brep(file_path: str) -> 'Topology':
        
        occt_shape = TopoDS_Shape()
        occt_brep_builder = BRep_Builder()

        return_value = BRepTools.Read(occt_shape, file_path, occt_brep_builder)
        topology = Topology.by_occt_shape(occt_shape, "")

        return topology

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_string(brep_string: str) -> 'Topology':
        
        occt_shape  = TopoDS_Shape()
        occt_brep_builder = BRep_Builder()

        iss = StringIO(brep_string)

        BRepTools.Read(occt_shape, iss, occt_brep_builder)

        topology = Topology.by_occt_shape(occt_shape, "")

        return topology

#--------------------------------------------------------------------------------------------------
    def string(self, version: int = 3) -> str:
        
        oss = StringIO()

        if version == 1:
            BRepTools.Write(self.get_occt_shape(), oss, False, True, TopTools_FormanVersion_1)

        elif version == 2:
            BRepTools.Write(self.get_occt_shape(), oss, False, True, TopTools_FormanVersion_2)

        elif version == 3:
            BRepTools.Write(self.get_occt_shape(), oss, False, True, TopTools_FormanVersion_3)

        return oss.getvalue()

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def filter(topologies: List['Topology'], type_filter: int, filtered_topologies: List['Topology']) -> None:

        for topology in topologies:

            shape_type: int = topology.get_shape_type()

            if shape_type != type_filter:
                continue

            filtered_topologies.append(topology)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def analyze(shape: TopoDS_Shape, level: int) -> str:
        
        occt_sub_topologies = TopTools_ListOfShape()
        Topology.sub_topologies(shape, occt_sub_topologies)

        occt_shape_name_singular: List[str] = []

        occt_shape_name_singular[0] = 'a_cluster'
        occt_shape_name_singular[1] = 'a_cellComplex'
        occt_shape_name_singular[2] = 'a_cell'
        occt_shape_name_singular[3] = 'a_shell'
        occt_shape_name_singular[4] = 'a_face'
        occt_shape_name_singular[5] = 'a_wire'
        occt_shape_name_singular[6] = 'an_edge'
        occt_shape_name_singular[7] = 'a_vertex'

        occt_shape_name_plural: List[str] = []
        occt_shape_name_plural[0] = 'clusters'
        occt_shape_name_plural[1] = 'cellComplexes'
        occt_shape_name_plural[2] = 'cells'
        occt_shape_name_plural[3] = 'shells'
        occt_shape_name_plural[4] = 'faces'
        occt_shape_name_plural[5] = 'wires'
        occt_shape_name_plural[6] = 'edges'
        occt_shape_name_plural[7] = 'vertices'

        occt_shape_type: TopAbs_ShapeEnum = shape.ShapeType()

        current_indent = "  " * level

        number_of_sub_entities = [0,0,0,0,0,0,0,0]

        member_iterator = TopTools_ListIteratorOfListOfShape(occt_sub_topologies)

        while member_iterator.More():

            member_topology: TopoDS_Shape = member_iterator.Value()

            occt_shape_member_type: TopAbs_ShapeEnum = member_topology.ShapeType()
            number_of_sub_entities[occt_shape_member_type] += 1

            member_iterator.Next()

        ss_current_result = StringIO()

        # For the topmost level only, print the overall subentities result
        if level == 0:

            occt_shape_analysis = ShapeAnalysis_ShapeContents()
            occt_shape_analysis.Perform()

            # No method is provided in ShapeAnalysis_ShapeContents to compute the number of CompSolids.
            # Do this manually.

            number_compSolids = 0

            occt_compSolids = TopTools_ListOfShape

            occt_explorer = TopExp_Explorer(shape, TopAbs_COMPSOLID)

            while occt_explorer.More():

                occt_current: TopoDS_Shape = occt_explorer.Current()

                if not occt_compSolids.Contains(occt_current):
                    occt_compSolids.Append(occt_current)
                    number_compSolids += 1

                occt_explorer.Next()

            ss_current_result.write(
            f"OVERALL ANALYSIS\n"
            f"================\n"
            f"The shape is {occt_shape_name_singular[occt_shape_type]}.\n"
            )

            if occt_shape_type == 0: # Only for cluster
                ss_current_result.write(f"Number of cell complexes = {number_compSolids}\n")

            if occt_shape_type <= 1: # Only up to cellcomplex
                ss_current_result.write(f"Number of cells = {occt_shape_analysis.NbSharedSolids()}\n")

            if occt_shape_type <= 2: # Only up to cell
                ss_current_result.write(f"Number of shells = {occt_shape_analysis.NbSharedShells()}\n")

            if occt_shape_type <= 3: # Only up to shell
                ss_current_result.write(f"Number of faces = {occt_shape_analysis.NbSharedFaces()}\n")

            if occt_shape_type <= 4: # Only up to face
                ss_current_result.write(f"Number of wires = {occt_shape_analysis.NbSharedWires()}\n")

            if occt_shape_type <= 5: # Only up to wire
                ss_current_result.write(f"Number of edges = {occt_shape_analysis.NbSharedEdges()}\n")

            if occt_shape_type <= 6: # Only up to shell
                ss_current_result.write(f"Number of vertices = {occt_shape_analysis.NbSharedVertices()}\n")

            ss_current_result.write("\n\nINDIVIDUAL ANALYSIS\n" + "================\n")

        ss_current_result.write(f"{current_indent}The shape is {occt_shape_name_singular[occt_shape_type]}.\n")

        for i in range(occt_shape_type + 1, 8):

            if number_of_sub_entities[i] > 0:

                ss_current_result.write(f"{current_indent}Number of {occt_shape_name_plural[i]} = {number_of_sub_entities[i]}\n")

        ss_current_result.write(f"{current_indent}================\n")

        member_iterator = TopTools_ListIteratorOfListOfShape(occt_sub_topologies)

        while member_iterator.More():

            member_topology: TopoDS_Shape = member_iterator.Value()
            str_member_analyze = Topology.analyze(member_topology, level + 1)
            ss_current_result.write(str_member_analyze)

            member_iterator.Next()

        return ss_current_result.getvalue() 

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def simplify(occt_shape: TopoDS_Shape) -> TopoDS_Shape:
        
        # Simplify needs to do the following.
		# 1. The input is a container type, otherwise return itself.
		# 2. If the input is an empty cluster: return null
		# 3. Else if the input just contains one container element: recursively dive deeper until a non-container element OR
		#    a container with more than one elements is found.
		# 4. Else if the input contains more than elements:
		#    a. For a non-container element: leave it.
		#    b. For a container element: recursively dive deeper until a non-container element OR
		#       a container with more than one elements is found.

        if not Topology.is_container_type(occt_shape):
            return occt_shape

        occt_sub_topologies = TopTools_ListOfShape()
        Topology.sub_topologies(occt_shape, occt_sub_topologies)

        if occt_sub_topologies.Size() == 0:
            return TopoDS_Shape()

        elif occt_sub_topologies.Size() == 1:

            occt_current_shape = occt_shape

            occt_shapes = TopTools_ListOfShape()
            occt_shapes_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes)

            is_simplest_shape_found = False

            while not is_simplest_shape_found:

                # Only do this for wire, shell, cellcomplex, cluster
                if not Topology.is_container_type(occt_current_shape):
                    break

                Topology.sub_topologies(occt_current_shape, occt_shapes)

                num_of_sub_topologies: int = occt_shapes.Size()

                if num_of_sub_topologies != 1:

                    # occtCurrentShape does not change.
                    is_simplest_shape_found = True

                else: # if (occtShapes.Size() == 1)
                    # Go deeper
                    occt_current_shape = occt_shapes_iterator.Next()

                occt_shapes.Clear()

            return occt_current_shape

        else: # occtSubTopologies.Size() > 1

            occt_shapes_to_remove = TopTools_ListOfShape()
            occt_shapes_to_add = TopTools_ListOfShape()

            occt_sub_topology_iterator = TopTools_ListIteratorOfListOfShape()

            while occt_sub_topology_iterator.More():

                occt_sub_shape: TopoDS_Shape = occt_sub_topology_iterator.Value()

                if not Topology.is_container_type(occt_sub_shape):
                    continue

                occt_current_shape: TopoDS_Shape = occt_sub_shape

                occt_shapes = TopTools_ListOfShape()
                occt_shapes_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes)

                is_simplest_shape_found = False

                while not is_simplest_shape_found:

                    # Only do this for wire, shell, cellcomplex, cluster
                    if not Topology.is_container_type(occt_current_shape):
                        break

                    Topology.sub_topologies(occt_current_shape, occt_shapes)

                    num_of_sub_topologies = occt_shapes.Size()

                    if num_of_sub_topologies != 1:

                        # occtCurrentShape does not change.
                        is_simplest_shape_found = True

                    else: # if (occtShapes.Size() == 1)

                        # Go deeper
                        occt_current_shape = occt_shapes_iterator.Next()

                    occt_shapes.Clear()

                if not occt_current_shape.IsSame(occt_sub_shape):

                    # Do this so as to not modify the list in the iteration.
                    occt_shapes_to_remove.Append(occt_sub_shape)
                    occt_shapes_to_add.Append(occt_current_shape)

            occt_sub_topology_to_remove_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes_to_remove)

            while occt_sub_topology_to_remove_iterator.More():

                occt_builder = TopoDS_Builder()

                try:
                    occt_builder.Remove(occt_shape, occt_sub_topology_to_remove_iterator.Value())

                except:
                    raise RuntimeError("Topology is locked, cannot remove subtopology. Please contact the developer.")

                occt_sub_topology_to_remove_iterator.Next()

            
            occt_sub_topology_to_add_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes_to_add)

            while occt_sub_topology_to_add_iterator.More():

                try:
                    occt_builder.Add(occt_shape, occt_sub_topology_to_add_iterator.Value())

                except:
                    raise RuntimeError("Cannot add incompatible subtopology.")

                occt_sub_topology_to_add_iterator.Next()

        return occt_shape

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def boolean_sub_topology_containment(occt_shape: TopoDS_Shape) -> TopoDS_Shape:
        
        # 1. Only for cluster
		# 2. If the input is an empty cluster: return null
		# 3. For each subtopology A, check against each other subtopology B. If A is inside B, remove A.

        if occt_shape.ShapeType() != TopAbs_COMPOUND:
            return occt_shape

        occt_sub_topologies = TopTools_ListOfShape()
        Topology.sub_topologies(occt_shape, occt_sub_topologies)

        if occt_sub_topologies.Size() == 0:
            return TopoDS_Shape()

        occt_shapes_to_remove = TopTools_MapOfShape()

        occt_sub_topology_iterator_A = TopTools_ListIteratorOfListOfShape(occt_sub_topologies)

        while occt_sub_topology_iterator_A.More():

            occt_sub_topology_A: TopoDS_Shape = occt_sub_topology_iterator_A.Value()

            is_shapeA_to_remove = False

            occt_sub_topology_iterator_B = TopTools_ListIteratorOfListOfShape(occt_sub_topologies)

            while occt_sub_topology_iterator_B.More():

                occt_sub_topology_B = occt_sub_topology_iterator_B.Value()

                if occt_sub_topology_A.IsSame(occt_sub_topology_B):
                    continue

                # Does B contain A?
                occt_sub_topologies_B = TopTools_MapOfShape()
                Topology.static_downward_navigation(occt_sub_topology_B, occt_sub_topology_A.ShapeType(), occt_sub_topologies_B)

                if occt_sub_topologies_B.Contains(occt_sub_topology_A):
                    is_shapeA_to_remove = True
                    occt_shapes_to_remove.Add(occt_sub_topology_A)

                occt_sub_topology_iterator_B.Next()

            occt_sub_topology_iterator_A.Next()

        # Remove the shapes
        occt_shapes_to_remove_iterator = TopTools_MapIteratorOfMapOfShape(occt_shapes_to_remove)

        while occt_shapes_to_remove_iterator.Move():

            occt_builder = TopoDS_Builder()

            try:
                occt_builder.Remove(occt_shape, occt_shapes_to_remove_iterator.Value())
            
            except:
                raise RuntimeError("Topology is locked, cannot remove subtopology. Please contact the developer.")

            occt_shapes_to_remove_iterator.Next()

        return occt_shape

#--------------------------------------------------------------------------------------------------
    def analyze(self) -> str:

        return Topology.analyze(self.get_occt_shape(), 0)

#--------------------------------------------------------------------------------------------------
    def non_regular_boolean_operation(self):
        pass

#--------------------------------------------------------------------------------------------------
    def non_regular_boolean_operation(self, other_topology: 'Topology',\
                                      occt_cells_builder: BOPAlgo_CellsBuilder,\
                                      occt_cells_builders_operands_A: TopTools_ListOfShape,\
                                      occt_cells_builders_operands_B: TopTools_ListOfShape,\
                                      occt_map_face_to_fixed_face_A: TopTools_DataMapOfShapeShape,\
                                      occt_map_face_to_fixed_face_B: TopTools_DataMapOfShapeShape) -> None:
        
        self.add_boolean_operands(other_topology, occt_cells_builder, occt_cells_builders_operands_A, occt_cells_builders_operands_B, occt_map_face_to_fixed_face_A, occt_map_face_to_fixed_face_B)

        # Split the arguments and tools
        try:
            occt_cells_builder.Perform()

        except:
            raise RuntimeError("Some error occured.")

        if occt_cells_builder.HasErrors():
            error_stream = StringIO()
            occt_cells_builder.DumpErrors(error_stream)
            error_message = error_stream.getvalue()
            raise RuntimeError(error_message)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def non_regular_boolean_operation(occt_arguments_A: TopTools_ListOfShape, occt_arguments_B: TopTools_ListOfShape, occt_cells_builder: BOPAlgo_CellsBuilder):
        
        occt_arguments = TopTools_ListOfShape()

        occt_argument_iterator_A = TopTools_ListIteratorOfListOfShape(occt_arguments_A)
        occt_argument_iterator_B = TopTools_ListIteratorOfListOfShape(occt_arguments_B)

        while occt_argument_iterator_A.More():

            occt_arguments.append(occt_argument_iterator_A.Value())

            occt_argument_iterator_A.Next()

        while occt_argument_iterator_B.More():

            occt_arguments.Append(occt_argument_iterator_B.Value())

            occt_argument_iterator_B.Next()

        occt_cells_builder.SetArguments(occt_arguments)

        # Split the arguments and tools
        try:
            occt_cells_builder.Perform()

        except:
            raise RuntimeError("Some error occured.")

        if occt_cells_builder.HasErrors():
            error_stream = StringIO()
            occt_cells_builder.DumpErrors(error_stream)
            error_message = error_stream.getvalue()
            raise RuntimeError(error_message)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def transfer_contents(occt_shape_1: TopoDS_Shape, topology_2: 'Topology') -> None:
        
        sub_contents: List[Topology] = []
        Topology.sub_contents(occt_shape_1, sub_contents)

        for sub_content in sub_contents:

            # Attach to the same context type
            context_type = 0
            contexts: List[Context] = []
            sub_content.contexts(contexts)

            for context in contexts:

                context_topology: Topology = context.topology()
                context_topology_type = context_topology.get_shape_type()

                context_type = context_type or context_topology_type

                # Remove content from current contexts
                context_topology.remove_content(sub_content)
                sub_content.remove_context(context)

            topology_2.add_content(sub_content, context_type)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def transfer_contents(occt_shape_1: TopoDS_Shape, topology_2: 'Topology', occt_delete_sub_shapes: TopTools_ListOfShape) -> None:

         sub_contents: List[Topology] = []
         Topology.sub_contents(occt_shape_1, sub_contents)

         for sub_content in sub_contents:

            # Check if the context topology is part of kpTopology2. Use OCCT IsDeleted()
            all_contexts_dissappear = True
            contexts: list[Context] = []
            sub_content.Contexts(contexts)

            for context in contexts:

                if not occt_delete_sub_shapes.Contains(context.topology().get_occt_shape()):
                    
                    all_contexts_dissappear = False
                    break

            if all_contexts_dissappear:
                continue

            # Attach to the same context type
            context_type = 0

            for context in contexts:

                context_topology: Topology = context.topology()
                context_topology_type = context_topology.get_shape_type()

                context_type = context_type or context_topology_type

                # Remove content from current contexts
                context_topology.remove_content(sub_content)
                sub_content.remove_context(context)

            topology_2.add_content(sub_content, context_type)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def regular_boolean_operation(occt_arguments_A: TopTools_ListOfShape, occt_arguments_B: TopTools_ListOfShape, occt_boolean_operation: BRepAlgoAPI_BooleanOperation) -> None:
        
        occt_boolean_operation.SetArguments(occt_arguments_A)
        occt_boolean_operation.SetTools(occt_arguments_B)
        occt_boolean_operation.SetNonDestructive(True)
        occt_boolean_operation.Build()

#--------------------------------------------------------------------------------------------------
    def post_process_boolean_result(self, occt_boolean_result: TopoDS_Shape) -> TopoDS_Shape:
        
        occt_post_processed_shape: TopoDS_Shape = Topology.simplify(occt_boolean_result)

        if not occt_post_processed_shape.IsNull():
            occt_post_processed_shape = Topology.boolean_sub_topology_containment(occt_post_processed_shape)

        if not occt_post_processed_shape.IsNull():
            occt_post_processed_shape = Topology.simplify(occt_post_processed_shape)

        return occt_post_processed_shape

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def transfer_make_shape_contents(occt_make_shape: BRepBuilderAPI_MakeShape, shapes: List['Topology']) -> None:
        
        occt_shapes = TopTools_ListOfShape()

        for shape in shapes:
            occt_shapes.Append(shape.get_occt_shape())

        Topology.transfer_make_shape_contents(occt_make_shape, occt_shapes)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def transfer_make_shape_contents(occt_make_shape: BRepBuilderAPI_MakeShape, 
                                     occt_shapes: TopTools_ListOfShape) -> None:
        """
        Transfers the contents of each shapes to newly created shapes.

        Args:
            occt_make_shape (BRepBuilderAPI_MakeShape): _description_
            occt_shapes (TopTools_ListOfShape): _description_
        """
        # 1. For each shape in occt_shapes, find the generated shapes in occt_make_shape
        shape_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes)

        while shape_iterator.More():
            occt_original_shape = shape_iterator.Value()
            original_shape = Topology.by_occt_shape(occt_original_shape, "")  # Assuming this method exists and has a similar signature
            occt_generated_shapes = occt_make_shape.Modified(occt_original_shape)

            # 2. Transfer the contents from the original shape to the generated shapes
            contents: List['Topology'] = []
            contents = original_shape.contents_(contents)

            generated_shape_iterator = TopTools_ListIteratorOfListOfShape(occt_generated_shapes)

            while generated_shape_iterator.More():
                occt_generated_shape = generated_shape_iterator.Value()
                generated_shape = Topology.by_occt_shape(occt_generated_shape, "")

                for content in contents:
                    generated_shape.add_content(content)

                generated_shape_iterator.Next()

            shape_iterator.Next()

#--------------------------------------------------------------------------------------------------
    def impose(self, tool: 'Topology', transfer_dictionary: bool) -> 'Topology':

        if tool == None:
            return Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())

        occt_arguments_A = TopTools_ListOfShape()
        occt_arguments_B = TopTools_ListOfShape()

        self.add_boolean_operands(tool, occt_arguments_A, occt_arguments_B)

        occt_cells_builder = BOPAlgo_CellsBuilder()
        Topology.non_regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_cells_builder)

        # 2. Select the parts to be included in the final result.
        occt_list_to_take = TopTools_ListOfShape()
        occt_list_to_avoid = TopTools_ListOfShape()

        # Get part of A not in B
        occt_shape_iterator_A = TopTools_ListIteratorOfListOfShape(occt_arguments_A)
        occt_shape_iterator_B = TopTools_ListIteratorOfListOfShape(occt_arguments_B)

        while occt_shape_iterator_A.More():

            occt_list_to_take.Clear()
            occt_list_to_avoid.Clear()
            occt_list_to_take.Append(occt_shape_iterator_A.Value())

            while occt_shape_iterator_B.More():

                occt_list_to_avoid.Append(occt_shape_iterator_B.Value())

                occt_shape_iterator_B.Next()
            
            occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid)

            occt_shape_iterator_A.Next()

        # Add B
        i = 1
        while occt_shape_iterator_B.More():

            i += 1

            occt_list_to_take.Clear()
            occt_list_to_avoid.Clear()

            occt_list_to_take.Append(occt_shape_iterator_B.Value())
            occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid, i, True)

            occt_shape_iterator_B.Next()

        occt_cells_builder.MakeContainers()

        occt_result_shape: TopoDS_Shape = occt_cells_builder.Shape()

        if occt_result_shape.IsNull():
            occt_post_processed_shape: TopoDS_Shape = occt_result_shape
        
        else:
            self.post_process_boolean_result(occt_result_shape)

        post_processed_shape: Topology = Topology.by_occt_shape(occt_post_processed_shape, "")

        if post_processed_shape == None:
            return None

        copy_post_processed_shape: Topology = post_processed_shape.deep_copy()

        Topology.transfer_contents(self.get_occt_shape(), copy_post_processed_shape)
        Topology.transfer_contents(tool.get_occt_shape(), copy_post_processed_shape)


        if transfer_dictionary:

            self.boolean_transfer_dictionary(self, tool, copy_post_processed_shape, True)

        return copy_post_processed_shape

#--------------------------------------------------------------------------------------------------
    def imprint(self, tool: 'Topology', transfer_dictionary: bool) -> 'Topology':
        
        if tool == None:
            return Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())

        occt_arguments_A = TopTools_ListOfShape()
        occt_arguments_B = TopTools_ListOfShape()
        self.add_boolean_operands(tool, occt_arguments_A, occt_arguments_B)

        occt_cells_builder = BOPAlgo_CellsBuilder()
        Topology.non_regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_cells_builder)

        # 2. Select the parts to be included in the final result.
        occt_list_to_take = TopTools_ListOfShape()
        occt_list_to_avoid = TopTools_ListOfShape()

        occt_shape_iterator_A = TopTools_ListIteratorOfListOfShape(occt_arguments_A)
        occt_shape_iterator_B = TopTools_ListIteratorOfListOfShape(occt_arguments_B)

        while occt_shape_iterator_A.More():

            while occt_shape_iterator_B.More():

                occt_list_to_take.Clear()
                occt_list_to_avoid.Clear()
                occt_list_to_take.Append(occt_shape_iterator_A.Value())
                occt_list_to_take.Append(occt_shape_iterator_B.Value())
                occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid)

                occt_shape_iterator_B.Next()

            occt_shape_iterator_A.Next()

        while occt_shape_iterator_A.More():

            occt_list_to_take.Clear()
            occt_list_to_avoid.Clear()
            occt_list_to_take.Append(occt_shape_iterator_A.Value())
            occt_list_to_avoid.Append(occt_arguments_B)
            occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid)

            occt_shape_iterator_A.Next()

        occt_cells_builder.MakeContainers()

        occt_result_shape = occt_cells_builder.Shape()
        
        if occt_result_shape.IsNull():
            occt_post_processed_shape = occt_result_shape

        else:
            self.post_process_boolean_result(occt_result_shape)

        post_processed_shape: Topology = Topology.by_occt_shape(occt_post_processed_shape, "")

        if post_processed_shape == None:
            return None

        copy_post_processed_shape: Topology = post_processed_shape.deep_copy()

        Topology.transfer_contents(self.get_occt_shape(), copy_post_processed_shape)
        Topology.transfer_contents(tool.get_occt_shape(). copy_post_processed_shape)

        if transfer_dictionary:
            self.boolean_transfer_dictionary(self, tool, copy_post_processed_shape, True)

        return copy_post_processed_shape

#--------------------------------------------------------------------------------------------------
    def intersect(self, other_topology: 'Topology', transfer_dictionary: bool) -> 'Topology':
        
        if other_topology == None:
            return Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())

        # Intersect = Common + Section
		# - Common gives intersection component with at least the same dimension
		# - Section gives the Vertices and Edges
		# NOTE: potential pitfall: 2 Cells intersecting on a Face

        occt_arguments_A = TopTools_ListOfShape()
        occt_arguments_B = TopTools_ListOfShape()
        self.add_boolean_operands(other_topology, occt_arguments_A, occt_arguments_B)

        occt_common = BRepAlgo_Common()
        Topology.regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_common)

        occt_section = BRepAlgo_Section()
        Topology.regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_section)

        # Create topology
        common_topology: Topology = Topology.by_occt_shape(occt_common)
        section_topology: Topology = Topology.by_occt_shape(occt_section)

        # Check isPracticallyEmpty: either nullptr, or (not a Vertex and there is no subtopologies)
        is_common_practically_empty = common_topology is None or (common_topology.num_of_sub_topologies() == 0)
        is_section_practically_empty = section_topology is None or (section_topology.num_of_sub_topologies() == 0)

        # Cases
        merge_topology: Topology = None

        if is_common_practically_empty:

            if is_section_practically_empty:
                return None

            else:
                merge_topology = section_topology

        else:

            if is_section_practically_empty:
                merge_topology = common_topology

            else:
                merge_topology = common_topology.merge(section_topology)

        if merge_topology == None:
            return None

        occt_result_merge_shape = merge_topology.get_occt_shape()
        
        if occt_result_merge_shape.IsNull():
            occt_post_processed_shape = occt_result_merge_shape

        else:
            occt_post_processed_shape = self.post_process_boolean_result(occt_result_merge_shape)

        post_processed_shape = Topology.by_occt_shape(occt_post_processed_shape, "")

        if post_processed_shape == None:
            return None

        copy_post_processed_shape: Topology = post_processed_shape.deep_copy()
        Topology.transfer_contents(self.get_occt_shape(), copy_post_processed_shape)
        Topology.transfer_contents(other_topology.get_occt_shape(), copy_post_processed_shape)

        if transfer_dictionary:
            self.boolean_transfer_dictionary(self, other_topology, copy_post_processed_shape, True)

        return copy_post_processed_shape

#--------------------------------------------------------------------------------------------------
    def merge(self, other_topology: 'Topology', transfer_dictionary: bool) -> 'Topology':
        
        if other_topology == None:
            return Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())

        occt_arguments_A = TopTools_ListOfShape()
        occt_arguments_B = TopTools_ListOfShape()
        self.add_boolean_operands(other_topology, occt_arguments_A, occt_arguments_B)

        occt_cells_builder = BOPAlgo_CellsBuilder()
        Topology.non_regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_cells_builder)

        # 2. Select the parts to be included in the final result.
        occt_list_to_take = TopTools_ListOfShape()
        occt_list_to_avoid = TopTools_ListOfShape()

        occt_shape_iterator_A = TopTools_ListIteratorOfListOfShape(occt_arguments_A)
        occt_shape_iterator_B = TopTools_ListIteratorOfListOfShape(occt_arguments_B)

        while occt_shape_iterator_A.More():

            occt_list_to_take.Clear()
            occt_list_to_avoid.Clear()
            occt_list_to_take.Append(occt_shape_iterator_A.Value())
            occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid)

            occt_shape_iterator_A.Next()

        while occt_shape_iterator_B.More():

            occt_list_to_take.Clear()
            occt_list_to_avoid.Clear()
            occt_list_to_take.Append(occt_shape_iterator_B.Value())
            occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid)

            occt_shape_iterator_B.Next()

        occt_cells_builder.MakeContainers()

        occt_result_shape: TopoDS_Shape = occt_cells_builder.Shape()

        if occt_result_shape.IsNull():
            occt_post_processed_shape = occt_result_shape

        else:
            occt_post_processed_shape = self.post_process_boolean_result(occt_result_shape)

        post_processed_shape = Topology.by_occt_shape(occt_post_processed_shape, "")

        if post_processed_shape == None:
            return None

        copy_post_processed_shape: Topology = post_processed_shape.deep_copy()
        Topology.transfer_contents(self.get_occt_shape(), copy_post_processed_shape)
        Topology.transfer_contents(other_topology.get_occt_shape(), copy_post_processed_shape)

        if transfer_dictionary:
            self.boolean_transfer_dictionary(self, other_topology, copy_post_processed_shape, True)

        return copy_post_processed_shape

#--------------------------------------------------------------------------------------------------
    def self_merge(self) ->'Topology':
        
        # 1
        occt_shapes = TopTools_ListOfShape()
        Topology.sub_topologies(self.get_occt_shape(), occt_shapes)

        # 2
        occt_cells_builder = BOPAlgo_CellsBuilder()
        occt_cells_builder.SetArguments(occt_shapes)

        try:
            occt_cells_builder.Perform()

        except:
            raise RuntimeError("Some error occured.")

        if occt_cells_builder.HasErrors() or occt_cells_builder.HasWarnings():

            error_stream = StringIO()
            occt_cells_builder.DumpErrors(error_stream)

            warning_stream = StringIO()
            occt_cells_builder.DumpWarnings(warning_stream)

            # Exit here and return occtShapes as a cluster.
            occt_compound = TopoDS_Compound()
            occt_builder = BRep_Builder()
            occt_builder.MakeCompound(occt_compound)

            occt_shape_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes)

            while occt_shape_iterator.More():

                occt_builder.Add(occt_compound, occt_shape_iterator.Value())

                occt_shape_iterator.Next()

            return Topology.by_occt_shape(occt_compound, "")

        occt_cells_builder.AddAllToResult()

        # 2b. Get discarded faces from Cells Builder
        occt_discarded_faces = TopTools_ListOfShape()
        occt_compound = TopoDS_Compound()
        occt_builder = BRep_Builder()
        occt_builder.MakeCompound(occt_compound)

        occt_face_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes)

        while occt_face_iterator.More():

            current: TopoDS_Shape = occt_face_iterator.Value()

            if occt_cells_builder.IsDeleted(current):
                occt_builder.Add(occt_compound, current)
                occt_discarded_faces.Append(current)

            occt_face_iterator.Next()

        # 3. Get Face[] from Topology[]
        occt_faces = TopTools_ListOfShape()
        occt_compound_3 = TopoDS_Compound()
        occt_builder_3 = BRep_Builder()
        occt_builder_3.MakeCompound(occt_compound_3)

        occt_explorer = TopExp_Explorer(occt_cells_builder.Shape(), TopAbs_SHAPE)

        while occt_explorer.More():

            occt_current = occt_explorer.Current()

            if not occt_faces.Contains(occt_current):
                occt_faces.Append(occt_current)
                occt_builder_3.Add(occt_compound_3, occt_current)

            occt_explorer.Next()

        # 5. Topology = VolumeMaker(Face[])--> first result
        occt_volume_maker = BOPAlgo_MakerVolume()
        run_parallel: bool = False # parallel or single mode (the default value is FALSE)
        intersect: bool = True # intersect or not the arguments (the default value is TRUE)
        tol: float = 0.0 # fuzzy option (default value is 0)

        occt_volume_maker.SetArguments(occt_faces)
        occt_volume_maker.SetRunParallel(run_parallel)
        occt_volume_maker.SetIntersect(intersect)
        occt_volume_maker.SetFuzzyValue(tol)

        occt_volume_maker.Perform() # perform the operation

        if occt_volume_maker.HasErrors() or occt_volume_maker.HasWarnings(): # check error status
            pass

        else:

            # 6. Get discarded faces from VolumeMaker--> second result
            occt_compound_2 = TopoDS_Compound()
            occt_builder_2 = BRep_Builder()

            occt_builder_2.MakeCompound(occt_compound_2)

            occt_face_iterator = TopTools_ListIteratorOfListOfShape(occt_faces)

            while occt_face_iterator.More():

                current: TopoDS_Shape = occt_face_iterator.Value()

                if occt_volume_maker.IsDeleted(current):
                    occt_discarded_faces.Append(current)
                    occt_builder_2.Add(occt_compound_2, current)

                occt_face_iterator.Next()

        # 7. Get the rest from Topology[] --> third result
        occt_other_shapes = TopTools_ListOfShape() # for step #7

        occt_shape_iterator = TopTools_ListIteratorOfListOfShape(occt_shapes)

        while occt_shape_iterator.More():

            if occt_shape_iterator.Value().ShapeType() != TopAbs_FACE:
                occt_other_shapes.Append(occt_shape_iterator.Value())

            occt_shape_iterator.Next()

        # 8. Merge results #1 #2 #3
        occt_final_arguments = TopTools_ListOfShape()

        if not occt_volume_maker.HasErrors() and not occt_volume_maker.HasWarnings():
            occt_final_arguments.Append(occt_volume_maker.Shape())

        occt_final_arguments.Append(occt_discarded_faces)
        occt_final_arguments.Append(occt_other_shapes)

        if occt_final_arguments.Size() == 1:
            return Topology.by_occt_shape(occt_volume_maker.Shape(), "")

        occt_cells_builder_2 = BOPAlgo_CellsBuilder()
        occt_cells_builder_2.SetArguments(occt_final_arguments)

        try:
            occt_cells_builder_2.Perform()
        
        except:
            raise RuntimeError("Some error occured.")

        if occt_cells_builder_2.HasErrors():
            error_stream = StringIO()
            occt_cells_builder_2.DumpErrors(error_stream)
            raise RuntimeError(error_stream.getvalue())

        occt_cells_builder_2.AddAllToResult()
        occt_cells_builder_2.MakeContainers()

        # 9. If there is still a discarded face, add to merge2Topology as a cluster.
        cluster_candidates = TopTools_ListOfShape()
        merge_2_topologies = TopTools_ListIteratorOfListOfShape(occt_final_arguments)

        while merge_2_topologies.more():

            if occt_cells_builder_2.IsDeleted(merge_2_topologies.Value()) and \
               merge_2_topologies.Value().ShapeType() == TopAbs_FACE: # currently only face

               modified_shapes: TopTools_ListOfShape = occt_cells_builder_2.Modified(merge_2_topologies.Value())
               generated_shapes: TopTools_ListOfShape = occt_cells_builder_2.Generated(merge_2_topologies.Value())
               cluster_candidates.Append(merge_2_topologies.Value())

            merge_2_topologies.Next()

        occt_final_result = TopoDS_Shape()

        if cluster_candidates.Size() > 0:

            Topology.sub_topologies(occt_cells_builder_2.Shape(), cluster_candidates)
            occt_final_compound = TopoDS_Compound()
            occt_final_builder = BRep_Builder()
            occt_final_builder.MakeCompound(occt_final_compound)

            cluster_candidate_iterator = TopTools_ListIteratorOfListOfShape(cluster_candidates)

            while cluster_candidate_iterator.More():

                occt_final_builder.Add(occt_final_compound, cluster_candidate_iterator.Value())

                cluster_candidate_iterator.Next()

            occt_final_result = occt_final_compound

        else:

            occt_current_shape: TopoDS_Shape = occt_cells_builder_2.Shape()

            if occt_current_shape.IsNull():
                occt_post_processed_shape: TopoDS_Shape = occt_current_shape
            else:
                self.post_process_boolean_result(occt_current_shape)

            occt_final_result = occt_post_processed_shape

        # Shape fix
        occt_shape_fix = ShapeFix_Shape(occt_final_result)
        occt_shape_fix.Perform()

        fixed_final_shape: TopoDS_Shape = occt_shape_fix.Shape()

        final_topology: Topology = Topology.by_occt_shape(fixed_final_shape, "")

        # Copy dictionaries
        AttributeManager.get_instance().deep_copy_attributes(self.get_occt_shape(), final_topology.get_occt_shape())

        # Copy contents
        Topology.transfer_contents(self.get_occt_shape(), final_topology)

        return final_topology

#--------------------------------------------------------------------------------------------------
    def slice(self, tool: 'Topology', transfer_dictionary: bool) -> 'Topology':
        
        if tool == None:
            return Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())

        occt_arguments_A = TopTools_ListOfShape()
        occt_arguments_B = TopTools_ListOfShape()
        self.add_boolean_operands(tool, occt_arguments_A, occt_arguments_B)

        occt_cells_builder = BOPAlgo_CellsBuilder()
        Topology.non_regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_cells_builder)

        # 2. Select the parts to be included in the final result.
        occt_list_to_take = TopTools_ListOfShape()
        occt_list_to_avoid = TopTools_ListOfShape()

        occt_shape_iterator_A = TopTools_ListIteratorOfListOfShape(occt_arguments_A)

        while occt_shape_iterator_A.More():

            occt_list_to_take.Clear()
            occt_list_to_avoid.Clear()
            occt_list_to_take.Append(occt_shape_iterator_A.Value())
            occt_cells_builder.AddAllToResult(occt_list_to_take, occt_list_to_avoid)

            occt_shape_iterator_A.Next()

        occt_cells_builder.MakeContainers()

        occt_result_shape: TopoDS_Shape = occt_cells_builder.Shape()

        if occt_result_shape.IsNull():
            occt_post_processed_shape = occt_result_shape

        else:
            occt_post_processed_shape = self.post_process_boolean_result(occt_result_shape)

        post_processed_shape = Topology.by_occt_shape(occt_post_processed_shape, "")

        if post_processed_shape == None:
            return None

        copy_post_processed_shape: Topology = post_processed_shape.deep_copy()
        Topology.transfer_contents(self.get_occt_shape(), copy_post_processed_shape)

        if transfer_dictionary:
            self.boolean_transfer_dictionary(self, tool, copy_post_processed_shape, True)

        return copy_post_processed_shape

#--------------------------------------------------------------------------------------------------
    def union(self, other_topology: 'Topology', transfer_dictionary: bool) -> 'Topology':
        
        if other_topology == None:
            return Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())

        occt_arguments_A = TopTools_ListOfShape()
        occt_arguments_B = TopTools_ListOfShape()
        self.add_boolean_operands(other_topology, occt_arguments_A, occt_arguments_B)

        occt_fuse = BRepAlgoAPI_Fuse()
        Topology.regular_boolean_operation(occt_arguments_A, occt_arguments_B, occt_fuse)

        occt_result_shape: TopoDS_Shape = occt_fuse.Shape()

        if occt_result_shape.IsNull():
            occt_post_processed_shape = occt_result_shape

        else:
            occt_post_processed_shape = self.post_process_boolean_result(occt_result_shape)

        post_processed_shape = Topology.by_occt_shape(occt_post_processed_shape, "")

        if post_processed_shape == None:
            return None

        copy_post_processed_shape: Topology = post_processed_shape.deep_copy()
        Topology.transfer_contents(self.get_occt_shape(), copy_post_processed_shape)
        Topology.transfer_contents(other_topology.get_occt_shape(), copy_post_processed_shape)

        if transfer_dictionary:
            self.boolean_transfer_dictionary(self, other_topology, copy_post_processed_shape, True)

        return copy_post_processed_shape

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_cell(self, occt_shape: TopoDS_Shape) -> TopoDS_Shape:
        
        occt_cells = TopTools_ListOfShape()
        occt_new_shape = TopoDS_Shape(occt_shape)

        occt_explorer = TopExp_Explorer(occt_shape, TopAbs_SOLID)

        while occt_explorer.More():

            occt_current_solid: TopoDS_Solid = topods.Solid(occt_explorer.Current())

            # create tools for fixing a face 
            occt_shape_fix_solid: ShapeFix_Solid = ShapeFix_Solid()

            # create tool for rebuilding a shape and initialize it by shape
            occt_context: ShapeBuild_ReShape = ShapeBuild_ReShape()
            occt_context.Apply(occt_new_shape)

            # set a tool for rebuilding a shape in the tool for fixing 
            occt_shape_fix_solid.SetContext(occt_context)

            # initialize the fixing tool by one face 
            occt_shape_fix_solid.Init(occt_current_solid)

            # fix the set face
            occt_shape_fix_solid.Perform()

            # get the result
            occt_new_shape = occt_context.Apply(occt_new_shape)

            occt_explorer.Next()

        return occt_new_shape

#--------------------------------------------------------------------------------------------------
    def add_union_internal_structure(self, occt_shape: TopoDS_Shape, union_arguments: TopTools_ListOfShape) -> None:
        
        occt_shape_type: TopAbs_ShapeEnum = occt_shape.ShapeType()
        topology = Topology.by_occt_shape(occt_shape)
        faces: List[Face] = []

        # Cell complex -> faces not part of the envelope
        # Cell -> inner shells
        # Shell --> inner wires of the faces
        # Face --> inner wires
        # Wire --> n/a
        # Edge --> n/a
        # Vertex --> n/a

        if occt_shape_type == TopAbs_COMPOUND:

            cluster: Cluster = TopologicalQuery.downcast(topology)
            immediate_members: List['Topology'] = []
            cluster.sub_topologies(immediate_members)

            for immediate_member in immediate_members:
                self.add_union_internal_structure(immediate_member.get_occt_shape(), union_arguments)

        elif occt_shape_type == TopAbs_COMPSOLID:
            cellComplex: CellComplex = TopologicalQuery.downcast(topology)
            cellComplex.internal_boundaries(faces)

            for internal_face in faces:
                union_arguments.Append(internal_face.get_occt_shape())

        elif occt_shape_type == TopAbs_SOLID:
            cell: Cell = TopologicalQuery.downcast(topology)
            shells: List[Shell] = []
            cell.internal_boundaries(shells)

            for internal_shell in shells:
                union_arguments.Append(internal_shell.get_occt_shape())

        elif occt_shape_type == TopAbs_SHELL:

            occt_shell_explorer = TopExp_Explorer(occt_shape, TopAbs_FACE)
            occt_face_explorer = TopExp_Explorer(occt_shape, TopAbs_WIRE)

            while occt_shell_explorer.More():

                # ???? occt_current_face ????
                occt_current_face = occt_shell_explorer.Current()

                # ???? occt_current_wire ????
                occt_outer_wire = BRepTools.OuterWire(topods.face(occt_current_face))

                while occt_face_explorer.More():

                    occt_current_face = occt_face_explorer.Current()

                    if not union_arguments.Contains(occt_current_face) and not occt_current_face.IsSame(occt_outer_wire):
                        union_arguments.Append(occt_current_face)

                    occt_face_explorer.Next()

                occt_shell_explorer.Next()

        elif occt_shape_type == TopAbs_FACE:

            occt_outer_wire = BRepTools.OuterWire(topods.face(occt_shape))

            occt_explorer = TopExp_Explorer(occt_shape, TopAbs_WIRE)

            while occt_explorer.More():

                occt_current = occt_explorer.Current()

                if not union_arguments.Contains(occt_current) and not occt_current.IsSame(occt_outer_wire):
                    union_arguments.Append(occt_current_face)

                occt_explorer.Next()

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_shell(self, occt_shape: TopoDS_Shape) -> TopoDS_Shape:
        
        occt_cells = TopTools_ListOfShape()
        occt_new_shape = TopoDS_Shape(occt_shape)

        occt_explorer = TopExp_Explorer(occt_shape, TopAbs_SHELL)

        while occt_explorer.More():

            occt_current_shell = topods.shell(occt_explorer.Current())

            # create tools for fixing a face
            occt_shape_fix_shell = ShapeFix_Shell()

            # create tool for rebuilding a shape and initialize it by shape
            occt_context = ShapeBuild_ReShape()
            occt_context.Apply(occt_new_shape)

            # set a tool for rebuilding a shape in the tool for fixing 
            occt_shape_fix_shell.SetContext(occt_context)

            # initialize the fixing tool by one face
            occt_shape_fix_shell.Init(occt_current_shell)

            # fix the set face
            occt_shape_fix_shell.Perform()

            # get the result 
            occt_new_shape = occt_context.Apply(occt_new_shape)

            occt_explorer.Next()

        return occt_new_shape

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_face(self, occt_shape: TopoDS_Shape, map_face_to_fixed_face: TopTools_DataMapOfShapeShape) -> TopoDS_Shape:
        
        occt_cells = TopTools_MapOfShape()
        occt_new_shape = TopoDS_Shape(occt_shape)

        occt_explorer = TopExp_Explorer(occt_shape, TopAbs_FACE)

        while occt_explorer.More():

            occt_current_face = topods.face(occt_explorer.Current())

            # create tools for fixing a face 
            occt_shape_fix_face = ShapeFix_Face()

            # create tool for rebuilding a shape and initialize it by shape
            occt_context = ShapeBuild_ReShape()
            occt_context.Apply(occt_new_shape)

            # set a tool for rebuilding a shape in the tool for fixing
            occt_shape_fix_face.SetContext(occt_context)

            # initialize the fixing tool by one face
            occt_shape_fix_face.Init(occt_current_face)

            # fix the set face
            occt_shape_fix_face.Perform()

            # Map occtCurrentFace and occtShapeFixFace.Shape()
            map_face_to_fixed_face.Bind(occt_current_face, occt_shape_fix_face.Face())

            # get the result
            occt_new_shape = occt_context.Apply(occt_new_shape)

            occt_explorer.Next()

        return occt_new_shape

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_face(self, occt_shape: TopoDS_Shape) -> TopoDS_Shape:
        
        map_face_to_fixed_face = TopTools_DataMapOfShapeShape()
        return self.fix_boolean_operand_face(occt_shape, map_face_to_fixed_face)

#--------------------------------------------------------------------------------------------------
    def get_deleted_boolean_sub_topologies(self, occt_shape: TopoDS_Shape, occt_cells_builder: BOPAlgo_CellsBuilder, occt_deleted_shapes: TopTools_ListOfShape) -> None:
        
        sub_shape_types = [TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE] 

        for i in range(3):

            if occt_shape.ShapeType() == sub_shape_types[i]:

                if occt_cells_builder.IsDeleted(occt_shape):

                    occt_deleted_shapes.Append(occt_shape)

                occt_modified_shapes: TopTools_ListOfShape = occt_cells_builder.Modified(occt_shape)

                if not occt_modified_shapes.IsEmpty():
                    occt_deleted_shapes.Append(occt_shape)

            occt_sub_shapes = TopTools_MapOfShape()

            Topology.static_downward_navigation(occt_shape, sub_shape_types[i], occt_sub_shapes)

            occt_sub_shape_iterator = TopTools_MapIteratorOfMapOfShape(occt_sub_shapes)

            while occt_sub_shape_iterator.More():

                if occt_cells_builder.IsDeleted(occt_sub_shape_iterator.Value()):
                    occt_deleted_shapes.Append(occt_sub_shape_iterator.Value())

                occt_modified_shapes: TopTools_ListOfShape = occt_cells_builder.Modified(occt_sub_shape_iterator.Value())

                if not occt_modified_shapes.IsEmpty():
                    occt_deleted_shapes.Append(occt_sub_shape_iterator.Value())

                occt_sub_shape_iterator.Next()

#--------------------------------------------------------------------------------------------------
    def get_deleted_boolean_sub_topologies(self, occt_shape: TopoDS_Shape, occt_boolean_operation: BRepAlgoAPI_BooleanOperation, occt_deleted_shapes: TopTools_ListOfShape) -> None:
        
        sub_shape_types = [TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE]

        for i in range(3):

            if occt_shape.ShapeType() == sub_shape_types[i]:

                if occt_boolean_operation.IsDeleted(occt_shape):

                    occt_deleted_shapes.Append(occt_shape)

            occt_sub_shapes = TopTools_MapOfShape()

            Topology.static_downward_navigation(occt_shape, sub_shape_types[i], occt_sub_shapes)

            occt_sub_shape_iterator = TopTools_MapIteratorOfMapOfShape(occt_sub_shapes)

            while occt_sub_shapes.More():

                if occt_boolean_operation.IsDeleted(occt_sub_shape_iterator.Value()):
                    occt_deleted_shapes.Append(occt_sub_shape_iterator.Value())

                occt_sub_shapes.Next()

#--------------------------------------------------------------------------------------------------
    def track_context_ancestor(self) -> 'Topology':
        
        contexts: List[Context] = []
        self.contexts(contexts)

        if len(contexts) == 1:

            # Go farther
            return contexts[0].topology.track_context_ancestor()

        # if empty or > 2
        return self

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def intersect_edge_shell(edge: Edge, shell: Shell) -> 'Topology':
        
        faces: List[Face] = []
        faces = shell.faces()

        intersection_vertices: List[Topology] = []

        for face in faces:

            merge_topology: Topology = edge.merge(face)

            cluster: Topology = Topology.intersect_edge_face(merge_topology, edge, face)

            if cluster == None:
                continue

            cluster_vertices: List[Vertex] = []
            vertices = cluster.vertices()

            intersection_vertices.extend(cluster_vertices)

        cluster: Cluster = Cluster.by_topologies(intersection_vertices)
        merged_cluster: Topology = cluster.self_merge()

        return merged_cluster

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def is_in_list(new_vertex: Vertex, old_vertices: List[Vertex], tolerance: float) -> bool:
        
        for old_vertex in old_vertices:

            occt_edge_distance = BRepExtrema_DistShapeShape(old_vertex.get_occt_shape(), new_vertex.get_occt_shape(), Extrema_ExtFlag_MINMAX)
            distance = occt_edge_distance.Value()

            if distance < tolerance:
                return True

        return False

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def intersect_edge_face(merge_topology: 'Topology', edge: Edge, face: Face) -> 'Topology':
        
        tolerance = 0.0001
        edge_vertices: List[Vertex] = []
        edge_vertices = edge.vetices()

        face_vertices: List[Vertex] = []
        face_vertices = face.vertices()

        merge_vertices: List[Vertex] = []
        merge_vertices = merge_topology.vertices()

        intersection_vertices = List[Topology] = []

        for merge_vertex in merge_vertices:

            is_in_edge_vertices: bool = Topology.is_in_list(merge_vertex, edge_vertices, tolerance)
            is_in_face_vertices: bool = Topology.is_in_list(merge_vertex, face_vertices, tolerance)

            if (not is_in_edge_vertices and not is_in_face_vertices) or \
               (is_in_edge_vertices and is_in_face_vertices):

               intersection_vertices.append(merge_vertex)

            else:
                occt_edge_distance = BRepExtrema_DistShapeShape(merge_vertex.get_occt_shape(), edge.get_occt_edge(), Extrema_ExtFlag_MINMAX)
                edge_distance = occt_edge_distance.Value()
                
                occt_face_distance = BRepExtrema_DistShapeShape(merge_vertex.get_occt_shape(), face.get_occt_face(), Extrema_ExtFlag_MINMAX)
                face_distance = occt_face_distance.Value()

                if edge_distance < tolerance and face_distance < tolerance:
                    intersection_vertices.append(merge_vertex)

        cluster: Cluster = Cluster.by_topologies(intersection_vertices)

        return cluster

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def intersect_face_face():
        pass

#--------------------------------------------------------------------------------------------------
    def add_boolean_operands():
        pass

#--------------------------------------------------------------------------------------------------
    def add_boolean_operands():
        pass

#--------------------------------------------------------------------------------------------------
    def xor(self):
        pass

#--------------------------------------------------------------------------------------------------
    def divide(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def sub_topologies():
        pass

#--------------------------------------------------------------------------------------------------
    def sub_topologies(sub_topologies: List['Topology']) -> None:
        pass

#--------------------------------------------------------------------------------------------------
    def num_of_sub_topologies(self):
        pass

#--------------------------------------------------------------------------------------------------
    def shells(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    def edges(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    def faces(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    def vertices(self, host_topology: 'Topology') -> List['Topology']:
        """
        Gets all vertices associated with the host topology.
        """
        if host_topology == None:
            host_topology = self
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    def wires(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    def cells(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    def cell_complexes(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    def is_container_type(self, occt_shape: TopoDS_Shape) -> bool:
        """
        Virtual method to be overridden by all descendant classes.
        TODO: What is container type?
        """
        shape_type = occt_shape.ShapeType()
        if shape_type == TopAbs_WIRE \
        or shape_type == TopAbs_SHELL \
        or shape_type == TopAbs_COMPSOLID \
        or shape_type == TopAbs_COMPOUND:
            return True
        else:
            return False

#--------------------------------------------------------------------------------------------------
    def global_cluster_sub_topologies(self):
        pass

#--------------------------------------------------------------------------------------------------
    def upward_navigation(self, 
                          host_topology: TopoDS_Shape,
                          topology_type: TopologyTypes) -> 'List[Topology]':
        """
        Returns:
            Looks up what higher order shapes contain this topology.
        """

        #-------------------------------------------------------------
        # The argument "topology_type" is unused. Is it really needed?
        #------------------------------------------------------------- 

        if host_topology.IsNull():
            raise RuntimeError("Host Topology cannot be None when searching for ancestors.")
        
        ret_ancestors: List[Topology] = []
        occt_shape_type = self.get_shape_type()
        occt_ancestor_map: TopTools_MapOfShape = None
        occt_shape_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndUniqueAncestors(
            host_topology,
            self.get_occt_shape().ShapeType(),
            TopAbs_ShapeEnum(occt_shape_type.value),
            occt_shape_map)
        
        occt_ancestors = TopTools_ListOfShape()
        is_in_shape = occt_shape_map.FindFromKey(self.get_occt_shape(), occt_ancestors)
        if not is_in_shape: return []

        shape_iterator = TopTools_ListIteratorOfListOfShape(occt_ancestors)
        while shape_iterator.More():
            occt_ancestor = shape_iterator.Value()
            is_ancestor_added = occt_ancestor_map.Contains(occt_ancestor)

            if occt_ancestor.ShapeType() == occt_shape_type and not is_ancestor_added:
                occt_ancestor_map.Add(occt_ancestor)

                p_topology = Topology.by_occt_shape(occt_ancestor, "")
                ret_ancestors.append(p_topology)

            shape_iterator.Next()

        return ret_ancestors

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def static_downward_navigation(occt_shape: TopoDS_Shape, shape_enum: TopAbs_ShapeEnum) -> List['Topology']:
        """
        Navigates downward through the sub-shapes of a given shape and retrieves
        the ones of a specified type.

        Parameters:
            occt_shape (TopoDS_Shape): The parent shape.
            shape_enum (TopAbs_ShapeEnum): The type of sub-shapes to be retrieved.

        Returns:
            TopTools_MapOfShape: Map containing the retrieved sub-shapes.
        """
        ret_members: List['Topology'] = []

        occt_members = TopTools_MapOfShape()
        occt_explorer = TopExp_Explorer(occt_shape, shape_enum)
        while occt_explorer.More():
            occt_current_shape = occt_explorer.Current()
            if not occt_members.Contains(occt_current_shape):
                occt_members.Add(occt_current_shape)
                child_topology = Topology.by_occt_shape(occt_current_shape, "")
                ret_members.append(child_topology)
            occt_explorer.Next()

        return ret_members

#--------------------------------------------------------------------------------------------------
    # EZT MIERT??? - A C++ KODBAN NINCS BENNE!!!!
    def downward_navigation(self, topabs_shape_enum: TopAbs_ShapeEnum=None) -> List['Topology']:
        """
        Appends collection of topology members that belong to current shape.
        """
        ret_members: List['Topology'] = []
        required_topabs_shapeenum: TopAbs_ShapeEnum = None
        if topabs_shape_enum == None:
            topology_type: TopologyTypes = self.get_shape_type()
            required_topabs_shapeenum = TopAbs_ShapeEnum(topology_type.value)
        else:
            required_topabs_shapeenum = topabs_shape_enum

        occt_shapes = TopTools_MapOfShape()
        occt_explorer = TopExp_Explorer(self.get_occt_shape(), required_topabs_shapeenum)

        while occt_explorer.More():
            occt_current_shape = occt_explorer.Current()
            if not occt_shapes.Contains(occt_current_shape):
                occt_shapes.Add(occt_current_shape)
                child_topology = Topology.by_occt_shape(occt_current_shape, "")
                ret_members.append(child_topology)
            occt_explorer.Next()

        return ret_members

#--------------------------------------------------------------------------------------------------
    def deep_copy_explode_shape(self):
        pass

#--------------------------------------------------------------------------------------------------
    def deep_copy_impl(self):
        pass

#--------------------------------------------------------------------------------------------------
    def deep_copy(self):
        pass

#--------------------------------------------------------------------------------------------------
    def shallow_copy(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def copy_occt():
        pass

#--------------------------------------------------------------------------------------------------
    def replace_sub_entity(self):
        pass

#--------------------------------------------------------------------------------------------------
    def is_same(self, test_topology: 'Topology') -> bool:
        """
        Returns:
            bool: True if the test_topology is identical to this topology.
        """
        is_same = self.get_occt_shape().IsSame(test_topology.get_occt_shape())
        return is_same
    
#--------------------------------------------------------------------------------------------------
    def is_reversed(self) -> bool:
        """
        Checks the shape orientation.
        """
        occt_orientation = self.get_occt_shape().Orientation()
        return occt_orientation == TopAbs_REVERSED

#--------------------------------------------------------------------------------------------------
    def deep_copy_attributes_from(self):
        pass

#--------------------------------------------------------------------------------------------------
    def members(self):
        pass

#--------------------------------------------------------------------------------------------------
    def members(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def members():
        pass

#--------------------------------------------------------------------------------------------------
    def get_instance_guid(self) -> str:
        """
        Instance-bound method to call static GUID getter.
        """
        return Topology.s_get_instance_guid(self.get_occt_shape())

#--------------------------------------------------------------------------------------------------
    def set_instance_guid(self, shape: TopoDS_Shape, guid: str) -> None:
        """
        For a new shape saves the shape and its guid.
        """
        instance_guid_manage = InstanceGUIDManager.get_instance_manager()
        instance_guid_manage.add(shape, guid)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def s_get_instance_guid(search_shape: TopoDS_Shape) -> str:
        """
        Looks up if the shape already exists, if so, its GUID will be returned.
        """
        instance_guid_manager = InstanceGUIDManager.get_instance_manager()
        found_guid: str = instance_guid_manager.find(search_shape)
        return found_guid

#--------------------------------------------------------------------------------------------------
    def set_dictionary(self):
        pass

#--------------------------------------------------------------------------------------------------
    def get_dictionary(self):
        pass

#--------------------------------------------------------------------------------------------------

# Additional methods, that are not included in Topology.cpp:

#--------------------------------------------------------------------------------------------------

    def get_occt_shape(self) -> TopoDS_Shape:
        """
        Returns:
            TopoDS_Shape: Underlying OCC shape.
        """
        return self.base_shape
    
    def navigate(self, host_topology: 'Topology') -> 'List[Topology]':
        """
        TODO: Review logic!
        Either navigates upward or downward based on the current type.   
        """

        is_current_type_higher_order = host_topology.get_shape_type() > self.get_shape_type()
        if is_current_type_higher_order:
            return self.downward_navigation()
        elif not is_current_type_higher_order:
            if not host_topology.is_null_shape():
                return self.upward_navigation()
            else:
                raise RuntimeError("Host Topology cannot be NULL when searching for ancestors.")
        else:
            return [Topology.by_occt_shape(self.get_occt_shape(), self.get_instance_guid())]
        
    def deep_copy_shape(self) -> 'TopoDS_Shape':
        """
        TODO!!
        """
        copy_shape = BRepBuilderAPI_Copy()
        copy_shape.Perform(self.get_occt_shape())
        return copy_shape.Shape()
    
    def geometry(self) -> Geom_Geometry:
        """
        Pure virtual method for geometry getting.
        """
        raise NotImplementedError("Topology is missing geometry getter!")
    
    def is_manifold(self) -> bool:
        """
        Pure virtual method for manifold querying.
        """
        raise NotImplementedError("Topology is missing manifold query method!")

    def clusters(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

    def get_contents(self, contents: List['Topology']) -> None:

        # self.get_occt_shape returns the base_shape (VERTEX, FACE, ...)
        Topology.contents(self.get_occt_shape, contents)

    @staticmethod
    def contents(occt_shape: TopoDS_Shape, contents: List['Topology']):
        content_manager = ContentManager.get_instance()

        # Finds Topology parent classes of entities of selected type. The instances will be added to the list "contents"
        content_manager.find(occt_shape, contents)


    def is_null_shape(self) -> bool:
        """
        Checks if the underlying TopoDS_Shape is not null.

        Returns:
            bool: True - if shape is null.
        """
        return self.base_shape.IsNull()
        
    def get_shape_type(self) -> TopologyTypes:
        """Public virtual method that returns the type of the shape.

        Returns:
            TopologyTypes: Type of shape
        """
        return self.shape_type
    
    def set_shape_type(self, shape_type: TopologyTypes) -> None:
        """Method to override topology type.
        This method is redundant for clarity only,
        so developers know to use this method to set the shape types.

        Args:
            shape_type (TopologyTypes): Shape type.
        """
        if shape_type == TopologyTypes.UNDEFINED:
            raise Exception('Cannot set undefined for shape type!')

        self.shape_type = shape_type

    def get_class_guid(self) -> str:
        """
        Looks up TopologyConstants module to determine constant guid related to shape.

        Returns:
            str: Found GUID to shape.
        """
        return TopologyTypes.get_guid_for_type(self.shape_type)
    
    def get_dimensionality(self) -> int:
        """
        Looks up TopologyConstants module to determine dimensionality related to shape.

        Returns:
            int: Found dimensionality to shape.
        """
        return TopologyTypes.get_dimensionality_for_type(self.shape_type)
    
    def get_type_as_string(self) -> str:
        """
        Pure virtual method that returns the name of the class.
        """
        raise NotImplementedError(f"Missing implementation for type name getter!")

    def __del__(self):
        """Deconstructor decrements counter.
        """
        Topology.topologic_entity_count -= 1




