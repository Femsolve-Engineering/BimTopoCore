
from graphlib import TopologicalSorter
import sys
from typing import List
from typing import Dict
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
    def set_dictionaries(self):
        pass

#--------------------------------------------------------------------------------------------------
    def set_dictionaries(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def occt_sew_faces(faces: List['Topology'], tolerance: float) -> List['Topology']:
        
        # To be implemented...
        occt_shapes = [] # dummy list
        return occt_shapes

#--------------------------------------------------------------------------------------------------
    def boolean_transfer_dictionary(self):
        pass

#--------------------------------------------------------------------------------------------------
    def difference(self):
        pass

#--------------------------------------------------------------------------------------------------
    def contents_(self, contents: List['Topology']) -> None:
        Topology.contents(self.get_occt_shape(), contents)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def contents(occt_shape: TopoDS_Shape, contents: List['Topology']) -> None:
        instance = ContentManager.get_instance()
        instance.find(occt_shape, contents)

#--------------------------------------------------------------------------------------------------
    def apertures(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def apertures():
        pass

#--------------------------------------------------------------------------------------------------
    def sub_contents(self) -> List['Topology']:
        """
        Returns:
            All topologies that are stored under this topology.
        """
        return self.static_sub_contents(self.get_occt_shape())

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def static_sub_contents(occt_shape: TopoDS_Shape) -> List['Topology']:
        """
        Finds all the topologies that are of lower type.
        """
        ret_members: List['Topology'] = []

        # ToDo: ContentManager
        # Topology.contents(occt_shape, sub_contents)

        occt_type = occt_shape.ShapeType()
        occt_type_int = int(occt_type) + 1  # +1 for the next lower type

        for occt_type_int_iteration in range(occt_type_int, int(TopAbs_SHAPE)):
            occt_type_iteration = TopAbs_ShapeEnum(occt_type_int_iteration)
            occt_members = TopTools_MapOfShape()
            ret_members.extend(
                Topology.static_downward_navigation(occt_shape, occt_type_iteration))

            # ToDo: ContentManager
            # for occt_member in occt_members:
            #     pass
                # ContentManager.get_instance().find(occt_member, sub_contents)

        return ret_members
#--------------------------------------------------------------------------------------------------
    def contexts(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def contexts():
        pass

#--------------------------------------------------------------------------------------------------
    def export_to_brep(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_import_brep():
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_string():
        pass

#--------------------------------------------------------------------------------------------------
    def string(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def filter():
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def analyze():
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def simplify():
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def boolean_sub_topology_containment():
        pass

#--------------------------------------------------------------------------------------------------
    def analyze(self):
        pass

#--------------------------------------------------------------------------------------------------
    def non_regular_boolean_operation(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def non_regular_boolean_operation():
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def transfer_contents():
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def transfer_contents():
        pass 

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def regular_boolean_operation():
        pass

#--------------------------------------------------------------------------------------------------
    def post_process_boolean_operation(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def transfer_make_shape_contents():
        pass

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
            # ToDo: Implement this when working on ContentManager
            # contents = original_shape.contents()

            generated_shape_iterator = TopTools_ListIteratorOfListOfShape(occt_generated_shapes)

            while generated_shape_iterator.More():
                occt_generated_shape = generated_shape_iterator.Value()
                generated_shape = Topology.by_occt_shape(occt_generated_shape, "")

                # ToDo: Implement this when working on ContentManager
                # for content in contents:
                #     generated_shape.add_content(content)

                generated_shape_iterator.Next()

            shape_iterator.Next()

#--------------------------------------------------------------------------------------------------
    def impose(self):
        pass

#--------------------------------------------------------------------------------------------------
    def imprint(self):
        pass

#--------------------------------------------------------------------------------------------------
    def intersect(self):
        pass

#--------------------------------------------------------------------------------------------------
    def merge(self):
        pass

#--------------------------------------------------------------------------------------------------
    def self_merge(self):
        pass

#--------------------------------------------------------------------------------------------------
    def slice(self):
        pass

#--------------------------------------------------------------------------------------------------
    def union(self):
        pass

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_cell(self):
        pass

#--------------------------------------------------------------------------------------------------
    def add_union_internal_structure(self):
        pass

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_shell(self):
        pass

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_face(self):
        pass

#--------------------------------------------------------------------------------------------------
    def fix_boolean_operand_face(self):
        pass

#--------------------------------------------------------------------------------------------------
    def get_deleted_boolean_sub_topologies(self):
        pass

#--------------------------------------------------------------------------------------------------
    def get_deleted_boolean_sub_topologies(self):
        pass

#--------------------------------------------------------------------------------------------------
    def track_context_ancestor(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def intersect_shell():
        pass

#--------------------------------------------------------------------------------------------------
    def is_in_list(self):
        pass

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def intersect_edge_face():
        pass

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

# Additional methods, that are not inluded in Topology.cpp:

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




