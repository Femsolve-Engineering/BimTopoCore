
# OCC
from typing import List
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopTools import TopTools_MapOfShape
from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShape
from OCC.Core.TopoDS import TopoDS_Shape

from Core.TopologyConstants import TopologyTypes
from Core.InstanceGUIDManager import InstanceGUIDManager
from Core.Factories.TopologyFactory import TopologyFactory
from Core.Factories.TopologyFactoryManager import TopologyFactoryManager

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

    def register_factory(self, guid: str, topology_factory: TopologyFactory) -> None:
        """
        Registers topological factory if it does not exist yet.
        """
        TopologyFactoryManager.get_instance().add(guid, topology_factory)

    def is_same(self, test_topology: 'Topology') -> bool:
        """
        Returns:
            bool: True if the test_topology is identical to this topology.
        """
        is_same = self.get_occt_shape().IsSame(test_topology.get_occt_shape())
        return is_same
    
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
        
    def deep_copy(self) -> 'Topology':
        """
        TODO!!
        """
        return None
    
    def upward_navigation(self, host_topology: TopoDS_Shape, shape_type: int) -> 'List[Topology]':
        """
        TODO
        Returns all upward ancestors.
        """
        # ToDo!!!
        return self.upward_navigation(host_topology)

        
    def upward_navigation(self, host_topology: TopoDS_Shape) -> 'List[Topology]':
        """
        Returns:
            Looks up what higher order shapes contain this topology.
        """
        if host_topology.IsNull():
            raise RuntimeError("Host Topology cannot be None when searching for ancestors.")
        
        ret_ancestors: List[Topology] = []
        occt_shape_type = self.get_shape_type()
        occt_ancestor_map: TopTools_MapOfShape = None
        occt_shape_map = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndUniqueAncestors(
            host_topology,
            self.get_occt_shape().ShapeType(),
            occt_shape_type,
            occt_shape_map)
        
        occt_ancestors: TopTools_ListOfShape = None
        is_in_shape = occt_shape_map.FindFromKey(self.get_occt_shape(), occt_ancestors)
        if not is_in_shape: return []

        shape_iterator = TopTools_ListIteratorOfListOfShape(occt_ancestors)
        current_ancestor_iter = shape_iterator.Begin()
        while shape_iterator.More():
            occt_ancestor = current_ancestor_iter.Value()
            is_ancestor_added = occt_ancestor_map.Contains(occt_ancestor)

            if occt_ancestor.ShapeType() == occt_shape_type and not is_ancestor_added:
                occt_ancestor_map.Add(occt_ancestor)

                p_topology = Topology.by_occt_shape(occt_ancestor, "")
                ret_ancestors.append(p_topology)

            shape_iterator.Next()

        return ret_ancestors

    @staticmethod
    def downward_navigation(occt_shape: TopoDS_Shape, shape_enum: TopAbs_ShapeEnum) -> TopTools_MapOfShape:
        """
        Navigates downward through the sub-shapes of a given shape and retrieves
        the ones of a specified type.

        Parameters:
            occt_shape (TopoDS_Shape): The parent shape.
            shape_enum (TopAbs_ShapeEnum): The type of sub-shapes to be retrieved.

        Returns:
            TopTools_MapOfShape: Map containing the retrieved sub-shapes.
        """
        occt_members = TopTools_MapOfShape()

        occt_explorer = TopExp_Explorer(occt_shape, shape_enum)
        while occt_explorer.More():
            occt_current = occt_explorer.Current()
            if not occt_members.Contains(occt_current):
                occt_members.Add(occt_current)
            occt_explorer.Next()

        return occt_members

    def downward_navigation(self) -> List['Topology']:
        """
        Appends collection of topology members that belong to current shape.
        """
        ret_members: List['Topology'] = []
        occt_shape_enum: TopAbs_ShapeEnum = self.get_shape_type()
        occt_shapes: TopTools_MapOfShape = []
        occt_explorer = TopExp_Explorer(self.get_occt_shape(), occt_shape_enum)

        while occt_explorer.More():
            occt_current_shape = occt_explorer.Current()
            if not occt_shapes.Contains(occt_current_shape):
                occt_shapes.Add(occt_current_shape)
                child_topology = Topology.by_occt_shape(occt_current_shape, "")
                ret_members.append(child_topology)

        return ret_members

    def shells(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)
    
    def edges(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

    def faces(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)
    
    def vertices(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)
    
    def wires(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)
    
    def cells(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)
    
    def cell_complexes(self, host_topology: 'Topology') -> List['Topology']:
        """
        TODO - M3
        """
        return self.navigate(host_topology)

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
        
    @staticmethod
    def by_occt_shape(occt_shape: TopoDS_Shape, instance_guid: str) -> 'Topology':
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
            topology_factory_manager.find(instance_guid, p_topology_factory)

        assert p_topology_factory is not None
        p_topology = p_topology_factory.create(occt_shape)

        return p_topology
    
    def set_instance_guid(self, shape: TopoDS_Shape, guid: str) -> None:
        """
        For a new shape saves the shape and its guid.
        """
        instance_guid_manage = InstanceGUIDManager.get_instance_manager()
        instance_guid_manage.add(shape, guid)
    
    def get_instance_guid(self) -> str:
        """
        Instance-bound method to call static GUID getter.
        """
        return Topology.s_get_instance_guid(self.get_occt_shape())

    @staticmethod
    def s_get_instance_guid(search_shape: TopoDS_Shape) -> str:
        """
        Looks up if the shape already exists, if so, its GUID will be returned.
        """
        instance_guid_manager = InstanceGUIDManager.get_instance_manager()
        found_guid: str = instance_guid_manager.find(search_shape)
        return found_guid

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

    def __del__(self):
        """Deconstructor decrements counter.
        """
        Topology.topologic_entity_count -= 1