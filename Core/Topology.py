
# OCC
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeFix import ShapeFix_Shape

from Core.TopologyConstants import TopologyTypes

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