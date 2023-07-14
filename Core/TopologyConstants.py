
import enum

class TopologyTypes(enum.Enum):
    """
    Static class that stores constant mappings of topology shapes.
    For example:
        Topology shape -> GUID identifier
        Topology shape -> OCCT type
    """

    UNDEFINED = 0
    VERTEX = 1
    EDGE = 2
    WIRE = 3
    FACE = 4

    TOPOLOGY_TO_GUID = {
        VERTEX : "9FE435F9-7AC9-4D94-852A-BB2FFBD3D720", # Original: c4a9b420-edaf-4f8f-96eb-c87fbcc92f2b
        EDGE : "AF050AFB-50BE-499C-8F26-DABB60105302", # Original: 1fc6e6e1-9a09-4c0a-985d-758138c49e35
        WIRE : "58486ECA-5CF4-4CC5-8C2F-3DA248C529FF", # Original: b99ccd99-6756-401d-ab6c-11162de541a3
        FACE : "7B80A066-5A0C-423E-ABAB-82343DBBD3E6" # Original: 3b0a6afe-af86-4d96-a30d-d235e9c98475
    }

    TOPOLOGY_TO_DIMENSIONALITY = {
        VERTEX : 0,
        EDGE : 1,
        WIRE : 1,
        FACE : 2
    }

    @staticmethod
    def get_guid_for_type(shape_type: "TopologyTypes") -> str:
        """
        Upon construction of each topological object we assign shape specific guids.
        Always use this method to lookup shape-specific guid!

        Args:
            shape_type (TopologyTypes): An enumeration entry relating to shapes

        Returns:
            str: Constant GUID for the given shape
        """
        return TopologyTypes.TOPOLOGY_TO_GUID.value[shape_type.value]
    
    @staticmethod
    def get_dimensionality_for_type(shape_type: "TopologyTypes") -> int:
        """
        Upon construction of each topological object we assign dimensionality.
        Alwyas use this method to lookup shape-specific dimensionality!

        Args:
            shape_type (TopologyTypes): An enumeration entry relating to shapes

        Returns:
            int: Dimensionality 
        """
        return TopologyTypes.TOPOLOGY_TO_DIMENSIONALITY.value[shape_type.value]
    

class EdgeEnd(enum.Enum):
    """Enumeration to relate to start or end of edge.
    """
    START = 0
    END = 1