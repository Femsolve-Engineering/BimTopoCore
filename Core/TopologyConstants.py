
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
    WIRE = 4
    FACE = 8
    SHELL = 16
    CELL = 32
    CELLCOMPLEX = 64
    CLUSTER = 128
    APERTURE = 256
    ALL = 511

    TOPOLOGY_TO_GUID = {
        VERTEX : "9FE435F9-7AC9-4D94-852A-BB2FFBD3D720", # Original: c4a9b420-edaf-4f8f-96eb-c87fbcc92f2b
        EDGE :   "AF050AFB-50BE-499C-8F26-DABB60105302", # Original: 1fc6e6e1-9a09-4c0a-985d-758138c49e35
        WIRE :   "58486ECA-5CF4-4CC5-8C2F-3DA248C529FF", # Original: b99ccd99-6756-401d-ab6c-11162de541a3
        FACE :   "7B80A066-5A0C-423E-ABAB-82343DBBD3E6", # Original: 3b0a6afe-af86-4d96-a30d-d235e9c98475
        SHELL:   "51c1e590-cec9-4e84-8f6b-e4f8c34fd3b3",
        CELL:    "8bda6c76-fa5c-4288-9830-80d32d283251",
        CELLCOMPLEX: "4ec9904b-dc01-42df-9647-2e58c2e08e78",
        CLUSTER: "7c498db6-f3e7-4722-be58-9720a4a9c2cc",
        APERTURE: "740d9d31-ca8c-47ef-825f-68c607af80aa"
    }

    TOPOLOGY_TO_DIMENSIONALITY = {
        VERTEX : 0,
        EDGE : 1,
        WIRE : 1,
        FACE : 2,
        SHELL: 2,
        CELL: 3,
        CELLCOMPLEX: 3,
        CLUSTER: 3,
        APERTURE : -1
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