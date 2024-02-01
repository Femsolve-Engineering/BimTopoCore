
from OCC.Core.TopoDS import TopoDS_Shape

class InstanceGUIDManager:

    _instance: 'InstanceGUIDManager' = None

    def __init__(self):
        """
        Singleton constructor.
        """

        self.occt_shape_to_guid_map: ShapeToGuidDict = {}

    @staticmethod
    def get_instance_manager() -> 'InstanceGUIDManager':
        """
        Returns:
            Singleton instance of GUID manager.
        """
        
        if InstanceGUIDManager._instance == None:
            InstanceGUIDManager._instance = InstanceGUIDManager()

        return InstanceGUIDManager._instance
    
    def add(self, occt_shape: TopoDS_Shape, guid: str) -> bool:
        """
        Returns:
            True - if addition was successful, false otherwise
        """
        if not occt_shape in self.occt_shape_to_guid_map:
            self.occt_shape_to_guid_map[occt_shape] = guid
            return True
        else:
            return False

    def remove(self, occt_shape: TopoDS_Shape) -> bool:
        """
        Returns: 
            True - if deletion was successful, false otherwise
        """
        if occt_shape in self.occt_shape_to_guid_map:
            del self.occt_shape_to_guid_map[occt_shape]
            return True
        else:
            return False

    def find(self, occt_shape: TopoDS_Shape) -> str:
        """
        Looks up a shape if it already exists in the collection of all shapes.

        Returns:
            "" - if shape was not found, otherwise guid's string
        """
        if occt_shape in self.occt_shape_to_guid_map:
            return self.occt_shape_to_guid_map[occt_shape]
        
        return ""

    def clear_all(self) -> None:
        """
        Clears the existing dictionary.
        """
        self.occt_shape_to_guid_map.clear()

#--------------------------------------------------------------------------------------------
class ShapeToGuid:
    def __init__(self, shape: TopoDS_Shape, guid: str):
        self.shape = shape
        self.guid = guid

def shape_comparator(e1: ShapeToGuid, e2: ShapeToGuid) -> bool:
    return e1.shape.IsSame(e2.shape)

class ShapeToGuidDict:
    def __init__(self):
        self._data = []

    def __contains__(self, key: TopoDS_Shape) -> bool:
        for k in self._data:
            if shape_comparator(k, ShapeToGuid(key, "")):
                return True
        return False

    def __getitem__(self, key: TopoDS_Shape) -> str:
        for k in self._data:
            if shape_comparator(k, ShapeToGuid(key, "")):
                return k.guid
        raise KeyError(f"{key} not found")

    def __setitem__(self, key: TopoDS_Shape, value: str):
        for k in self._data:
            if shape_comparator(k, ShapeToGuid(key, "")):
                k.guid = value
                return
        self._data.append(ShapeToGuid(key, value))

    def clear(self):
        self._data.clear()
