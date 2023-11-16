from typing import Dict, List, Union, Tuple, Any


class Attribute:
    def __init__(self):
        self.attribute_value = None
        pass

    def value(self) -> Any:
        return self.attribute_value
    
class IntAttribute(Attribute):
    def __init__(self, integer: int):
        self.attribute_value = integer

class DoubleAttribute(Attribute):
    def __init__(self, float: float):
        self.attribute_value = float

class StringAttribute(Attribute):
    def __init__(self, string: str):
        self.attribute_value = string

class ListAttribute(Attribute):
    def __init__(self, list_attributes: Attribute):
        self.attribute_value = list_attributes

class Dictionary:
    """
    This dictionary class is customize to store Attribute informations 
    of Topological objects.
    """
    def __init__(self):
        self._dict: Dict[str, Attribute] = {}

    def add(self, key: Union[str, Tuple[str, Attribute]], value: Attribute = None) -> None:
        if isinstance(key, tuple):
            key, value = key
        if key in self._dict:
            raise ValueError("Key already exists")
        self._dict[key] = value

    def clear(self) -> None:
        self._dict.clear()

    def contains_key(self, key: str) -> bool:
        return key in self._dict

    def contains(self, key: str, value: Attribute) -> bool:
        return self._dict.get(key) == value

    def remove(self, key: Union[str, Tuple[str, Attribute]]) -> bool:
        if isinstance(key, tuple):
            key, value = key
            if self._dict.get(key) == value:
                del self._dict[key]
                return True
        elif key in self._dict:
            del self._dict[key]
            return True
        return False

    def try_add(self, key: str, value: Attribute) -> bool:
        if key not in self._dict:
            self._dict[key] = value
            return True
        return False

    def try_get_value(self, key: str) -> Tuple[bool, Union[Attribute, None]]:
        if key in self._dict:
            return True, self._dict[key]
        return False, None

    def keys(self) -> List[str]:
        return list(self._dict.keys())

    def values(self) -> List[Attribute]:
        return list(self._dict.values())

    def copy_to(self, target: List[Tuple[str, Attribute]], index: int, length: int) -> None:
        if length - index < len(self._dict):
            raise ValueError("Array is not big enough")
        for i, (k, v) in enumerate(self._dict.items(), start=index):
            target[i] = (k, v)

    @staticmethod
    def by_keys_values(keys: List[str], values: List[Attribute]) -> 'Dictionary':
        if len(keys) != len(values):
            raise ValueError("Keys and values have a different size")
        d = Dictionary()
        for k, v in zip(keys, values):
            d.add(k, v)
        return d

    def value_at_key(self, key: str) -> Attribute:
        return self._dict[key]

    def count(self) -> int:
        return len(self._dict)

    def is_read_only(self) -> bool:
        return False

    def is_fixed_size(self) -> bool:
        return False

    def get_enumerator(self):
        return iter(self._dict.items())
