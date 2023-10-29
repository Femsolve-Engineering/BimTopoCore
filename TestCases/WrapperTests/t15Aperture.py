# Aperture class unit test

# Core
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Dictionary import Dictionary as coreDictionary

# Wrapper 
from Wrapper.Topology import Topology
from Wrapper.Vertex import Vertex
from Wrapper.Edge import Edge
from Wrapper.Wire import Wire
from Wrapper.Face import Face
from Wrapper.Shell import Shell
from Wrapper.Cell import Cell
from Wrapper.Cluster import Cluster
from Wrapper.CellComplex import CellComplex
from Wrapper.Dictionary import Dictionary

def test_15aperture() -> bool:

    try:
        return True
    
    except Exception as ex:
        print(f'Failure Occured: {ex}')
        return False