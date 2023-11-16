# Aperture class unit test

# Core
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Dictionary import Dictionary as coreDictionary
from Core.Context import Context as coreContext
from Core.Aperture import Aperture as coreAperture
from Core.Topology import Topology as coreTopology

# Wrapper 
from Wrapper.Topology import Topology
from Wrapper.Aperture import Aperture
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
        v1 = Vertex.ByCoordinates(0, 10, 1)         # create vertex
        v2 = Vertex.ByCoordinates(10, 10, 1)        # create vertex
        
        # Case 1 - Contextless aperture construction
        # test 1
        ap1 = Aperture.ConstructAperture(v1, None)
        assert isinstance(ap1, coreAperture), "Aperture.ConstructAperture. Should be coreAperture"
        
        # Case 2 - Context-bound aperture construction
        # test 1
        print('TestToDo-Context: Skipping test because of missing methods at the time of writing this. (Context)')
        # context = ...
        # assert isinstance(context, coreContext), "Context ctor. Should be coreContext."
        # ap1 = Aperture.ConstructAperture(v1, context)
        # assert isinstance(ap1, coreAperture), "Aperture.ConstructAperture. Should be coreAperture"
        
        # Case 3 - ByTopologyContext
        # test 1 
        print('TestToDo-Context: Skipping test because of missing methods at the time of writing this. (Context)')
        # ap1 = Aperture.ByTopologyContext()

        # Case 4 - GetApertureTopology
        tp1 = Aperture.GetApertureTopology(ap1)
        assert isinstance(tp1, coreTopology), "Aperture.GetApertureTopology. Should be coreTopology."

        return True
    
    except Exception as ex:
        print(f'Failure Occured: {ex}')
        return False