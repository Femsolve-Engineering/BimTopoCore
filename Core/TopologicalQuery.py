

from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes

class TopologicalQuery(Topology):


    def downcast(parent: Topology) -> None:
        """
        parent class is known. Let's find its child based on the child's topology type.
        """
        
        # az occt_shape jelenthet kapcsolatot a Topology instance es a Subshape kozott. mindkettot egyanazzal az occt_shap-pe. inicializaljuk
        # ismerjuk tovabba a TopologyType-ot

        # 1)
        # meg kell vizsgalni hogy a register_factory pontosan mit csinal. lehet hogy ott map-peljuk az adatokat es valahogy megvan ott az asszociacio

        # minden shape __init__ fuggvenyeben benne van a self.register_factory()
        # -> Ez a TopologyFactorymanager.Add() segitsegevel beleteszi a az adott shape guid-jat es az az adott shape factory instanc-et egy map-be (= dict)

        # A register factory igy mukodhet:
        # 1) topology instance -> guid 
        # 2) subshape instance -> guid

        # 3) __init__fv-ben a register_factory() triggers -> Topology.register_factory(): TopologyFactoryManager.get_instance().add(guid, topology_factory)
        # a TopologyFactory inicializálása során le kellene futni a Create() fv-nek, ami az adott occt_shape-hez egy specific topology-t ad vissza (Vertex, Face, Wire, ...) -> DE EZ NE MTORTENIK MEG ES INNENTOL NEM ERTEM HOGY HOGYAN MUKODIK

        # ------

        # A TopologyFactoryManager add() fv-e hozzaadja egy dict-hez a guid-ot es a topologyFactory-t, ami egy specific topology-re jellemzo.

        # 4) A topologyFactory instance-en keresztul kellene elernem az  adott specific topology instance-et.
        # Az instance-en keresztul a type-ot is elerem es le tudom ellenorizni hogy stimmel-e a kert tipusus subshap-pel
        
        # Problema 1) Mi general guid-ot?
        # Problema 2) A <shape>Factory (VertexFactory, FaceFactory, ..) nem tarol semmit. Pedig rajta kellene lennie az adott spcific topology instance-nek.
        pass

    def upcast():
        pass