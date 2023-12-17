
from Topology import Topology

class Context:

    def __init__(self, topology: Topology, param_u: float, param_v: float, param_w: float) -> None:
        """
        Constructor.
        """

        self.base_shape = topology.get_occt_shape()
        self.u = param_u
        self.v = param_v
        self.w = param_w 

#--------------------------------------------------------------------------------------------------
    def topology(self) -> 'Topology':
        """
        Getter for Topology object.
        """

        return Topology.by_occt_shape(self.base_shape, "")

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def topology_by_parameters(topology: Topology, param_u: float, param_v: float, param_w: float) -> 'Context':
        
        return Context(topology, param_u, param_v, param_w)