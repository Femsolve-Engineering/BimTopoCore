# Core
from Core.Topology import Topology as coreTopology
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Shell import Shell as coreShell
from Core.Cluster import Cluster as coreCluster
from Core.Aperture import Aperture as coreAperture
from Core.Context import Context as coreContext

from Core.Dictionary import Dictionary as coreDictionary
from Core.Utilities.TopologicUtilities import VertexUtility, EdgeUtility, FaceUtility

# Wrapper
from Wrapper.Vertex import Vertex
from Wrapper.Vector import Vector

class Aperture(coreAperture):
    @staticmethod
    def ApertureTopology(aperture: coreAperture) -> coreTopology:
        """
        Returns the topology of the input aperture.
        
        Parameters
        ----------
        aperture : coreAperture
            The input aperture.

        Returns
        -------
        coreTopology
            The topology of the input aperture.

        """
        return coreAperture.topology(aperture)

    @staticmethod
    def ByTopologyContext(topology: coreTopology, context: coreContext) -> coreAperture:
        """
        Creates an aperture object represented by the input topology and one that belongs to the input context.

        Parameters
        ----------
        topology : coreTopology
            The input topology that represents the aperture.
        context : coreContext
            The context of the aperture. See Context class.

        Returns
        -------
        coreAperture
            The created aperture.

        """
        aperture = None
        try:
            aperture = coreAperture.by_topology_context(topology, context)
        except:
            aperture = None
        return aperture