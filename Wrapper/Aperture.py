import topologicpy
import topologic

class Aperture(coreAperture):
    @staticmethod
    def ApertureTopology(aperture: coreAperture) -> topologic.Topology:
        """
        Returns the topology of the input aperture.
        
        Parameters
        ----------
        aperture : coreAperture
            The input aperture.

        Returns
        -------
        topologic.Topology
            The topology of the input aperture.

        """
        return coreAperture.Topology(aperture)

    @staticmethod
    def ByTopologyContext(topology: topologic.Topology, context: topologic.Context) -> coreAperture:
        """
        Creates an aperture object represented by the input topology and one that belongs to the input context.

        Parameters
        ----------
        topology : topologic.Topology
            The input topology that represents the aperture.
        context : topologic.Context
            The context of the aperture. See Context class.

        Returns
        -------
        coreAperture
            The created aperture.

        """
        aperture = None
        try:
            aperture = coreAperture.ByTopologyContext(topology, context)
        except:
            aperture = None
        return aperture