

from typing import Tuple
from typing import List
from xmlrpc.client import boolean



# BimTopoCore
from Core.TopologyConstants import TopologyTypes

class EdgeUtility:

    @staticmethod
    def adjacent_faces(edge, parent_topology, core_adjacent_faces) -> List['Face']:

        from Core.Face import Face
        
        # Create list of adjacent faces
        core_adjacent_topologies = edge.upward_navigation(parent_topology.get_occt_shape(), TopologyTypes.FACE)

        for adjacent_topology in core_adjacent_topologies:

            face = Face(adjacent_topology)
            core_adjacent_faces.append(face)

