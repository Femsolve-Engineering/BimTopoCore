
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Wire import Wire
from Core.Face import Face

# Utility
from TestCases.Visualization import show_topology

def run_coverage_test():
    """
    Checks if all imports and initialization work fine.
    And then we construct objects.
    """

    try:
        # Create vertices
        v1 = Vertex.by_coordinates(0, 0, 0)
        v2 = Vertex.by_coordinates(10, 0, 0)
        v3 = Vertex.by_coordinates(10, 10, 0)
        v4 = Vertex.by_coordinates(0, 10, 0)

        # Create edges
        e1 = Edge.by_start_vertex_end_vertex(v1, v2)
        e2 = Edge.by_start_vertex_end_vertex(v2, v3)
        e3 = Edge.by_start_vertex_end_vertex(v3, v4)
        e4 = Edge.by_start_vertex_end_vertex(v4, v1)

        # Create wire
        wire = Wire.by_edges([e1, e2, e3, e4])

        # Create face
        face = Face.by_external_boundary(wire)

        # Show face
        show_topology(face)

    except Exception as ex:
        print(f'Error occured: {ex}')
        return False