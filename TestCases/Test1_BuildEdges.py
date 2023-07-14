
from Core.Vertex import Vertex
from Core.Edge import Edge


def run_build_edges() -> bool:
    """Baseline test case to build edges from vertices.

    Returns:
        bool: True - if all successful
    """

    try:

        # Create vertices
        v1 = Vertex.by_coordinates(0, 0, 0)
        v2 = Vertex.by_coordinates(10, 0, 0)
        v3 = Vertex.by_coordinates(10, 10, 0)
        v4 = Vertex.by_coordinates(0, 10, 0)
        v5 = Vertex.by_coordinates(0, 0, 10)
        v6 = Vertex.by_coordinates(10, 0, 10)
        v7 = Vertex.by_coordinates(10, 10, 10)
        v8 = Vertex.by_coordinates(0, 10, 10)
        v9 = Vertex.by_coordinates(0, 0, 20)
        v10 = Vertex.by_coordinates(10, 0, 20)
        v11 = Vertex.by_coordinates(10, 10, 20)
        v12 = Vertex.by_coordinates(0, 10, 20)

        # Create edges
        e1 = Edge.by_start_vertex_end_vertex(v1, v2)
        e2 = Edge.by_start_vertex_end_vertex(v2, v3)
        e3 = Edge.by_start_vertex_end_vertex(v3, v4)
        e4 = Edge.by_start_vertex_end_vertex(v4, v1)
        e5 = Edge.by_start_vertex_end_vertex(v5, v6)
        e6 = Edge.by_start_vertex_end_vertex(v6, v7)
        e7 = Edge.by_start_vertex_end_vertex(v7, v8)
        e8 = Edge.by_start_vertex_end_vertex(v8, v5)
        e9 = Edge.by_start_vertex_end_vertex(v1, v5)
        e10 = Edge.by_start_vertex_end_vertex(v2, v6)
        e11 = Edge.by_start_vertex_end_vertex(v3, v7)
        e12 = Edge.by_start_vertex_end_vertex(v4, v8)
        e13 = Edge.by_start_vertex_end_vertex(v9, v10)
        e14 = Edge.by_start_vertex_end_vertex(v10, v11)
        e15 = Edge.by_start_vertex_end_vertex(v11, v12)
        e16 = Edge.by_start_vertex_end_vertex(v12, v9)
        e17 = Edge.by_start_vertex_end_vertex(v5, v9)
        e18 = Edge.by_start_vertex_end_vertex(v6, v10)
        e19 = Edge.by_start_vertex_end_vertex(v7, v11)
        e20 = Edge.by_start_vertex_end_vertex(v8, v12)

        # Edges
        edges = [e1, e2, e3, e4, e5]
        for edge_count, edge in enumerate(edges):
            print(f'Edge #{edge_count}')
            (start_vertex, end_vertex) = edge.vertices()
            print(f"\tStart Vertex: X={start_vertex.x()}, Y={start_vertex.y()}, Z={start_vertex.z()}")
            print(f"\tEnd Vertex: X={end_vertex.x()}, Y={end_vertex.y()}, Z={end_vertex.z()}")

        print("Done!")
        return True
            
    except Exception as ex:
        print(f"Exception occured: {ex}")
        return False