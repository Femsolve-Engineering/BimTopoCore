import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")
from typing import Tuple

# Edge Class unit test

# Core
from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace

# Wrapper 
from Wrapper.Topology import Topology
from Wrapper.Vertex import Vertex
from Wrapper.Edge import Edge
from Wrapper.Face import Face
from Wrapper.Cell import Cell
from Wrapper.Cluster import Cluster
from Wrapper.CellComplex import CellComplex

def test_02edge() -> bool:
    try:

        # Object for test case
        v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
        v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
        v3 = Vertex.ByCoordinates(0, -10, 0)        # create vertex
        v4 = Vertex.ByCoordinates(5, 0, 0)          # create vertex
        list_v = [v0,v1]                            # create list of vertices
        list_v1 = [v1,v3]                           # create list of vertices
        print("TestToDo-Edge: Skipping test because some methods that are required are not yet available.")
        # cluster_1 = Cluster.ByTopologies(list_v)    # create cluster of vertices
        # cluster_2 = Cluster.ByTopologies(list_v1)   # create cluster of vertices    
        eB = Edge.ByStartVertexEndVertex(v0,v3)     # create edge

        # Case 1 - Create an edge ByStartVertexEndVertex
        # test 1
        e1 = Edge.ByStartVertexEndVertex(v0,v1)                    # without tolerance
        assert isinstance(e1, coreEdge), "Edge.ByStartVertexEndVertex. Should be coreEdge"
        # test 2
        e1 = Edge.ByStartVertexEndVertex(v0,v1, tolerance=0.001)   # with tolerance (optional)
        assert isinstance(e1, coreEdge), "Edge.ByStartVertexEndVertex. Should be coreEdge"
        # test 3
        e2 = Edge.ByStartVertexEndVertex(v1, v3)                   # without tolerance
        assert isinstance(e2, coreEdge), "Edge.ByStartVertexEndVertex. Should be coreEdge"

        # Case 2 - Create an edge ByVertices
        # test 1
        e3 = Edge.ByVertices(list_v)                    # without tolerance
        assert isinstance(e3, coreEdge), "Edge.ByVertices. Should be coreEdge"
        # test 2
        e3 = Edge.ByVertices(list_v, tolerance=0.001)   # with tolerance (optional) 
        assert isinstance(e3, coreEdge), "Edge.ByVertices. Should be coreEdge"
        # test 3
        e4 = Edge.ByVertices(list_v1, tolerance=0.001)  # with tolerance (optional) 
        assert isinstance(e4, coreEdge), "Edge.ByVertices. Should be coreEdge"                                                  

        # Case 3 - Create an edge ByVerticesCluster
        # test 1
        print("TestToDo-Edge: Skipping test because some methods that are required are not yet available.")
        # e5 = Edge.ByVerticesCluster(cluster_1)                  # without tolerance
        # assert isinstance(e5, coreEdge), "Edge.ByVerticesCluster. Should be coreEdge"
        # # test 2
        # e5 = Edge.ByVerticesCluster(cluster_1, tolerance=0.001) # with tolerance (optional)
        # assert isinstance(e5, coreEdge), "Edge.ByVerticesCluster. Should be coreEdge"
        # # test 3
        # e6 = Edge.ByVerticesCluster(cluster_2)                  # without tolerance
        # assert isinstance(e6, coreEdge), "Edge.ByVerticesCluster. Should be coreEdge"

        # Case 4 - Angle
        e7 = Edge.ByStartVertexEndVertex(v0,v4)         #create edge
        # test 1
        angle = Edge.Angle(e1,e7)                                # without optional inputs
        assert isinstance(angle, float), "Edge.Angle. Should be float"
        # test 2
        angle = Edge.Angle(e1,e7,mantissa=2, bracket=True)       # with optional inputs
        assert isinstance(angle, float), "Edge.Angle. Should be float"
        # test 3
        print("TestToDo-Edge: Skipping test because some methods that are required are not yet available.")
        # angle1 = Edge.Angle(e1,e5)                               # without optional inputs
        # assert isinstance(angle1, float), "Edge.Angle. Should be float"

        # Case 5 - Bisect
        # test 1
        e_bis = Edge.Bisect(e1, eB)                                                # without optional inputs
        assert isinstance(e_bis, coreEdge), "Edge.Bisect. Should be coreEdge" 
        # test 2
        e_bis = Edge.Bisect(e1, eB, length=1.0, placement=1, tolerance=0.001)      # with optional inputs
        assert isinstance(e_bis, coreEdge), "Edge.Bisect. Should be coreEdge" 

        # Case 6 - Direction
        # test 1
        direction = Edge.Direction(e2)                   # without optional inputs
        assert isinstance(direction, list), "Edge.Direction. Should be list"
        # test 2
        direction1 = Edge.Direction(e1)                  # without optional inputs
        assert isinstance(direction1, list), "Edge.Direction. Should be list"
        # test 3
        direction1 = Edge.Direction(e1,mantissa=3)       # with optional inputs
        assert isinstance(direction1, list), "Edge.Direction. Should be list"

        # Case 7 - EndVertex
        # test 1
        end2 = Edge.EndVertex(e2)
        assert isinstance(end2, coreVertex), "Edge.EndVertex. Should be coreVertex"
        # test 2
        end3 = Edge.EndVertex(e3)
        assert isinstance(end3, coreVertex), "Edge.EndVertex. Should be coreVertex"

        # Case 8 - Extend
        # test 1
        extend2 = Edge.Extend(e2)                                                               # without optional inputs
        assert isinstance(extend2, coreEdge), "Edge.Extend. Should be coreEdge"
        # test 2
        extend3 = Edge.Extend(e3)                                                               # without optional inputs
        assert isinstance(extend3, coreEdge), "Edge.Extend. Should be coreEdge"
        # test 3
        extend3 = Edge.Extend(e3,distance=2, bothSides=False, reverse=True, tolerance=0.001)    # with optional inputs
        assert isinstance(extend3, coreEdge), "Edge.Extend. Should be coreEdge"

        # Case 9 - IsCollinear (True)
        e5 = Edge.ByStartVertexEndVertex(v0,v3)
        # test 1
        col_1 = Edge.IsCollinear(e1,e5, mantissa=3)                                         # without optional inputs
        assert isinstance(col_1, bool), "Edge.IsCollinear. Should be bool"
        # test 2
        col_2 = Edge.IsCollinear(e1,e3, mantissa=3)                                         # without optional inputs
        assert isinstance(col_2, bool), "Edge.IsCollinear. Should be bool"
        # test 3
        col_1 = Edge.IsCollinear(e1,e5, mantissa=3, angTolerance=0.01, tolerance=0.001)     # with optional inputs
        assert isinstance(col_1, bool), "Edge.IsCollinear. Should be bool"

        # Case 10 - IsParallel
        # test 1
        par_1 = Edge.IsParallel(e1,e4)                                  # without optional inputs
        assert isinstance(par_1, bool), "Edge.IsParallel. Should be bool"
        # test 2
        par_2 = Edge.IsParallel(e1,e3)                                  # without optional inputs
        assert isinstance(par_2, bool), "Edge.IsParallel. Should be bool"
        # test 3
        par_1 = Edge.IsParallel(e1,e4, mantissa=2, angTolerance=0.01)   # with optional inputs
        assert isinstance(par_1, bool), "Edge.IsParallel. Should be bool"

        # Case 11 - Length
        # test 1
        len_1 = Edge.Length(e1)               # without optional inputs
        assert isinstance(len_1, float), "Edge.Length. Should be float"
        # test 2
        len_2 = Edge.Length(e2)               # without optional inputs
        assert isinstance(len_2, float), "Edge.Length. Should be float"
        # test 3
        len_1 = Edge.Length(e1, mantissa=3)   # with optional inputs
        assert isinstance(len_1, float), "Edge.Length. Should be float"

        # Case 12 - Normalize
        # test 1
        normal_3 = Edge.Normalize(e3)                     # without optional inputs
        assert isinstance(normal_3, coreEdge), "Edge.Normalize. Should be coreEdge"
        # test 2
        normal_4 = Edge.Normalize(e4)                     # without optional inputs
        assert isinstance(normal_4, coreEdge), "Edge.Normalize. Should be coreEdge"
        # test 3
        normal_4 = Edge.Normalize(e4, useEndVertex=True)  # with optional inputs
        assert isinstance(normal_4, coreEdge), "Edge.Normalize. Should be coreEdge"

        # Case 13 - ParameterAtVertex
        # test 1
        param1 = Edge.ParameterAtVertex(e2,v1)              # without optional inputs
        assert isinstance(param1, float), "Edge.ParameterAtVertex. Should be float"
        # test 2
        param2 = Edge.ParameterAtVertex(e1,v1, mantissa=3)  # with optional inputs
        assert isinstance(param2, float), "Edge.ParameterAtVertex. Should be float"

        # Case 14 - Reverse
        # test 1
        reverse3 = Edge.Reverse(e3)
        assert isinstance(reverse3, coreEdge), "Edge.Reverse. Should be coreEdge"
        # test 2
        reverse4 = Edge.Reverse(e4)
        assert isinstance(reverse4, coreEdge), "Edge.Reverse. Should be coreEdge"

        # Case 15 - SetLength
        # test 1
        SetLen1 = Edge.SetLength(e1)                                                            # without optional inputs
        assert isinstance(SetLen1, coreEdge), "Edge.SetLength. Should be coreEdge"
        # test 2
        SetLen2 = Edge.SetLength(e2)                                                            # without optional inputs
        assert isinstance(SetLen2, coreEdge), "Edge.SetLength. Should be coreEdge"
        # test 3
        SetLen1 = Edge.SetLength(e1, length=2, bothSides=False, reverse=True, tolerance=0.001)  # with optional inputs
        assert isinstance(SetLen1, coreEdge), "Edge.SetLength. Should be coreEdge"

        # Case 16 - StartVertex
        # test 1
        iV = Edge.StartVertex(e1)
        assert isinstance(iV, coreVertex), "Edge.StartVertex. Should be coreVertex"
        # test 2
        iV1 = Edge.StartVertex(e2)
        assert isinstance(iV1, coreVertex), "Edge.StartVertex. Should be coreVertex"

        # Case 17 - Trim
        # test 1
        trim3 = Edge.Trim(e3)                                                               # without optional inputs
        assert isinstance(trim3, coreEdge), "Edge.Trim. Should be coreEdge"
        # test 2
        trim4 = Edge.Trim(e4)                                                               # without optional inputs
        assert isinstance(trim4, coreEdge), "Edge.Trim. Should be coreEdge"
        # test 3
        trim4 = Edge.Trim(e4, distance=1, bothSides=False, reverse=True, tolerance=0.001)   # with optional inputs
        assert isinstance(trim4, coreEdge), "Edge.Trim. Should be coreEdge"

        # Case 18 - VertexByDistance
        # test 1
        dist1 = Edge.VertexByDistance(e1)                                           # without optional inputs
        assert isinstance(dist1, coreVertex), "Edge.VertexByDistance. Should be coreVertex"
        # test 2
        dist2 = Edge.VertexByDistance(e2)                                           # without optional inputs
        assert isinstance(dist2, coreVertex), "Edge.VertexByDistance. Should be coreVertex"
        # test 3
        dist2 = Edge.VertexByDistance(e2, distance=1, origin=v3, tolerance=0.001)   # with optional inputs
        assert isinstance(dist2, coreVertex), "Edge.VertexByDistance. Should be coreVertex"

        # Case 19 - VertexByParameter
        # test 1
        ByParam3 = Edge.VertexByParameter(e3)                  # without optional inputs
        assert isinstance(ByParam3, coreVertex), "Edge.VertexByParameter. Should be coreVertex"
        # test 2
        ByParam4 = Edge.VertexByParameter(e4)                  # without optional inputs
        assert isinstance(ByParam4, coreVertex), "Edge.VertexByParameter. Should be coreVertex"
        # test 3
        ByParam4 = Edge.VertexByParameter(e4, parameter=0.7)   # with optional inputs
        assert isinstance(ByParam4, coreVertex), "Edge.VertexByParameter. Should be coreVertex"

        #Case 20 - Vertices
        # test 1
        v_e5 = Edge.Vertices(e5)
        assert isinstance(v_e5, list), "Edge.Vertices. Should be list"
        # test 2
        # print("TestToDo-Edge: Skipping test because some methods that are required are not yet available.")
        # v_e6 = Edge.Vertices(e6)
        # assert isinstance(v_e6, list), "Edge.Vertices. Should be list"

        #Case 21 - ByFaceNormal
        # test 1
        from Wrapper.Face import Face
        # face = Face.Rectangle()
        print("TestToDo-Edge: Originally created face with 'Face.Rectangle()' but that uses 'Topologic.SelfMerge' method which is not yet available.")
        base_edge1 = coreEdge.by_start_vertex_end_vertex(coreVertex.by_coordinates(0,0,0), coreVertex.by_coordinates(0,1,0))
        base_edge2 = coreEdge.by_start_vertex_end_vertex(coreVertex.by_coordinates(0,1,0), coreVertex.by_coordinates(1,1,0))
        base_edge3 = coreEdge.by_start_vertex_end_vertex(coreVertex.by_coordinates(1,1,0), coreVertex.by_coordinates(1,0,0))
        base_edge4 = coreEdge.by_start_vertex_end_vertex(coreVertex.by_coordinates(1,0,0), coreVertex.by_coordinates(0,0,0))
        face = coreFace.by_edges([base_edge1, base_edge2, base_edge3, base_edge4])
        edge = Edge.ByFaceNormal(face)
        assert isinstance(edge, coreEdge), "Edge.ByFaceNormal. Should be coreEdge"
        # test 2
        # face = Face.Rectangle()
        print("TestToDo-Edge: Originally created face with 'Face.Rectangle()' but that uses 'Topologic.SelfMerge' method which is not yet available.")
        face = coreFace.by_edges([base_edge1, base_edge2, base_edge3, base_edge4])
        edge = Edge.ByFaceNormal(face, length=3)
        assert Edge.Length(edge) == 3, "Edge.ByFaceNormal. Length should be 3"

        #Case 22 - ByOffset2D
        # test 1
        from Wrapper.Topology import Topology
        v1 = Vertex.ByCoordinates(0,0,0)
        v2 = Vertex.ByCoordinates(10,0,0)
        edge = Edge.ByVertices([v1, v2])
        edge2 = Edge.ByOffset2D(edge, offset=1)
        assert isinstance(edge2, coreEdge), "Edge.ByOffset2D. Should be coreEdge"
        centroid = Topology.Centroid(edge2)
        assert Vertex.X(centroid) == 5, "Edge.ByOffset2D. X Should be 5"
        assert Vertex.Y(centroid) == 1, "Edge.ByOffset2D. Y Should be 1"

        #Case 23 - ExtendToEdge2D
        # test 1
        v1 = Vertex.ByCoordinates(0,0,0)
        v2 = Vertex.ByCoordinates(10,0,0)
        edge = Edge.ByVertices([v1, v2])
        v1 = Vertex.ByCoordinates(20,-10,0)
        v2 = Vertex.ByCoordinates(20,10,0)
        edge2 = Edge.ByVertices([v1, v2])
        edge3 = Edge.ExtendToEdge2D(edge, edge2)
        assert isinstance(edge3, coreEdge), "Edge.ExtendToEdge2D. Should be coreEdge"
        assert Edge.Length(edge3) == 20, "Edge.ExtendToEdge2D. Length should be 3"
        centroid = Topology.Centroid(edge3)
        assert Vertex.X(centroid) == 10, "Edge.ExtendToEdge2D. X Should be 5"
        assert Vertex.Y(centroid) == 0, "Edge.ExtendToEdge2D. Y Should be 1"

        #Case 24 - Intersect2D
        # test 1
        v1 = Vertex.ByCoordinates(0,0,0)
        v2 = Vertex.ByCoordinates(10,0,0)
        edge = Edge.ByVertices([v1, v2])
        v1 = Vertex.ByCoordinates(5,-10,0)
        v2 = Vertex.ByCoordinates(5,10,0)
        edge2 = Edge.ByVertices([v1, v2])
        v3 = Edge.Intersect2D(edge, edge2)
        assert isinstance(v3, coreVertex), "Edge.Intersect2D. Should be coreEdge"
        assert Vertex.X(v3) == 5, "Edge.Intersect2D. X Should be 5"
        assert Vertex.Y(v3) == 0, "Edge.Intersect2D. Y Should be 0"
        return True

    except Exception as ex:
        print(f'Failure Occured: {ex}')
        return False
