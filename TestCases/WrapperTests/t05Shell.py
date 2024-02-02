# Shell Classes unit test

# importing libraries
import sys
sys.path.append("C:/Users/wassimj/Documents/GitHub")

# import topologicpy # Wrapper
# import topologic # Core

# Core
from Core.Topology import Topology as coreTopology

from Core.Vertex import Vertex as coreVertex
from Core.Edge import Edge as coreEdge
from Core.Wire import Wire as coreWire
from Core.Face import Face as coreFace
from Core.Shell import Shell as coreShell
from Core.Cell import Cell as coreCell
from Core.CellComplex import CellComplex as coreCellComplex
from Core.Cluster import Cluster as coreCluster

from Wrapper.Vertex import Vertex
from Wrapper.Edge import Edge
from Wrapper.Cluster import Cluster
from Wrapper.Wire import Wire
from Wrapper.Face import Face
from Wrapper.Shell import Shell
from Wrapper.Topology import Topology

def test_05shell() -> bool:
    # try:
    # Object for test case
    v0 = Vertex.ByCoordinates(0, 0, 0)          # create vertex
    v1 = Vertex.ByCoordinates(0, 10, 0)         # create vertex
    v2 = Vertex.ByCoordinates(10, 10, 0)        # create vertex
    v3 = Vertex.ByCoordinates(10, 0, 0)         # create vertex
    v5 = Vertex.ByCoordinates(-10, 10, 0)       # create vertex
    v6 = Vertex.ByCoordinates(-10, 0, 0)        # create vertex
    v_list0 = [v0,v1,v2,v3,v0]                  # create list
    v_list1 = [v0,v1,v5,v6,v0]                  # create list
    wire0 = Wire.ByVertices(v_list0)            # create wire
    wire1 = Wire.ByVertices(v_list1)            # create wire
    w_list = [wire0,wire1]                      # create list
    w_cluster = Cluster.ByTopologies(w_list)    # create cluster // Topology.by_occt_shape() occt_shape is a compound. Compound has no base_shape -> it fails
    face0 = Face.ByVertices(v_list0)            # create face
    face1 = Face.ByVertices(v_list1)            # create face
    f_list = [face0,face1]                      # create list
    c_faces = Cluster.ByTopologies(f_list)      # create cluster // Topology.by_occt_shape() occt_shape is a compound. Compound has no base_shape -> it fails

    # Case 1 - ByFaces

    # test 1
    # Remark: self.occt_shape_to_attributes_map is empty --> AttributeManager.py. How is the dict initialized? it should contain the sub-shapes of the current shape.
    shell_f = Shell.ByFaces(f_list)             # without tolerance
    assert isinstance(shell_f, coreShell), "Shell.ByFaces. Should be coreShell"

    # test 2
    # Remark: origin_attribute_map in Topology.deep_copy_attributes_from() is empty --> AttributeManager.py
    shell_f = Shell.ByFaces(f_list,0.001)       # with tolerance
    assert isinstance(shell_f, coreShell), "Shell.ByFaces. Should be coreShell"

    # Case 2 - ByFacesCluster

    # test 1
    # Remark: Clusters cannot be created
    shell_fc = Shell.ByFacesCluster(c_faces)
    assert isinstance(shell_fc, coreShell), "Shell.ByFacesCluster. Should be coreShell"

    # Case 3 - ByWires

    # test 1
    print("Shell-C3T1: The resulting wire consists of a single edge. Explanation: the edges making up the wire are coincident!")
    shell_w = Shell.ByWires(w_list)                                     # without optional inputs
    assert isinstance(shell_w, coreShell), "Shell.ByFaces. Should be coreShell"

    # test 2
    print("Shell-C3T2: Error message: The input wire is open!")
    # shell_w = Shell.ByWires(w_list, triangulate=True, tolerance=0.001)  # with optional inputs
    # assert isinstance(shell_w, coreShell), "Shell.ByFaces. Should be coreShell"

    # test 3
    print("Shell-C3T3: Error message: The input wire is open!")
    # shell_w = Shell.ByWires(w_list, triangulate=False, tolerance=0.001)  # with optional inputs
    # assert isinstance(shell_w, coreShell), "Shell.ByFaces. Should be coreShell"

    # Case 4 - ByWiresCluster

    # test 1
    # Remark: Topology.simplify() fails
    # shell_wc = Shell.ByWiresCluster(w_cluster)               # without optional inputs
    # assert isinstance(shell_wc, coreShell), "Shell.ByFaces. Should be coreShell"

    # test 2
    # Remark: Topology.simplify() fails
    # shell_wc = Shell.ByWiresCluster(w_cluster, triangulate=True, tolerance=0.001)   # with optional inputs
    # assert isinstance(shell_wc, coreShell), "Shell.ByFaces. Should be coreShell"

    # # test 3
    # # Remark: Topology.simplify() fails
    # shell_wc = Shell.ByWiresCluster(w_cluster, triangulate=False, tolerance=0.001)   # with optional inputs
    # assert isinstance(shell_wc, coreShell), "Shell.ByFaces. Should be coreShell"

    # # Case 5 - Circle

    # test 1
    # Remark: passed
    shell_c = Shell.Circle()                                                                 # without optional inputs
    assert isinstance(shell_c, coreShell), "Shell.Circle. Should be coreShell"

    # test 2
    # Remark: passed
    shell_c = Shell.Circle(v1, radius=2, sides=64, fromAngle=90, toAngle=180,
                            direction = [0,0,1], placement='lowerleft', tolerance=0.001)  # with optional inputs
    assert isinstance(shell_c, coreShell), "Shell.Circle. Should be coreShell"

    # Case 6 - Edges

    # # test 1
    # # Remark: shell_w has not been created
    # e_shell = Shell.Edges(shell_w)
    # assert isinstance(e_shell, list), "Shell.Edges. Should be list"

    # test 2
    # Remark: passed
    e_shell2 = Shell.Edges(shell_fc)
    assert isinstance(e_shell2, list), "Shell.Edges. Should be list"

    # # Case 7 - ExternalBoundary

    # # test 1
    # # Remark: edges should be a list of topologies but it is a list of shapes; static_downward_navigation returns occt_members
    # eb_shell = Shell.ExternalBoundary(shell_c)
    # assert isinstance(eb_shell, coreWire), "Shell.ExternalBoundary. Should be Wire"

    # # test 2
    # # Remark: edges should be a list of topologies but it is a list of shapes; static_downward_navigation returns occt_members
    # eb_shell2 = Shell.ExternalBoundary(shell_wc)
    # assert isinstance(eb_shell2, coreWire), "Shell.ExternalBoundary. Should be Wire"

    # # Case 8 - Faces

    # # test 1
    # # Remark: shell_wc has not been created
    # f_shell = Shell.Faces(shell_wc)
    # assert isinstance(f_shell, list), "Shell.Faces. Should be list"

    # # test 2
    # # Remark: shell_c has not been created
    # f_shell2 = Shell.Faces(shell_c)
    # assert isinstance(f_shell2, list), "Shell.Faces. Should be list"

    # # Case 9 - HyperbolicParaboloidCircularDomain

    # # test 1
    # # Remark: TopExp_Explorer in static_downward_navigation returns new instances of occt_shapes.
    # shell_hpcd = Shell.HyperbolicParaboloidCircularDomain()                                                 # without optional inputs
    # assert isinstance(shell_hpcd, coreShell), "Shell.HyperbolicParaboloidCircularDomain. Should be coreShell"

    # # test 2
    # # Remark: TopExp_Explorer in static_downward_navigation returns new instances of occt_shapes.
    # shell_hpcd = Shell.HyperbolicParaboloidCircularDomain(v2, radius=3.7, sides=64, rings=21, A=3, B=-3,
    #                                                         direction = [0,0,1], placement='lowerleft')  # with optional inputs
    # assert isinstance(shell_hpcd, coreShell), "Shell.HyperbolicParaboloidCircularDomain. Should be coreShell"

    # # Case 10 - HyperbolicParaboloidRectangularDomain

    # # test 1
    # # Remark: origin_attribute_map in Topology.deep_copy_attributes_from() is empty --> AttributeManager.py
    # shell_hprd = Shell.HyperbolicParaboloidRectangularDomain()                                                      # without optional inputs
    # assert isinstance(shell_hprd, coreShell), "Shell.HyperbolicParaboloidRectangularDomain. Should be coreShell"

    # # test 2
    # # Remark: origin_attribute_map in Topology.deep_copy_attributes_from() is empty --> AttributeManager.py
    # shell_hprd = Shell.HyperbolicParaboloidRectangularDomain(v3, llVertex=None, lrVertex=None, ulVertex=None, urVertex=None, uSides=20,
    #                                                         vSides=20, direction = [0,0,1], placement='lowerleft')    # with optional inputs
    # assert isinstance(shell_hprd, coreShell), "Shell.HyperbolicParaboloidRectangularDomain. Should be coreShell"

    # # Case 11 - InternalBoundaries

    # # test 1
    # # Remark: shell_hpcd has not been created
    # ib_shell = Shell.InternalBoundaries(shell_hpcd)
    # assert isinstance(ib_shell, coreTopology), "Shell.InternalBoundaries. Should be Topology"

    # # test 2
    # # Remark: shell_hprd has not been created
    # ib_shell2 = Shell.InternalBoundaries(shell_hprd)
    # assert isinstance(ib_shell2, coreTopology), "Shell.InternalBoundaries. Should be Topology"

    # # Case 12 - IsClosed

    # # test 1
    # # Remark: shell_hprd has not been created
    # bool_shell = Shell.IsClosed(shell_hprd)
    # assert isinstance(bool_shell, bool), "Shell.IsClosed. Should be bool"
    # # test 2
    # # Remark: shell_hpcd has not been created
    # bool_shell2 = Shell.IsClosed(shell_hpcd)
    # assert isinstance(bool_shell2, bool), "Shell.IsClosed. Should be bool"

    # # Case 13 - Pie

    # # test 1
    # # Remark: origin_attribute_map in Topology.deep_copy_attributes_from() is empty --> AttributeManager.py
    # shell_p = Shell.Pie()                                                           # without optional inputs
    # assert isinstance(shell_p, coreShell), "Shell.Pie. Should be coreShell"

    # # test 2
    # # Remark: origin_attribute_map in Topology.deep_copy_attributes_from() is empty --> AttributeManager.py
    # shell_p = Shell.Pie(v1, radiusA=10, radiusB=5, sides=64, rings=2, fromAngle=0, toAngle=90,
    #                     direction = [0,0,1], placement='lowerleft', tolerance=0.001)             # with optional inputs
    # assert isinstance(shell_p, coreShell), "Shell.Pie. Should be coreShell"

    # Case 14 - Rectangle

    # # test 1
    # # Remark: ContentManager is not correctly implementes --> occt_shape_to_contents_map is empty!
    # #         origin_attribute_map in Topology.deep_copy_attributes_from() is empty --> AttributeManager.py
    # shell_r = Shell.Rectangle()                                             # without optional inputs
    # assert isinstance(shell_r, coreShell), "Shell.Rectangle. Should be coreShell"

    # # test 2
    # # Remark: origin_attribute_map in Topology.deep_copy_attributes_from() is empty --> AttributeManager.py
    # shell_r = Shell.Rectangle(v2, width=2, length=4, uSides=3, vSides=3, direction = [0,0,1],
    #                         placement='lowerleft', tolerance=0.001)         # with optional inputs
    # assert isinstance(shell_r, coreShell), "Shell.Rectangle. Should be coreShell"

    # # Case 15 - SelfMerge

    # # test 1
    # # Remark: shell_f has not been created
    # f_smshell = Shell.SelfMerge(shell_f,0.1)
    # assert isinstance(f_smshell, coreFace), "Shell.SelfMerge. Should be list coreFace"

    # # test 2
    # # Remark: shell_r has not been created
    # f_smshell2 = Shell.SelfMerge(shell_r,0.1)
    # assert isinstance(f_smshell2, coreFace), "Shell.SelfMerge. Should be list coreFace"

    # # Case 16 - Vertices

    # # test 1
    # # Remark: shell_r has not been created
    # v_shell = Shell.Vertices(shell_r)
    # assert isinstance(v_shell, list), "Shell.Vertices. Should be list"

    # # test 2
    # # Remark: shell_c has not been created
    # v_shell2 = Shell.Vertices(shell_c)
    # assert isinstance(v_shell2, list), "Shell.Vertices. Should be list"

    # # Case 17 - Wires

    # # test 1
    # # Remark: shell_hprd has not been created
    # w_shell = Shell.Wires(shell_hprd)
    # assert isinstance(w_shell, list), "Shell.Wires. Should be list"

    # # test 2
    # # Remark: shell_c has not been created
    # w_shell2 = Shell.Wires(shell_c)
    # assert isinstance(w_shell2, list), "Shell.Wires. Should be list"
    return True

    # except Exception as ex:
    #     print(f'Failure Occured: {ex}')
    #     return False