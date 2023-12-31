

from typing import Tuple
from typing import List
from xmlrpc.client import boolean

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, topods
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_MapIteratorOfMapOfShape, TopTools_ListOfShape, TopTools_IndexedDataMapOfShapeListOfShape
from OCC.core.TopExp import TopExp_MapShapesAndAncestors, TopExp_Explorer
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Surface, Geom_Geometry
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Solid
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_EdgeDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NoFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NotPlanar
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_CurveProjectionFailed
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_ParametersOutOfRange
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, SurfaceProperties, VolumeProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeSolid
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError
from OCC.Core.BRepClass3d import BRepClass3d_OuterShell
from OCC.Core.BOPAlgo import BOPAlgo_MakerVolume

# BimTopoCore
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Wire import Wire
from Core.Face import Face
from Core.Cell import Cell
from Core.CellComplex import CellComplex
from Core.Shell import Shell

from Core.Utilities import EdgeUtility
from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.Factories.AllFactories import CellFactory

# Not implemented yet
# import Core.AttributeManager

class Cell(Topology):
    
    def __init__(self, occt_solid: TopoDS_Solid, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_solid (TopoDS_Solid): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        instance_topology = super().__init__(occt_solid, TopologyTypes.CELL)
        self.base_shape_solid = occt_solid
        self.register_factory(self.get_class_guid(), CellFactory())

        # Register the instances
        Topology.topology_to_subshape[instance_topology] = self
        Topology.subshape_to_topology[self] = instance_topology

#--------------------------------------------------------------------------------------------------
    def is_container_type(self) -> bool:
        """
        Determines if this topology is container type.
        """
        return False

#--------------------------------------------------------------------------------------------------
    def get_type(self) -> TopologyTypes:
        """
        Returns:
            TopologyTypes: Internal definition for types.
        """
        return TopologyTypes.CELL

#--------------------------------------------------------------------------------------------------
    def adjacent_cells(self, host_topology: Topology):
        """
        Returns the Cells adjacent to this Cell.
        """
        
        adjacent_cells = []

        # Get a map of Face->Solid[]
        occt_face_solid_map = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp_MapShapesAndAncestors(host_topology.get_occt_shape(), TopAbs_FACE, TopAbs_SOLID, occt_face_solid_map)

        # Find the constituent Faces
        occt_faces = TopTools_MapOfShape() # Create an empty list for shapes
        self.downward_navigation(host_topology.get_occt_shape(), TopAbs_FACE, occt_faces) # Put FACEs into the list

        # Iterate through the Faces and find the incident cells which are not this cell.
        occt_solid = self.get_occt_solid()
        occt_adjacent_solids = TopTools_MapOfShape()
        for occt_face in occt_faces:
            try:
                incident_cells = occt_face_solid_map.FindFromKey(occt_face)

                for incident_cell in incident_cells:
                    if not occt_solid.IsSame(incident_cell):
                        occt_adjacent_solids.Add(incident_cell)

            except:
                assert "Cannot find a Face in the global Cluster."
                raise RuntimeError("Cannot find a Face in the global Cluster.")

        # Output the adjacent Cells
        adjacent_cells = []
        for occt_adjacent_solid in occt_adjacent_solids:
            adjacent_cells.append(Cell(TopoDS_Solid(occt_adjacent_solid.Value())))



        return adjacent_cells

#--------------------------------------------------------------------------------------------------
    def cell_complexes(self, host_topology: Topology) -> List[Topology]:
        """
        Returns the CellComplexes which contain this Cell.
        """

        print('REMINDER!!! Base Topology method of upward_navigation() is not yet correctly implemented.')
        if not host_topology.is_null_shape():
            return self.upward_navigation(host_topology.get_occt_shape())
        else:
            raise RuntimeError("Host Topology cannot be None when searching for ancestors.")

#--------------------------------------------------------------------------------------------------
    def shells(self):
        """
        Returns the Shell contituents to this Cell
        """
        
        return self.downward_navigation(TopologyTypes.SHELL)

#--------------------------------------------------------------------------------------------------
    def edges(self):
        """
        Returns the Edge contituents to this Cell
        """
        
        return self.downward_navigation(TopologyTypes.EDGE)

#--------------------------------------------------------------------------------------------------
    def faces(self):
        """
        Returns the Face contituents to this Cell
        """
        
        return self.downward_navigation(TopologyTypes.FACE)

#--------------------------------------------------------------------------------------------------
    def vertices(self):
        """
        Returns the Vrtex contituents to this Cell
        """
        
        return self.downward_navigation(TopologyTypes.VERTEX)

#--------------------------------------------------------------------------------------------------
    def wires(self):
        """
        Returns the Wire contituents to this Cell
        """
        
        return self.downward_navigation(TopologyTypes.WIRE)

#--------------------------------------------------------------------------------------------------
    def center_of_mass(self) -> 'Vertex':
        """
        Returns the Vertex at the center of mass of this Cell.
        """

        occt_vertex = Cell.make_pnt_at_center_of_mass(self.get_occt_solid())
        vertex = Vertex(occt_vertex)
        
        # occt_center_of_mass = Cell.make_pnt_at_center_of_mass(self.get_occt_solid())        
        # occt_vertex = Vertex.center_of_mass(occt_center_of_mass)
        # vertex = Topology.by_occt_shape(occt_vertex)

        return vertex


#--------------------------------------------------------------------------------------------------
    @staticmethod
    def make_pnt_at_center_of_mass(occt_solid: TopoDS_Solid) -> TopoDS_Vertex:
        """
        Returns the OCCT vertex at the center of mass of this Cell.
        """

        occt_shape_properties = GProp_GProps()
        VolumeProperties(occt_solid, occt_shape_properties)

        center_of_mass_point = occt_shape_properties.CenterOfMass()
        return BRepBuilderAPI_MakeVertex(center_of_mass_point).Vertex()

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_face(faces: List[Face], tolerance: float, copy_attributes: boolean):
        """
        Creates a Cell by a set of faces.
        """
        
        if tolerance <= 0.0:
            # raise RuntimeError("The tolerance must have a positive value.")
            return None

        if not faces:
            # raise RuntimeError("The input Face list is empty.")
            return None

        occt_maker_volume = BOPAlgo_MakerVolume()
        occt_shapes = TopTools_ListOfShape()

        for face in faces:
            occt_shapes.Append(face.get_occt_shape())

        is_parallel = False
        does_intersection = True

        occt_maker_volume.SetArguments(occt_shapes)
        occt_maker_volume.SetRunParallel(is_parallel)
        occt_maker_volume.SetIntersect(does_intersection)
        occt_maker_volume.SetFuzzyValue(tolerance)

        occt_maker_volume.Perform()
        if occt_maker_volume.HasErrors():
            # raise RuntimeError("The input Faces do not form a Cell.")
            return None

        occt_result = occt_maker_volume.Shape()

        # The result is either:
        # - A solid

        occt_solid = None

        if occt_result.ShapeType() == TopAbs_SOLID:
            # Return the Cell
            occt_solid = topods.Solid(occt_result)
        # - A compound, collect the solids
        elif occt_result.ShapeType() == TopAbs_COMPOUND:

            print('txt is unused! Topology.analyze() not implemented!')
            # txt = Topology.analyze(occt_result)

            # Must only have 1 shape
            occt_shape = None

            for occt_explorer in TopExp_Explorer(occt_result):
                occt_current = occt_explorer.Current()
                if occt_shape is None:
                    occt_shape = topods.Solid(occt_current)
                else:
                    # raise RuntimeError("The input Faces do not form a Cell.")
                    return None

            if occt_shape is None:
                # raise RuntimeError("The input Faces do not form a Cell.")
                return None
            else:
                occt_solid = topods.Solid(occt_shape)

        # Shape fix the solid
        occt_fixed_solid = self.occt_shape_fix(occt_solid)
        fixed_cell = Cell(occt_fixed_solid)

        # Deep copy the Cell
        copy_fixed_cell = fixed_cell.deep_copy_shape()

        # Register to Global Cluster
        # GlobalCluster::GetInstance().AddTopology(fixed_cell->GetOcctSolid())

        # Copy the Dictionaries
        if copy_attributes:

            faces_as_topologies = []

            for face in faces:
                faces_as_topologies.append(face)
                
                # AttributeManager not implemented yet!

                # C++ code:
                # AttributeManager::GetInstance().DeepCopyAttributes(kpFace->GetOcctFace(), copyFixedCell->GetOcctSolid());
                
                # Python code:
                # instance = AttributeManager.get_instance()
                # instance.deep_copy_attributes(occt_face, occt_shell)


            # Topology.deep_copy_attributes_from() not implemented yet!

            # C++ code:
            # copyFixedCell->DeepCopyAttributesFrom(facesAsTopologies);

            # Python code:
            # copy_fixed_cell.deep_copy_attributes_from(faces_as_topologies)

        return copy_fixed_cell

#--------------------------------------------------------------------------------------------------
    def by_shell(self, shell: Shell, copy_attributes: boolean):
        """
        Creates a Cell by a Shell.
        """
        
        if not shell.is_closed():
            # raise RuntimeError("The input Shell is open.")
            return None

        try:
            occt_make_solid = BRepBuilderAPI_MakeSolid(shell.get_occt_shell())
        except:
            # raise RuntimeError("The input Shell does not form a valid Cell.")
            return None

        # Shape fix the solid
        occt_fixed_solid = self.occt_shape_fix(occt_make_solid.Solid())
        fixed_cell = Cell(occt_fixed_solid)

        # Deep copy the Cell
        copy_fixed_cell = fixed_cell.deep_copy_shape()

        # Register to Global Cluster
        # GlobalCluster::GetInstance().AddTopology(fixed_cell.GetOcctSolid())

        # Copy the Dictionaries
        if copy_attributes:
            pass
            
            # AttributeManager not implemented yet!

            # C++ code:
            # AttributeManager::GetInstance().DeepCopyAttributes(kpShell->GetOcctShell(), copyFixedCell->GetOcctSolid());
            
            # Python code:
            # instance = AttributeManager.get_instance()
            # instance.deep_copy_attributes(shell.get_occt_shell(), copy_fixed_cell.get_occt_solid())

        return copy_fixed_cell

#--------------------------------------------------------------------------------------------------
    def shared_edges(self, another_cell: Cell) -> List[Edge]:
        """
        Identify Edges shared by two cells.
        """
        
        occt_shape1 = self.get_occt_shape()
        occt_edges1 = TopTools_MapOfShape()
        occt_edges1 = Topology.static_downward_navigation(occt_shape1, TopAbs_EDGE)

        occt_shape2 = another_cell.get_occt_shape()
        occt_edges2 = TopTools_MapOfShape()
        occt_edges2 = Topology.static_downward_navigation(occt_shape2, TopAbs_EDGE)

        shared_edges_list = []

        for occt_edge_iterator1 in TopTools_MapIteratorOfMapOfShape(occt_edges1):

            for occt_edge_iterator2 in TopTools_MapIteratorOfMapOfShape(occt_edges2):

                if occt_edge_iterator1.Value().IsSame(occt_edge_iterator2.Value()):

                    edge = Edge(topods.Edge(occt_edge_iterator1.Value()))
                    shared_edges_list.append(edge)

        return shared_edges_list

#--------------------------------------------------------------------------------------------------
    def shared_faces(self, another_cell: Cell) -> List[Face]:
        """
        Identify Faces shared by two cells.
        """

        occt_shape1 = self.get_occt_shape()
        occt_faces1 = TopTools_MapOfShape()
        occt_faces1 = Topology.static_downward_navigation(occt_shape1, TopAbs_FACE)

        occt_shape2 = another_cell.get_occt_shape()
        occt_faces2 = TopTools_MapOfShape()
        occt_faces2 = Topology.static_downward_navigation(occt_shape2, TopAbs_FACE)

        shared_faces_list = []

        for occt_face_iterator1 in TopTools_MapIteratorOfMapOfShape(occt_faces1):

            for occt_face_iterator2 in TopTools_MapIteratorOfMapOfShape(occt_faces2):

                if occt_face_iterator1.Value().IsSame(occt_face_iterator2.Value()):

                    face = Face(topods.Face(occt_face_iterator1.Value()))
                    shared_faces_list.append(face)

        return shared_faces_list

#--------------------------------------------------------------------------------------------------
    def shared_vertices(self, another_cell: Cell) -> List[Vertex]:
        """
        Identify Vertices shared by two cells.
        """
        
        occt_shape1 = self.get_occt_shape()
        occt_vertices1 = TopTools_MapOfShape()
        occt_vertices1 = Topology.static_downward_navigation(occt_shape1, TopAbs_VERTEX)

        occt_shape2 = another_cell.get_occt_shape()
        occt_vertices2 = TopTools_MapOfShape()
        occt_vertices2 = Topology.static_downward_navigation(occt_shape2, TopAbs_VERTEX)

        shared_vertices_list = []

        for occt_vertex_iterator1 in TopTools_MapIteratorOfMapOfShape(occt_vertices1):

            for occt_vertex_iterator2 in TopTools_MapIteratorOfMapOfShape(occt_vertices2):

                if occt_vertex_iterator1.Value().IsSame(occt_vertex_iterator2.Value()):

                    vertex = Vertex(topods.Vertex(occt_vertex_iterator1.Value()))
                    shared_vertices_list.append(vertex)

        return shared_vertices_list

#--------------------------------------------------------------------------------------------------
    def external_boundary(self):
        """
        Returns the external boundary (= Shell) of this Cell.
        """
        
        occt_outer_shell = BRepClass3d_OuterShell(topods.Solid(self.get_occt_shape()))
        return Shell(occt_outer_shell)

#--------------------------------------------------------------------------------------------------
    def internal_boundaries(self):
        """
        Returns the internal boundary (= Shells) of this Cell.
        """
        
        # Identify the external boundary
        external_boundary = self.external_boundary()

        # Identify the boundaries which are not the external boundary
        shells = self.downward_navigation(TopologyTypes.SHELL)

        internal_boundaries = []

        for shell in shells:

             if not shell.is_same(external_boundary):
                internal_boundaries.append(shell)
        
        return internal_boundaries

#--------------------------------------------------------------------------------------------------
    def is_manifold(self, host_topology: Topology):
        """
        Returns True, if this Cell is a manifold, otherwise a False.
        """
        
        # Create a Shell object representing the external boundary
        external_boundary = self.external_boundary()

        # Create list of external boundary faces (List[Topology])
        external_boundary_faces = external_boundary.faces()
        
        # Get the faces of the current Cell object
        cell_faces = self.faces()
        
        if len(cell_faces) > len(external_boundary_faces):
            return False
        
        # Get edges of bounding shell object
        edges = external_boundary.edges()
        
        for edge in edges:

            # Get adjacent faces 
            edge_faces = EdgeUtility.adjacent_faces(edge, external_boundary)
            
            if len(edge_faces) != 2:
                return False
        
        return True

#--------------------------------------------------------------------------------------------------
    def get_occt_shape(self) -> TopoDS_Shape:
        """
        Returns the underlying OCCT shape.
        """

        return self.get_occt_solid()

#--------------------------------------------------------------------------------------------------
    def get_occt_solid(self) -> TopoDS_Solid:
        """
        Returns the underlying OCCT solid.
        """
        
        if self.base_shape_solid.IsNull():
            raise RuntimeError("A null Solid is encountered.")

        return self.base_shape_solid

#--------------------------------------------------------------------------------------------------
    def set_occt_shape(self, occt_shape: TopoDS_Shape):
        """
        Sets the underlying OCCT shape.
        """
        
        self.set_occt_solid(topods.Solid(occt_shape))

#--------------------------------------------------------------------------------------------------
    def set_occt_solid(self, occt_solid: TopoDS_Solid):
        """
        Sets the underlying OCCT solid.
        """

        self.base_shape_solid = occt_solid

#--------------------------------------------------------------------------------------------------
    def geometry(self) -> List[Geom_Geometry]:
        """
        Creates a geometry from this Cell.
        """

        occt_geometries = []
        
        # Returns a list of faces
        faces: List[Face] = []
        faces = self.faces()
        
        # Get Geom_Surface for the OCC surface.
        for face in faces:
            occt_geometries.append(face.surface())

        return occt_geometries

#--------------------------------------------------------------------------------------------------
    def get_type_as_string(self):
        """
        Returns the type of this Cell as a string.
        """

        return 'Cell'

#--------------------------------------------------------------------------------------------------
    def occt_shape_fix(self, occt_input_solid: TopoDS_Solid):
        """
        Fixes the input OCCT solid.
        """
        
        occt_solid_fix = ShapeFix_Solid(occt_input_solid)
        occt_solid_fix.Perform()
        return topods.Solid(occt_solid_fix.Solid())