

from typing import Tuple
from typing import List
from xmlrpc.client import boolean

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_CompSolid, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, topods
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopTools import TopTools_MapOfShape, TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Surface, Geom_Geometry
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_EdgeDone
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NoFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NotPlanar
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_CurveProjectionFailed
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_ParametersOutOfRange
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties, VolumeProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeFace
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError
from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder
from OCC.Core.ShapeAnalysis import ShapeAnalysis_ShapeContents
from OCC.Core.IntTools import IntTools_Context
from OCC.Core.BOPTools import BOPTools_AlgoTools

# BimTopoCore
from Core.Vertex import Vertex
from Core.Edge import Edge
from Core.Wire import Wire
from Core.Face import Face
from Core.Cell import Cell
from Core.Shell import Shell
from Core.Cluster import Cluster

from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.Factories.AllFactories import CellComplexFactory

class CellComplex(Topology):
    
    def __init__(self, occt_compSolid: TopoDS_CompSolid, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_solid (TopoDS_Solid): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        instance_topology = super().__init__(occt_compSolid, TopologyTypes.CELLCOMPLEX)
        self.base_shape_compSolid = occt_compSolid
        self.register_factory(self.get_class_guid(), CellComplexFactory())

        # Register the instances
        Topology.topology_to_subshape[instance_topology] = self
        Topology.subshape_to_topology[self] = instance_topology

    def is_container_type(self) -> bool:
        """
        Determines if this topology is container type.
        """
        return True
    
    def get_type(self) -> TopologyTypes:
        """
        Returns:
            TopologyTypes: Internal definition for types.
        """
        return TopologyTypes.CELLCOMPLEX

#--------------------------------------------------------------------------------------------------
    def cells(self) -> List[Cell]:
        """
        Returns the Cell contituents to this CellComplex
        """

        return self.downward_navigation(TopologyTypes.CELL)

#--------------------------------------------------------------------------------------------------
    def faces(self) -> List[Face]:
        """
        Returns the Face contituents to this CellComplex
        """

        return self.downward_navigation(TopologyTypes.FACE)

#--------------------------------------------------------------------------------------------------
    def shells(self) -> List[Shell]:
        """
        Returns the Shell contituents to this CellComplex
        """

        return self.downward_navigation(TopologyTypes.SHELL)

#--------------------------------------------------------------------------------------------------
    def edges(self) -> List[Edge]:
        """
        Returns the Edge contituents to this CellComplex
        """

        return self.downward_navigation(TopologyTypes.EDGE)

#--------------------------------------------------------------------------------------------------
    def vertices(self) -> List[Vertex]:
        """
        Find vertices making up the base shape.
        """

        return self.downward_navigation(TopologyTypes.VERTEX)

#--------------------------------------------------------------------------------------------------
    def wires(self) -> List[Wire]:
        """
        Returns the Wire contituents to this CellComplex
        """

        return self.downward_navigation(TopologyTypes.WIRE)

#--------------------------------------------------------------------------------------------------
    def by_cells(self, cells: List[Cell], copy_attributes: boolean) -> 'CellComplex':
        """
        Creates a CellComplex by a set of Cells.
        """

        # ByOcctSolids does the actual construction. This method extracts the OCCT structures from the input
		# and wrap the output in a Topologic class.

        occt_shapes = TopTools_ListOfShape()

        for cell in cells:
            occt_shapes.Append(cell.get_occt_shape())

        # Create CellComplex from Solid and Copy it 
        occt_comp_solid = self.by_occt_solids(occt_shapes)
        cell_complex = CellComplex(occt_comp_solid)
        copy_cell_complex = cell_complex.deep_copy()

        if copy_attributes:

            cells_as_topologies = []

            for cell in cells:
                cells_as_topologies.append(cell)

                # AttributeManager not implemented!

                # C++ code:
                # AttributeManager.get_instance().deep_copy_attributes(cell.get_occt_solid(), copy_cell_complex.get_occt_compSolid())

                # Python code:
                # instance = AttributeManager.get_instance()
                # instance.deep_copy_attributes(cell.get_occt_solid(), copy_cell_complex.get_occt_compSolid())

        # Topology.DeepCopyAttributesFrom not implemented!
            
        # C++ code:
        # CellComplex::Ptr pCopyCellComplex = TopologicalQuery::Downcast<CellComplex>(pCellComplex->DeepCopyAttributesFrom(cellsAsTopologies));

        # Python code:
        # ...

        return copy_cell_complex

#--------------------------------------------------------------------------------------------------
    def by_occt_solids(self, occt_solids: TopTools_ListOfShape) -> TopoDS_CompSolid:
        """
        Creates an OCCT CompSolid by a set of OCCT solids
        """

        occt_builder = BRep_Builder()
        occt_compSolid = topods.CompSolid()

        if occt_solids.IsEmpty():
            occt_builder.MakeCompSolid(occt_compSolid)
            return occt_compSolid

        occt_builder.MakeCompSolid(occt_compSolid)

        occt_solid_iterator = occt_solids.begin()
        p_cellComplex = None

        if occt_solids.Size() == 1:
            try:
                occt_builder.Add(occt_compSolid, occt_solid_iterator.Value())
            except:
                occt_compSolid = topods.CompSolid()
                occt_builder.MakeCompSolid(occt_compSolid)
                return occt_compSolid

            p_cellComplex = CellComplex(occt_compSolid)

        else:
            # Merge the first cell with the rest.
            first_topology = Topology.by_occt_shape(occt_solid_iterator.Value(), "")

            topologies = []

            occt_solid_iterator = occt_solids.begin()
            next_solid = next(occt_solid_iterator)  # Move iterator to the next element
            last_solid = occt_solids.end()

            if next_solid != last_solid:
                next_solid = next(occt_solid_iterator)  # Move iterator to the next element 
                topologies.append(Topology.by_occt_shape(next_solid, ""))

            other_cells_as_cluster = Cluster.by_topologies(topologies)
            p_merge_topology = first_topology.Merge(other_cells_as_cluster)

            if p_merge_topology.get_type() != TopologyTypes.CELLCOMPLEX:
                occt_builder.MakeCompSolid(occt_compSolid)
                return occt_compSolid

            # C++ code:
            # pCellComplex = TopologicalQuery::Downcast<CellComplex>(pMergeTopology);

            # Python code:
            # ...

        # Deep copy the CellComplex
        p_copy_cellComplex = p_cellComplex.deep_copy_shape()
        return p_copy_cellComplex.get_occt_compSolid()

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_faces(faces: List[Face], tolerance: float, copy_attributes: boolean) -> 'CellComplex':
        """
        Creates a CellComplex from a space enclosed by a set of Faces.
        """

        occt_maker_volume = BRepAlgoAPI_Fuse()

        occt_shapes = TopTools_ListOfShape()

        for face in faces:
            occt_shapes.Append(face.get_occt_shape())

        is_parallel = False
        does_intersection = True

        occt_maker_volume.SetArguments(occt_shapes)
        occt_maker_volume.SetRunParallel(is_parallel)
        occt_maker_volume.SetOperation(BRepAlgoAPI_Fuse)  # or any other BRepAlgoAPI operation
        occt_maker_volume.SetFuzzyValue(tolerance)
        occt_maker_volume.Perform()

        if occt_maker_volume.HasWarnings():
            raise RuntimeError("Warnings.")

        if occt_maker_volume.HasErrors():
            cellComplex = None
            return cellComplex

        occt_result = occt_maker_volume.Shape()

        cells: List[Cell] = []

        if occt_result.ShapeType() == TopAbs_SOLID:
            cells.append(Cell(topods.Solid(occt_result)))

        elif occt_result.ShapeType() == TopAbs_COMPOUND:
            occt_shapes = TopTools_MapOfShape()
            
            for occt_explorer in TopExp_Explorer(occt_result, TopAbs_SOLID):
                occt_current = occt_explorer.Current()
                
                if not occt_shapes.Contains(occt_current):
                    occt_shapes.Add(occt_current)
                    cells.append(Cell(topods.Solid(occt_current)))

        cellComplex = CellComplex.by_cells(cells, False)  # Since these are new Cells, no need to copy dictionaries

        occt_fixed_compSolid = CellComplex.occt_shape_fix(cellComplex.get_occt_compSolid())

        fixed_cellComplex = CellComplex(occt_fixed_compSolid)

        # TopologicalQuery.Downcast not implemented 
        # copy_fixed_cellComplex = TopologicalQuery.Downcast(fixedCellComplex.DeepCopy())

        if copy_attributes:

            faces_as_topologies: List[Topology] = []

            for face in faces:

                faces_as_topologies.append(face)

                # AttributeManager.get_instance not implemented!
                # instance = AttributeManager.get_instance()
                # instance.deep_copy_attributes(face.get_occt_face(), copy_fixed_cellComplex.get_occt_compSolid)
            
            # Topology.deep_copy_attributes_from not implemented
            # copy_fixed_cellComplex.deep_copy_attributes_from(faces_as_topologies)

        # return copy_fixed_cellComplex

#--------------------------------------------------------------------------------------------------
    def external_boundary(self) -> 'Cell':
        """
        Returns the external boundary (= Cell) of a CellComplex.
        """

        # Get the Cells
        occt_cells_builders_arguments = TopTools_ListOfShape()
        cells: List[Cell] = []
        cells = self.cells()

        for cell in cells:
            occt_cells_builders_arguments.Append(cell.get_occt_shape())

        # Do a Union
        occt_cells_builder = BOPAlgo_CellsBuilder()
        occt_cells_builder.SetArguments(occt_cells_builders_arguments)
        occt_cells_builder.Perform()

        if occt_cells_builder.HasErrors():
            # errorStream = StringIO()
            # occt_cells_builder.DumpErrors(errorStream)
            # raise RuntimeError(errorStream.getvalue())
            raise RuntimeError('BOPAlgo_CellsBuilder has errors!')

        occt_list_to_take = TopTools_ListOfShape()
        occt_list_to_avoid = TopTools_ListOfShape()
        
        for occt_shape_iterator in TopTools_ListIteratorOfListOfShape(occt_cells_builders_arguments):
            occt_list_to_take.Clear()
            occt_list_to_take.Append(occt_shape_iterator.Value())
            occt_cells_builder.AddToResult(occt_list_to_take, occt_list_to_avoid, 1, True)

        # A cell complex is a contiguous shape, so there can be at maximum only one envelope cell.
        occt_envelope_shape = occt_cells_builder.Shape()
        occt_shape_analysis = ShapeAnalysis_ShapeContents()
        occt_shape_analysis.Perform(occt_envelope_shape)

        number_of_solids = occt_shape_analysis.NbSharedSolids()
        ssErrorMessage = "There can be only 0 or 1 envelope cell, but this cell complex has " + str(number_of_solids) + " cells."
        assert number_of_solids < 2, ssErrorMessage

        # Return the first Cell
    
        explorer = TopExp_Explorer(occt_envelope_shape, TopAbs_SOLID)

        while explorer.More():

            current_solid = topods.Solid(explorer.Current())
            
            # Perform operations:
            p_cell = Cell(current_solid)

            # TopologicalQuery::Downcast not implemented!

            # C++ code:
            # Cell::Ptr pCellCopy = TopologicalQuery::Downcast<TopologicCore::Cell>(pCell->DeepCopy());
            # return p_cell_copy

            explorer.Next()

        return None

#--------------------------------------------------------------------------------------------------
    def internal_boundaries(self) -> List[Face]:
        """
        Returns the internal boundary (= Faces) of a CellComplex.
        """

        internal_faces: List[Face] = []
        
        # Compute the envelope Cell
        envelope_cell = self.external_boundary()

        # Get the envelope Faces
        envelope_faces: List[Face] = []
        envelope_faces = envelope_cell.faces()

        # Get the original Faces
        faces: List[Face] = []
        faces = self.faces()

        # An internal Face can be found in the original Face list, but not in the envelope Face list.
        occt_intTools_context = IntTools_Context()
        
        for face in faces:

            is_envelope_face = False
            
            for envelope_face in envelope_faces:

                if BOPTools_AlgoTools.AreFacesSameDomain(face.get_occt_face(), envelope_face.get_occt_face(), occt_intTools_context):
                    is_envelope_face = True
                    break

            if not is_envelope_face:
                internal_faces.append(face)

        return internal_faces

#--------------------------------------------------------------------------------------------------
    def is_manifold(self):
        """
        Returns True, if this CellComplex is a manifold, otherwise a False.
        """

        # Not implemented yet
        return False

#--------------------------------------------------------------------------------------------------
    def non_manifold_faces(self) -> List[Face]:
        """
        Returns the non-manifold Faces of this CellComplex.
        """

        faces: List[Face] = []
        faces = self.faces()

        non_manifold_faces: List[Face] = []

        for face in faces:

            cellComplex = CellComplex(self.get_occt_compSolid())

            if not face.is_manifold(cellComplex):
                non_manifold_faces.append(face)

        return non_manifold_faces

#--------------------------------------------------------------------------------------------------
    def get_occt_shape(self) -> TopoDS_Shape:
        """
        Returns the underlying OCCT shape.
        """

        return self.get_occt_compSolid()

#--------------------------------------------------------------------------------------------------
    def get_occt_compSolid(self):
        """
        Returns the underlying OCCT compSolid.
        """
        
        if self.base_shape_compSolid.IsNull():
            raise RuntimeError("A null CellComplex is encountered.")

        return self.base_shape_compSolid

#--------------------------------------------------------------------------------------------------
    def set_occt_shape(self, occt_shape: TopoDS_Shape):
        """
        Sets the underlying OCCT shape.
        """

        self.set_occt_compSolid(topods.CompSolid(occt_shape))

#--------------------------------------------------------------------------------------------------
    def geometry(self) -> List[Geom_Geometry]:
        """
        Creates a geometry from this CellComplex.
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
    def get_type_as_string(self) -> str:
        """
        Returns the type of this CellComplex as a string.
        """

        return 'CellComplex'

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def occt_shape_fix(occt_input_compSolid: TopoDS_CompSolid) -> TopoDS_CompSolid:
        """
        Fixes the input OCCT compSolid.
        """
        
        occt_compSolid_fix = ShapeFix_Shape(occt_input_compSolid)
        occt_compSolid_fix.Perform()

        return topods.CompSolid(occt_compSolid_fix.Shape())
#--------------------------------------------------------------------------------------------------
    def set_occt_compSolid(self, occt_compSolid: TopoDS_CompSolid):
        """
        Sets the underlying OCCT compSolid.
        """
        
        self.base_shape_compSolid = occt_compSolid


#--------------------------------------------------------------------------------------------------
    def center_of_mass(self) -> 'Vertex':
        """
        Returns the Vertex at the center of mass of this OCCT compSolid.
        """

        occt_vertex = CellComplex.make_pnt_at_center_of_mass(self.get_occt_compSolid())
        vertex = Vertex(occt_vertex)
        
        return vertex

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def make_pnt_at_center_of_mass(occt_compSolid: TopoDS_CompSolid) -> TopoDS_Vertex:
        """
        Returns the OCCT vertex at the center of mass of this OCCT compSolid.
        """
        
        occt_shape_properties = GProp_GProps()
        VolumeProperties(occt_compSolid, occt_shape_properties)

        center_of_mass_point = occt_shape_properties.CenterOfMass()
        return BRepBuilderAPI_MakeVertex(center_of_mass_point).Vertex()
