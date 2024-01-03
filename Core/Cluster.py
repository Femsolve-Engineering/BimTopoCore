

from typing import Tuple
from typing import List
from xmlrpc.client import boolean

# OCC
from OCC.Core.Standard import Standard_Failure
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_CompSolid, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, TopoDS_Compound, topods
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPOUND
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopTools import toptools, TopTools_MapOfShape, TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Core.TopExp import TopExp_Explorer
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
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeFace
from OCC.Core.BRepCheck import BRepCheck_Wire, BRepCheck_NoError
from OCC.Core.BOPAlgo import BOPAlgo_CellsBuilder
from OCC.Core.ShapeAnalysis import ShapeAnalysis_ShapeContents
from OCC.Core.IntTools import IntTools_Context
from OCC.Core.BOPTools import BOPTools_AlgoTools

# BimTopoCore
from Core.CellComplex import CellComplex

from Core.Topology import Topology
from Core.TopologyConstants import TopologyTypes
from Core.Factories.AllFactories import ClusterFactory

from Core.AttributeManager import AttributeManager

class Cluster(Topology):
    
    def __init__(self, occt_compound: TopoDS_Compound, guid=""):
        """Constructor saves shape and processes GUID.

        Args:
            occt_solid (TopoDS_Solid): base_shape
            guid (str, optional): Shape specific guid. Defaults to "".
        """

        # This constructor does not initialise the compound with MakeCompound.
        instance_topology = super().__init__(occt_compound, TopologyTypes.CLUSTER)
        self.base_shape_compound = occt_compound
        self.register_factory(self.get_class_guid(), ClusterFactory())

        # Register the instances
        Topology.topology_to_subshape[instance_topology] = self
        Topology.subshape_to_topology[self] = instance_topology

        self.occt_builder = BRep_Builder() 


#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_topologies(topologies: list[Topology], copy_attributes: boolean) -> 'Cluster':
        
        if not topologies:
            return None

        occt_compound = TopoDS_Compound()
        topods.Compound(occt_compound)
        occt_builder = BRep_Builder()
        occt_builder.MakeCompound(occt_compound)

        cluster = Cluster(occt_compound)

        for topology in topologies:
            cluster.add_topology(topology)

        # Deep copy
        copy_cluster = cluster.deep_copy()

        # Register shapes in Topology.topology_to_subshape and Topology.subshape_to_topology
        # ...

        # dynamic pointer cast -> convert TopoDS_Compound to Cluster
        # ...

        # Transfer the attributes
        if copy_attributes:

            for topology in topologies:
                pass

                # AttributeManager not implemented!

                # C++ code:
                # AttributeManager::GetInstance().DeepCopyAttributes(kpTopology->GetOcctShape(), pCopyCluster->GetOcctCompound());

                # Python code:
                instance = AttributeManager.get_instance()
                instance.deep_copy_attributes(topology.get_occt_shape(), copy_cluster.get_occt_compound())

            # Topology.DeepCopyAttributesFrom not implemented!

            copy_cluster.deep_copy_attributes_from(topologies)

        return copy_cluster

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def by_occt_topologies(occt_map_of_shapes: TopTools_MapOfShape) -> TopoDS_Compound:
        
        occt_compound = topods.Compound()
        occt_builder = BRep_Builder()
        occt_builder.MakeCompound(occt_compound)

        occt_shape_iterator = toptools.TopTools_MapIteratorOfMapOfShape(occt_map_of_shapes)

        while occt_shape_iterator.More():

            try:
                occt_builder.Add(occt_compound, occt_shape_iterator.Value())

            except:
                raise RuntimeError("Error making an OCCT compound.")

            occt_shape_iterator.Next()

        return occt_compound

#--------------------------------------------------------------------------------------------------
    def add_topology(self, topology: Topology) -> boolean:
        
        returnValue = True

        try:
            self.occt_builder.Add(self.get_occt_shape(), topology.get_occt_shape())

        except:
            returnValue = False

        return returnValue

#--------------------------------------------------------------------------------------------------
    def remove_topology(self, topology: Topology):
        
        try:
            self.occt_builder.Remove(self.get_occt_shape(), topology.get_occt_shape())
            return True
        
        except:
            return False

#--------------------------------------------------------------------------------------------------
    def get_occt_shape(self):
        
        return self.get_occt_compound()

#--------------------------------------------------------------------------------------------------
    def get_occt_compound(self):
        
        if self.base_shape_compound.IsNull():
            raise RuntimeError("A null Cluster is encountered.")

        return self.base_shape_compound

#--------------------------------------------------------------------------------------------------
    def set_occt_shape(self, occt_shape: TopoDS_Shape):
        
        self.set_occt_compound(topods.Compound(occt_shape))

#--------------------------------------------------------------------------------------------------
    def set_occt_compound(self, occt_compound: TopoDS_Compound):

        self.base_shape_compound = occt_compound

#--------------------------------------------------------------------------------------------------
    def geometry(self) -> Geom_Geometry:

        raise RuntimeError("No implementation for Cluster entity.")

#--------------------------------------------------------------------------------------------------
    def shells(self) -> List['Shell']:
        
        return self.downward_navigation(TopologyTypes.SHELL)

#--------------------------------------------------------------------------------------------------
    def edges(self) -> List['Edge']:
        
        return self.downward_navigation(TopologyTypes.EDGE)

#--------------------------------------------------------------------------------------------------
    def faces(self) -> List['Face']:
        
        return self.downward_navigation(TopologyTypes.FACE)

#--------------------------------------------------------------------------------------------------
    def vertices(self) -> List['Vertex']:
        
        return self.downward_navigation(TopologyTypes.VERTEX)

#--------------------------------------------------------------------------------------------------
    def wires(self) -> List['Wire']:
        
        return self.downward_navigation(TopologyTypes.WIRE)

#--------------------------------------------------------------------------------------------------
    def cells(self) -> List['Cell']:
        
        return self.downward_navigation(TopologyTypes.CELL)

#--------------------------------------------------------------------------------------------------
    def cellComplexes(self) -> List[CellComplex]:
        
        return self.downward_navigation(TopologyTypes.CELLCOMPLEX)

#--------------------------------------------------------------------------------------------------
    def is_inside(self, topology: Topology) -> boolean:
        
        occt_added_shape = topology.get_occt_shape()
        # occt_shapes = TopTools_MapOfShape() # Not needed

        occt_explorer = TopExp_Explorer(self.get_occt_shape(), occt_added_shape.get_shape_type())

        while occt_explorer.More():

            occt_current = occt_explorer.Current()

            if occt_added_shape.IsSame(occt_current):
                return True
            
            occt_explorer.Next()

        return False

#--------------------------------------------------------------------------------------------------
    def center_of_mass(self) -> 'Vertex':

        from Core.Vertex import Vertex
        from Core.Cluster import Cluster
        
        occt_vertex = Cluster.make_pnt_at_center_of_mass(self.get_occt_compound())
        vertex = Vertex(occt_vertex)
        
        return vertex

#--------------------------------------------------------------------------------------------------
    @staticmethod
    def make_pnt_at_center_of_mass(occt_compound: TopoDS_Compound) -> TopoDS_Vertex:
        
        # Compute the average of the centers of mass.
        occt_subTopologies = TopTools_ListOfShape()

        # Topology.subTopologies(occt_compound, occt_subTopologies)
        occt_subTopologies = Topology.subTopologies(occt_compound)

        if occt_subTopologies.IsEmpty():
            raise RuntimeError("The input Cluster is empty.")

        size = float(occt_subTopologies.Size())
        occt_centroid_sum = gp_Pnt()

        occt_iterator = TopTools_ListIteratorOfListOfShape(occt_subTopologies)

        while occt_iterator.More():

            subTopology = Topology.by_occt_shape(occt_iterator.Value(), "")
            subTopology_center_of_mass = subTopology.center_of_mass()
            occt_subTopology_center_of_mass = subTopology_center_of_mass.Point().Pnt()

            occt_centroid_sum.SetX(occt_centroid_sum.X() + occt_subTopology_center_of_mass.X())
            occt_centroid_sum.SetY(occt_centroid_sum.Y() + occt_subTopology_center_of_mass.Y())
            occt_centroid_sum.SetZ(occt_centroid_sum.Z() + occt_subTopology_center_of_mass.Z())

            occt_iterator.Next()

        occt_centroid = gp_Pnt(
            occt_centroid_sum.X() / size,
            occt_centroid_sum.Y() / size,
            occt_centroid_sum.Z() / size
        )

        return BRepBuilderAPI_MakeVertex(occt_centroid).Vertex()

#--------------------------------------------------------------------------------------------------
    def is_manifold(self) -> bool:
        
        raise RuntimeError("Not implemented yet")

#--------------------------------------------------------------------------------------------------
    def get_type_as_string(self) -> str:
        
        return 'Cluster'

#--------------------------------------------------------------------------------------------------