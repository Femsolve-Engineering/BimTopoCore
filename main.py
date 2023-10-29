from TestCases.CustomTests.Test0_Coverages import run_coverage_test
from TestCases.CustomTests.Test1_BuildEdges import run_build_edges

# BimTopoCore Phase-1: Validation tests
from TestCases.WrapperTests.t01Vertex import test_01vertex
from TestCases.WrapperTests.t02Edge import test_02edge
from TestCases.WrapperTests.t03Wire import test_03wire
from TestCases.WrapperTests.t04Face import test_04face
from TestCases.WrapperTests.t10Dictionary import test_10dictionary
from TestCases.WrapperTests.t15Aperture import test_15aperture

was_success = test_01vertex()
if was_success:
    print('Test case was success!')
else:
    print('Test case failed!')