from TestCases.CustomTests.Test0_Coverages import run_coverage_test
from TestCases.CustomTests.Test1_BuildEdges import run_build_edges

from TestCases.WrapperTests.t01Vertex import test_01vertex

was_success = test_01vertex()
if was_success:
    print('Test case was success!')
else:
    print('Test case failed!')