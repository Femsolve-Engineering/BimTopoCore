from TestCases.Test0_Coverages import run_coverage_test
from TestCases.Test1_BuildEdges import run_build_edges

was_success = run_coverage_test()
if was_success:
    print('Test case was success!')
else:
    print('Test case failed!')