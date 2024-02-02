from typing import List

from TestCases.CustomTests.Test0_Coverages import run_coverage_test
from TestCases.CustomTests.Test1_BuildEdges import run_build_edges

# BimTopoCore Phase-1: Validation tests
from TestCases.WrapperTests.t01Vertex import test_01vertex
from TestCases.WrapperTests.t02Edge import test_02edge
from TestCases.WrapperTests.t03Wire import test_03wire
from TestCases.WrapperTests.t04Face import test_04face
from TestCases.WrapperTests.t05Shell import test_05shell
from TestCases.WrapperTests.t06Cell import test_06cell
from TestCases.WrapperTests.t10Dictionary import test_10dictionary
from TestCases.WrapperTests.t15Aperture import test_15aperture

def test_runner(functors: list) -> dict[str, bool]:
    """
    Returns the outcome of all the tests.
    """

    test_to_outcome: dict[str, bool] = {}

    for functor in functors:
        if functor():
            test_to_outcome[functor.__name__] = True
        else:
            test_to_outcome[functor.__name__] = False

    return test_to_outcome
    
# Test collections
test_to_outcome = test_runner([
    # test_01vertex,
    # test_02edge,
    # test_03wire,
    # test_04face,
    # test_05shell,
    test_06cell,
    # test_10dictionary,
    # test_15aperture
])

print('\n------------------------- Test Summary -------------------------\n')
for test_name in test_to_outcome.keys():
    print(f"Name: {test_name}, Outcome: {'PASS' if test_to_outcome[test_name] else 'FAIL'}")
print('\n------------------------- Test(s) Finished -------------------------')
