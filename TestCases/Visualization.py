from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display

from Core.Topology import Topology

def show_topology(topology: 'Topology', skip_visualization=False) -> bool:

    if skip_visualization:
        return True
    
    try: 
        display, start_display, add_menu, add_function_to_menu = init_display()
        display.DisplayShape(topology.get_occt_shape(), update=True)
        start_display()
        return True
    except Exception as ex:
        print(f'Exception occured: {ex}')
        return False
        

def run_box_visualization() -> bool:
    """Simple box visualizer test.

    Returns:
        bool: True - if all finished successfully
    """

    try: 
        display, start_display, add_menu, add_function_to_menu = init_display()
        my_box = BRepPrimAPI_MakeBox(10., 20., 30.).Shape()
        display.DisplayShape(my_box, update=True)
        start_display()
        return True
    except Exception as ex:
        print(f'Exception occured: {ex}')
        return False