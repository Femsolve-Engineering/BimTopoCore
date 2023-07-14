from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Display.SimpleGui import init_display

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