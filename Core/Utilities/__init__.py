def handle_exception_decorator(method):
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as ex:
            host_class = args[0].__class__.__name__
            method_name = method.__qualname__
            print('\n****************** EXCEPTION CAUGHT ******************')
            print(f'{method_name} failed.\nException:{ex}')
            print('******************************************************\n')
            return None

    return wrapper
