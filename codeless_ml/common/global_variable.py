from absl import logging
from typing import Any, Callable, List, Mapping, TypeVar

from codeless_ml.common import callable_pb2

ValueType = TypeVar("ValueType", None, int, float, str, bytes, List[int],
                    List[float], List[str], List[bytes])


def _extract_argument_value(
        argument_value: callable_pb2.ArgumentValue) -> ValueType:
    value_field = argument_value.WhichOneof("value")
    if value_field == "int32_value":
        return argument_value.int32_value
    elif value_field == "int64_value":
        return argument_value.int64_value
    elif value_field == "float_value":
        return argument_value.float_value
    elif value_field == "double_value":
        return argument_value.double_value
    elif value_field == "string_value":
        return argument_value.string_value
    elif value_field == "bytes_value":
        return argument_value.bytes_value
    elif value_field == "int32_list":
        return argument_value.int32_list.value
    elif value_field == "int64_list":
        return argument_value.int64_list.value
    elif value_field == "float_list":
        return argument_value.float_list.value
    elif value_field == "double_list":
        return argument_value.double_list.value
    elif value_field == "string_list":
        return argument_value.string_list.value
    elif value_field == "bytes_list":
        return argument_value.bytes_list.value
    else:
        return None


def _closure_argument_to_map(
        arguments: Mapping[str,
                           callable_pb2.ArgumentValue]) -> Mapping[str, Any]:
    argument_map = {}
    for name, argument_value in arguments.items():
        argument_map[name] = _extract_argument_value(argument_value)
    return argument_map


class GlobalVariableRepository(object):

    def __init__(self):
        self._callable_map = {}
        self._variable_map = {}

    def register_callable(self, name: str, fn: Callable[..., Any]) -> bool:
        if name in self._callable_map:
            logging.error("Entry name already exists: %s" % name)
            return False
        self._callable_map[name] = fn
        return True

    def callable_exists(self, name: str) -> bool:
        return self._callable_map.get(name)

    def retrieve_callable(
            self,
            registry: callable_pb2.CallableRegistry) -> Callable[..., Any]:
        function = registry.WhichOneof("function")
        if function == "function_name":
            return self._callable_map[registry.function_name]
        elif function == "closure":
            closure = self._callable_map[registry.closure.function_name]
            argument_map = _closure_argument_to_map(registry.closure.argument)
            return closure(**argument_map)
        else:
            logging.fatal("You must specify one way to retrieve the callable.")

    def register_variable(self, name: str, variable: Any) -> bool:
        if name in self._variable_map:
            logging.error("Entry name already exists: %s" % name)
            return False
        self._variable_map[name] = variable
        return True

    def variable_exists(self, name: str) -> bool:
        return self._variable_map.get(name)

    def retrieve_variable(self, name: str) -> Any:
        return self._variable_map[name]


GLOBAL_VARIABLE_REPOSITORY = GlobalVariableRepository()
