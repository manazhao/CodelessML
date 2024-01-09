import unittest

from codeless_ml.common import callable_pb2
import codeless_ml.common.global_variable as gv


def _test_callable():
  return "hello"


def _get_sum_function(a: int, b: int):

  def _sum():
    return a + b

  return _sum


class GlobalVariableTest(unittest.TestCase):

  def testRegisterCallable(self):
    name = "/callable/test"
    self.assertFalse(gv.GLOBAL_VARIABLE_REPOSITORY.callable_exists(name))
    gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(name, _test_callable)
    self.assertTrue(gv.GLOBAL_VARIABLE_REPOSITORY.callable_exists(name))
    callable_register = callable_pb2.CallableRegistry()
    callable_register.function_name = name
    self.assertEqual(
        gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(callable_register)(),
        "hello")
    self.assertFalse(
        gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(name, _test_callable))

  def testClosureBasedCallable(self):
    name = "/callable/closure"
    self.assertFalse(gv.GLOBAL_VARIABLE_REPOSITORY.callable_exists(name))
    gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(name, _get_sum_function)
    self.assertTrue(gv.GLOBAL_VARIABLE_REPOSITORY.callable_exists(name))
    callable_register = callable_pb2.CallableRegistry()
    callable_register.closure.function_name = name
    callable_register.closure.argument["a"].int32_value = 10
    callable_register.closure.argument["b"].float_value = 20
    self.assertEqual(
        gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(callable_register)(),
        30)

  def testRegisterVariable(self):
    name = "/variable/test"
    variable = "hello"
    self.assertFalse(gv.GLOBAL_VARIABLE_REPOSITORY.variable_exists(name))
    gv.GLOBAL_VARIABLE_REPOSITORY.register_variable(name, variable)
    self.assertTrue(gv.GLOBAL_VARIABLE_REPOSITORY.variable_exists(name))
    self.assertEqual(
        gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_variable(name), variable)
    self.assertFalse(
        gv.GLOBAL_VARIABLE_REPOSITORY.register_variable(name, variable))


if __name__ == '__main__':
  unittest.main()
