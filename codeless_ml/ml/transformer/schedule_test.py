import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

import codeless_ml.common.global_variable as gv
import codeless_ml.ml.transformer.schedule as s

from codeless_ml.common import callable_pb2


class TestSchedule(unittest.TestCase):

    def test_retrieve_schedule(self):
        name = s.CREATE_SCHEDULE_REGESTRY_KEY
        self.assertTrue(gv.GLOBAL_VARIABLE_REPOSITORY.callable_exists(name))
        callable_register = callable_pb2.CallableRegistry()
        callable_register.closure.function_name = name
        callable_register.closure.argument["d_model"].int32_value = 32
        callable_register.closure.argument["warmup_steps"].int32_value = 128
        schedule = gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(
            callable_register)
        prev_rate = .0
        rate_has_peaked = False
        # the learning rate should first mono-increase and then mono-decrease
        # after reaching the peak.
        for step in range(1, 1280):
            rate = schedule(step)
            print(f"rate: {rate}")
            if rate < prev_rate and not rate_has_peaked:
                rate_has_peaked = True
            if not rate_has_peaked:
                self.assertGreater(rate, prev_rate)
            else:
                self.assertLess(rate, prev_rate)
            prev_rate = rate


if __name__ == '__main__':
    unittest.main()
