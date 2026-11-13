import importlib.util
import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("math_verify"),"Math-Verify is installed in the Modal environment")
class MathVerifyTests(unittest.TestCase):
    def test_actual_equivalence_engine_on_known_mathematical_examples(self):
        from pa3.scoring import math_equivalent,parsed_math
        import math_verify
        def assertions():
            self.assertTrue(math_equivalent(r"\frac{1}{2}","0.5"))
            self.assertTrue(math_equivalent("2/4","1/2"))
            self.assertFalse(math_equivalent("2","3"))
            self.assertFalse(math_equivalent(None,"3"))
        if os.name=="nt":
            # Only these fixed, tiny test expressions bypass the library's Windows
            # multiprocessing timeout. Production scoring refuses Windows instead
            # of silently accepting parser failures; Modal uses bounded Linux calls.
            original_verify=math_verify.verify
            def parse_known(text):
                return math_verify.parse("$"+text+"$",parsing_timeout=None)
            def verify_known(a,b):
                return original_verify(a,b,timeout_seconds=None)
            with patch("pa3.scoring.parsed_math",side_effect=parse_known),patch("math_verify.verify",side_effect=verify_known):
                assertions()
        else:
            assertions()
        math_equivalent.cache_clear()
        parsed_math.cache_clear()


if __name__=="__main__":
    unittest.main()

