# test_attention.py

import unittest
import numpy as np
from attention_numpy import attention_numpy
import attention


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.n = 128
        self.d = 64
        self.dk = 64

        self.Q32 = np.random.rand(self.n, self.dk).astype(np.float32)
        self.K32 = np.random.rand(self.n, self.dk).astype(np.float32)
        self.V32 = np.random.rand(self.n, self.d).astype(np.float32)

        self.Q64 = self.Q32.astype(np.float64)
        self.K64 = self.K32.astype(np.float64)
        self.V64 = self.V32.astype(np.float64)

    def test_v0_float32(self):
        out_ref = attention_numpy(self.Q32, self.K32, self.V32)
        out_test = attention_numpy(self.Q32, self.K32, self.V32)
        np.testing.assert_allclose(out_ref, out_test, atol=1e-4)

    def test_v1_float32(self):
        out_ref = attention_numpy(self.Q32, self.K32, self.V32)
        out_test = attention.attention(self.Q32, self.K32, self.V32, version=1)
        np.testing.assert_allclose(out_ref, out_test, atol=1e-4)

    def test_v1_float64(self):
        out_ref = attention_numpy(self.Q64, self.K64, self.V64)
        out_test = attention.attention(self.Q64, self.K64, self.V64, version=1)
        np.testing.assert_allclose(out_ref, out_test, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
