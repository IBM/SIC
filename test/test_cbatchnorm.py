from __future__ import division
import unittest
import torch
from torch.autograd import gradcheck


class test_modules(unittest.TestCase):

    def test_conditional_batchnorm(self):
        from modules.generators import ConditionalBatchNorm1d

        torch.manual_seed(1)

        num_classes = 10
        num_features = 20
        num_samples = 100

        cbn = ConditionalBatchNorm1d(num_features, num_classes)

        # cast parameters to double precision
        cbn.embed.weight.data = cbn.embed.weight.data.double()
        cbn.bn.running_mean = cbn.bn.running_mean.double()
        cbn.bn.running_var = cbn.bn.running_var.double()

        for _ in range(10):
            x = torch.randn(num_samples, num_features, dtype=torch.float64, requires_grad=True)
            labels = torch.randint(0, num_classes, (num_samples,))
            func = lambda x: cbn.forward((x, labels))
            self.assertTrue(gradcheck(func, (x,), eps=1e-4, atol=1e-3))

        self.assertEqual(x.shape, cbn((x, labels)).shape)


if __name__ == '__main__':
    unittest.main()
