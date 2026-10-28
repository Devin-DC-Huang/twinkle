"""Real backward/optimizer updates must match complete large-batch windows."""
import contextlib
import inspect
import unittest

import torch

from twinkle.model.optimizer_group import BaseOptimizerGroup
from twinkle.model.transformers.transformers import OptimizerGroup, TransformersModel


class TinyLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False)
        self.weight.data.fill_(0.25)
        self.no_sync_calls = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


class GradientAccumulationRegression(unittest.TestCase):
    def test_microbatches_match_large_batches_and_scheduler(self):
        for ga in (1, 2, 4):
            with self.subTest(ga=ga):
                model, reference = TinyLinear(), TinyLinear()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
                ref_scheduler = torch.optim.lr_scheduler.StepLR(ref_optimizer, step_size=1, gamma=0.9)
                group = OptimizerGroup(optimizer=optimizer, lr_scheduler=scheduler,
                                       gradient_accumulation_steps=ga)
                wrapper = object.__new__(TransformersModel)
                wrapper.__dict__['model'] = model
                wrapper.optimizer_group = {'': group}
                wrapper.device_mesh = None
                wrapper.mixed_precision = 'bf16'
                updates = []
                for micro in range(1, 3 * ga + 1):
                    before = model.weight.detach().clone()
                    x = torch.tensor([[float(micro % 3 + 1)]])
                    group.train_status.loss_value = ((model(x) - 1) ** 2).mean() / ga
                    inspect.unwrap(TransformersModel.backward)(wrapper, gradient_accumulation_steps=ga)
                    inspect.unwrap(TransformersModel.step)(wrapper, gradient_accumulation_steps=ga)
                    inspect.unwrap(TransformersModel.zero_grad)(wrapper, gradient_accumulation_steps=ga)
                    inspect.unwrap(TransformersModel.lr_step)(wrapper, gradient_accumulation_steps=ga)
                    if micro % ga == 0:
                        xs = torch.tensor([[float(i % 3 + 1)] for i in range(micro - ga + 1, micro + 1)])
                        ((reference(xs) - 1) ** 2).mean().backward()
                        ref_optimizer.step()
                        ref_optimizer.zero_grad()
                        ref_scheduler.step()
                        updates.append(micro)
                        torch.testing.assert_close(model.weight, reference.weight)
                        self.assertIsNone(model.weight.grad)
                    else:
                        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
                        self.assertIsNotNone(model.weight.grad)
                    self.assertEqual(scheduler.last_epoch, micro // ga)
                self.assertEqual(updates, [ga, 2 * ga, 3 * ga])
                self.assertEqual(group.cur_step, 3 * ga)
                self.assertEqual(model.no_sync_calls, 3 * (ga - 1))

    def test_no_update_before_backward_and_boundary_resume(self):
        for ga in (1, 2, 4):
            group = BaseOptimizerGroup(gradient_accumulation_steps=ga)
            self.assertFalse(group.do_grad_sync())
            group.cur_step = 2 * ga
            boundaries = []
            for _ in range(ga):
                group.cur_step += 1
                if group.do_grad_sync():
                    boundaries.append(group.cur_step)
            self.assertEqual(boundaries, [3 * ga])

    def test_metrics_report_completed_updates(self):
        for ga in (1, 2, 4):
            group = OptimizerGroup(gradient_accumulation_steps=ga, cur_step=ga)
            group.train_status.inputs = {}
            group.train_status.outputs = {'loss': torch.tensor(1.0)}
            metrics = group.calculate_metrics(True)
            self.assertEqual(metrics['iters'], 1)


if __name__ == '__main__':
    unittest.main()
