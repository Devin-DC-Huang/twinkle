import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from twinkle.model.transformers.transformers import TransformersModel


def test_resident_reset_removes_previous_templates_forward_hook():
    class ScalingTemplate:
        def __init__(self, scale):
            self.scale = scale

        def pre_forward_hook(self, model, args, kwargs):
            return (args[0] * self.scale,), kwargs

    def group(scale):
        return SimpleNamespace(template=ScalingTemplate(scale), _handler=None)

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    wrapper = object.__new__(TransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.model_id = 'local-reset-test'
    wrapper._model_wrapped = True
    wrapper.strategy = SimpleNamespace(unwrap_model=lambda value: value,
                                      load_full_state_dict=lambda value, state: value.load_state_dict(state))
    wrapper.optimizer_group = {'': group(3)}
    object.__setattr__(wrapper, '_construct_default_optimizer_group', lambda: group(1))
    wrapper.register_mm_forward_hook(wrapper.optimizer_group[''])
    assert model(torch.ones(1, 1)).item() == 3
    initial = {'weight': torch.ones(1, 1)}
    with patch('twinkle.model.transformers.transformers._read_hf_state_dict', return_value=initial):
        for _ in range(2):
            model.weight.grad = torch.ones_like(model.weight)
            wrapper.reload_initial_weights(reset_training_state=True)
            assert model(torch.ones(1, 1)).item() == 1
            assert model.weight.grad is None
            assert len(model._forward_pre_hooks) == 1


def test_blank_model_still_resets_training_state():
    wrapper = object.__new__(TransformersModel)
    wrapper.model_id = ''
    wrapper._model_wrapped = False
    previous = SimpleNamespace(template=None, _handler=None, optimizer=object())
    fresh = SimpleNamespace(template=None, _handler=None, optimizer=None)
    wrapper.optimizer_group = {'': previous}
    object.__setattr__(wrapper, '_construct_default_optimizer_group', lambda: fresh)
    wrapper.reload_initial_weights(reset_training_state=True)
    assert wrapper.optimizer_group == {'': fresh}
    assert wrapper.active_group == '' and fresh.adapter_name == ''


@pytest.mark.parametrize('reload_kwargs', [{}, {'reset_training_state': False}], ids=['default', 'explicit-false'])
def test_transformers_reload_preserves_training_progress_by_default(reload_kwargs):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(3)
    model.weight.grad = torch.ones_like(model.weight)
    previous = SimpleNamespace(optimizer=object(), cur_step=7)
    wrapper = object.__new__(TransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.model_id = 'local-reset-test'
    wrapper.optimizer_group = {'': previous}
    wrapper.strategy = SimpleNamespace(load_full_state_dict=lambda value, state: value.load_state_dict(state))
    initial = {'weight': torch.ones_like(model.weight)}
    with patch('twinkle.model.transformers.transformers._read_hf_state_dict', return_value=initial):
        wrapper.reload_initial_weights(**reload_kwargs)
    torch.testing.assert_close(model.weight, initial['weight'])
    assert model.weight.grad is None
    assert wrapper.optimizer_group[''] is previous and previous.cur_step == 7


def test_transformers_reload_propagates_checkpoint_failure():
    wrapper = object.__new__(TransformersModel)
    wrapper.model_id = 'unreadable-checkpoint'
    with patch('twinkle.model.transformers.transformers._read_hf_state_dict',
               side_effect=OSError('checkpoint unavailable')):
        with pytest.raises(OSError, match='checkpoint unavailable'):
            wrapper.reload_initial_weights()


@pytest.mark.parametrize('reload_kwargs', [{}, {'reset_training_state': False}, {'reset_training_state': True}],
                         ids=['default', 'explicit-false', 'reset'])
def test_megatron_reload_resets_optimizer_state_only_when_requested(reload_kwargs):
    # Exercise the repository method without importing the optional Megatron stack.
    path = Path('src/twinkle/model/megatron/megatron.py')
    model_class = next(node for node in ast.parse(path.read_text()).body
                       if isinstance(node, ast.ClassDef) and node.name == 'MegatronModel')
    method = next(node for node in model_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'reload_initial_weights')
    method.decorator_list = []
    namespace = {'_default_adapter_name': '', 'dist': SimpleNamespace(is_initialized=lambda: False)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    original_weights = {'weight': torch.ones_like(model.weight)}

    def load_weights(chunks, checkpoint, *, peft_format):
        assert checkpoint == 'initial-checkpoint' and peft_format is False
        for chunk in chunks:
            chunk.load_state_dict(original_weights)

    previous = SimpleNamespace(optimizer=object(), cur_step=7)
    fresh = SimpleNamespace(optimizer=None, cur_step=0)
    wrapper = SimpleNamespace(
        model=[model], _model_path='initial-checkpoint',
        optimizer_group={'': previous},
        strategy=SimpleNamespace(unwrap_model=lambda value: value,
                                 bridge=SimpleNamespace(load_weights=load_weights)),
        _is_model_ddp_wrapped=lambda: False,
        _construct_default_optimizer_group=lambda: fresh,
    )
    namespace['reload_initial_weights'](wrapper, **reload_kwargs)
    assert model.weight.grad is None
    torch.testing.assert_close(model.weight, original_weights['weight'])
    assert wrapper.optimizer_group[''] is (fresh if reload_kwargs.get('reset_training_state') else previous)
    if reload_kwargs.get('reset_training_state'):
        assert fresh.adapter_name == '' and fresh.optimizer is None and fresh.cur_step == 0
