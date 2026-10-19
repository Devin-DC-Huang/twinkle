# Copyright (c) ModelScope Contributors. All rights reserved.
"""Executable dependency architecture for every ``twinkle.server`` component."""
import grimp

_ROOT = 'twinkle.server'
_ROOT_COMPONENT = '__root__'

# Dependencies must point from a later layer to an earlier layer. Components in
# the same layer are independent and therefore may not import one another.
_LAYERS = (
    frozenset({'exceptions'}),
    frozenset({'config', 'middleware', 'resources', 'runtime', 'task_errors', 'utils', 'validation'}),
    frozenset({'telemetry'}),
    frozenset({'env_propagation', 'service_proxy', 'state'}),
    frozenset({'deployment', 'lifecycle', 'session_resource'}),
    frozenset({'task_queue'}),
    frozenset({'checkpoint', 'data_plane'}),
    frozenset({'gateway', 'model', 'processor'}),
    frozenset({'sampler', 'training'}),
    frozenset({'launcher'}),
    frozenset({'cli'}),
    frozenset({'__main__', _ROOT_COMPONENT}),
)


def _component(module: str) -> str:
    if module == _ROOT:
        return _ROOT_COMPONENT
    return module.removeprefix(f'{_ROOT}.').split('.', 1)[0]


def test_server_components_form_declared_dag():
    """Reject cycles, upward imports, same-layer coupling, and unclassified components."""
    graph = grimp.build_graph('twinkle', include_external_packages=False)
    server_modules = {
        module for module in graph.modules if module == _ROOT or module.startswith(f'{_ROOT}.')
    }
    observed_components = {_component(module) for module in server_modules}
    declared_components = set().union(*_LAYERS)
    assert observed_components == declared_components, (
        f'Update the server DAG when adding or removing components: '
        f'undeclared={sorted(observed_components - declared_components)}, '
        f'missing={sorted(declared_components - observed_components)}')

    layer_by_component = {
        component: layer_index for layer_index, layer in enumerate(_LAYERS) for component in layer
    }
    component_edges = {
        (_component(importer), _component(imported))
        for importer in server_modules
        for imported in graph.find_modules_directly_imported_by(importer)
        if imported in server_modules and _component(importer) != _component(imported)
    }
    violations = sorted(
        (importer, imported) for importer, imported in component_edges
        if layer_by_component[importer] <= layer_by_component[imported])
    assert not violations, (
        'Server dependencies must point strictly down the declared DAG; '
        f'invalid edges={violations}')
