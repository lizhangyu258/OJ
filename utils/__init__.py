__all__ = ['setup_environment', 'run_benchmark', 'get_default_prof_config', 'run_full_benchmark']


def __getattr__(name):
    if name in __all__:
        from .benchmark import (
            setup_environment,
            run_benchmark,
            get_default_prof_config,
            run_full_benchmark
        )
        return locals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
