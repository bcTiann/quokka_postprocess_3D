"""QUOKKA post-processing package.

Public functionality is imported from explicit submodules. Keeping the package
root lightweight lets the optional DESPOTIC table builder run without loading
the yt-based simulation pipeline.
"""

from importlib import import_module


_LAZY_EXPORTS = {
    "YTDataProvider": (".data_handling", "YTDataProvider"),
    "get_attenuation_factor": (".analysis", "get_attenuation_factor"),
    "along_sight_cumulation": (".analysis", "along_sight_cumulation"),
    "calculate_cumulative_column_density": (".analysis", "calculate_cumulative_column_density"),
    "calculate_attenuation": (".analysis", "calculate_attenuation"),
    "create_plot": (".plotting", "create_plot"),
    "plot_multiview_grid": (".plotting", "plot_multiview_grid"),
}
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
