import law

from columnflow.util import DotDict

from columnflow.tasks.framework.base import MultiConfigTask
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, WeightProducerMixin,
    CategoriesMixin, HistHookMixin, MultiConfigDatasetsProcessesMixin,
    # ShiftSourcesMixin,
)
from columnflow.tasks.framework.plotting import (
    ProcessPlotSettingMixin, VariablePlotSettingMixin, PlotBase,
)

from hbw.util import timeit_multiple


class ResolveDummy(
    HistHookMixin,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    MultiConfigDatasetsProcessesMixin,
    CategoriesMixin,
    MLModelsMixin,
    WeightProducerMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    MultiConfigTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_2d.plot_2d",
        add_default_to_description=True,
    )

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

    @classmethod
    @timeit_multiple
    def resolve_param_values(cls, params):
        params = super().resolve_param_values(params)
        return params

    def run(self):
        pass

    def output(self):
        output = {
            "always_incomplete_dummy": self.target("dummy.txt"),
        }
        return output
