import warnings
from time import time

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2, Session
from qiskit_ibm_runtime.options import SamplerOptions, SimulatorOptions

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, SLSQP
from qiskit_machine_learning.utils import algorithm_globals

def report(results, n_top=3):
    """Report the top parameters for each search strategy."""
    results = pd.DataFrame(results)
    if "iter" in results:
        results = results[results["iter"] == results["iter"].max()]

    for rank, (_, candidate) in enumerate(
        results.nlargest(n_top, "mean_test_score").iterrows(), start=1
    ):
        print(
            f"Model with rank: {rank}\n"
            f"Mean validation score: "
            f"{candidate['mean_test_score']:.3f} "
            f"(std: {candidate['std_test_score']:.3f})\n"
            f"Parameters: {candidate['params']}\n"
        )

n_classes = 2

x, y = make_classification(
    n_samples=10,
    n_features=2,
    n_classes=n_classes,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=5.0,
    random_state=algorithm_globals.random_seed,
)
x = MinMaxScaler().fit_transform(x)


feature_map = zz_feature_map(2)
ansatz = real_amplitudes(num_qubits=2, reps=1)

parity_n_classes = lambda x: "{:b}".format(x).count("1") % n_classes
initial_point = np.array([0.5] * ansatz.num_parameters) if ansatz is not None else None

backend = GenericBackendV2(
    num_qubits=3,
    noise_info=False,
    seed=123,
)

session = Session(backend=backend)

simopts = SimulatorOptions(seed_simulator=123)
sampler_opts = SamplerOptions(default_shots=1e3, simulator=simopts)

sampler = SamplerV2(mode=session, options=sampler_opts)
pm = generate_preset_pass_manager(optimization_level=0, backend=backend)

vqc = VQC(
    num_qubits=2,
    feature_map=feature_map,
    ansatz=ansatz,
    initial_point=initial_point,
    output_shape=n_classes,
    interpret=parity_n_classes,
    sampler=sampler,
    pass_manager=pm
)

param_grid = {
    "warm_start": [True, False],
    "optimizer": [SLSQP(), COBYLA(25)]
}

grid_search = GridSearchCV(vqc, param_grid=param_grid, scoring="roc_auc_ovr")
start = time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    grid_search.fit(x, y)

print(
    f"GridSearchCV took {time() - start:.2f} seconds for "
    f"{len(grid_search.cv_results_['params'])} candidate parameter settings."
)
report(grid_search.cv_results_)
