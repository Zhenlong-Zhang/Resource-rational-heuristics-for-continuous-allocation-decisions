"""Redraw the accepted compact scientific inputs without fabricating a new epoch.

The optional input files are externally supplied, immutable historical data.
Their content pins authenticate this dataset, not its original execution. Fresh
reproduction uses paper_reproduction and needs none of these files.
"""
from pathlib import Path

from . import coefficient_original as O
from . import personal_time_inputs as I
from . import personal_time_sensitivity as S

INPUTS = {
    'saved_summary.json': '1ec3a0b16f584918d16e9714ada350974ff14340edcaa099a8b5632d3b25a2b0',
    'public_summary.json': O.INPUTS['public_summary.json'],
}
BOUNDARY = {
    'inputKind': 'authenticated_original_compact_figure_inputs',
    'historicalExecutionReconstructed': False,
    'historicalAcceptanceRevalidated': False,
    'newPublicPolicyEpoch': False,
    'paperBundleCreated': False,
    'newPolicySimulations': 0,
    'newFitInvocations': 0,
    'actionQuestionTableIncluded': False,
}


def redraw(source, output, config_path):
    """Render exact original summaries under a new, explicitly named renderer."""
    from . import paper_figures_plot as plot
    source, output = Path(source).resolve(), S.safe_output(output)
    I.need(not source.is_relative_to(output.resolve())
           and not output.resolve().is_relative_to(source), 'Separate input and output required')
    configs = Path(config_path).resolve().parent
    identity = O.scientific_identity(configs)
    inputs = {}
    for name, pin in INPUTS.items():
        path = source / name
        I.need(path.is_file() and not path.is_symlink(), 'Ordinary historical input required: ' + name)
        I.need(S.file_sha(path) == pin, 'Changed historical scientific bytes: ' + name)
        inputs[name] = I.read(path)
    recipe = I.read(config_path)
    design = I.read(configs / 'personal_time_design.json')
    allocation, true_need = inputs['saved_summary.json'], inputs['public_summary.json']
    data = plot.figure_data(allocation, true_need, design, recipe)
    manifest = {'schema': 'paper-original-redraw-v1', 'inputs': dict(INPUTS),
                'scientificIdentity': identity, 'renderer': S.file_sha(plot.__file__),
                'adapter': S.file_sha(__file__), 'recipe': S.file_sha(config_path),
                'boundary': dict(BOUNDARY)}
    I.write(output / 'historical_redraw.json', manifest)
    I.write(output / 'figure_data.json', data)
    paths = plot.render(allocation, true_need, design, output, recipe)
    return {**BOUNDARY, 'figures': [str(p) for p in paths.values()],
            'vectorFigures': [str(p.with_suffix('.pdf')) for p in paths.values()],
            'validationScope': 'exact_historical_summary_bytes_and_complete_plot_roster',
            'rendererSHA256': manifest['renderer']}
