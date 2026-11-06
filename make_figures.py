"""Figures for the report: learning curves and held-out composition results.

python make_figures.py RUN_DIR [RUN_DIR ...] --out results_attention_comparison/figures

Reads history.csv (train/validation loss per epoch), summary.json and the most
recent evaluation_*/summary.json of each run. Writes PNG files only; nothing is
trained or evaluated. Needs matplotlib (pip install matplotlib).
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['baseline', 'attention', 'bi_baseline', 'bi_attention']
LABEL = {'baseline': 'Baseline (uni LSTM)', 'attention': 'Attention (uni LSTM)',
         'bi_baseline': 'Bidirectional, no attention', 'bi_attention': 'Bidirectional + attention'}
# Validated categorical slots 1-4 (adjacent CVD ΔE >= 9.1 on the light surface).
COLOR = {'baseline': '#2a78d6', 'attention': '#eb6834',
         'bi_baseline': '#1baf7a', 'bi_attention': '#eda100'}
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    variant = config.get('variant') or ('attention' if config['model_config'].get('attention') else 'baseline')
    with (run / 'history.csv').open(encoding='utf-8') as handle:
        history = list(csv.DictReader(handle))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    evaluation = json.loads(sorted(run.glob('evaluation_*/summary.json'))[-1].read_text(encoding='utf-8'))
    return {'variant': variant, 'seed': config['seed'], 'best_epoch': summary['best_epoch'],
            'epoch': [int(r['epoch']) for r in history],
            'train': [float(r['train_loss']) for r in history],
            'val': [float(r['validation_loss']) for r in history],
            'evaluation': evaluation['splits']}


def learning_curves(runs, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True, facecolor=SURFACE)
    for ax, variant in zip(axes, ORDER):
        style(ax)
        mine = [r for r in runs if r['variant'] == variant]
        for r in mine:
            ax.plot(r['epoch'], r['train'], color=COLOR[variant], linewidth=1.2,
                    linestyle=(0, (3, 2)), alpha=0.8)
            ax.plot(r['epoch'], r['val'], color=COLOR[variant], linewidth=2)
            ax.plot([r['best_epoch']], [r['val'][r['epoch'].index(r['best_epoch'])]], 'o',
                    color=COLOR[variant], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yscale('log')
        ax.set_title(LABEL[variant], fontsize=10.5, color=INK, loc='left')
        ax.set_xlabel('Epoch', color=INK_2, fontsize=9)
    axes[0].set_ylabel('Token cross-entropy (log scale)', color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], color=INK_2, linewidth=2, label='Validation loss'),
               plt.Line2D([], [], color=INK_2, linewidth=1.2, linestyle=(0, (3, 2)), label='Training loss'),
               plt.Line2D([], [], color=INK_2, marker='o', linestyle='', label='Selected checkpoint')]
    fig.legend(handles=handles, loc='upper right', ncol=3, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle('Learning curves, 5 seeds per variant (one line per seed)', x=0.01, ha='left',
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / 'learning_curves.png', dpi=180, facecolor=SURFACE)
    plt.close(fig)


def composition_bars(runs, out):
    structures = sorted(runs[0]['evaluation']['test_composition']['by_structure'])
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    style(ax)
    width = 0.19
    for i, variant in enumerate(ORDER):
        mine = [r for r in runs if r['variant'] == variant]
        for j, structure in enumerate(structures):
            values = [100 * r['evaluation']['test_composition']['by_structure'][structure]['all_tests_pass_rate']
                      for r in mine]
            x = j + (i - 1.5) * (width + 0.02)
            ax.bar(x, statistics.mean(values), width=width, color=COLOR[variant],
                   label=LABEL[variant] if j == 0 else None)
            ax.scatter([x] * len(values), values, s=14, color=INK, zorder=3, linewidths=0)
    ax.set_xticks(range(len(structures)))
    ax.set_xticklabels([{'count_le': 'count + "<="', 'filter_ne': 'filter + "!="',
                         'sum_gt': 'sum + ">"'}.get(s, s) for s in structures], color=INK, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Programs passing all tests (%)', color=INK_2, fontsize=9)
    ax.set_title('Held-out combinations (test_composition): bar = mean of 5 seeds, dots = individual seeds',
                 loc='left', fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1),
              labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / 'composition_by_structure.png', dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--out', default='results_attention_comparison/figures')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load(r) for r in args.runs
            if json.loads((Path(r) / 'config.json').read_text(encoding='utf-8')).get('mode') != 'overfit']
    learning_curves(runs, out)
    composition_bars(runs, out)
    print(f'Saved figures to {out}')


if __name__ == '__main__':
    main()
