import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import yaml
import os
import argparse


plt.style.use("matplotlibrc")


# names for columns and labels
N_WEDGE = "\\ensuremath{N_{\\wedge}}"
DATASET_SIZE = "\\ensuremath{|\\mathcal{D}|}"
EXPECTED_RETURN = "Expected Return"
METRIC = "Metric"
ALGORITHM = "Algorithm"
BEHAVIOR_POLICY = r"$\pi_b$"
OPTIMAL_POLICY = r"$\pi^*$"


def main(conf):
    conf["out_dir"] = os.path.join("plots", conf['experiment_id'])
    conf["results"] = os.path.join("results", conf['experiment_id'])
    csv_pattern = os.path.join(conf['results'], "*.csv")

    yaml_file = os.path.join(conf['results'], "config.yaml")
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    conf.update(data)

    print(conf)
    dfs = [
        pd.read_csv(csv_file, index_col=0)
        for csv_file in glob.glob(csv_pattern)
    ]
    df = pd.concat(dfs).reset_index(drop=True)

    df['algorithm'] = df['algorithm'].replace(['Pi_b_SPIBB'], 'SPIBB')

    df.rename(inplace=True, columns={
        "algorithm": ALGORITHM,
        "performance": EXPECTED_RETURN,
        "nb_trajectories": DATASET_SIZE,
    })


    os.makedirs(conf["out_dir"], exist_ok=True)

    conf['n_seeds'] = len(df["seed"].unique())
    print(f"Plotting {conf['experiment_id']}. Found {conf['n_seeds']}.")
    df[METRIC] = "Mean"
    cols = [ALGORITHM, DATASET_SIZE]
    cvar10_df = df.sort_values(['Expected Return'], ascending=False).groupby(cols).tail(tail_size(10, conf['n_seeds']))
    cvar10_df[METRIC] = "10\\%-CVaR"
    cvar1_df = df.sort_values(['Expected Return'], ascending=False).groupby(cols).tail(tail_size(1, conf['n_seeds']))
    cvar1_df[METRIC] = "1\\%-CVaR"

    combined_df = pd.concat([df, cvar1_df, cvar10_df]).reset_index(drop=True)
    # combined_df = pd.concat([df, cvar10_df]).reset_index(drop=True)

    ax = sns.lineplot(
        data=combined_df, x=DATASET_SIZE, y=EXPECTED_RETURN,
        hue=ALGORITHM,
        markers=True,
        style=METRIC,
        style_order=[
            'Mean',
            '10\\%-CVaR',
            '1\\%-CVaR'
        ],
        errorbar=None,
    )

    ax = sns.lineplot(
        data=combined_df,
        x=DATASET_SIZE,
        y=conf["baseline_perf"],
        errorbar=None,
        ax=ax,
        linestyle="-.",
        label=BEHAVIOR_POLICY,
        color='gray',
    )
    ax = sns.lineplot(
        data=combined_df,
        x=DATASET_SIZE,
        y=conf["pi_star_perf"],
        errorbar=None,
        ax=ax,
        linestyle=':',
        label=OPTIMAL_POLICY,
        color='purple',
    )
    ax.set(xscale="log")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    if conf['legend']:
        export_legend(ax, os.path.join(conf["out_dir"], f"legend.pdf"))

    ax.legend([], [], frameon=False)

    if conf['png']:
        ax.figure.savefig(os.path.join(conf["out_dir"], f"curve.png"), bbox_inches='tight')
    if conf['pdf']:
        ax.figure.savefig(os.path.join(conf["out_dir"], f"curve.pdf"), bbox_inches='tight')


    if conf['show']:
        plt.tight_layout()
        plt.show()




def tail_size(cvar, n):
    assert isinstance(cvar, int)
    return int(max(cvar * n / 100, 1))


def export_legend(ax, filename):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[:9] + handles[0:1] + handles[9:]
    labels = labels[:9] + ['Policy'] + labels[9:]
    legend = ax2.legend(handles,labels, frameon=False, loc='lower center')

    for x in [0, 5, 9]:
        ax2.legend_.get_texts()[x].set_position((-20, 0))
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", type=str, default="test")
    parser.add_argument("--show", default=False, action='store_true')
    parser.add_argument("--pdf", default=False, action='store_true')
    parser.add_argument("--png", default=False, action='store_true')
    parser.add_argument("--legend", default=False, action='store_true')
    main(vars(parser.parse_args()))
