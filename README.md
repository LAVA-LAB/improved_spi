## Implementation of the IJCAI 2023 paper More for Less: Safe Policy Improvement with Stronger Performance Guarantees

This code reproduces the experimental results presented in the paper and is built on top of the implementation of the ICML 2019 paper: Safe Policy Improvement with Baseline Bootstrapping, by Romain Laroche, Paul Trichelair, and Rémi Tachet des Combes.

Developers:
- Patrick Wienhöft
- Marnix Suilen
- Thiago D. Simão

Contributers:
- Clemens Dubslaff
- Christel Baier
- Nils Jansen


![teaser](https://github.com/lava-lab/improved_spi/blob/main/assets/teaser.gif?raw=true)


## Usage

Set up a virtual environment and install the required packages:
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

To run the Gridworld experiments:

```
$ python3 gridworld_main.py <output_experiment_name> <seeds> --Nwedges <nwedge> --cpus <cpus>
```
where 
- `<output_experiment_name>` is the name of the folder in /results/ where the results will be stored,
- `<seeds>` is a (space-separated) list of seeds, e.g., `1 2 3 4` or `{1..1000}`,
- `<nwedges>` is the Nwedge to use for standard SPIBB, or a list of Nwedges, e.g. `60 120 180` to run simulatinously,
- `<cpus>` indicates how many seeds to run in parallel.



To run the Wet Chicken or Resource Gathering experiments:
```
$ python3 main.py <output_experiment_name> <seeds> -e <env_name> --Nwedges <nwedges> --cpus <integer>
```
where 
- `<output_experiment_name>` is the name of the folder in /results/ where the results will be stored,
- `<seeds>` is a (space-separated) list of seeds, e.g., `1 2 3 4` or `{1..1000}`,
- `<env_name>` is the environment to run, either `"chicken"` or `resource` for the Wet Chicken or Resource Gathering environment, respectively. 
- `<nwedges>` is the Nwedge to use for standard SPIBB, or a list of Nwedges, e.g. `60 120 180` to run simulatinously,
- `<cpus>` indicates how many seeds to run in parallel.


The results will be stored in a unique .csv file per seed.

To plot the results:
```
$ python3 plot_results.py <output_experiment_name> --png --pdf --legend
```
where `<output_experiment_name>` is the name from the experiment to be plotted. Note that in case of the Wet Chicken environemnt, one should add the Nwedge to the experiment name, e.g, `<output_experiment_name>/Nwedge_500`.

To generate the plot that shows how the different Nwedges scale when the number of states increases, simply run
```
python3 generate_theoretical_plots.py 
```


All resulting plots will be stored in the /plots/ folder.


