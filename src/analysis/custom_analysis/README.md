# Plotting with Pantheon

### This is a folder to replace `pantheon/src/analysis/`. It fulfills extra graph plotting methods.

* We need to install following dependencies first:
1. `pip install seaborn`





* It gathers the time-varying throughput and time-varying delay of all cc schemes in each run in **all_throughput_run_id.log** and **all_delay_run_id.log** respectively, and then plots graphs **all_throughput_run_id.pdf** and **all_delay_run_id.pdf**, saved in the specified output directory `--data-dir DIR`.


> example: <a><img src="https://github.com/feiwang98/glider/blob/master/pantheon_related/analysis/example_graphs/all_throughput_run1.svg" align="top" height="500" width="500" ></a>

* It calculates the average performance of all runs (e.g. `pantheon/src/experiments/test.py (..other args..) --runtimes 10`) for every tested cc scheme in `analysis/plot.py: run()` and gather all the data into one log file **all_perf.log** located in `DIR`.
This helps show the average performance of cc schemes under different experiment scenarios, e.g. when testing on mahimahi links of different *random loss rate* or *buffer size*. For example, if we want to compare how cc schemes perform on links with different buffer size, we can follow these steps:
  1. create a directory named `buffer_size` under `DIR`, and then a directory named `results` under `buffer_size`;
  2. test cc schemes on pantheon, with a specified buffer size;
  3. analyze test results;
  4. move the log file **all_perf.log** into `DIR/buffer_size/` and rename it using that specified buffer size (KB);
  5. repeat *steps 2-4* by passing another buffer size we plan to test with as argument;
  6. run `analysis/plot_mm.py` to gather all the data into **buffer_size.log** and plot graphs (**throughput_buffer_size.pdf**, **delay_buffer_size.pdf**, **loss_rate_buffer_size.pdf**), saved in directory `DIR/buffer_size/result/`.
  
> example: <a><img src="https://github.com/feiwang98/glider/blob/master/pantheon_related/analysis/example_graphs/throughput_buffer_size.svg" align="top" height="500" width="500" ></a>
  
  


The structure of the specified output directory `DIR` will be like this.
```bash
  .
  ├── all_delay_run1.pdf
  ├── all_throughput_run1.pdf
  ├── all_delay_run1.log
  ├── all_throughput_run1.log
  ├── all_perf.log
  ├── buffer_size
  │   ├── 60.log
  │   ├── 90.log
  │   ├── 375.log
  │   └── results
  │       ├── buffer_size.log
  │       ├── delay_buffer_size.pdf
  │       ├── loss_rate_buffer_size.pdf
  │       └── throughput_buffer_size.pdf
  ├── pantheon_metadata.json
  ├── pantheon_perf.json
  ├── pantheon_report.pdf
  ├── pantheon_summary_mean.pdf
  ├── pantheon_summary_mean.svg
  ├── pantheon_summary.pdf
  ├── pantheon_summary.svg
  ├── random_loss
  │   ├── 0.01.log
  │   ├── 0.02.log
  │   ├── 0.1.log
  │   └── results
  │       ├── delay_random_loss.pdf
  │       ├── loss_rate_random_loss.pdf
  │       ├── random_loss.log
  │       └── throughput_random_loss.pdf
  ├── vivace_acklink_run1.log
  ├── vivace_datalink_delay_run1.png
  ├── vivace_datalink_run1.log
  ├── vivace_datalink_throughput_run1.png
  ├── vivace_mm_acklink_run1.log
  ├── vivace_mm_datalink_run1.log
  ├── vivace_stats_run1.log
  ├── ...
  └── ...
```
