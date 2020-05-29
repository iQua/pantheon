#!/usr/bin/env python

from os import path
import sys
import math
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
import numpy as np
import matplotlib_agg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

import arg_parser
import tunnel_graph
import context
from helpers import utils


class Plot(object):
    def __init__(self, args):
        self.data_dir = path.abspath(args.data_dir)
        self.include_acklink = args.include_acklink
        self.no_graphs = args.no_graphs

        metadata_path = path.join(self.data_dir, 'pantheon_metadata.json')
        meta = utils.load_test_metadata(metadata_path)
        self.cc_schemes = utils.verify_schemes_with_meta(args.schemes, meta)

        self.run_times = meta['run_times']
        self.flows = meta['flows']
        self.runtime = meta['runtime']
        self.expt_title = self.generate_expt_title(meta)

    def generate_expt_title(self, meta):
        if meta['mode'] == 'local':
            expt_title = 'local test in mahimahi, '
        elif meta['mode'] == 'remote':
            txt = {}
            for side in ['local', 'remote']:
                txt[side] = []

                if '%s_desc' % side in meta:
                    txt[side].append(meta['%s_desc' % side])
                else:
                    txt[side].append(side)

                txt[side] = ' '.join(txt[side])

            if meta['sender_side'] == 'remote':
                sender = txt['remote']
                receiver = txt['local']
            else:
                receiver = txt['remote']
                sender = txt['local']

            expt_title = 'test from %s to %s, ' % (sender, receiver)

        runs_str = 'run' if meta['run_times'] == 1 else 'runs'
        expt_title += '%s %s of %ss each per scheme\n' % (
            meta['run_times'], runs_str, meta['runtime'])

        if meta['flows'] > 1:
            expt_title += '%s flows with %ss interval between flows' % (
                meta['flows'], meta['interval'])

        return expt_title

    def parse_tunnel_log(self, cc, run_id, all_tput_path, all_delay_path):
        log_prefix = cc
        if self.flows == 0:
            log_prefix += '_mm'

        error = False
        ret = None

        link_directions = ['datalink']
        if self.include_acklink:
            link_directions.append('acklink')

        for link_t in link_directions:
            log_name = log_prefix + '_%s_run%s.log' % (link_t, run_id)
            log_path = path.join(self.data_dir, log_name)

            if not path.isfile(log_path):
                sys.stderr.write('Warning: %s does not exist\n' % log_path)
                error = True
                continue

            if self.no_graphs:
                tput_graph_path = None
                delay_graph_path = None
            else:
                tput_graph = cc + '_%s_throughput_run%s.png' % (link_t, run_id)
                tput_graph_path = path.join(self.data_dir, tput_graph)

                delay_graph = cc + '_%s_delay_run%s.png' % (link_t, run_id)
                delay_graph_path = path.join(self.data_dir, delay_graph)

            sys.stderr.write('$ tunnel_graph %s\n' % log_path)
            try:
                tunnel_results = tunnel_graph.TunnelGraph(
                    cc=cc,
                    all_tput_log=all_tput_path,
                    all_delay_log=all_delay_path,
                    tunnel_log=log_path,
                    throughput_graph=tput_graph_path,
                    delay_graph=delay_graph_path
                ).run()
            except Exception as exception:
                sys.stderr.write('Error: %s\n' % exception)
                sys.stderr.write('Warning: "tunnel_graph %s" failed but '
                                 'continued to run.\n' % log_path)
                error = True

            if error:
                continue

            if link_t == 'datalink':
                ret = tunnel_results
                duration = tunnel_results['duration'] / 1000.0

                if duration < 0.8 * self.runtime:
                    sys.stderr.write(
                        'Warning: "tunnel_graph %s" had duration %.2f seconds '
                        'but should have been around %s seconds. Ignoring this'
                        ' run.\n' % (log_path, duration, self.runtime))
                    error = True

        if error:
            return None

        return ret

    def update_stats_log(self, cc, run_id, stats):
        stats_log_path = path.join(
            self.data_dir, '%s_stats_run%s.log' % (cc, run_id))

        if not path.isfile(stats_log_path):
            sys.stderr.write('Warning: %s does not exist\n' % stats_log_path)
            return None

        saved_lines = ''

        # back up old stats logs
        with open(stats_log_path) as stats_log:
            for line in stats_log:
                if any([x in line for x in [
                        'Start at:', 'End at:', 'clock offset:']]):
                    saved_lines += line
                else:
                    continue

        # write to new stats log
        with open(stats_log_path, 'w') as stats_log:
            stats_log.write(saved_lines)

            if stats:
                stats_log.write('\n# Below is generated by %s at %s\n' %
                                (path.basename(__file__), utils.utc_time()))
                stats_log.write('# Datalink statistics\n')
                stats_log.write('%s' % stats)

    def eval_performance(self):
        perf_data = {}
        stats = {}

        for cc in self.cc_schemes:
            perf_data[cc] = {}
            stats[cc] = {}

        cc_id = 0
        run_id = 1
        pool = ThreadPool(processes=multiprocessing.cpu_count())

        while cc_id < len(self.cc_schemes):
            cc = self.cc_schemes[cc_id]

            # gather all time-varying tput and delay into one file of each run for all cc
            all_tput_path = path.join(self.data_dir, 'all_throughput_run' + str(run_id) + '.log')
            all_delay_path = path.join(self.data_dir, 'all_delay_run' + str(run_id) + '.log')
            with open(all_tput_path, 'w') as all_tput_log:
                all_tput_log.write("Scheme\tTraffic\tTime (s)\tThroughput (Mbit/s)\n")
            with open(all_delay_path, 'w') as all_delay_log:
                all_delay_log.write("Scheme\tTime (s)\t95th Percentile Delay (ms)\t99th Percentile Delay (ms)\tMean Delay (ms)\n")

            perf_data[cc][run_id] = pool.apply_async(
                self.parse_tunnel_log, args=(cc, run_id, all_tput_path, all_delay_path))

            run_id += 1
            if run_id > self.run_times:
                run_id = 1
                cc_id += 1

        for cc in self.cc_schemes:
            for run_id in xrange(1, 1 + self.run_times):
                perf_data[cc][run_id] = perf_data[cc][run_id].get()

                if perf_data[cc][run_id] is None:
                    continue

                stats_str = perf_data[cc][run_id]['stats']
                self.update_stats_log(cc, run_id, stats_str)
                stats[cc][run_id] = stats_str

        sys.stderr.write('Appended datalink statistics to stats files in %s\n'
                         % self.data_dir)

        return perf_data, stats

    def xaxis_log_scale(self, ax, min_delay, max_delay):
        if min_delay < -2:
            x_min = int(-math.pow(2, math.ceil(math.log(-min_delay, 2))))
        elif min_delay < 0:
            x_min = -2
        elif min_delay < 2:
            x_min = 0
        else:
            x_min = int(math.pow(2, math.floor(math.log(min_delay, 2))))

        if max_delay < -2:
            x_max = int(-math.pow(2, math.floor(math.log(-max_delay, 2))))
        elif max_delay < 0:
            x_max = 0
        elif max_delay < 2:
            x_max = 2
        else:
            x_max = int(math.pow(2, math.ceil(math.log(max_delay, 2))))

        symlog = False
        if x_min <= -2:
            if x_max >= 2:
                symlog = True
        elif x_min == 0:
            if x_max >= 8:
                symlog = True
        elif x_min >= 2:
            if x_max > 4 * x_min:
                symlog = True

        if symlog:
            ax.set_xscale('symlog', basex=2, linthreshx=2, linscalex=0.5)
            ax.set_xlim(x_min, x_max)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    def plot_throughput_delay(self, measure, data):
        min_raw_delay = sys.maxint
        min_mean_delay = sys.maxint
        max_raw_delay = -sys.maxint
        max_mean_delay = -sys.maxint

        fig_raw, ax_raw = plt.subplots()
        fig_mean, ax_mean = plt.subplots()

        schemes_config = utils.parse_config()['schemes']
        for cc in data:
            if not data[cc]:
                sys.stderr.write('No performance data for scheme %s\n' % cc)
                continue

            # measure is among 95th percentile, 99th percentile, mean
            value = data[cc][measure]
            cc_name = schemes_config[cc]['name']
            color = schemes_config[cc]['color']
            marker = schemes_config[cc]['marker']
            y_data, x_data = zip(*value)

            # update min and max raw delay
            min_raw_delay = min(min(x_data), min_raw_delay)
            max_raw_delay = max(max(x_data), max_raw_delay)

            # plot raw values
            ax_raw.scatter(x_data, y_data, color=color, marker=marker,
                           label=cc_name, clip_on=False)

            # plot the average of raw values
            x_mean = np.mean(x_data)
            y_mean = np.mean(y_data)

            # update min and max mean delay
            min_mean_delay = min(x_mean, min_mean_delay)
            max_mean_delay = max(x_mean, max_mean_delay)

            ax_mean.scatter(x_mean, y_mean, color=color, marker=marker,
                            clip_on=False)
            ax_mean.annotate(cc_name, (x_mean, y_mean))

        for fig_type, fig, ax in [('raw', fig_raw, ax_raw),
                                  ('mean', fig_mean, ax_mean)]:
            if fig_type == 'raw':
                self.xaxis_log_scale(ax, min_raw_delay, max_raw_delay)
            else:
                self.xaxis_log_scale(ax, min_mean_delay, max_mean_delay)
            ax.invert_xaxis()

            yticks = ax.get_yticks()
            if yticks[0] < 0:
                ax.set_ylim(bottom=0)

            if measure == '95th':
                tag = '95th percentile'
            elif measure == '99th':
                tag = '99th percentile'
            elif measure == 'mean':
                tag = 'Average'
            xlabel = tag + ' one-way delay (ms)'
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Average throughput (Mbit/s)', fontsize=12)
            ax.grid()

        # save pantheon_summary.svg and .pdf
        ax_raw.set_title(self.expt_title.strip(), y=1.02, fontsize=12)
        lgd = ax_raw.legend(scatterpoints=1, bbox_to_anchor=(1, 0.5),
                            loc='center left', fontsize=12)

        for graph_format in ['svg', 'pdf']:
            raw_summary = path.join(
                self.data_dir, 'pantheon_summary_delay_%s.%s' % (measure, graph_format))
            fig_raw.savefig(raw_summary, dpi=300, bbox_extra_artists=(lgd,),
                            bbox_inches='tight', pad_inches=0.2)

        # save pantheon_summary_mean.svg and .pdf
        ax_mean.set_title(self.expt_title +
                          ' (mean of all runs by scheme)', fontsize=12)

        for graph_format in ['svg', 'pdf']:
            mean_summary = path.join(
                self.data_dir, 'pantheon_summary_mean_delay_%s.%s' % (measure, graph_format))
            fig_mean.savefig(mean_summary, dpi=300,
                             bbox_inches='tight', pad_inches=0.2)

        sys.stderr.write(
            'Saved throughput graphs, delay graphs, and summary '
            'graphs in %s\n' % self.data_dir)

    def plot_all_ingress_graph(self):
        sns.set(style="whitegrid")
        # sns.despine()
        for i in range(1, self.run_times + 1):
            data_path = path.join(self.data_dir, 'all_throughput_run' + str(i) + '.log')
            data = pd.read_csv(data_path, iterator=True, sep="\t", chunksize=1000)
            ingress = pd.concat([chunk[chunk['Traffic'] == 'ingress'] for chunk in data])
            sns.lineplot(x="Time (s)", y="Throughput (Mbit/s)", ci=None, hue="Scheme", style="Scheme", dashes=True, data=ingress)
            plt.ylabel('Sending Rate (Mbit/s)')
            plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
            plt.savefig(path.join(self.data_dir, 'all_ingress_run' + str(i) + '.pdf'), bbox_inches='tight')

    def plot_all_delay_graph(self):
        sns.set(style="ticks")
        for i in range(1, self.run_times + 1):
            data_path = path.join(self.data_dir, 'all_delay_run' + str(i) + '.log')
            data = pd.read_csv(data_path, sep="\t")
            sns.lineplot(x="Time (s)", y="Delay (ms)", ci=None, hue="Scheme", data=data)
            plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
            plt.savefig(path.join(self.data_dir, 'all_delay_run' + str(i) + '.pdf'), bbox_inches='tight')


    def run(self):
        perf_data, stats_logs = self.eval_performance()

        all_perf_path = path.join(self.data_dir, 'all_perf.log')
        with open(all_perf_path, 'w') as all_perf_log:
            all_perf_log.write('Scheme(all runs)\tAvg throughput (Mbit/s)\tAvg 95th delay (ms)\tAvg 99th delay (ms)\tAvg mean delay (ms)\tAvg loss rate\n')

        data_for_plot = {}
        data_for_json = {}

        for cc in perf_data:
            data_for_plot[cc] = {}
            data_for_plot[cc]['95th'] = []
            data_for_plot[cc]['99th'] = []
            data_for_plot[cc]['mean'] = []
            data_for_json[cc] = {}
            sum_tput = 0
            sum_delay_95th = 0
            sum_delay_99th = 0
            sum_delay_mean = 0
            sum_loss = 0
            valid_run_times = 0
            

            for run_id in perf_data[cc]:
                if perf_data[cc][run_id] is None:
                    continue

                tput = perf_data[cc][run_id]['throughput']
                delay_95th = perf_data[cc][run_id]['delay_95th']
                delay_99th = perf_data[cc][run_id]['delay_99th']
                delay_mean = perf_data[cc][run_id]['delay_mean']
                loss = perf_data[cc][run_id]['loss']

                if tput is None or delay_95th is None:
                    continue
                data_for_plot[cc]['95th'].append((tput, delay_95th))

                if tput is None or delay_99th is None:
                    continue
                data_for_plot[cc]['99th'].append((tput, delay_99th))

                if tput is None or delay_mean is None:
                    continue
                data_for_plot[cc]['mean'].append((tput, delay_mean))

                flow_data = perf_data[cc][run_id]['flow_data']
                if flow_data is not None:
                    data_for_json[cc][run_id] = flow_data

                # calculate the sum performance of all runs for every cc
                valid_run_times += 1
                sum_tput += tput
                sum_delay_95th += delay_95th
                sum_delay_99th += delay_99th
                sum_delay_mean += delay_mean
                sum_loss += loss

                # gather cc performance data into one file
                # with open(all_perf_path, 'a') as all_perf_log:
                #     all_perf_log.write('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.6f\n' %
                #                        (cc, run_id, tput, delay_95th, delay_99th, delay_mean, loss))

            # calculate the average performance of all runs for every cc
            avg_tput = (float(sum_tput) / valid_run_times) if valid_run_times > 0 else 0 
            avg_delay_95th = float(sum_delay_95th) / valid_run_times if valid_run_times > 0 else 0
            avg_delay_99th = float(sum_delay_99th) / valid_run_times if valid_run_times > 0 else 0
            avg_delay_mean = float(sum_delay_mean) / valid_run_times if valid_run_times > 0 else 0
            avg_loss = float(sum_loss) / valid_run_times if valid_run_times > 0 else 0 

            # gather avg cc performance data of all run into one file
            with open(all_perf_path, 'a') as all_perf_log:
                all_perf_log.write('%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.6f\n' %
                                   (cc, avg_tput, avg_delay_95th, avg_delay_99th, avg_delay_mean, avg_loss))

        if not self.no_graphs:
            self.plot_all_ingress_graph()
            # self.plot_all_delay_graph()
            self.plot_throughput_delay('95th', data_for_plot)
            self.plot_throughput_delay('99th', data_for_plot)
            self.plot_throughput_delay('mean', data_for_plot)

        plt.close('all')

        perf_path = path.join(self.data_dir, 'pantheon_perf.json')
        with open(perf_path, 'w') as fh:
            json.dump(data_for_json, fh)


def main():
    args = arg_parser.parse_plot()
    Plot(args).run()


if __name__ == '__main__':
    main()
