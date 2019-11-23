import matplotlib.pyplot as plt
from os import listdir
from os.path import isdir, isfile, join
import sys
import seaborn as sns
import pandas as pd


class PlotMM():
    def __init__(self, type):
        self.type = type
        self.dir = '../experiments/data/' + type
        self.graph_log_path = join(self.dir, 'results/' + self.type + '.log')
        sns.set(style="ticks")

        if self.type == 'random_loss':
            self.title = 'Random Loss Rate'
        elif self.type == 'buffer_size':
            self.title = 'Buffer Size (KB)'

    def gather_data_from_files(self):
        open(self.graph_log_path, 'w').write('Scheme\tThroughput (Mbit/s)\tDelay (ms)\tLoss Rate\t' + self.title + '\n')

        log = open(self.graph_log_path, 'a')
        files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        for f in files:
            n = f.strip('.log')
            with open(join(self.dir, f), 'r') as in_f:
                # skip title
                next(in_f)
                for l in in_f:
                    log.write(l.strip('\n') + '\t' + n + '\n')

    def plot(self):
        data = pd.read_table(self.graph_log_path, sep='\t')
        sns.lineplot(x=self.title, y="Throughput (Mbit/s)", ci=None, hue="Scheme", style=None, data=data)
        plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
        plt.savefig(self.dir + '/results/throughput_' + self.type + '.pdf', bbox_inches='tight')

        sns.lineplot(x=self.title, y="Delay (ms)", ci=None, hue="Scheme", data=data)
        plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
        plt.savefig(self.dir + '/results/delay_' + self.type + '.pdf', bbox_inches='tight')

        sns.lineplot(x=self.title, y="Loss Rate", ci=None, hue="Scheme", data=data)
        plt.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0)
        plt.savefig(self.dir + '/results/loss_rate_' + self.type + '.pdf', bbox_inches='tight')

        plt.close()

    def run(self):
        if not isdir(self.dir):
            sys.stderr.write('Warning: %s does not exist\n' % self.dir)
            return None
        if not isdir(join(self.dir, 'results')):
            sys.stderr.write('Warning: %s does not exist\n' % join(self.dir, 'results'))
            return None
        self.gather_data_from_files()
        self.plot()


def main():
    plot_random_loss = PlotMM('random_loss')
    plot_buffer_size = PlotMM('buffer_size')
    plot_random_loss.run()
    plot_buffer_size.run()


if __name__ == '__main__':
    main()