#!/usr/bin/env python
from os import path
from subprocess import check_call
import subprocess

import arg_parser
import context
import time


def main():
        args = arg_parser.sender_first()
        cc_repo = path.join(context.third_party_dir, 'eagle-v3')
        send_src = '/home/zi/PycharmProjects/gold-IRL/sender-receiver/sender-receiver/sender_receiver/DeepRL/deep_rl_2/ml_algos/run.py'
        recv_src = path.join(cc_repo, 'sender-receiver/sender_receiver/envs', 'run_receiver.py')

        if args.option == 'setup':
            return

        if args.option == 'sender':
            cmd = ['python3', send_src, args.port, "ppo", "Gradient", "0.6", "ppo-0.6-70iter-rw252.pt"]
            subprocess.check_output(cmd)
            return

        if args.option == 'receiver':
            cmd = ['python3', recv_src, args.ip, args.port]
            subprocess.check_output(cmd)
            return

if __name__ == "__main__":
        main()
