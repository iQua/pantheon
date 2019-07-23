#!/usr/bin/env python
from os import path
from subprocess import check_call
import subprocess

import arg_parser
import context

def main():
        args = arg_parser.sender_first()
        cc_repo = path.join(context.third_party_dir, 'gold-IRL')
        send_src = path.join(cc_repo, 'sender-receiver/sender-receiver/sender_receiver/envs', 'example_a2c.py')
        recv_src = path.join(cc_repo, 'sender-receiver/sender-receiver/sender_receiver/envs', 'run_receiver.py')
        #dependencies = path.join(cc_repo, 'dependencies.sh')
        if args.option == 'setup':
            #check_call(dependencies, shell = True)
            return

        if args.option == 'sender':
            cmd = ['python3' , send_src, args.port, '--sending_rate']
            subprocess.check_output(cmd)
            return

        if args.option == 'receiver':
            cmd = ['python3', recv_src, args.ip, args.port]
            subprocess.check_output(cmd)
            return

if __name__ == "__main__":
        main()
