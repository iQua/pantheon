#!/usr/bin/env python
from os import path
from subprocess import check_call
import subprocess

import arg_parser
import context

def main():
        args = arg_parser.sender_first()
        cc_repo = path.join(context.third_party_dir, 'eagle-plus')
        send_src = path.join(cc_repo, 'net-em/net-em/net_em/envs', 'example_xentropy.py')
        recv_src = path.join(cc_repo, 'net-em/net-em/net_em/envs/connect-Eagle', 'connect')
        model_src = 'training_models/model-xentropy-505iter-rw97.pt'
        if args.option == 'setup':
            return

        if args.option == 'sender':
            cmd = ['python3', send_src, args.port, '--return_recent_meas', '--pass_hidden_state', '----with_elite_buffer', '--model', model_src]
            check_call(cmd)
            return

        if args.option == 'receiver':
            cmd = [recv_src, 'receive', args.ip, args.port]
            check_call(cmd)
            return

if __name__ == "__main__":
        main()
