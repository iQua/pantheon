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
        conn_src = path.join(cc_repo, 'net-em/net-em/net_em/envs/connect-Eagle')
        cmake_src = path.join(conn_src, 'build.sh')
        recv_src = path.join(conn_src, 'connect')
        # designate a trained model here (default path: cc_repo/net-em/.../envs/models/)
        model_src = 'training_models/model-xentropy-505iter-rw97.pt'
        
        if args.option == 'setup':
            # build C++ connection program
            cmd = ['bash', cmake_src]
            check_call(cmd, cwd=conn_src)
            return

        if args.option == 'sender':
            cmd = ['python3', send_src, args.port, '--return_recent_meas', '--pass_hidden_states', '--model', model_src]
            check_call(cmd)
            return

        if args.option == 'receiver':
            cmd = [recv_src, 'receive', args.ip, args.port]
            check_call(cmd)
            return

if __name__ == "__main__":
        main()
