#!/usr/bin/env python

import os
from os import path
from subprocess import check_call

import arg_parser
import context


def main():
    args = arg_parser.receiver_first()

    cc_repo = path.join(context.third_party_dir, 'aurora')
    src_dir = path.join(cc_repo, 'src')
    lib_dir = path.join(src_dir, 'core')
    app_dir = path.join(src_dir, 'app')
    send_src = path.join(app_dir, 'pccclient')
    recv_src = path.join(app_dir, 'pccserver')

    if args.option == 'setup':
        cmd = ['git', 'checkout', 'deep-learning']
        check_call(cmd)
        check_call(['make'], cwd=src_dir)
        return

    if args.option == 'receiver':
        os.environ['LD_LIBRARY_PATH'] = path.join(lib_dir)
        cmd = [recv_src, 'recv', args.port]
        check_call(cmd)
        return

    if args.option == 'sender':
        os.environ['LD_LIBRARY_PATH'] = path.join(lib_dir)
        arguments = '--pcc-rate-control=python -pyhelper=loaded_client -pypath=/path/to/pcc-rl/src/udt-plugins/testing/ --history-len=10 --pcc-utility-calc=linear --model-path=/path/to/your/model/'
        cmd = [send_src, 'send', args.ip, args.port, arguments]
        check_call(cmd)
        return


if __name__ == '__main__':
    main()
