#!/usr/bin/env python
from os import path
from subprocess import check_call

import arg_parser
import context

def main():
	args = arg_parser.sender_first()
	cc_repo = path.join(context.third_party_dir, 'gold')
	send_src = path.join(cc_repo, 'environment', 'learner.py')
	recv_src = path.join(cc_repo, 'environment', 'run_receiver.py')
	dependencies = path.join(cc_repo, 'dependencies.sh')
	if args.option == 'setup':
		check_call(dependencies, shell = True)
                return 
	if args.option == 'sender':
		cmd = ['python3', '-O', send_src,'action_space1', args.port, '--test']
		check_call(cmd)
                return
	if args.option == 'receiver':
		cmd = [recv_src, args.ip, args.port]
		check_call(cmd)
		return
if __name__ == "__main__":
	main()
