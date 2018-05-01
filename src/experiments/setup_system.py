#!/usr/bin/env python

import arg_parser

import context
from helpers import kernel_ctl


def main():
    args = arg_parser.parse_sysctl()

    # enable IP forwarding
    kernel_ctl.enable_ip_forwarding()

    if args.reset_rmem:
        # change socket receive buffer sizes to Linux default ones
        kernel_ctl.set_sock_recv_buf(212992, 212992)
    elif args.set_rmem:
        # change socket receive buffer sizes to Pantheon's required ones
        kernel_ctl.set_sock_recv_buf(16777216, 33554432)

    # disable reverse path filtering
    if args.interface is not None:
        kernel_ctl.disable_rp_filter()


if __name__ == '__main__':
    main()
