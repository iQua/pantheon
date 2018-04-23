#!/usr/bin/env python

import os
from os import path
import sys
import time
import uuid
import random
import signal
import traceback
from subprocess import PIPE
from parse_arguments import parse_arguments
import project_root
from helpers.helpers import (
    Popen, call, TMPDIR, kill_proc_group, parse_config,
    timeout_handler, TimeoutError, utc_time, get_open_port,
    get_default_qdisc, set_default_qdisc)
from test_helpers import (
    who_runs_first, parse_remote_path, query_clock_offset, get_git_summary,
    save_test_metadata, get_recv_sock_bufsizes, set_recv_sock_bufsizes,
    get_cmds_to_revert_conf, set_conf, revert_conf)
from collections import namedtuple

Flow = namedtuple('Flow', ['cc', # replace self.cc
                           'cc_src_local', # replace self.cc_src
                           'cc_src_remote', # replace self.r[cc_src]
                           'run_first', # replace self.run_first
                           'run_second']) # replace self.run_second

class Test(object):    
    def __init__(self, args, run_id, cc):
        self.mode = args.mode
        self.run_id = run_id
        self.cc = cc
        self.data_dir = path.abspath(args.data_dir)

        # shared arguments between local and remote modes
        self.flows = args.flows
        self.runtime = args.runtime
        self.interval = args.interval
        self.run_times = args.run_times

        # used for cleanup
        self.proc_first = None
        self.proc_second = None
        self.ts_manager = None
        self.tc_manager = None

        self.test_start_time = None
        self.test_end_time = None

        # local mode
        if self.mode == 'local':
            self.datalink_trace = args.uplink_trace
            self.acklink_trace = args.downlink_trace
            self.prepend_mm_cmds = args.prepend_mm_cmds
            self.append_mm_cmds = args.append_mm_cmds
            self.extra_mm_link_args = args.extra_mm_link_args

            # for convenience
            self.sender_side = 'remote'
            self.server_side = 'local'
                      
        # remote mode
        if self.mode == 'remote':
            self.sender_side = args.sender_side
            self.server_side = args.server_side
            self.local_addr = args.local_addr
            self.local_if = args.local_if
            self.remote_if = args.remote_if
            self.local_desc = args.local_desc
            self.remote_desc = args.remote_desc

            self.ntp_addr = args.ntp_addr
            self.local_ofst = None
            self.remote_ofst = None

            self.r = parse_remote_path(args.remote_path, self.cc)

        # arguments when there's a config
        self.test_config = None
        if hasattr(args, 'test_config'):
            self.test_config = args.test_config

        if self.test_config is not None:
            self.cc = self.test_config['test-name']
            self.flow_objs = []
            if self.mode == 'remote':
                cc_src_remote_dir = r['pantheon_dir']
            for flow in args.test_config['flows']:
                cc = flow['scheme']
                run_first, run_second = who_runs_first(cc)
                self.flow_objs.append(Flow(cc=cc, 
                                      cc_src_local=path.join(project_root.DIR,
                                                             'src',
                                                             cc + '.py'),
                                      cc_src_remote=path.join(cc_src_remote_dir,
                                                              'src',
                                                              cc + '.py'),
                                      run_first=run_first,
                                      run_second=run_second)) 
                
    def setup_mm_cmd(self):
        mm_datalink_log = self.cc + '_mm_datalink_run%d.log' % self.run_id
        mm_acklink_log = self.cc + '_mm_acklink_run%d.log' % self.run_id
        self.mm_datalink_log = path.join(self.data_dir, mm_datalink_log)
        self.mm_acklink_log = path.join(self.data_dir, mm_acklink_log)

        if self.run_first == 'receiver' or self.flows > 0:
            # if receiver runs first OR if test inside pantheon tunnel
            uplink_log = self.mm_datalink_log
            downlink_log = self.mm_acklink_log
            uplink_trace = self.datalink_trace
            downlink_trace = self.acklink_trace
        else:
            # if sender runs first AND test without pantheon tunnel
            uplink_log = self.mm_acklink_log
            downlink_log = self.mm_datalink_log
            uplink_trace = self.acklink_trace
            downlink_trace = self.datalink_trace

        self.mm_cmd = []

        if self.prepend_mm_cmds:
            self.mm_cmd += self.prepend_mm_cmds.split()

        self.mm_cmd += [
            'mm-link', uplink_trace, downlink_trace,
            '--uplink-log=' + uplink_log,
            '--downlink-log=' + downlink_log]

        if self.extra_mm_link_args:
            self.mm_cmd += self.extra_mm_link_args.split()

        if self.append_mm_cmds:
            self.mm_cmd += self.append_mm_cmds.split()

    def setup(self):
        # setup commonly used paths
        self.cc_src = path.join(project_root.DIR, 'src', self.cc + '.py')
        self.test_dir = path.join(project_root.DIR, 'test')
        self.tunnel_manager = path.join(self.test_dir, 'tunnel_manager.py')

        # record who runs first
        if self.test_config is None:
            self.run_first, self.run_second = who_runs_first(self.cc)
        else:
            self.run_first = None
            self.run_second = None
            
        # wait for 3 seconds until run_first is ready
        self.run_first_setup_time = 3

        # setup output logs
        self.datalink_name = self.cc + '_datalink_run%d' % self.run_id
        self.acklink_name = self.cc + '_acklink_run%d' % self.run_id

        self.datalink_log = path.join(
            self.data_dir, self.datalink_name + '.log')
        self.acklink_log = path.join(
            self.data_dir, self.acklink_name + '.log')

        if self.flows > 0:
            self.datalink_ingress_logs = []
            self.datalink_egress_logs = []
            self.acklink_ingress_logs = []
            self.acklink_egress_logs = []

            for tun_id in xrange(1, self.flows + 1):
                uid = uuid.uuid4()

                self.datalink_ingress_logs.append(path.join(
                    TMPDIR, '%s_flow%s_uid%s.log.ingress'
                    % (self.datalink_name, tun_id, uid)))
                self.datalink_egress_logs.append(path.join(
                    TMPDIR, '%s_flow%s_uid%s.log.egress'
                    % (self.datalink_name, tun_id, uid)))
                self.acklink_ingress_logs.append(path.join(
                    TMPDIR, '%s_flow%s_uid%s.log.ingress'
                    % (self.acklink_name, tun_id, uid)))
                self.acklink_egress_logs.append(path.join(
                    TMPDIR, '%s_flow%s_uid%s.log.egress'
                    % (self.acklink_name, tun_id, uid)))

        if self.mode == 'local':
            self.setup_mm_cmd()
        else:
            # record local and remote clock offset
            if self.ntp_addr is not None:
                self.local_ofst, self.remote_ofst = query_clock_offset(
                    self.ntp_addr, self.r['ssh_cmd'])

    # test congestion control without running pantheon tunnel
    def run_without_tunnel(self):
        port = get_open_port()

        # run the side specified by self.run_first
        cmd = ['python', self.cc_src, self.run_first, port]
        sys.stderr.write('Running %s %s...\n' % (self.cc, self.run_first))
        self.proc_first = Popen(cmd, preexec_fn=os.setsid)

        # sleep just in case the process isn't quite listening yet
        # the cleaner approach might be to try to verify the socket is open
        time.sleep(self.run_first_setup_time)

        self.test_start_time = utc_time()
        # run the other side specified by self.run_second
        sh_cmd = 'python %s %s $MAHIMAHI_BASE %s' % (
            self.cc_src, self.run_second, port)
        sh_cmd = ' '.join(self.mm_cmd) + " -- sh -c '%s'" % sh_cmd
        sys.stderr.write('Running %s %s...\n' % (self.cc, self.run_second))
        self.proc_second = Popen(sh_cmd, shell=True, preexec_fn=os.setsid)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.runtime)

        try:
            self.proc_first.wait()
            self.proc_second.wait()
        except TimeoutError:
            pass
        else:
            signal.alarm(0)
            sys.stderr.write('Warning: test exited before time limit\n')
        finally:
            self.test_end_time = utc_time()

        return True

    def run_tunnel_managers(self):
        # run tunnel server manager
        if self.mode == 'remote':
            if self.server_side == 'local':
                ts_manager_cmd = ['python', self.tunnel_manager]
            else:
                ts_manager_cmd = self.r['ssh_cmd'] + [
                    'python', self.r['tunnel_manager']]
        else:
            ts_manager_cmd = ['python', self.tunnel_manager]

        sys.stderr.write('[tunnel server manager (tsm)] ')
        self.ts_manager = Popen(ts_manager_cmd, stdin=PIPE, stdout=PIPE,
                                preexec_fn=os.setsid)
        ts_manager = self.ts_manager

        while True:
            running = ts_manager.stdout.readline()
            if 'tunnel manager is running' in running:
                sys.stderr.write(running)
                break

        ts_manager.stdin.write('prompt [tsm]\n')
        ts_manager.stdin.flush()

        # run tunnel client manager
        if self.mode == 'remote':
            if self.server_side == 'local':
                tc_manager_cmd = self.r['ssh_cmd'] + [
                    'python', self.r['tunnel_manager']]
            else:
                tc_manager_cmd = ['python', self.tunnel_manager]
        else:
            tc_manager_cmd = self.mm_cmd + ['python', self.tunnel_manager]

        sys.stderr.write('[tunnel client manager (tcm)] ')
        self.tc_manager = Popen(tc_manager_cmd, stdin=PIPE, stdout=PIPE,
                                preexec_fn=os.setsid)
        tc_manager = self.tc_manager

        while True:
            running = tc_manager.stdout.readline()
            if 'tunnel manager is running' in running:
                sys.stderr.write(running)
                break

        tc_manager.stdin.write('prompt [tcm]\n')
        tc_manager.stdin.flush()

        return ts_manager, tc_manager

    def run_tunnel_server(self, tun_id, ts_manager):
        if self.server_side == self.sender_side:
            ts_cmd = 'mm-tunnelserver --ingress-log=%s --egress-log=%s' % (
                self.acklink_ingress_logs[tun_id - 1],
                self.datalink_egress_logs[tun_id - 1])
        else:
            ts_cmd = 'mm-tunnelserver --ingress-log=%s --egress-log=%s' % (
                self.datalink_ingress_logs[tun_id - 1],
                self.acklink_egress_logs[tun_id - 1])

        if self.mode == 'remote':
            if self.server_side == 'remote':
                if self.remote_if is not None:
                    ts_cmd += ' --interface=' + self.remote_if
            else:
                if self.local_if is not None:
                    ts_cmd += ' --interface=' + self.local_if

        ts_cmd = 'tunnel %s %s\n' % (tun_id, ts_cmd)
        ts_manager.stdin.write(ts_cmd)
        ts_manager.stdin.flush()

        # read the command to run tunnel client
        readline_cmd = 'tunnel %s readline\n' % tun_id
        ts_manager.stdin.write(readline_cmd)
        ts_manager.stdin.flush()

        cmd_to_run_tc = ts_manager.stdout.readline().split()
        return cmd_to_run_tc

    def run_tunnel_client(self, tun_id, tc_manager, cmd_to_run_tc):
        if self.mode == 'local':
            cmd_to_run_tc[1] = '$MAHIMAHI_BASE'
        else:
            if self.server_side == 'remote':
                cmd_to_run_tc[1] = self.r['ip']
            else:
                cmd_to_run_tc[1] = self.local_addr

        cmd_to_run_tc_str = ' '.join(cmd_to_run_tc)

        if self.server_side == self.sender_side:
            tc_cmd = '%s --ingress-log=%s --egress-log=%s' % (
                cmd_to_run_tc_str,
                self.datalink_ingress_logs[tun_id - 1],
                self.acklink_egress_logs[tun_id - 1])
        else:
            tc_cmd = '%s --ingress-log=%s --egress-log=%s' % (
                cmd_to_run_tc_str,
                self.acklink_ingress_logs[tun_id - 1],
                self.datalink_egress_logs[tun_id - 1])

        if self.mode == 'remote':
            if self.server_side == 'remote':
                if self.local_if is not None:
                    tc_cmd += ' --interface=' + self.local_if
            else:
                if self.remote_if is not None:
                    tc_cmd += ' --interface=' + self.remote_if

        tc_cmd = 'tunnel %s %s\n' % (tun_id, tc_cmd)
        readline_cmd = 'tunnel %s readline\n' % tun_id

        # re-run tunnel client after 20s timeout for at most 3 times
        max_run = 3
        curr_run = 0
        got_connection = ''
        while 'got connection' not in got_connection:
            curr_run += 1
            if curr_run > max_run:
                sys.stderr.write('Unable to establish tunnel\n')
                return False

            tc_manager.stdin.write(tc_cmd)
            tc_manager.stdin.flush()
            while True:
                tc_manager.stdin.write(readline_cmd)
                tc_manager.stdin.flush()

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(20)

                try:
                    got_connection = tc_manager.stdout.readline()
                    sys.stderr.write('Tunnel is connected\n')
                except TimeoutError:
                    sys.stderr.write('Tunnel connection timeout\n')
                    break
                except IOError:
                    sys.stderr.write('Tunnel client failed to connect to '
                                     'tunnel server\n')
                    return False
                else:
                    signal.alarm(0)
                    if 'got connection' in got_connection:
                        break

        return True

    def run_first_side(self, tun_id, send_manager, recv_manager,
                       send_pri_ip, recv_pri_ip):            
        
        first_src = self.cc_src
        second_src = self.cc_src

        if self.run_first == 'receiver':
            if self.mode == 'remote':
                if self.sender_side == 'local':
                    first_src = self.r['cc_src']
                else:
                    second_src = self.r['cc_src']

            port = get_open_port()

            first_cmd = 'tunnel %s python %s receiver %s\n' % (
                tun_id, first_src, port)
            second_cmd = 'tunnel %s python %s sender %s %s\n' % (
                tun_id, second_src, recv_pri_ip, port)

            recv_manager.stdin.write(first_cmd)
            recv_manager.stdin.flush()
        elif self.run_first == 'sender':  # self.run_first == 'sender'
            if self.mode == 'remote':
                if self.sender_side == 'local':
                    second_src = self.r['cc_src']
                else:
                    first_src = self.r['cc_src']

            port = get_open_port()

            first_cmd = 'tunnel %s python %s sender %s\n' % (
                tun_id, first_src, port)
            second_cmd = 'tunnel %s python %s receiver %s %s\n' % (
                tun_id, second_src, send_pri_ip, port)

            send_manager.stdin.write(first_cmd)
            send_manager.stdin.flush()
            
        # get run_first and run_second from the flow object
        else:
            assert(hasattr(self, 'flow_objs'))
            flow = self.flow_objs[tun_id - 1] # tunnel id starts with 1

            first_src = flow.cc_src_local
            second_src = flow.cc_src_local
            
            if flow.run_first == 'receiver':
                if self.mode == 'remote':
                    if self.sender_side == 'local':
                        first_src = flow.cc_src_remote
                    else:
                        second_src = flow.cc_src_remote

                port = get_open_port()

                first_cmd = 'tunnel %s python %s receiver %s\n' % (
                    tun_id, first_src, port)
                second_cmd = 'tunnel %s python %s sender %s %s\n' % (
                    tun_id, second_src, recv_pri_ip, port)

                recv_manager.stdin.write(first_cmd)
                recv_manager.stdin.flush()
            else:  # flow.run_first == 'sender'
                if self.mode == 'remote':
                    if self.sender_side == 'local':
                        second_src = flow.cc_src_remote
                    else:
                        first_src = flow.cc_src_remote

                port = get_open_port()
                
                first_cmd = 'tunnel %s python %s sender %s\n' % (
                    tun_id, first_src, port)
                second_cmd = 'tunnel %s python %s receiver %s %s\n' % (
                    tun_id, second_src, send_pri_ip, port)

                send_manager.stdin.write(first_cmd)
                send_manager.stdin.flush()
                        
        return second_cmd

    def run_second_side(self, send_manager, recv_manager, second_cmds):
        time.sleep(self.run_first_setup_time)

        start_time = time.time()
        self.test_start_time = utc_time()

        # start each flow self.interval seconds after the previous one
        for i in xrange(len(second_cmds)):
            if i != 0:
                time.sleep(self.interval)
            second_cmd = second_cmds[i]

            if self.run_first == 'receiver':
                send_manager.stdin.write(second_cmd)
                send_manager.stdin.flush()
            elif self.run_first == 'sender':
                recv_manager.stdin.write(second_cmd)
                recv_manager.stdin.flush()
            else:
                assert(hasattr(self, 'flow_objs'))
                flow = self.flow_objs[i]
                if flow.run_first == 'receiver':
                    send_manager.stdin.write(second_cmd)
                    send_manager.stdin.flush()
                elif flow.run_first == 'sender':
                    recv_manager.stdin.write(second_cmd)
                    recv_manager.stdin.flush()

        elapsed_time = time.time() - start_time
        if elapsed_time > self.runtime:
            sys.stderr.write('Interval time between flows is too long')
            return False

        time.sleep(self.runtime - elapsed_time)
        self.test_end_time = utc_time()

        return True

    # test congestion control using tunnel client and tunnel server
    def run_with_tunnel(self):
        # run pantheon tunnel server and client managers
        ts_manager, tc_manager = self.run_tunnel_managers()

        # create alias for ts_manager and tc_manager using sender or receiver
        if self.sender_side == self.server_side:
            send_manager = ts_manager
            recv_manager = tc_manager
        else:
            send_manager = tc_manager
            recv_manager = ts_manager

        # run every flow
        second_cmds = []
        for tun_id in xrange(1, self.flows + 1):
            # run tunnel server for tunnel tun_id
            cmd_to_run_tc = self.run_tunnel_server(tun_id, ts_manager)

            # run tunnel client for tunnel tun_id
            if not self.run_tunnel_client(tun_id, tc_manager, cmd_to_run_tc):
                return False

            tc_pri_ip = cmd_to_run_tc[3]  # tunnel client private IP
            ts_pri_ip = cmd_to_run_tc[4]  # tunnel server private IP

            if self.sender_side == self.server_side:
                send_pri_ip = ts_pri_ip
                recv_pri_ip = tc_pri_ip
            else:
                send_pri_ip = tc_pri_ip
                recv_pri_ip = ts_pri_ip

            # run the side that runs first and get cmd to run the other side
            second_cmd = self.run_first_side(
                tun_id, send_manager, recv_manager, send_pri_ip, recv_pri_ip)
            second_cmds.append(second_cmd)

        # run the side that runs second
        if not self.run_second_side(send_manager, recv_manager, second_cmds):
            return False

        # stop all the running flows and quit tunnel managers
        ts_manager.stdin.write('halt\n')
        ts_manager.stdin.flush()
        tc_manager.stdin.write('halt\n')
        tc_manager.stdin.flush()

        # process tunnel logs
        self.process_tunnel_logs()

        return True

    def process_tunnel_logs(self):
        datalink_tun_logs = []
        acklink_tun_logs = []

        apply_ofst = False
        if self.mode == 'remote':
            if self.remote_ofst is not None and self.local_ofst is not None:
                apply_ofst = True

                if self.sender_side == 'remote':
                    data_e_ofst = self.remote_ofst
                    ack_i_ofst = self.remote_ofst
                    data_i_ofst = self.local_ofst
                    ack_e_ofst = self.local_ofst
                else:
                    data_i_ofst = self.remote_ofst
                    ack_e_ofst = self.remote_ofst
                    data_e_ofst = self.local_ofst
                    ack_i_ofst = self.local_ofst

        for i in xrange(self.flows):
            tun_id = i + 1

            if self.mode == 'remote':
                # download logs from remote side
                cmd = 'scp -C %s:' % self.r['host_addr']
                cmd += '%(log)s %(log)s'

                if self.sender_side == 'remote':
                    call(cmd % {'log': self.datalink_egress_logs[i]},
                         shell=True)
                    call(cmd % {'log': self.acklink_ingress_logs[i]},
                         shell=True)
                else:
                    call(cmd % {'log': self.datalink_ingress_logs[i]},
                         shell=True)
                    call(cmd % {'log': self.acklink_egress_logs[i]},
                         shell=True)

            uid = uuid.uuid4()
            datalink_tun_log = path.join(
                TMPDIR, '%s_flow%s_uid%s.log.merged'
                % (self.datalink_name, tun_id, uid))
            acklink_tun_log = path.join(
                TMPDIR, '%s_flow%s_uid%s.log.merged'
                % (self.acklink_name, tun_id, uid))

            cmd = ['merge-tunnel-logs', 'single',
                   '-i', self.datalink_ingress_logs[i],
                   '-e', self.datalink_egress_logs[i],
                   '-o', datalink_tun_log]
            if apply_ofst:
                cmd += ['-i-clock-offset', data_i_ofst,
                        '-e-clock-offset', data_e_ofst]
            call(cmd)

            cmd = ['merge-tunnel-logs', 'single',
                   '-i', self.acklink_ingress_logs[i],
                   '-e', self.acklink_egress_logs[i],
                   '-o', acklink_tun_log]
            if apply_ofst:
                cmd += ['-i-clock-offset', ack_i_ofst,
                        '-e-clock-offset', ack_e_ofst]
            call(cmd)

            datalink_tun_logs.append(datalink_tun_log)
            acklink_tun_logs.append(acklink_tun_log)

        cmd = ['merge-tunnel-logs', 'multiple', '-o', self.datalink_log]
        if self.mode == 'local':
            cmd += ['--link-log', self.mm_datalink_log]
        cmd += datalink_tun_logs
        call(cmd)

        cmd = ['merge-tunnel-logs', 'multiple', '-o', self.acklink_log]
        if self.mode == 'local':
            cmd += ['--link-log', self.mm_acklink_log]
        cmd += acklink_tun_logs
        call(cmd)

    def run_congestion_control(self):
        if self.flows > 0:
            try:
                return self.run_with_tunnel()
            finally:
                kill_proc_group(self.ts_manager)
                kill_proc_group(self.tc_manager)
        else:
            # test without pantheon tunnel when self.flows = 0
            try:
                return self.run_without_tunnel()
            finally:
                kill_proc_group(self.proc_first)
                kill_proc_group(self.proc_second)

    def record_time_stats(self):
        stats_log = path.join(
            self.data_dir, '%s_stats_run%s.log' % (self.cc, self.run_id))
        stats = open(stats_log, 'w')

        # save start time and end time of test
        if self.test_start_time is not None and self.test_end_time is not None:
            test_run_duration = (
                'Start at: %s\nEnd at: %s\n' %
                (self.test_start_time, self.test_end_time))
            sys.stderr.write(test_run_duration)
            stats.write(test_run_duration)

        if self.mode == 'remote':
            ofst_info = ''
            if self.local_ofst is not None:
                ofst_info += 'Local clock offset: %s ms\n' % self.local_ofst

            if self.remote_ofst is not None:
                ofst_info += 'Remote clock offset: %s ms\n' % self.remote_ofst

            if ofst_info:
                sys.stderr.write(ofst_info)
                stats.write(ofst_info)

        stats.close()

    # run congestion control test
    def run(self):
        msg = 'Testing scheme %s for experiment run %d/%d...' % (
            self.cc, self.run_id, self.run_times)
        sys.stderr.write(msg + '\n')

        # setup before running tests
        self.setup()

        # run receiver and sender
        if not self.run_congestion_control():
            sys.stderr.write('Error in testing scheme %s with run ID %d\n' %
                             (self.cc, self.run_id))
            return

        # write runtimes and clock offsets to file
        self.record_time_stats()
        
        sys.stderr.write('Done testing %s\n' % self.cc)

            

def run_tests(args):
    git_summary = get_git_summary(
        args.mode, getattr(args, 'remote_path', None))

    config = parse_config()
    schemes_config = config['schemes']

    if args.all:
        cc_schemes = schemes_config.keys()
        if args.random_order:
            random.shuffle(cc_schemes)
    elif args.schemes is not None:
        cc_schemes = args.schemes.split()
        if args.random_order:
            random.shuffle(cc_schemes)
    else:
        assert(args.test_config is not None)
        if args.random_order:
            random.shuffle(args.test_config['flows'])
        cc_schemes = [flow['scheme'] for flow in args.test_config['flows']]
            
    ssh_cmd = None
    if args.mode == 'remote':
        r = parse_remote_path(args.remote_path)
        ssh_cmd = r['ssh_cmd']

    # For each run of each scheme, change the queueing discipline and
    # receiving socket buffer sizes before and after the test.
    # Check config.yml for values.
    for run_id in xrange(args.start_run_id,
                         args.start_run_id + args.run_times):
        # clean the contents in /tmp/pantheon-tmp
        clean_tmp_cmd = 'rm -rf /tmp/pantheon-tmp/*'
        if args.mode == 'remote':
            call(ssh_cmd + [clean_tmp_cmd])
        call(clean_tmp_cmd, shell=True)

        for cc in cc_schemes:
            default_qdisc = get_default_qdisc(ssh_cmd)
            old_recv_bufsizes = get_recv_sock_bufsizes(ssh_cmd)

            if 'qdisc' in schemes_config[cc]:
                test_qdisc = schemes_config[cc]['qdisc']
            else:
                test_qdisc = config['kernel_attrs']['default_qdisc']

            test_recv_sock_bufs = config['kernel_attrs']['sock_recv_bufs']

            try:
                if default_qdisc != test_qdisc:
                    set_default_qdisc(test_qdisc, ssh_cmd)

                set_recv_sock_bufsizes(test_recv_sock_bufs, ssh_cmd)

                Test(args, run_id, cc).run()
            finally:
                set_default_qdisc(default_qdisc, ssh_cmd)
                set_recv_sock_bufsizes(old_recv_bufsizes, ssh_cmd)

    if not args.no_metadata:
        meta = vars(args).copy()
        meta['cc_schemes'] = sorted(cc_schemes)
        save_test_metadata(meta, path.abspath(args.data_dir), git_summary)


def pkill(args):
    sys.stderr.write('Cleaning up using pkill...'
                     '(enabled by --pkill-cleanup)\n')

    if args.mode == 'remote':
        r = parse_remote_path(args.remote_path)
        remote_pkill_src = path.join(r['pantheon_dir'], 'helpers', 'pkill.py')

        cmd = r['ssh_cmd'] + [
            'python', remote_pkill_src, '--kill-dir', r['pantheon_dir']]
        call(cmd)

    pkill_src = path.join(project_root.DIR, 'helpers', 'pkill.py')
    cmd = ['python', pkill_src, '--kill-dir', project_root.DIR]
    call(cmd)


def main():
    args = parse_arguments(path.basename(__file__))
    print(args)
    try:
        run_tests(args)
    except:  # intended to catch all exceptions
        # dump traceback ahead in case pkill kills the program
        sys.stderr.write(traceback.format_exc())

        if args.pkill_cleanup:
            pkill(args)

        sys.exit('Error in tests!')
    else:
        sys.stderr.write('All tests done!\n')


if __name__ == '__main__':
    main()
