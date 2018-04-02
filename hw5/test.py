# OpenSSH_7.2p2 Ubuntu-4ubuntu2.4, OpenSSL 1.0.2g  1 Mar 2016\r
# debug1: Reading configuration data /etc/ssh/ssh_config\r
# debug1: /etc/ssh/ssh_config line 19: Applying options for *\r
# debug1: auto-mux: Trying existing master\r
# debug2: fd 3 setting O_NONBLOCK\r
# debug2: mux_client_hello_exchange: master version 4\r
# debug3: mux_client_forwards: request forwardings: 0 local, 0 remote\r
# debug3: mux_client_request_session: entering\r
# debug3: mux_client_request_alive: entering\r
# debug3: mux_client_request_alive: done pid = 29378\r
# debug3: mux_client_request_session: session request sent\r
# debug1: mux_client_request_session: master session id: 2\r
# debug3: mux_client_read_packet: read header failed: Broken pipe\r
# debug2: Received exit status from master 0\r
# Shared connection to s03-k8s-master.maas closed.\r
# ",
#     "module_stdout": "Traceback (most recent call last):\r
#   File \"/tmp/ansible_8Jblhd/ansible_module_lineinfile.py\", line 461, in <module>\r
#     main()\r
#   File \"/tmp/ansible_8Jblhd/ansible_module_lineinfile.py\", line 458, in main\r
#     absent(module, path, params['regexp'], params.get('line', None), backup)\r
#   File \"/tmp/ansible_8Jblhd/ansible_module_lineinfile.py\", line 360, in absent\r
#     f = open(b_dest, 'rb')\r
# IOError: [Errno 13] Permission denied: '/etc/sudoers'\r
