import icenet.algo
import icenet.deep
import icenet.optim
import icenet.tools

from datetime import datetime
import socket
import os

__version__    = '0.0.8.6'
__release__    = 'alpha'
__date__       = '29/05/2023'
__author__     = 'm.mieskolainen@imperial.ac.uk'
__repository__ = 'github.com/mieskolainen/icenet'
__asciiart__   = \
"""
ICENET
"""
print(f'{__asciiart__} version: {__version__} | date: {__date__} | author: {__author__} | repository: {__repository__}')
print(f' {datetime.now()} | hostname: {socket.gethostname()} | CPU cores: {os.cpu_count()} ')
print('')
