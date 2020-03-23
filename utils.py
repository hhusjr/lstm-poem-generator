"""
工具
@author Junru Shen
"""


def log(message, pause=False):
    print('Message: {}'.format(message))
    if pause:
        print('Press enter to continue...')
        input()
