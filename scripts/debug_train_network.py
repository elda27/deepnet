from logging import basicConfig, DEBUG


import os.path
import sys

def main():
    basicConfig(level=DEBUG)

    root = os.path.dirname(os.path.abspath(__file__))
    script_filename = os.path.join(root, 'train_network.py')

    with open(script_filename, 'r') as fp:
        exec(compile(fp.read(), script_filename, 'exec'), globals())
        
if __name__ == '__main__':
    main()
