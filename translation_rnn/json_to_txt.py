import json
import sys


def main(path):
    for line in open(path, 'rt'):
        item = json.loads(line)
        if item['en']:
            print(item['en'])
        if item['ru']:
            print(item['ru'])


if __name__ == '__main__':
    main(sys.argv[1])

