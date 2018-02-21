from args import parse_args



def main():
    args = parse_args()
    print(args)
    if args is None:
        exit()

if __name__ == '__main__':
    main()
