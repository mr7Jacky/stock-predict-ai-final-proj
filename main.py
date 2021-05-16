from run_handler import *

if __name__ == '__main__':
    # Obtain argments from command line
    parser = argparse.ArgumentParser(description='Stock Prediction')
    parser.add_argument('-path', default='./data/AAPL.csv', type=str)
    parser.add_argument('-model', type=str, default='lstm')
    parser.add_argument('-load', type=int, default=1)
    parser.add_argument('-oname', type=str, default='result')
    parser.add_argument('-save_model', type=int, default=0)

    args = parser.parse_args()

    data = pd.read_csv(args.path, date_parser=True)
    if args.load == 1:
        exe_load(args, data)
    else:
        exe_new(args, data)
    exe_q_l(args)
    
    

    
    