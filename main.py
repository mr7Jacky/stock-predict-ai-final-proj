from run_handler import *
import sys

if __name__ == '__main__':
    """
    System Launcher

    options:
        -path: Directory path of stock data, default is the provided AAPL.csv
        -model: Type of NN model can be chosen from lstm or drnn. Default is `lstm`.
        -load: 0 or 1, if load saved model or train a new model. Default is `0`.
        -oname: name for saved files, any string is legal
        -save_model: 0 or 1, if save the newly trained model. Default is `0`.
    """
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
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
