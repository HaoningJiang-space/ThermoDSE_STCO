
import argparse
import sys
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

sys.path.append('../')


from core.chiplet_eva import chiplet_evaluator

def argparser():
    ''' Argument parser. '''

    ap = argparse.ArgumentParser()

    ap.add_argument('net', nargs='+',
                    help='network name, should be a .py file under examples, could be a list', default=['toy_net'])

    ap.add_argument('-b', '--batch', type=int, default=1,
                    help='batch size')
    ap.add_argument('-w', '--word', type=int, default=8,
                    help='word size in bits')

    return ap

def target_function(cs, cc, ci, hsa, wsa, ubf):
    chipletSize = int(cs) * int(cc)
    chipletCut = int(cc)
    chipletIntvl = 0.0005 + int(ci) * 0.0001
    h_sa = 16 * int(hsa)  
    w_sa = 16 * int(wsa)
    ubuf_size = 128 * 2** int(ubf) *1024
    nop_bw = 64
    sys_info = [chipletSize, chipletCut, chipletIntvl, h_sa, w_sa, ubuf_size, nop_bw]
    print(f'sys_info:{sys_info}, cost:')
    evaluator = chiplet_evaluator(nn_wld, interposer_cons=0.03, hotspot_path='../../HotSpot', sim_path='../tmp', sys_info= sys_info)
    evaluator.generate_hardware()
    delay, energy, die_yield = evaluator.evaluate(batch=2)
    # print(f'{EDY_cost}\n')
    EDY_cost = - delay * energy / die_yield
    print(f'E: {energy} D: {delay}, Yield: {die_yield}. The EDYP cost is {EDY_cost}')
    return EDY_cost


if __name__ == '__main__':
    arg = argparser().parse_args()
    nn_wld = arg.net[0]
    batch = arg.batch
    ## hardware search space
    pbounds = {'cs':(1,4+1), 'cc':(1,6+1), 'ci': (1, 25+1), 'hsa':(1, 8+1), 'wsa': (1,8+1), 'ubf':(1, 6)}
    optimizer = BayesianOptimization(
        f = target_function,
        pbounds = pbounds,
        random_state = 7777,
        verbose = 0
    )
    optimizer.probe(params={'cs':1, 'cc':6, 'ci': 20, 'hsa':2, 'wsa': 4, 'ubf':2}, lazy=True,)
    optimizer.probe(params={'cs':1, 'cc':6, 'ci': 25, 'hsa':2, 'wsa': 4, 'ubf':2}, lazy=True,)
    optimizer.probe(params={'cs':2, 'cc':4, 'ci': 25, 'hsa':2, 'wsa': 2, 'ubf':2}, lazy=True,)
    optimizer.probe(params={'cs':1, 'cc':1, 'ci': 1 , 'hsa':8, 'wsa': 8, 'ubf':5}, lazy=True,)

    # optimizer.maximize(init_points= 5, n_iter= 50)

    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))

    # new_optimizer = BayesianOptimization(
    # f = target_function,
    # pbounds = pbounds,
    # verbose=2,
    # random_state=7,)
    # load_logs(optimizer, logs=["./logs.log"])
    
    # print(len(optimizer.space))
    logger = JSONLogger(path="./logs.log", reset=True)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points= 10, n_iter= 500)




    