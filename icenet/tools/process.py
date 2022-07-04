# Common input & data reading routines
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import argparse
import yaml
import numpy as np
from termcolor import colored, cprint
import os
import copy
import sys
import pickle
import torch
import xgboost
from pprint import pprint

import icenet.deep.train as train
import icenet.deep.predict as predict


from icenet.tools import io
from icenet.tools import prints
from icenet.tools import aux
from icenet.tools import reweight
from icenet.tools import plots


import matplotlib.pyplot as plt


# ******** GLOBALS *********
roc_mstats        = []
roc_labels        = []
ROC_binned_mstats = []
ROC_binned_mlabel = []

# **************************

def read_cli():

    parser = argparse.ArgumentParser()
    
    ## argparse.SUPPRESS removes argument from the namespace if not passed
    parser.add_argument("--runmode",    type=str, default='null')
    parser.add_argument("--config",     type=str, default='tune0')
    parser.add_argument("--datapath",   type=str, default=".")
    parser.add_argument("--datasets",   type=str, default="*.root")
    parser.add_argument("--tag",        type=str, default='tag0')

    parser.add_argument("--maxevents",  type=int, default=argparse.SUPPRESS)
    parser.add_argument("--inputfiles", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--input_tag",  type=str, default='input0')
    parser.add_argument("--model_tag",  type=str, default='model0')

    cli      = parser.parse_args()
    cli_dict = vars(cli)

    return cli, cli_dict


def read_config(config_path='configs/xyz/', runmode='all'):
    """
    Commandline and YAML configuration reader
    """

    # -------------------------------------------------------------------
    ## Create folders
    aux.makedir('tmp/')

    # -------------------------------------------------------------------
    ## Parse command line arguments
    cli, cli_dict = read_cli()

    # -------------------------------------------------------------------
    ## Read yaml configuration

    # This allows to use "!include foo.yaml" syntax
    from yamlinclude import YamlIncludeConstructor
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir="")

    args = {}
    config_yaml_file = cli.config
    with open(f'{config_path}/{config_yaml_file}', 'r') as f:
        try:
            args = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    

    ## Inputfiles .yaml
    if args["genesis_runmode"]["inputfiles"] is not None:
        with open(f'{config_path}/{args["genesis_runmode"]["inputfiles"]}', 'r') as f:
            try:
                inputfiles = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        inputfiles = {}
    
    # -------------------------------------------------------------------
    # Mode setup

    new_args = {}
    new_args['rootname'] = args['rootname']
    new_args['rngseed']  = args['rngseed']
    
    # -----------------------------------------------------
    if   runmode == 'genesis':
        new_args.update(args['genesis_runmode'])
        new_args.update(inputfiles)
    elif runmode == 'train':
        new_args.update(args['train_runmode'])
        new_args['plot_param'] = args['plot_param']
    elif runmode == 'eval':
        new_args.update(args['eval_runmode'])
        new_args['plot_param'] = args['plot_param']
    elif runmode == 'all':
        for key in args.keys():
            if "_runmode" in key:
                new_args.update(args[key])
        new_args['plot_param'] = args['plot_param']
    else:
        raise Exception(__name__ + f'.read_config: Unknown runmode = {runmode}')
    # -----------------------------------------------------

    old_args = copy.deepcopy(args)
    args     = copy.deepcopy(new_args)

    # -------------------------------------------------------------------
    ## Commandline override of yaml variables
    for key in cli_dict.keys():
        if key in args:
            cprint(__name__ + f'.read_config: Override {config_yaml_file} input with --{key} {cli_dict[key]}', 'red')
            args[key] = cli_dict[key]
    print()

    # -------------------------------------------------------------------
    ## Create a hash

    hash_args = {}
    hash_args.update(old_args['genesis_runmode']) # This first !
    hash_args['rngseed']   = args['rngseed']
    hash_args['maxevents'] = args['maxevents']
    hash_args.update(inputfiles)

    args['__hash__'] = io.make_hash_sha256(hash_args)


    # -------------------------------------------------------------------
    ## Create new variables

    args["config"]     = cli_dict['config']
    args['datadir']    = aux.makedir(f'output/{args["rootname"]}/{cli_dict["config"]}')
    args['modeldir']   = aux.makedir(f'checkpoint/{args["rootname"]}/{cli_dict["config"]}')
    args['root_files'] = io.glob_expand_files(datasets=cli.datasets, datapath=cli.datapath)

    # Technical
    args['__use_cache__']       = True
    args['__raytune_running__'] = False

    # -------------------------------------------------------------------
    ## Create directories
    aux.makedir(args['datadir'])
    aux.makedir(args['modeldir'])

    # -------------------------------------------------------------------
    # Set random seeds for reproducability and train-validate-test splits

    print('')
    print(" torch.__version__: " + torch.__version__)

    cprint(__name__ + f'.read_config: Setting random seed: {args["rngseed"]}', 'yellow')
    np.random.seed(args['rngseed'])
    torch.manual_seed(args['rngseed'])

    # ------------------------------------------------
    print(__name__ + f'.read_config: Created arguments dictionary with runmode = <{runmode}> :')    
    pprint(args)
    # ------------------------------------------------

    return args, cli


def read_data(args, func_loader, impute_vars, runmode):
    """
    Load input data
    
    Args:
        args:  main argument dictionary
        func_loader:  application specific root file loader function
        
    Returns:
        data
    """
    cache_filename = f'{args["datadir"]}/data_{args["__hash__"]}.pkl'

    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):

        load_args = {'entry_start': 0, 'entry_stop': args['maxevents'], 'args': args}

        if runmode != "genesis":
            raise Exception(__name__ + f'.read_data: Data not in cache (or __use_cache__ == False) but --runmode is not "genesis"')

        # N.B. This loop is needed, because certain applications have each root file loaded here,
        # whereas some apps do all the multi-file processing under 'func_loader'
        for k in range(len(args['root_files'])):
            X_,Y_,W_,ids = func_loader(root_path=args['root_files'][k], **load_args)

            if k == 0:
                X,Y,W = copy.deepcopy(X_), copy.deepcopy(Y_), copy.deepcopy(W_)
            else:
                X = np.concatenate((X, X_), axis=0)
                Y = np.concatenate((Y, Y_), axis=0)
                if W is not None:
                    W = np.concatenate((W, W_), axis=0)

        trn, val, tst = io.split_data(X=X, Y=Y, W=W, ids=ids, frac=args['frac'])

        with open(cache_filename, 'wb') as handle:
            print(__name__ + f'.read_data: saving to cache: "{cache_filename}"')
            pickle.dump([trn, val, tst, args], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if args['imputation_param']['active']:
            trn, val, tst, imputer = impute_datasets(trn=trn, val=val, tst=tst, features=impute_vars, args=args['imputation_param'], imputer=None)
            pickle.dump(imputer, open(args["modeldir"] + f'/imputer_{args["__hash__"]}.pkl', 'wb'))

    else:
        with open(cache_filename, 'rb') as handle:
            print(__name__ + f'.read_data: loading from cache: "{cache_filename}"')
            trn, val, tst, genesis_args = pickle.load(handle)

            print(__name__ + f'.read_data: cached data was generated with arguments:')
            pprint(genesis_args)

    return {'trn': trn, 'val': val, 'tst': tst}


def process_data(args, data, func_factor, runmode):

    trn, val, tst = data['trn'], data['val'], data['tst']

    ### Split and factor data
    output = {}
    if   runmode == 'train':

        ### Compute reweighting weights (before funcfactor because we need the variables !)
        if args['reweight']:
            trn.w, pdf = reweight.compute_ND_reweights(pdf=None, x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args['reweight_param'])
            val.w,_    = reweight.compute_ND_reweights(pdf=pdf,  x=val.x, y=val.y, w=val.w, ids=val.ids, args=args['reweight_param'])
            pickle.dump(pdf, open(args["modeldir"] + '/reweight_pdf.pkl', 'wb'))
        
        output['trn'] = func_factor(x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args)
        output['val'] = func_factor(x=val.x, y=val.y, w=val.w, ids=val.ids, args=args)

    elif runmode == 'eval':
        
        ### Compute reweighting weights (before funcfactor because we need the variables !)
        if args['reweight']:
            pdf      = pickle.load(open(args["modeldir"] + '/reweight_pdf.pkl', 'rb'))
            tst.w, _ = reweight.compute_ND_reweights(pdf=pdf, x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args['reweight_param'])
        
        output['tst'] = func_factor(x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args)

    return output

def make_plots(data, args):

    ### Plot variables
    if args['plot_param']['basic']['active']:

        ###
        if data['data_kin'] is not None:
            targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/reweight/1D_kinematic/')
            for k in data['data_kin'].ids:
                plots.plotvar(x = data['data_kin'].x[:, data['data_kin'].ids.index(k)],
                    y = data['data_kin'].y, weights = data['data_kin'].w, var = k, nbins = args['plot_param']['basic']['nbins'],
                    targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")

        ### Plot correlations
        targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/')
        fig,ax    = plots.plot_correlations(X=data['data'].x, weights=data['data'].w, netvars=data['data'].ids, classes=data['data'].y, targetdir=targetdir)
        
        ### Plot basic plots
        targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/1D_distributions/')
        plots.plotvars(X = data['data'].x, y = data['data'].y, weights = data['data'].w, nbins = args['plot_param']['basic']['nbins'], ids = data['data'].ids,
            targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")



def impute_datasets(trn, val, tst, args, features, imputer=None):
    """
    Dataset imputation

    Args:
        trn,val,tst: .x, .y, .w, .ids type object
        args:        imputer parameters
        features:    variables to impute (list)
        imputer:     imputer object (scikit-type)

    Return:
        imputed data
    """

    imputer_trn = None

    if args['active']:

        special_values = args['values'] # possible special values
        print(__name__ + f'.impute_datasets: Imputing data for special values {special_values} for variables in <{args["var"]}>')

        # Choose active dimensions
        dim = np.array([i for i in range(len(trn.ids)) if trn.ids[i] in features], dtype=int)

        # Parameters
        param = {
            "dim":        dim,
            "values":     special_values,
            "labels":     trn.ids,
            "algorithm":  args['algorithm'],
            "fill_value": args['fill_value'],
            "knn_k":      args['knn_k']
        }
        
        trn.x, imputer_trn = io.impute_data(X=trn.x, imputer=imputer,     **param)
        tst.x, _           = io.impute_data(X=tst.x, imputer=imputer_trn, **param)
        val.x, _           = io.impute_data(X=val.x, imputer=imputer_trn, **param)
        
    else:
        # No imputation, but fix spurious NaN / Inf
        trn.x[np.logical_not(np.isfinite(trn.x))] = 0
        val.x[np.logical_not(np.isfinite(val.x))] = 0
        tst.x[np.logical_not(np.isfinite(tst.x))] = 0

    return trn, val, tst, imputer_trn


def train_models(data_trn, data_val, args=None) :
    """
    Train ML/AI models wrapper with pre-processing.
    
    Args:
        Different datatype objects (see the code)
    
    Returns:
        Saves trained models to disk
    """
    
    args["modeldir"] = aux.makedir(f'./checkpoint/{args["rootname"]}/{args["config"]}/')


    # @@ Tensor normalization @@
    if data_trn['data_tensor'] is not None and (args['varnorm_tensor'] == 'zscore'):
            
        print('\nZ-score normalizing tensor variables ...')
        X_mu_tensor, X_std_tensor = io.calc_zscore_tensor(data_trn['data_tensor'])
        
        data_trn['data_tensor'] = io.apply_zscore_tensor(data_trn['data_tensor'], X_mu_tensor, X_std_tensor)
        data_val['data_tensor'] = io.apply_zscore_tensor(data_val['data_tensor'], X_mu_tensor, X_std_tensor)
        
        # Save it for the evaluation
        pickle.dump([X_mu_tensor, X_std_tensor], open(args["modeldir"] + '/zscore_tensor.dat', 'wb'))    
    
    # --------------------------------------------------------------------

    # @@ Truncate outliers (component by component) from the training set @@
    if args['outlier_param']['algo'] == 'truncate' :
        for j in range(data_trn['data'].x.shape[1]):

            minval = np.percentile(data_trn['data'].x[:,j], args['outlier_param']['qmin'])
            maxval = np.percentile(data_trn['data'].x[:,j], args['outlier_param']['qmax'])

            data_trn['data'].x[data_trn['data'].x[:,j] < minval, j] = minval
            data_trn['data'].x[data_trn['data'].x[:,j] > maxval, j] = maxval

    # @@ Variable normalization @@
    if   args['varnorm'] == 'zscore' :

        print('\nZ-score normalizing variables ...')
        X_mu, X_std = io.calc_zscore(data_trn['data'].x)
        data_trn['data'].x  = io.apply_zscore(data_trn['data'].x, X_mu, X_std)
        data_val['data'].x  = io.apply_zscore(data_val['data'].x, X_mu, X_std)

        # Save it for the evaluation
        pickle.dump([X_mu, X_std], open(args['modeldir'] + '/zscore.dat', 'wb'))
        
        prints.print_variables(data_trn['data'].x, data_trn['data'].ids)

    elif args['varnorm'] == 'madscore' :

        print('\nMAD-score normalizing variables ...')
        X_m, X_mad  = io.calc_madscore(data_trn['data'].x)
        data_trn['data'].x  = io.apply_zscore(data_trn['data'].x, X_m, X_mad)
        data_val['data'].x  = io.apply_zscore(data_val['data'].x, X_m, X_mad)

        # Save it for the evaluation
        pickle.dump([X_m, X_mad], open(args['modeldir'] + '/madscore.dat', 'wb'))
    
        prints.print_variables(data_trn['data'].x, data_trn['data'].ids)


    # Loop over active models
    for i in range(len(args['active_models'])):

        ID    = args['active_models'][i]
        param = args['models'][ID]
        print(f'Training <{ID}> | {param} \n')

        ## Different model
        if   param['train'] == 'torch_graph':
            
            inputs = {'data_trn': data_trn['data_graph'],
                      'data_val': data_val['data_graph'],
                      'args':     args,
                      'param':    param}
            
            #### Add distillation, if turned on
            #if args['distillation']['drains'] is not None:
            #    if ID in args['distillation']['drains']:
            #        inputs['y_soft'] = y_soft
            
            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_torch_graph)
            else:
                model = train.train_torch_graph(**inputs)
        
        elif param['train'] == 'xgb':

            inputs = {'data_trn': data_trn['data'],
                      'data_val': data_val['data'],
                      'args':     args,
                      'param':    param}
            
            #### Add distillation, if turned on
            if args['distillation']['drains'] is not None:
                if ID in args['distillation']['drains']:
                    inputs['y_soft'] = y_soft
            
            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_xgb)
            else:
                model = train.train_xgb(**inputs)

        elif param['train'] == 'torch_deps':
            
            inputs = {'X_trn':       torch.tensor(data_trn['data_deps'].x, dtype=torch.float),
                      'Y_trn':       torch.tensor(data_trn['data'].y,      dtype=torch.long),
                      'X_val':       torch.tensor(data_val['data_deps'].x, dtype=torch.float),
                      'Y_val':       torch.tensor(data_val['data'].y,      dtype=torch.long),
                      'X_trn_2D':    None,
                      'X_val_2D':    None,
                      'trn_weights': torch.tensor(data_trn['data'].w, dtype=torch.float),
                      'val_weights': torch.tensor(data_val['data'].w, dtype=torch.float),
                      'args':        args,
                      'param':       param}
            
            #### Add distillation, if turned on
            #if args['distillation']['drains'] is not None:
            #    if ID in args['distillation']['drains']:
            #        inputs['y_soft'] = y_soft

            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_torch_generic)
            else:
                model = train.train_torch_generic(**inputs)        

        elif param['train'] == 'torch_generic':
            
            inputs = {'X_trn':       torch.tensor(data_trn['data'].x, dtype=torch.float),
                      'Y_trn':       torch.tensor(data_trn['data'].y, dtype=torch.long),
                      'X_val':       torch.tensor(data_val['data'].x, dtype=torch.float),
                      'Y_val':       torch.tensor(data_val['data'].y, dtype=torch.long),
                      'X_trn_2D':    None if data_trn['data_tensor'] is None else torch.tensor(data_trn['data_tensor'], dtype=torch.float),
                      'X_val_2D':    None if data_val['data_tensor'] is None else torch.tensor(data_val['data_tensor'], dtype=torch.float),
                      'trn_weights': torch.tensor(data_trn['data'].w, dtype=torch.float),
                      'val_weights': torch.tensor(data_val['data'].w, dtype=torch.float),
                      'args':  args,
                      'param': param}

            #### Add distillation, if turned on
            #if args['distillation']['drains'] is not None:
            #    if ID in args['distillation']['drains']:
            #        inputs['y_soft'] = y_soft
            
            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_torch_generic)
            else:
                model = train.train_torch_generic(**inputs)

        elif param['train'] == 'graph_xgb':
            train.train_graph_xgb(data_trn=data_trn['data_graph'], data_val=data_val['data_graph'], 
                trn_weights=data_trn['data'].w, val_weights=data_val['data'].w, args=args, param=param)  
        
        elif param['train'] == 'flr':
            train.train_flr(data_trn=data_trn['data'], args=args, param=param)

        elif param['train'] == 'flow':
            train.train_flow(data_trn=data_trn['data'], data_val=data_val['data'], args=args, param=param)

        elif param['train'] == 'cut':
            None
        elif param['train'] == 'cutset':
            None
        else:
            raise Exception(__name__ + f'.Unknown param["train"] = {param["train"]} for ID = {ID}')

        # --------------------------------------------------------
        # If distillation
        if ID == args['distillation']['source']:
            print(__name__ + '.train.models: Computing distillation soft targets ...')
            
            if args['num_classes'] != 2:
                raise Exception(__name__ + f'.train_models: Distillation supported now only for 2-class classification')
            
            if   param['train'] == 'xgb':
                y_soft = model.predict(xgboost.DMatrix(data = data_trn['data'].x))[:, args['signalclass']]
            elif param['train'] == 'torch_graph':
                y_soft = model.softpredict(data_trn['data_graph'])[:, args['signalclass']]
            else:
                raise Exception(__name__ + f".train_models: Unsupported distillation source <{param['train']}>")
        # --------------------------------------------------------

    return


def evaluate_models(data=None, args=None):
    """
    Evaluate ML/AI models.

    Args:
        Different datatype objects (see the code)

    Returns:
        Saves evaluation plots to the disk
    """

    print(__name__ + ".evaluate_models: Evaluating models")
    print('')
    
    
    # -----------------------------
    # ** GLOBALS **

    global roc_mstats
    global roc_labels
    global ROC_binned_mstats
    global ROC_binned_mlabel

    #mva_mstats = []
    targetdir  = None

    #MVA_binned_mstats = []
    #MVA_binned_mlabel = []

    ROC_binned_mstats = [list()] * len(args['active_models'])
    ROC_binned_mlabel = [list()] * len(args['active_models'])

    # -----------------------------
    # Prepare output folders

    targetdir  = f'./figs/{args["rootname"]}/{args["config"]}/eval'

    subdirs = ['', 'ROC', 'MVA', 'COR']
    for sd in subdirs:
        os.makedirs(targetdir + '/' + sd, exist_ok = True)
    
    args["modeldir"] = aux.makedir(f'./checkpoint/{args["rootname"]}/{args["config"]}/')

    # --------------------------------------------------------------------
    # Collect data

    X       = None
    X_RAW   = None
    ids_RAW = None

    y       = None
    weights = None
    
    if data['data'] is not None:

        ## Set feature indices for simple cut classifiers
        args['features'] = data['data'].ids

        X        = copy.deepcopy(data['data'].x)
        X_RAW    = data['data'].x
        ids_RAW  = data['data'].ids

        y        = data['data'].y
        weights  = data['data'].w

    X_2D = None
    if data['data_tensor'] is not None:
        X_2D    = data['data_tensor']

    X_graph = None
    if data['data_graph'] is not None:
        X_graph = data['data_graph']

    X_deps = None
    if data['data_deps'] is not None:
        X_deps  = data['data_deps'].x
    
    X_kin    = None
    VARS_kin = None
    if data['data_kin'] is not None:
        X_kin    = data['data_kin'].x
        VARS_kin = data['data_kin'].ids
    # --------------------------------------------------------------------

    if weights is not None: print(__name__ + ".evaluate_models: -- per event weighted evaluation ON ")
    
    try:
        ### Tensor variable normalization
        if data['data_tensor'] is not None and (args['varnorm_tensor'] == 'zscore'):

            print('\nZ-score normalizing tensor variables ...')
            X_mu_tensor, X_std_tensor = pickle.load(open(args["modeldir"] + '/zscore_tensor.dat', 'rb'))
            X_2D = io.apply_zscore_tensor(X_2D, X_mu_tensor, X_std_tensor)
        
        ### Variable normalization
        if   args['varnorm'] == 'zscore':

            print('\nZ-score normalizing variables ...')
            X_mu, X_std = pickle.load(open(args["modeldir"] + '/zscore.dat', 'rb'))
            X = io.apply_zscore(X, X_mu, X_std)

        elif args['varnorm'] == 'madscore':

            print('\nMAD-score normalizing variables ...')
            X_m, X_mad = pickle.load(open(args["modeldir"] + '/madscore.dat', 'rb'))
            X = io.apply_madscore(X, X_m, X_mad)

    except:
        cprint('\n' + __name__ + f' WARNING: {sys.exc_info()[0]} in normalization. Continue without! \n', 'red')
        
    # --------------------------------------------------------------------
    # For pytorch based
    if X is not None:
        X_ptr      = torch.from_numpy(X).type(torch.FloatTensor)

    if X_2D is not None:
        X_2D_ptr   = torch.from_numpy(X_2D).type(torch.FloatTensor)
        
    if X_deps is not None:
        X_deps_ptr = torch.from_numpy(X_deps).type(torch.FloatTensor)
        

    # ====================================================================
    # **  MAIN LOOP OVER MODELS **
    #
    for i in range(len(args['active_models'])):
        
        ID    = args['active_models'][i]
        param = args['models'][ID]
        print(f'Evaluating <{ID}> | {param} \n')
        
        inputs = {'weights': weights, 'label': param['label'],
                 'targetdir': targetdir, 'args':args, 'X_kin': X_kin, 'VARS_kin': VARS_kin, 'X_RAW': X_RAW, 'ids_RAW': ids_RAW}
        
        if   param['predict'] == 'torch_graph':
            func_predict = predict.pred_torch_graph(args=args, param=param)

            # Geometric type -> need to use batch loader, get each graph, node or edge prediction
            import torch_geometric
            loader  = torch_geometric.loader.DataLoader(X_graph, batch_size=len(X_graph), shuffle=False)
            for batch in loader: # Only one big batch
                plot_XYZ_wrap(func_predict = func_predict, x_input = X_graph, y = batch.to('cpu').y.detach().cpu().numpy(), **inputs)
        
        elif param['predict'] == 'graph_xgb':
            func_predict = predict.pred_graph_xgb(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_graph,    y = y, **inputs)
            
        elif param['predict'] == 'torch_vector':
            func_predict = predict.pred_torch_generic(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_ptr,      y = y, **inputs)

        elif param['predict'] == 'torch_scalar':
            func_predict = predict.pred_torch_scalar(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_ptr,      y = y, **inputs)
        
        elif param['predict'] == 'torch_deps':
            func_predict = predict.pred_torch_generic(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_deps_ptr, y = y, **inputs)

        elif param['predict'] == 'torch_image':
            func_predict = predict.pred_torch_generic(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_2D_ptr,   y = y, **inputs)
            
        elif param['predict'] == 'torch_image_vector':
            func_predict = predict.pred_torch_generic(args=args, param=param)

            X_dual      = {}
            X_dual['x'] = X_2D_ptr # image tensors
            X_dual['u'] = X_ptr    # global features
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_dual, y = y, **inputs)
            
        elif param['predict'] == 'flr':
            func_predict = predict.pred_flr(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X,      y = y, **inputs)
            
        elif param['predict'] == 'xgb':
            func_predict = predict.pred_xgb(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X,      y = y, **inputs)

        #elif param['predict'] == 'xtx':
        # ...   
        #
        
        elif param['predict'] == 'torch_flow':
            func_predict = predict.pred_flow(args=args, param=param, n_dims=X_ptr.shape[1])
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_ptr, y = y, **inputs)
            
        elif param['predict'] == 'cut':
            func_predict = predict.pred_cut(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_RAW, y = y, **inputs)
            
        elif param['predict'] == 'cutset':
            func_predict = predict.pred_cutset(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_RAW, y = y, **inputs)
                    
        else:
            raise Exception(__name__ + f'.Unknown param["predict"] = {param["predict"]} for ID = {ID}')

    ## Multiple model comparisons
    plot_XYZ_multiple_models(targetdir=targetdir, args=args)

    return


def plot_XYZ_wrap(func_predict, x_input, y, weights, label, targetdir, args,
    X_kin, VARS_kin, X_RAW, ids_RAW):
    """ 
    Arbitrary plot wrapper function.
    """

    global roc_mstats
    global roc_labels
    global ROC_binned_mstats
    global ROC_binned_mlabel

    # Compute predictions once and for all here
    y_pred = func_predict(x_input)

    # --------------------------------------
    ## Total ROC Plot
    metric = aux.Metric(y_true=y, y_pred=y_pred, weights=weights)
    
    roc_mstats.append(metric)
    roc_labels.append(label)
    # --------------------------------------

    # --------------------------------------
    ### ROC, MVA binned plots
    for i in range(100): # Loop over plot types
        try:
            var   = args['plot_param'][f'plot_ROC_binned[{i}]']['var']
            edges = args['plot_param'][f'plot_ROC_binned[{i}]']['edges']
        except:
            break # No more this type of plots 

        ## 1D
        if   len(var) == 1:

            met_1D, label_1D = plots.binned_1D_AUC(y_pred=y_pred, y=y, weights=weights, X_kin=X_kin, \
                VARS_kin=VARS_kin, edges=edges, label=label, ids=var[0])

            # Save for multiple comparison
            ROC_binned_mstats[i].append(met_1D)
            ROC_binned_mlabel[i].append(label_1D)

            # Plot this one
            plots.ROC_plot(met_1D, label_1D, title = f'{label}', filename=aux.makedir(f'{targetdir}/ROC/{label}') + f'/ROC_binned[{i}]')
            plots.MVA_plot(met_1D, label_1D, title = f'{label}', filename=aux.makedir(f'{targetdir}/MVA/{label}') + f'/MVA_binned[{i}]')

        ## 2D
        elif len(var) == 2:

            fig, ax, met = plots.binned_2D_AUC(y_pred=y_pred, y=y, weights=weights, X_kin=X_kin, \
                VARS_kin=VARS_kin, edges=edges, label=label, ids=var)

            plt.savefig(aux.makedir(f'{targetdir}/ROC/{label}') + f'/ROC_binned[{i}].pdf', bbox_inches='tight')
            
        else:
            print(var)
            raise Exception(__name__ + f'.plot_AUC_wrap: Unknown dimensionality {len(var)}')

    # ----------------------------------------------------------------
    ### MVA  1D plot
    hist_edges = args['plot_param'][f'MVA_output']['edges']

    inputs = {'y_pred': y_pred, 'y': y, 'weights': weights, 'hist_edges': hist_edges, \
        'label': f'{label}', 'path': targetdir + '/MVA/'}

    plots.density_MVA_wclass(**inputs)

    # ----------------------------------------------------------------
    ### COR 2D plots

    for i in range(100): # Loop over plot types
        try:
            var   = args['plot_param'][f'plot_COR[{i}]']['var']
            edges = args['plot_param'][f'plot_COR[{i}]']['edges']
        except:
            break # No more this type of plots 

        inputs = {'y_pred': y_pred, 'weights': weights, 'X_RAW': X_RAW, 'ids_RAW': ids_RAW, \
            'label': f'{label}', 'hist_edges': edges, 'path': targetdir + '/COR/'}

        plots.density_COR_wclass(y=y, **inputs)
        #plots.density_COR(**inputs) 

    return True


def plot_XYZ_multiple_models(targetdir, args):

    global roc_mstats
    global roc_labels
    global ROC_binned_mstats

    # ===================================================================
    # ** Plots for multiple model comparison **

    ### Plot all ROC curves
    plots.ROC_plot(roc_mstats, roc_labels, title = '', filename=aux.makedir(targetdir + '/ROC/__ALL__') + '/ROC')

    ### Plot all MVA outputs (not implemented)
    #plots.MVA_plot(mva_mstats, mva_labels, title = '', filename=aux.makedir(targetdir + '/MVA/__ALL__') + '/MVA')

    ### Plot all binned ROC curves
    for i in range(100):
        try:
            var   = args['plot_param'][f'plot_ROC_binned[{i}]']['var']
            edges = args['plot_param'][f'plot_ROC_binned[{i}]']['edges']
        except:
            return # No more plots 

        if len(var) == 1:

            # Over different bins
            for b in range(len(edges)-1):

                # Over different models
                xy,legs = [],[]
                for k in range(len(ROC_binned_mstats[i])):
                    xy.append(ROC_binned_mstats[i][k][b])

                    # Take label for the legend
                    ID    = args['active_models'][k]
                    label = args['models'][ID]['label']
                    legs.append(label)

                ### ROC
                title = f'BINNED ROC: {var[0]}$ \\in [{edges[b]:0.1f}, {edges[b+1]:0.1f})$'
                plots.ROC_plot(xy, legs, title=title, filename=targetdir + f'/ROC/__ALL__/ROC_binned[{i}]_bin[{b}]')

                ### MVA (not implemented)
                #title = f'BINNED MVA: {var[0]}$ \\in [{edges[b]:0.1f}, {edges[b+1]:0.1f})$'
                #plots.MVA_plot(xy, legs, title=title, filename=targetdir + f'/MVA/__ALL__/MVA_binned[{i}]_bin[{b}]')

    return True
