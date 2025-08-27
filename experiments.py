import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from topopipeline import topopipeline



def param_range_experiment(rf_params, tp_params, param_name, param_range, 
                           feature_dicts, num_seeds, filename=None):
    

    num_param_values = len(param_range)

    tp_params_ = tp_params.copy()

    results = np.zeros(shape=(num_param_values, num_seeds, len(feature_dicts), 2))

    for i, param_value in enumerate(param_range):

        if param_name == 'r_range':

            tp_params_['r_min'] = 0.25 - param_value / 2
            tp_params_['r_max'] = 0.25 + param_value / 2
        
        else:

            tp_params_[param_name] = param_value
        

        for j in range(num_seeds):

            tp = topopipeline.random(seed=j, **tp_params_)

            for k, fs in enumerate(feature_dicts):

                model = RandomForestClassifier(random_state=j, **rf_params)

                if isinstance(fs, list):

                    X = np.column_stack( [ getattr(tp, f['feature_name'])(**f['params']) if 'params' in f
                                           else getattr(tp, f['feature_name'])() for f in fs ] )
                
                else:

                    X = getattr(tp, fs['feature_name'])(**fs['params']) if 'params' in fs else getattr(tp, fs['feature_name'])()
                
                y = tp.get_labels()

                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=j)

                model.fit(X_train, y_train)

                results[i,j,k,0] = model.score(X_train, y_train)
                results[i,j,k,1] = model.score(X_test, y_test)

            print(f"\rcurrent param idx: {i+1}/{len(param_range)} , current seed: {j+1}/{num_seeds}", end="")

    print('\n')
    
    if filename is not None:

        np.save(filename, results)
    
    return results





def pc_subsample_experiment(rf_params, tp_params, bootstrap_params, 
                            feature_dicts, num_seeds, filename=None):


    tp_params_ = tp_params.copy()
    tpb_params_ = tp_params.copy()

    for param in bootstrap_params:

        tpb_params_[param] = bootstrap_params[param]
    

    results = np.zeros(shape=(2, num_seeds, len(feature_dicts), 2))

    for j in range(num_seeds):

        tp = topopipeline.random(seed=j, **tp_params_)
        tpb = topopipeline.random(seed=j, bootstrap=True, **tpb_params_)

        for k, fs in enumerate(feature_dicts):
            
            model = RandomForestClassifier(random_state=j, **rf_params)
            modelb = RandomForestClassifier(random_state=j, **rf_params)

            if isinstance(fs, list):

                X = np.column_stack( [ getattr(tp, f['feature_name'])(**f['params']) if 'params' in f
                                       else getattr(tp, f['feature_name'])() for f in fs ] )
                
                Xb = np.column_stack( [ getattr(tpb, f['feature_name'])(**f['params']) if 'params' in f
                                       else getattr(tp, f['feature_name'])() for f in fs ] )
            
            else:

                X = getattr(tp, fs['feature_name'])(**fs['params']) if 'params' in fs else getattr(tp, fs['feature_name'])()
                
                Xb = getattr(tpb, fs['feature_name'])(**fs['params']) if 'params' in fs else getattr(tpb, fs['feature_name'])()
            
            y = tp.get_labels()

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=j)
            Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, y, stratify=y, test_size=0.25, random_state=j)

            model.fit(X_train, y_train)
            modelb.fit(Xb_train, yb_train)

            results[0,j,k,0] = model.score(X_train, y_train)
            results[0,j,k,1] = model.score(X_test, y_test)
            results[1,j,k,0] = modelb.score(Xb_train, yb_train)
            results[1,j,k,1] = modelb.score(Xb_test, yb_test)

        print(f"\rcurrent seed: {j+1}/{num_seeds}", end="")
    
    print('\n')
    
    if filename is not None:

        np.save(filename, results)

    return results