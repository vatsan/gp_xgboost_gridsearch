-------------------------------------------------------------------------------------------------------------------------------
--                                      Build XGBoost models in-database                                                     --
--                                   Srivatsan Ramanujam<vatsan.cs@utexas.edu>                                               --
-------------------------------------------------------------------------------------------------------------------------------
--
-----------
-- Note: --
-----------
-- 1) The design of this pipeline uses XGBoost (https://github.com/dmlc/xgboost)
--    A grid-search on model parameters is distributed across all nodes such that each node will build a model for a specific 
--    set of parameters. In this sense, training happens in parallel on all nodes. However we're limited by the maximum 
--    field-size in Greenplum/Postgres which is currently 1 GB. 
-- 2) If your dataset is much larger (> 1 GB), it is strongly recommended that you use MADlib's models so that 
--    training & scoring will happen in-parallel on all nodes.
-------------------------------------------------------------------------------------------------------------------------------

create schema xgbdemo;

---------------------------------------------------------------------------------------------------------
--1) XGBoost parallel grid-search
---------------------------------------------------------------------------------------------------------

--1) Create function to construct a dataframe and serialize it.
drop function if exists xgbdemo.__serialize_pandas_dframe_as_bytea__(
    text,
    text,
    text,
    text,
    text[]
);

create or replace function xgbdemo.__serialize_pandas_dframe_as_bytea__(
    features_schema text,
    features_tbl text,
    id_column text,
    class_label text,
    exclude_columns text[]
)
returns bytea
as
$$
    import pandas as pd
    import cPickle as pickle
    from sklearn.preprocessing import Imputer    
    #http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn/25562948#25562948
    from sklearn.base import TransformerMixin
    import numpy as np
    class DataFrameImputer(TransformerMixin):
        def __init__(self):
            """Impute missing values.

            Columns of dtype object are imputed with the most frequent value 
            in column.

            Columns of other types are imputed with mean of column.

            """
        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                index=X.columns)

            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)    
    #1) Extract feature names from information_schema
    discard_features = exclude_columns + [class_label]
    sql = """
        select
            column_name
        from
            information_schema.columns
        where
            table_schema = '{features_schema}' and
            table_name = '{features_tbl}' and
            column_name not in {exclude_columns}
        group by
            column_name
        order by
            column_name
    """.format(
        features_schema = features_schema,
        features_tbl = features_tbl,
        exclude_columns = str(discard_features).replace('[','(').replace(']',')')
    )
    result = plpy.execute(sql)
    features = [r['column_name'] for r in result]
    #2) Fetch dataset for model training
    mdl_train_sql = """
        select
            {id_column},
            {features},
            {class_label}
        from
            {features_schema}.{features_tbl}
    """.format(
        features_schema = features_schema,
        features_tbl = features_tbl,
        features = ','.join(features),
        id_column = id_column,
        class_label = class_label
    )

    result = plpy.execute(mdl_train_sql)
    df = pd.DataFrame.from_records(result)
    #Drop any columns which are all null
    df_filtered = df.dropna(axis=1, how='all')
    #Impute missing values before persisting this DFrame
    imp = DataFrameImputer()
    imp.fit(df_filtered)
    df_imputed = imp.transform(df_filtered)
    return pickle.dumps(df_imputed)
$$ language plpythonu;


--2) UDF to train XGBoost
drop type if exists xgbdemo.mdl_gridsearch_train_results_type cascade;
create type xgbdemo.mdl_gridsearch_train_results_type
as
(
    metrics text,
    features text[],
    mdl bytea,
    params text
);

drop function if exists xgbdemo.__xgboost_train_parallel__(
    bytea,
    text[],
    text,
    text,
    text
);

create or replace function xgbdemo.__xgboost_train_parallel__(
    dframe bytea,
    features_all text[],
    class_label text,
    params text,
    class_weights text    
)
returns xgbdemo.mdl_gridsearch_train_results_type
as
$$
    import plpy, re 
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix    
    import xgboost as xgb
    import numpy
    import cPickle as pickle
    import ast
    from sklearn.preprocessing import Imputer
    #
    def print_prec_rec_fscore_support(mat, metric_labels, class_labels):
        """
           pretty print precision, recall, fscore & support using pandas dataframe
        """
        tbl = pd.DataFrame(mat, columns=metric_labels)
        tbl['class'] = class_labels
        tbl = tbl[['class']+metric_labels]
        return tbl    
    #1) Load the dataset for model training
    df = pickle.loads(dframe)
    #2) Train XGBoost model & return a serialized representation to store in table
    features = filter(lambda x: x in df.columns, features_all)
    X = df[features]
    y = df[class_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #
    sample_representation = y_train.value_counts()
    total_samples = sum(sample_representation)
    sample_weight = None
    if not class_weights:
        sample_weight = map(
                lambda s: total_samples*1.0/sample_representation[s]
                                /
                sum([total_samples*1.0/sample_representation[c] for c in sample_representation.keys()])
                ,
                y_train
            )
    else:
        #User-supplied class-weights
        class_weights_dict = ast.literal_eval(re.sub("[\\t]","",class_weights).strip())
        sample_weight = map(lambda s: class_weights_dict[s], y_train)
    #Train gradient boosted trees
    p_list = [p.split('=') for p in ast.literal_eval(re.sub("[\\t]","",params).strip())]
    params_dict = dict([(k, ast.literal_eval(v.strip())) for k,v in p_list])
    gbm = xgb.XGBClassifier(**params_dict)
    #Fit model
    gbm.fit(
        X_train, 
        y_train, 
        eval_metric = 'auc',
        sample_weight = sample_weight
    )    
    #3) Compute and return model metrics score
    y_pred_train = gbm.predict(X_train)
    y_pred_test = gbm.predict(X_test)
    cmat_train = confusion_matrix(y_train, y_pred_train)
    cmat_test = confusion_matrix(y_test, y_pred_test)
    scores = numpy.array(precision_recall_fscore_support(y_test, y_pred_test)).transpose()
    metric_labels = ['precision', 'recall', 'fscore', 'support']
    model_metrics = print_prec_rec_fscore_support(scores, metric_labels, gbm.classes_)
    return (model_metrics.to_string(), features, pickle.dumps(gbm), params)
$$ language plpythonu;


--3) XGBoost grid search
drop function if exists xgbdemo.xgboost_grid_search(
        text, 
        text, 
        text, 
        text, 
        text[], 
        text, 
        text, 
        text,
        text
);

create or replace function xgbdemo.xgboost_grid_search(
    features_schema text,
    features_tbl text,
    id_column text,
    class_label text,
    exclude_columns text[],
    params_str text,
    grid_search_params_temp_tbl text,
    grid_search_results_tbl text,
    class_weights text    
)
returns text
as
$$
    import plpy
    #1) Expand the grid-search parameters
    import collections, itertools, ast, re
    #Expand the params to run-grid search
    params = ast.literal_eval(re.sub("[\\t]","",params_str).strip())
    def expand_grid(params):
        """
           Expand a dict of parameters into a grid
        """
        import collections, itertools
        #Expand the params to run-grid search
        params_list = []
        for key, val  in params.items():
            #If supplied param is a list of values, expand it out
            if(val and isinstance(val, collections.Iterable)):
                r = ["""{k}={v}""".format(k=key,v=v) for v in val]
            else:
                r = ["""{k}={v}""".format(k=key,v=val)]
            params_list.append(r)
        params_grid = [l for l in itertools.product(*params_list)]
        return params_grid

    params_grid = expand_grid(params)
    #2) Save each parameter list in the grid as a row in a distributed table
    sql = """
        drop table if exists {grid_search_params_temp_tbl};
        create temp table {grid_search_params_temp_tbl}
        (
            params_indx int,
            params text
        ) distributed by (params_indx);
    """.format(grid_search_params_temp_tbl=grid_search_params_temp_tbl)
    plpy.execute(sql)
    sql = """
        insert into {grid_search_params_temp_tbl}
            values ({params_indx}, $X${val}$X$);
    """
    for indx, val in enumerate(params_grid):
        plpy.execute(
            sql.format(
                val=val, 
                params_indx = indx+1, #postgres indices start from 1, so keeping it consistent
                grid_search_params_temp_tbl=grid_search_params_temp_tbl
            )
        )
    #3) Extract feature names from information_schema
    discard_features = exclude_columns + [class_label]
    sql = """
        select
            column_name
        from
            information_schema.columns
        where
            table_schema = '{features_schema}' and
            table_name = '{features_tbl}' and
            column_name not in {exclude_columns}
        group by
            column_name
        order by
            column_name
    """.format(
        features_schema = features_schema,
        features_tbl = features_tbl,
        exclude_columns = str(discard_features).replace('[','(').replace(']',')')
    )
    result = plpy.execute(sql)
    features = [r['column_name'] for r in result]
    #4) Extract features from table and persist as serialized dataframe
    sql = """
        drop table if exists {grid_search_params_temp_tbl}_df;
        create temp table {grid_search_params_temp_tbl}_df
        as
        (
            select
                df,
                generate_series(1, {grid_size}) as params_indx
            from
            (
                select
                    xgbdemo.__serialize_pandas_dframe_as_bytea__(
                        '{features_schema}',
                        '{features_tbl}',
                        '{id_column}',
                        '{class_label}',
                        ARRAY[{exclude_columns}]
                    ) as df 
            )q
        ) distributed by (params_indx);
    """.format(
        grid_search_params_temp_tbl = grid_search_params_temp_tbl,
        grid_size = len(params_grid),
        features_schema = features_schema,
        features_tbl = features_tbl,
        id_column = id_column,
        class_label = class_label,
        exclude_columns = str(exclude_columns).replace('[','').replace(']','')
    )
    plpy.execute(sql)

    #5) Invoke XGBoost's train by passing each row from parameter list table. This will run in parallel.
    sql = """
        drop table if exists {grid_search_results_tbl};
        create table {grid_search_results_tbl}
        as
        (
            select
                now() as mdl_train_ts,
                '{features_schema}.{features_tbl}'||'_xgboost' as mdl_name,
                (mdl_results).metrics,
                (mdl_results).features,
                (mdl_results).mdl,
                (mdl_results).params,
                params_indx
            from
            (
                select
                    xgbdemo.__xgboost_train_parallel__(
                        df,
                        ARRAY[
                            {features}
                        ],
                        '{class_label}',
                        params,
                        $CW${class_weights}$CW$
                    ) as mdl_results,
                    t1.params_indx
                from 
                    {grid_search_params_temp_tbl} t1,
                    {grid_search_params_temp_tbl}_df t2
                where 
                    t1.params_indx = t2.params_indx
            )q
        ) distributed by (params_indx);    
    """.format(
        grid_search_results_tbl = grid_search_results_tbl,
        grid_search_params_temp_tbl = grid_search_params_temp_tbl,
        features = str(features).replace('[','').replace(']','').replace(',',',\n'),
        features_schema = features_schema,
        features_tbl = features_tbl,
        class_label = class_label,
        class_weights = class_weights
    )
    plpy.execute(sql)
    return """Grid search results saved in {tbl}""".format(tbl = grid_search_results_tbl)
$$ language plpythonu;

---------------------------------------------------------------------------------------------------------
--2) XGBoost : Model Scoring
---------------------------------------------------------------------------------------------------------

drop function if exists xgbdemo.xgboost_mdl_score(text, text, text, text, text, text);
create or replace function xgbdemo.xgboost_mdl_score(
    scoring_tbl text,
    id_column text,
    class_label text,
    mdl_table text,
    mdl_filters text,
    mdl_output_tbl text
)
returns text
as
$$
    import plpy
    import pandas as pd
    from operator import itemgetter
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import confusion_matrix
    import xgboost as xgb
    import numpy
    import cPickle as pickle
    # Confusion Matrix
    def print_prec_rec_fscore_support(mat, metric_labels, class_labels):
        """
           pretty print precision, recall, fscore & support using pandas dataframe
        """
        tbl = pd.DataFrame(mat, columns=metric_labels)
        tbl['class'] = class_labels
        tbl = tbl[['class']+metric_labels]
        return tbl
    #1) Load the serialized XGBoost model from the table
    mdl_sql = """
        select
            mdl,
            features
        from
            {mdl_table}
        where
            {mdl_filters}
        """.format(
            mdl_table = mdl_table,
            mdl_filters = mdl_filters
        )
    result = plpy.execute(mdl_sql)
    mdl = result[0]['mdl']
    features = result[0]['features']
    #Train gradient boosted trees
    gbm = pickle.loads(mdl)        
    #2) Fetch features from test dataset for scoring
    mdl_score_sql = ""
    if(class_label):
        mdl_score_sql = """
            select
                {id_column},
                {features},
                {class_label}
            from
                {scoring_tbl}
        """.format(
            scoring_tbl = scoring_tbl,
            id_column = id_column,
            features = ','.join(features),
            class_label = class_label
        )
    else:
        mdl_score_sql = """
            select
                {id_column},
                {features}
            from
                {scoring_tbl}
        """.format(
            scoring_tbl = scoring_tbl,
            id_column = id_column,
            features = ','.join(features)
        )
    result = plpy.execute(mdl_score_sql)
    df = pd.DataFrame.from_records(result)
    X_test = df[features]
    y_test = df[class_label] if class_label else None    
    #3) Score the test set
    y_pred_test = gbm.predict(X_test)
    if(class_label):
        cmat_test = confusion_matrix(y_test, y_pred_test)
        scores = numpy.array(precision_recall_fscore_support(y_test, y_pred_test)).transpose()
        metric_labels = ['precision', 'recall', 'fscore', 'support']
        model_metrics = print_prec_rec_fscore_support(scores, metric_labels, gbm.classes_).to_string()
    else:
        model_metrics = 'NA'
    predicted_class_label = class_label+'_predicted' if class_label else 'class_label_predicted'
    res_df = df.join(pd.Series(y_pred_test, index = X_test.index).to_frame(predicted_class_label))
    #4) Feature importance scores
    importance = gbm.booster().get_fscore()
    fnames_importances = sorted(
                [(features[int(k.replace('f',''))], importance[k]) for k in importance], 
                key=itemgetter(1), 
                reverse=True
            )
    fnames, f_importance_scores = zip(*fnames_importances)
    ret_dict = res_df.to_dict('records')
    ret_result = (
            (
                r[id_column], 
                r[predicted_class_label], 
                model_metrics,
                str(fnames).replace('(','{').replace(')','}').replace('\'','\"'),
                str(f_importance_scores).replace('(','{').replace(')','}').replace('\'','\"'),
                str([str(r[key]) for key in fnames]).replace('[','{').replace(']','}').replace('\'','\"')
            )
            for r in ret_dict
        )
    sql = """
        drop table if exists {mdl_output_tbl};
        create table {mdl_output_tbl}
        (
            {id_column} text,
            {predicted_class_label} text,
            metrics text,
            feature_names text[],
            feature_importance_scores float8[],
            feature_values text[]
        ) distributed by ({id_column});
    """.format(
        mdl_output_tbl = mdl_output_tbl,
        id_column = id_column,
        predicted_class_label = predicted_class_label
    )
    plpy.execute(sql)
    sql = """
        insert into {mdl_output_tbl}
        values {row};
    """
    for row in ret_result:
        plpy.execute(sql.format(mdl_output_tbl = mdl_output_tbl, row = row))
    return 'Scoring results written to {mdl_output_tbl}'.format(mdl_output_tbl = mdl_output_tbl)
$$language plpythonu;

---------------------------------------------------------------------------------------------------------
--  Sample invocation                                                                                  --
---------------------------------------------------------------------------------------------------------

---------------------
-- Model training
---------------------

select
    xgbdemo.xgboost_grid_search(
        'sr',--training table_schema
        'features_tbl',--training table_name
        'id', -- id column
        'multi_class_label', -- class label column
        -- Columns to exclude from features (independent variables)
        ARRAY[
            'id', 
            'multi_class_label', 
            'year'
        ],
        --XGBoost grid search parameters
        $$
            {
                'learning_rate': [0.1, 0.3, 0.4], #Regularization on weights (eta). For smaller values, increase n_estimators
                'max_depth': [12, 14],#Larger values could lead to overfitting
                'subsample': [0.9, 1.0],#introduce randomness in samples picked to prevent overfitting
                'colsample_bytree': [0.9, 1.0],#introduce randomness in features picked to prevent overfitting
                'min_child_weight':[1, 4],#larger values will prevent over-fitting
                'n_estimators':[200, 400, 600] #More estimators, lesser variance (better fit on test set)
            }
        $$,
        --Grid search parameters temp table (will be dropped when session ends)
        'xgb_params_temp_tbl',
        --Grid search results table.
        'xgbdemo.xgb_mdl_results',
        --class weights (set it to empty string '' if you want it to be automatic)
        $$
            {
                'class_a':0.25,
                'class_b':0.50,
                'class_c':0.25
            }
        $$
    );

---------------------
-- Model scoring
---------------------

select
    xgbdemo.xgboost_mdl_score(
        'xgbdemo.scoring_tbl', -- scoring table
        'id', -- id column
        NULL, -- class label column, NULL if unavailable
        'xgbdemo.xgb_mdl_results', -- model table
        'params_indx = 5', -- model filter, set to 'True' if no filter
        'xgbdemo.xgb_mdl_scoring_results'
    );    

---------------------------------------------------------------------------------------------------------
--                                                                                                     --
---------------------------------------------------------------------------------------------------------    
