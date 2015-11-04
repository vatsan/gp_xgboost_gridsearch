##In-database parallel grid-search wrapper for XGBoost

### Dependencies 

1. [XGBoost library](https://github.com/dmlc/xgboost)
2. [scikit-learn](http://scikit-learn.org/stable/)

### Installation

1. Download XGBoost and sklearn libraries to all nodes of your Greenplum cluster (using `gpscp -f hostfile <source> =:<destination>`)
2. Compile and install XGBoost and sklearn libraries on all nodes (using `gpssh -f hostfile` followed by the compile and install commands)
3. Run the above SQL file (it will create a schema called `xgbdemo`). 
4. Invoke the UDFs as shown in the sample snippet.

### Note: XGBoost and Python 2.6 
Since the XGBoost implementation in https://github.com/dmlc/xgboost is not Python 2.6 compatible, I recommend you clone my version from https://github.com/vatsan/xgboost and use it instead (Python 2.6 compatible, will work with PL/Python on Greenplum/HAWQ).

### Implementation details
![In-database parallel grid-search](https://raw.githubusercontent.com/vatsan/gp_xgboost_gridsearch/master/img/gp_xgboost.png)
