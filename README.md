##In-database parallel grid-search wrapper for XGBoost

### Dependencies 

1. [XGBoost library](https://github.com/dmlc/xgboost)
2. [scikit-learn](http://scikit-learn.org/stable/)

### Installation

1. Download XGBoost and sklearn libraries to all nodes of your Greenplum cluster (using `gpscp -f hostfile <source> =:<destination>`)
2. Compile and install XGBoost and sklearn libraries on all nodes (using `gpssh -f hostfile` followed by the compile and install commands)
3. Run the above SQL file (it will create a schema called `xgbdemo`). 
4. Invoke the UDFs as shown in the sample snippet.




