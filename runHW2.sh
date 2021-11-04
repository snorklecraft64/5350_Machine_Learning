if [ $1 -lt 4 ]
then
	python3 EnsembleLearning/main.py $1
else
	python3 LinearRegression/main.py $1
fi

read name