
#weights_array=(1 4 20 30 40 50 60 70 80 120 140 200 400 600 800 1000)
weights_array=(1 10 40 100)

for wit in ${weights_array[@]}
do
	echo "=====testing 1:${wit}======"
	/home/haoz/deepfilter/cmd/test_weight.sh ${wit} 
	echo "===========done============"
done
