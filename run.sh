
weights_array=(1 2 4 10 20 30 40 50 60 70 80 100 120 140 200 400 600 800 1000)

for wit in ${weights_array[@]}
do
	echo "=====testing 1:${wit}======"
	./test_weight.sh ${wit} >> ./snv_test_weight_result
	echo "===========done============"
done
