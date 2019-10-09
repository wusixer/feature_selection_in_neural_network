pop=$1    # aa or ea
model=$2  # model is no_drug, all_drug, no_op

echo $pop
echo $model

head -1 /phenotype/dev_val_for_dnn/$model/${pop}_train_x.csv|tr ',' '\n' |awk '{print $0}'>${pop}.txt

paste -d'\t' ${pop}.txt ${pop}.ci_plus_average.txt |sort -gk 2,2 -r|awk '{print $1"\t"$2"\t"NR}' >${pop}.ci_plus_average.final.txt
