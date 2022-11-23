function run()
{
  lan=$1
  model="/work/codes/mlama/info/mbert.json"
  cmd="python scripts/run_experiments_mBERT_ranked.py -l $lan -m $model"
  echo $cmd
  $cmd
}

export PYTHONPATH=${PYTHONPATH}:/work/codes/mlama
lcs=(ar en fi fr id ja ru vi zh)
for ((i=$1;i<=$2;i++))
do 
  lc=${lcs[$i]}
  run $lc
done
