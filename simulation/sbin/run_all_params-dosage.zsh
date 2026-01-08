repeat=$1
start=$2
output=$3
effect=$4
dosage=$5

echo "useage: run_all_params.zsh <repeat> <start> <output> <effect>"
echo " |- repeat: number of repeats"
echo " |-  start: starting repeation number"
echo " |- output: output directory"
echo " |- effect: treatment effect"
echo "Generating commands for $repeat repeats starting from repeation number $start with treatment effect $effect"

policies=(null high-risk low-risk age-first)
# policies=(null high-risk low-risk)
# policies=(null high-risk low-risk age-first high-risk-young-first)
# policies=(null high-risk low-risk age-first age-tolerance)
# policies=("age-first-high-risk")
# policies=(high-risk-cutoff)
scale_factors=(0.05 0.1 0.2 0.3 0.4 0.5 0.7 0.9 1.0)
# scale_factors=(0.05 2.0 3.0 5.0)
# scale_factors=(10.0)
# scale_factors=(3.0 5.0)
# scale_factors=(0.05 0.1 0.2)
# scale_factors=(0.3 0.4 0.5 0.7 0.9 1.0)
term_lengths=(1000)
arrival=5
cap=80

# Generate commands to cmd.sh
if [[ -f $output/cmd.sh ]]; then
  rm $output/cmd.sh
fi
if [[ -d $output/logs ]]; then
  rm -r $output/logs
fi
mkdir -p $output/logs
for ((k=$start; k<$start+$repeat; k++)); do
  for scale_factor in $scale_factors; do
    for term_length in $term_lengths; do
      nm=tl-${term_length}-sf-${scale_factor}
      outd=$output/$nm
      if [[ -d $outd ]]; then
        rm -r $outd/logs
      fi
      mkdir -p $outd/logs
      for policy in $policies; do
        echo "python -u run_policy.py \
    --prison_rate_scaler $scale_factor \
    --length_scaler 1.0 \
    --beta_arrival $arrival \
    --beta_initial 2000 \
    --max_returns 30 \
    --max_offenses 35 \
    --T_max 40000 \
    --p_length 100 \
    --p_freeze 100 \
    --rel_off_probation ${term_length} \
    --treatment_capacity $cap \
    --treatment_effect $effect \
    --treatment_dosage $dosage \
    $policy 1 $k $outd &> $outd/logs/${policy}_${k}_${nm}.log " >> $output/cmd.sh
      done
    done
  done
done

echo "Commands generated in cmd.sh."
echo "see:  $output/cmd.sh"
echo 'parallel run with: \
cat $output/cmd.sh | xargs -I {} -P 30 bash -c "{}" &'
