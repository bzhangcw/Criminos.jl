repeat=$1
start=$2
output=$3
scale_factor=$4
echo "useage: run_all_params_cap.zsh <repeat> <start> <output> <scale_factor>"
echo " |- repeat: number of repeats"
echo " |-  start: starting repeation number"
echo " |- output: output directory"
echo " |- scale_factor: scale factor"
echo "Generating commands for $repeat repeats starting from repeation number $start with scale factor $scale_factor"

# policies=(null random high-risk low-risk high-risk-only-young age-first age-tolerance)
# policies=(high-risk age-tolerance high-risk-only-young)
# policies=(null random low-risk)
# policies=(age-first)
# policies=(null high-risk low-risk age-first age-first-high-risk)
policies=(null high-risk low-risk age-first)
# policies=(age-first-high-risk)
capacities=(50 100 200 300 400 500)
# effects=(0.3 0.4 0.7 0.8)
effects=(0.6 0.9)


# Generate commands to cmd.sh
if [[ -f $output/cmd.sh ]]; then
  rm $output/cmd.sh
fi
if [[ -d $output/logs ]]; then
  rm -r $output/logs
fi
mkdir -p $output/logs
for ((k=$start; k<$start+$repeat; k++)); do
  for capacity in $capacities; do
    for effect in $effects; do
      outd=$output/c-${capacity}-e-${effect}
      if [[ -d $outd ]]; then
        rm -r $outd/logs
      fi
      mkdir -p $outd/logs
      for policy in $policies; do
       echo "python -u run_policy.py \
    --prison_rate_scaler $scale_factor \
    --length_scaler 1.0 \
    --beta_arrival 5 \
    --max_returns 30 \
    --max_offenses 35 \
    --T_max 60000 --p_length 100 \
    --rel_off_probation 1500 \
    --treatment_capacity $capacity \
    --treatment_effect $effect \
    $policy 1 $k $outd &> $outd/logs/${policy}_${k}_${capacity}_${effect}.log " >> $output/cmd.sh
      done
    done
  done
done

echo "Commands generated in cmd.sh."
echo "see:  $output/cmd.sh"
echo 'parallel run with: \
cat $output/cmd.sh | xargs -I {} -P 30 bash -c "{}" &'
