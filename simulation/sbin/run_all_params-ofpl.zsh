repeat=$1
start=$2
output=$3
effect=$4
allowrtn=$5
echo "useage: run_all_params.zsh <repeat> <start> <output> <effect> <allowrtn>"
echo " |- repeat: number of repeats"
echo " |-  start: starting repeation number"
echo " |- output: output directory"
echo " |- effect: treatment effect"
echo " |- allowrtn: whether to allow returning individuals to be treated"
echo "Generating commands for $repeat repeats \n\t starting from $start with effect $effect\n\t and allowrtn $allowrtn"

# policies=(null random high-risk low-risk high-risk-only-young age-first age-tolerance)
# policies=(high-risk age-tolerance high-risk-only-young)
# policies=(null random high-risk low-risk)
policies=(null high-risk low-risk age-first)
# policies=(high-risk-cutoff)
# policies=(null high-risk low-risk age-first)
# policies=(age-first-high-risk)

# normal
# scale_factors=(0.05 0.1 0.2 0.3 0.4 0.5 0.7 0.9 1.0)
# very large
scale_factors=(0.05 0.2 0.4 0.9 2.0 4.0 8.0 9.0)
# term_lengths=(500 1000 1500 2000)
# term_lengths=(365 730 1000 2000)
term_lengths=(365 730)
# term_lengths=(1000)


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
    --beta_arrival 5 \
    --max_returns 30 \
    --max_offenses 35 \
    --T_max 40000 --p_length 100 \
    --rel_off_probation ${term_length} \
    --treatment_capacity 80 \
    --treatment_effect $effect \
    --bool_return_can_be_treated $allowrtn \
    $policy 1 $k $outd &> $outd/logs/${policy}_${k}_${nm}.log " >> $output/cmd.sh
      done
    done
  done
done

echo "Commands generated in cmd.sh."
echo "see:  $output/cmd.sh"
echo "parallel run with: \n
cat $output/cmd.sh | xargs -I {} -P 30 bash -c \"{}\" &"
