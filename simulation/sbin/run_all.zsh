repeat=$1
start=$2
output=$3
echo "Generating commands for $repeat repeats starting from repeation number $start"

# policies=(null random high-risk low-risk high-risk-only-young age-first age-tolerance)
# policies=(high-risk age-tolerance high-risk-only-young)
policies=(null high-risk low-risk age-first)
# policies=(null random low-risk)
# policies=(age-first)

# Generate commands to cmd.sh
if [[ -f $output/cmd.sh ]]; then
  rm $output/cmd.sh
fi
if [[ -d $output/logs ]]; then
  rm -r $output/logs
fi
mkdir -p $output/logs
for ((k=$start; k<$start+$repeat; k++)); do
  for policy in $policies; do
    echo "python -u run_policy.py \
    --prison_rate_scaler 0.2 --length_scaler 1.0 \
    --beta_arrival 5 \
    --max_returns 30 \
    --max_offenses 35 \
    --T_max 60000 --p_length 100 \
    --rel_off_probation 1500 \
    $policy 1 $k $output &> $output/logs/${policy}_${k}.log " >> $output/cmd.sh
  done
done

echo "Commands generated in $output/cmd.sh. Run with: bash cmd.sh"
