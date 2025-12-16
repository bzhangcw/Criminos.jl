repeat=$1
start=$2
output=$3
echo "Generating commands for $repeat repeats starting from repeation number $start"

policies=(null random high-risk low-risk high-risk-only-young age-tolerance)

# Generate commands to cmd.sh
if [[ -f cmd.sh ]]; then
  rm cmd.sh
fi
if [[ -d $output/logs ]]; then
  rm -r $output/logs
fi
mkdir -p $output/logs
for ((k=$start; k<$start+$repeat; k++)); do
  for policy in $policies; do
    echo "python -u run_policy.py $policy 1 $k $output &> $output/logs/${policy}_${k}.log " >> cmd.sh
  done
done

echo "Commands generated in cmd.sh. Run with: bash cmd.sh"
