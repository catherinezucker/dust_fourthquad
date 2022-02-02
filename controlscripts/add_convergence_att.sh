fname="$1"

bayestar_dir=/n/fink2/czucker/Plane_Final

IFS=$'\n'
for pixel in `h5ls "${fname}" | awk '{ print $2 }'`; do
    echo "/pixel ${pixel}/discrete-los"
    python ${bayestar_dir}/final_scripts/chain_convergence.py -i "${fname}" -d "/pixel ${pixel}/discrete-los" --add-attribute
done
