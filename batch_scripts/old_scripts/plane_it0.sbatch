#!/bin/sh

# clouds.sbatch
#SBATCH -J G314 # A single job name for the array
#SBATCH -p shared,itc_cluster,serial_requeue # Partition
#SBATCH --mail-user=catherine.zucker@cfa.harvard.edu
#SBATCH --mail-type=FAIL
#SBATCH -n 1 # one core
#SBATCH --array=2221
#SBATCH -t 7-00:00 # Running time of 4 days
#SBATCH --mem 10000 # Memory request of 10 GB
#SBATCH -o /n/fink2/czucker/Plane_Final/stout/G314/out_%A_%a.out # Standard output
#SBATCH -e /n/fink2/czucker/Plane_Final/sterr/G314/out_%A_%a.err # Standard error
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

module load Anaconda/5.0.1-fasrc02

# Detect CentOS version number and CPU model
centos_ver=7
six_str=`grep "release 6" /etc/centos-release`
if [[ ! -z ${six_str} ]]; then
    centos_ver=6
fi

cpu_spec=`grep "model name" /proc/cpuinfo | head -n 1 `
cpu_flags=`grep "flags" /proc/cpuinfo | head -n 1 `

full_spec="${centos_ver} ${cpu_info} ${cpu_flags} ${bayestar_ver}"
full_hash=`echo ${full_spec} | md5sum | awk '{print $1}' | cut -c -8`

echo "CentOS ${centos_ver}"
echo "CPU spec: ${cpu_spec}"
echo "CPU flags: ${cpu_flags}"
grep "cache size" /proc/cpuinfo | head -n 1
grep "cpu MHz" /proc/cpuinfo | head -n 1
grep "cpu cores" /proc/cpuinfo | head -n 1
echo "Bayestar version: ${bayestar_ver}"
echo "Hash: ${full_hash}"


source activate plane

# On termination, propagate signal to entire process group
job_termination_handler()
{
    echo "+ Job termination handler:"
    echo "+   * Propagating signal to subprocesses ..."
    kill "${PID}"
    echo "+   * Waiting 160 seconds ..."
    sleep 60
    echo "+   * Exiting ..."
    exit 17
}

trap 'job_termination_handler' USR1 # Job script should specify --signal=USR1@120

# Temporary working directory
work_dir=`env TMPDIR="/n/holyscratch01/finkbeiner_lab/Lab/czucker" mktemp -d -p /n/holystratch01/finkbeiner_lab/Lab/czucker/ -t bayestar.XXXXXXX`
echo "# Working directory: ${work_dir}"
cd ${work_dir}

export work_dir=${work_dir}

cp /n/fink2/czucker/Plane_Final/models_and_offsets/grid_mist_v9.h5 ${work_dir}
cp /n/fink2/czucker/Plane_Final/models_and_offsets/offsets_mist_v9.txt ${work_dir}

#python /n/fink2/czucker/Plane_Final/final_scripts/decaps_vvv_tmass_unwise_perstar.py ${SLURM_ARRAY_TASK_ID}

# Set up environment
if [[ ${centos_ver} -eq 6 ]]; then
    echo "Sourcing CentOS 6 environment ..."
    source /n/fink2/czucker/terra/activate-bayestar-centos6.sh
else
    echo "Sourcing CentOS 7 environment ..."
    source /n/fink2/czucker/terra/activate-bayestar-centos7.sh
fi

cp /n/home12/czucker/projects/bayestar_load_surfs.tar.gz ${work_dir}
cd ${work_dir}
tar -xzf bayestar_load_surfs.tar.gz
mv bayestar bayestar_load_surfs
cd bayestar_load_surfs
bash recompile.sh
cp bayestar ../
cd ${work_dir}

cp /n/fink2/czucker/terra/PS1_2MASS_Extinction_APOGEE.dat ${work_dir}
cp /n/fink2/czucker/terra/PS1_qz_2MASS_colors.dat ${work_dir}


fn=$(printf "G314.%05d.h5" ${SLURM_ARRAY_TASK_ID})
./bayestar /n/fink2/czucker/Plane_Final/perstar/G314/${fn} /n/fink2/czucker/Plane_Final/output/it0/G314/${fn} --load-surfs --config /n/fink2/czucker/terra/config/it0/config.cfg.0

# Delete working directory
echo "# Deleting the working directory ..."
rm -rf "${work_dir}"
