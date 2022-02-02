#!/bin/sh

# clouds.sbatch
#SBATCH -J G314_it4 # A single job name for the array
#SBATCH -p shared,itc_cluster,serial_requeue # Partition
#SBATCH --mail-user=catherine.zucker@cfa.harvard.edu
#SBATCH --mail-type=FAIL
#SBATCH -n 1 # one core
#SBATCH --array=87
#SBATCH -t 2-0:00 # Running time of 2 days
#SBATCH --mem 5000 # Memory request of 5 GB
#SBATCH -o /n/fink2/czucker/Plane_Final/stout/G314/it4/out_%A_%a.out # Standard output
#SBATCH -e /n/fink2/czucker/Plane_Final/sterr/G314/it4/out_%A_%a.err # Standard error
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

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
cp /n/fink2/czucker/terra/PSMrLF.dat ${work_dir}

fn=$(printf "G314.%05d.h5" ${SLURM_ARRAY_TASK_ID})
./bayestar /n/fink2/czucker/Plane_Final/perstar/G314/${fn} /n/fink2/czucker/Plane_Final/output/it4/G314/${fn} --load-surfs --config /n/fink2/czucker/terra/config/it4/config.cfg.0



