import os, sys
# Expected input argument layout: name, types, types, ..., types, -<flag>, <flag content>, ...
pbs_name = sys.argv[1]
for i in range(len(sys.argv)):
    if sys.argv[i][0] == '-':
        break
pbs_ids = sys.argv[2:i] # All flags and other value settings
pbs_opts = sys.argv[i:]   # Optimizer options to try

name_the_rest = ''
filename_the_rest = ''
the_rest = ' '
for j,p_opt in enumerate(pbs_opts):
    the_rest += p_opt
    the_rest += ' '
    filename_the_rest += p_opt
    if "-S" == p_opt:
        name_the_rest += 'SameInit'
    if "-U" == p_opt:
        name_the_rest += 'Unb'
    if "-SM" == p_opt:
        name_the_rest += '_segType_'+pbs_opts[j+1]
        filename_the_rest += '_segType_'+pbs_opts[j+1]

# Here's what's going to happen here:
# The user customly determine which one to loop around about, and put that flag at the end of the flag list.
# The user also manually specify the list of values that it should take on. 
# Right now only allow one varying parameter for simplicity (and to reduce # of jobs generated).
# pbs_opts.append('-K')
# pbs_opts.append('-EE')
pbs_opts.append('-PSS')
# last_opt_vals = [20]
# last_opt_vals = [0.05, 0.5, 1, 5]
last_opt_vals = [0.1, 0.25, 0.5, 0.8, 1.0]


filename_the_rest += pbs_opts[-1]
# Start creating jobs
for pbs_id in pbs_ids:
    for opt_val in last_opt_vals:
        filename = pbs_name+'_MNISTredcom_'+pbs_id+name_the_rest+filename_the_rest+pbs_opts[-1]+'{}.pbs'.format(opt_val)
        with open(filename, 'w') as fp:
            fp.writelines(['#!/bin/bash\n',
                       '#PBS -N redcom_'+pbs_opts[-1]+'{}'.format(opt_val)+pbs_id+'\n',
                       '#PBS -A GT-cabdallah3\n',
                       '#PBS -l nodes=1:ppn=16\n',
                       '#PBS -l pmem=2gb\n',
                       '#PBS -l walltime=07:45:00\n',
                       '#PBS -j oe\n',
                       '#PBS -o redcommunMNIST_'+pbs_name+pbs_id+filename_the_rest+pbs_opts[-1]+'{}.out\n'.format(opt_val),
                       '#PBS -m abe\n',
                       '#PBS -M zzhang433@gatech.edu\n\n',
                       'cd $PBS_O_WORKDIR\n',
                       'module load pytorch\n',
                       'python3 MNIST_DFL_reducedSharing.py '+pbs_name+' '+pbs_id+the_rest+' '+pbs_opts[-1]+' {}'.format(opt_val),
                       '\n\n#END OF SCRIPT'
                      ])
        print('Generaged: ', filename)
        os.system('qsub -q embers '+filename)
        print('Job submitted')

