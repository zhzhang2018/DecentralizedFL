import os, sys
# Expected input argument layout: name, types, types, ..., types, -<flag>, <flag content>, ...
pbs_name = sys.argv[1]
for i in range(len(sys.argv)):
    if sys.argv[i][0] == '-':
        break
pbs_ids = sys.argv[2:i] # All flags and other value settings
pbs_opts = sys.argv[i:]   # Optimizer options to try
# pbs_ids = ['G_'+p for p in pbs_ids]

name_the_rest = ''
filename_the_rest = ''
the_rest = ' '
for p_opt in pbs_opts:
    the_rest += p_opt
    the_rest += ' '
    filename_the_rest += p_opt
    if "-S" == p_opt:
        name_the_rest += 'SameInit'
    if "-U" == p_opt:
        name_the_rest += 'Unb'

# Here's what's going to happen here:
# The user customly determine which one to loop around about, and put that flag at the end of the flag list.
# The user also manually specify the list of values that it should take on. 
# Right now only allow one varying parameter for simplicity (and to reduce # of jobs generated).
pbs_opts.append('-K')
# pbs_opts.append('-EE')
# pbs_opts.append('-AET')
# pbs_opts.append('-k')
# pbs_opts.append('-p')
last_opt_vals = [10, 30, 50, 75, 100, 200]
# last_opt_vals = [ 1, 5, 10]
# last_opt_vals = [0.1, 1, 10, 20, 100]
# last_opt_vals = [2, 8, 16, 24]
# last_opt_vals = [0.1, 0.4, 0.6, 0.9]

filename_the_rest += pbs_opts[-1]
# Start creating jobs
for pbs_id in pbs_ids:
    for opt_val in last_opt_vals:
        filename = pbs_name+'_'+pbs_id+name_the_rest+pbs_opts[-1]+'{}.pbs'.format(opt_val)
        with open(filename, 'w') as fp:
            fp.writelines(['#!/bin/bash\n',
                       '#PBS -N commun_'+pbs_opts[-1]+'{}'.format(opt_val)+pbs_id+'\n',
                       '#PBS -A GT-cabdallah3\n',
                       '#PBS -l nodes=1:ppn=8\n',
                       '#PBS -l pmem=2gb\n',
                       '#PBS -l walltime=06:59:00\n',
                       '#PBS -j oe\n',
                       '#PBS -o commun_'+pbs_id+filename_the_rest+pbs_opts[-1]+'{}.out\n'.format(opt_val),
                       '#PBS -m abe\n',
                       '#PBS -M zzhang433@gatech.edu\n\n',
                       'cd $PBS_O_WORKDIR\n',
                       'module load pytorch\n',
                       'python3 MNIST_DFL_extremeCommun.py '+pbs_name+' '+pbs_id+the_rest+' '+pbs_opts[-1]+' {}'.format(opt_val),
                       '\n\n#END OF SCRIPT'
                      ])
        print('Generaged: ', filename)
        os.system('qsub -q embers '+filename)
        print('Job submitted')

