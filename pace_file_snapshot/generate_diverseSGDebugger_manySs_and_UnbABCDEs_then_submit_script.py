import os, sys

pbs_name = sys.argv[1]
pbs_ids = sys.argv[2:]
#pbs_ids = ['F_'+p+'s' for p in pbs_ids]
S_vals = [0, 0.1, 0.5, 1, 2] #[0.01,0.05,0.1,0.2]#[0.5, 1, 2] #[0,4,9]
name_the_rest = ''
the_rest = ''
if sys.argv[-1] == "-S":
    name_the_rest = 'SameInit'
    the_rest = ' -S'
    pbs_ids = pbs_ids[:-1]
name_the_rest += 'Unb'

for pbs_id in pbs_ids:
    for S in S_vals:
        with open(pbs_name+'_'+pbs_id+name_the_rest+'S{}.pbs'.format(S), 'w') as fp:
            fp.writelines(['#!/bin/bash\n',
                       '#PBS -N DFL_divSGDebug_'+pbs_id+name_the_rest+'\n',
                       '#PBS -A GT-cabdallah3\n',
                       '#PBS -l nodes=1:ppn=8\n',
                       '#PBS -l pmem=2gb\n',
                       '#PBS -l walltime=07:29:00\n',
                       '#PBS -j oe\n',
                       '#PBS -o divSGDebug_1.0_'+pbs_id+name_the_rest+'S{}.out\n'.format(S),
                       '#PBS -m abe\n',
                       '#PBS -M zzhang433@gatech.edu\n\n',
                       'cd $PBS_O_WORKDIR\n',
                       'module load pytorch\n',
                       'python3 MNIST_DFL_diverseUnbSGDebugger.py divUnbE1Class10 '+pbs_id+the_rest+' -s {}'.format(S),
                       '\n\n#END OF SCRIPT'
                      ])
        print('Generaged: ', pbs_name+'_'+pbs_id+name_the_rest+'S{}.pbs'.format(S))
        os.system('qsub -q embers '+(pbs_name+'_'+pbs_id+name_the_rest+'S{}.pbs'.format(S)))
        print('Job submitted')
