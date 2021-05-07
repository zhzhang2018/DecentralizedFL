import os, sys

pbs_name = sys.argv[1]
# pbs_ids = sys.argv[2:]
#pbs_ids = ['F_'+p+'s' for p in pbs_ids]
# S_vals = [0,1,4,9]
pbs_ids = ['A','C','D','E']
name_the_rest = ''
the_rest = ''
if sys.argv[-1] == "-S":
    name_the_rest = 'SameInit'
    the_rest = ' -S'
#     pbs_ids = pbs_ids[:-1]
name_the_rest += 'Unb'

for pbs_id in pbs_ids:
#     for S in S_vals:
#     with open(pbs_name+'_'+pbs_id+name_the_rest+'S{}.pbs'.format(S), 'w') as fp:
    filename = pbs_name+'_'+pbs_id+name_the_rest+'.pbs'
    with open(filename, 'w') as fp:
        fp.writelines(['#!/bin/bash\n',
                   '#PBS -N DFL_split_divSGD_'+pbs_id+name_the_rest+'\n',
                   '#PBS -A GT-cabdallah3\n',
                   '#PBS -l nodes=1:ppn=16\n',
                   '#PBS -l pmem=2gb\n',
                   '#PBS -l walltime=07:59:00\n',
                   '#PBS -j oe\n',
                   '#PBS -o divSGD_'+pbs_id+name_the_rest+'.out\n',
#                    '#PBS -o divSGD_'+pbs_id+name_the_rest+'S{}.out\n'.format(S),
                   '#PBS -m abe\n',
                   '#PBS -M zzhang433@gatech.edu\n\n',
                   'cd $PBS_O_WORKDIR\n',
                   'module load pytorch\n',
#                    'python3 MNIST_DFL_diverseSGD.py divSGD_n_varS_ '+pbs_id+the_rest+' -s {}'.format(S),
                   'python3 MNIST_DFL_splitDataUnbalanced'+pbs_id+'.py split_uniSGD_ '+the_rest,
                   '\n\n#END OF SCRIPT'
                  ])
    print('Generaged: ', filename)
    os.system('qsub -q embers '+filename)
    print('Job submitted')
