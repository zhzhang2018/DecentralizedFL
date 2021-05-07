import os, sys

pbs_name = sys.argv[1]
pbs_ids = sys.argv[2:]
# pbs_ids = ['F_'+p+'n' for p in pbs_ids]
name_the_rest = ''
the_rest = ''
if sys.argv[-1] == "-S":
    name_the_rest = 'SameInit'
    the_rest = ' -S'
    pbs_ids = pbs_ids[:-1]

for pbs_id in pbs_ids:
    for x_val in ['00','02','05','08','10']:
        filename = pbs_name+'_X{}_'.format(x_val)+pbs_id+name_the_rest+'.pbs'
        with open(filename, 'w') as fp:
            fp.writelines(['#!/bin/bash\n',
                           '#PBS -N DFL_X{}_',format(x_val)+pbs_id+name_the_rest+'\n',
                           '#PBS -A GT-cabdallah3\n',
                           '#PBS -l nodes=1:ppn=8\n',
                           '#PBS -l pmem=1gb\n',
                           '#PBS -l walltime=07:59:00\n',
                           '#PBS -j oe\n',
                           '#PBS -o divLoss_X'+x_val+'_'+pbs_id+name_the_rest+'.out\n',
                           '#PBS -m abe\n',
                           '#PBS -M zzhang433@gatech.edu\n\n',
                           'cd $PBS_O_WORKDIR\n',
                           'module load pytorch\n',
                           'python3 MNIST_DFL_diverseSGD_diverseLoss.py loss_first_try '+'X'+x_val+pbs_id+the_rest,
                           '\n\n#END OF SCRIPT'
                          ])
        print('Generaged: ', filename)
        os.system('qsub -q embers '+filename)
        print('Job submitted')
    
    
