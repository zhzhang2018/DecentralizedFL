This repo documents the files used in running federated-learning-related experiments.

Folder contents: 

./data: Folder that stores the MNIST and CIFAR datasets. Depending on the code, you probably also need to have this data folder within the root folder (one step above this repo) as well: 
* ./data
  * ./MNIST
  * ./cifar-...
* ./DecentralizedFL

./FL_Scripts: Auxiliary scripts developed in 2021/01-02 and seldom used anymore.

./npy_datas: Folder where recorded data from PACE would be copied to. Should not be visible.


Individual files:

CIFAR_* and MNIST_* files differ in the dataset that is used, and the model that is created for the dataset. The mechanisms should be similar. Both are files that split and train the dataset.

generate_* are files that generate PBS scripts that PACE accepts.

\*.ipynb are files mostly used for visualizations and/or debugging. 

Each file under each category is explained in more detail in the following. To understand how each file works on their own, please open it and read its comments. 

**generate_* files**: The usage for those files is generally something like: 
`python3 generate_*_script.py <experiment identifier> <flags>`.

For example, this command: `python3 generate_reducedCommunMNIST_then_submit_script.py tryMNIST_has_CRS_record_layer0_EE01_cuSW_ B1 -k 4  -Adj SW -nr 100 -S -K 20 -SM cu -EE 0.1 -CRS` would generate several PBS files corresponding to all experiments specified by the command, if run in the PACE cluster. In this line, `generate_reducedCommunMNIST_then_submit_script.py` would generage scripts for different `PSS` values (as mentioned inside the code), sharing the same flags for all other settings. The generated files would bear the marker `tryMNIST_has_CRS_record_layer0_EE01_cuSW_` for identification. The generated PBS files would then be automatically submitted to PACE. The flags attached at the end of the command would be copied into the generated PBS files, so that those files, when submitted, would pass the same flags to the training script.

Below explains what each file does. If the file is not listed, then it's outdated and shouldn't be used.
* generate_diverseNorm_manySs_and_ABCDEs_then_submit_script.py: Corresponds to `MNIST_DFL_diverseNorm.py`. Last modified 2021/03. 
  * Usage: `python3 generate_diverseNorm_manySs_and_ABCDEs_then_submit_script.py <identifier> <list of optimizer identifiers> [-S]`
  * Generates one file for each combination of optimizer identifier (e.g. A, B1, D45) and each dataset overlapping proportion (marked by "-s" inside the file; default is from [0,1,4,9]). The dataset partition is NOT skewed by default (but you can opt to add "-Unb" by changing the code). User specifies if models start from uniform weight by including "-S" at the end. 
  
* generate_diverseSGD_manySs_and_ABCDEs_then_submit_script.py: Corresponds to `MNIST_DFL_diverseSGD.py`. Last modified 2021/03.
  * Usage: `python3 generate_diverseSGD_manySs_and_ABCDEs_then_submit_script.py <identifier> <list of optimizer identifiers> [-S]`
  * Generates one file for each combination of optimizer identifier (e.g. A, B1, D45) and each dataset overlapping proportion (marked by "-s" inside the file; default is from [0,1,4,9]). The dataset partition is skewed by default ("-Unb"). User specifies if models start from uniform weight by including "-S" at the end. 
  
* generate_diverseSGD_pbs_and_submit_script.py: Corresponds to `MNIST_DFL_diverseSGD.py`. Last modified 2021/02.
  * Usage: `python3 generate_diverseSGD_pbs_and_submit_script.py <identifier> <list of optimizer identifier combinationss> [-S]`
  * Generates one file for each combination of optimizer identifiers (e.g. AB, DE, AC). User specifies if models start from uniform weight by including "-S" at the end. 
  
* generate_diverseSGDloss_pbs_and_submit_script.py: Corresponds to `MNIST_DFL_diverseSGD_diverseLoss.py`. Last modified 2021/03.
  * Usage: `python3 generate_diverseSGDloss_pbs_and_submit_script.py <identifier> <list of optimizer identifier combinationss> [-S]`
  * Generates one file for each combination of optimizer identifier combination (e.g. AB, DE, AC) and number of clients using each loss function (from [10+0,8+2,5+5,2+8,0+10], where the first number is the number of clients using nLL, and the second is for using cross-entropy). User specifies if models start from uniform weight by including "-S" at the end. 

* generate_extremeCommun_then_submit_script.py: Corresponds to `MNIST_DFL_extremeCommun.py`. Last modified 2021/03. 
  * Usage: `python3 generate_extremeCommun_then_submit_script.py <identifier> <List of optimizer identifiers> <List of flags, each followed by its corresponding values> <List of flags that does not specify values>`
  * Generates one file for each combination of optimizer identifier (e.g. A, B1, D45) and each varying parameter (by default it's "-K" with [10, 30, 50, 75, 100, 200], but you can change it by modifying the code). 
  * The second-to-last list of flags should look like something like this: `-k 4  -Adj SW -nr 100`. Each flag (-...) is followed by the specified value (int, string, or other types).
  * The dataset partition is NOT skewed OR using same initial weights OR any other stuff like that by default, but you can opt to add those flags in the final list of flags, like "-U" for skewed and "-S" for same initial values. The code should still work if you disperse those flags in between the other list of flags-with-values, though. 

* generate_reducedCommunMNIST_then_submit_script.py: Corresponds to `MNIST_DFL_reducedSharing.py`. Last modified 2021/04. 
  * Usage: `python3 generate_reducedCommunMNIST_then_submit_script.py <identifier> <List of optimizer identifiers> <List of flags, each followed by its corresponding values> <List of flags that does not specify values>`
  * Generates one file for each combination of optimizer identifier (e.g. A, B1, D45) and each varying parameter (by default it's "-PSS" with [0.1, 0.25, 0.5, 0.8, 1.0], but you can change it by modifying the code). 
  * The second-to-last list of flags should look like something like this: `-k 4  -Adj SW -nr 100`. Each flag (-...) is followed by the specified value (int, string, or other types).
  * The dataset partition is NOT skewed OR using same initial weights OR any other stuff like that by default, but you can opt to add those flags in the final list of flags, like "-U" for skewed and "-S" for same initial values. The code should still work if you disperse those flags in between the other list of flags-with-values, though. 

**MNIST_DFL_* files**: If you run it with `python3 MNIST_DFL_*.py <filename identifier> <list of flags>`, then it would start the training on the head node (command window), which could trigger a PACE usage warning. Instead, please use them by submitting a corresponding PBS script to the PACE computation cluster. To create a PBS script and then submit it, refer to the `generate_*.py` files above.

Note that most files are mostly similar, sharing the same bulk of code sections while differing at small details. As a result, only a few of them are fully documented inside the file. I'll mark those files in bold. 
* `MNIST_DFL_diverseClient.py`: Trains a bunch of clients with different parameters. Last modified 2021/02. Because it hasn't been used for so long, it still maintains what the code would look like at the start of the exploration, and you can use it as a simpler reference file.
* `MNIST_DFL_diverseNorm.py`: Uses different normalization parameters for each client's dataset. Last modified 2021/03.
* `MNIST_DFL_diverseSGD_diverseLoss.py`: Uses different loss functions for different clients. Last modified 2021/03.
* `MNIST_DFL_diverseSGD.py`: Offers different optimizers. Also offers possibility for mixing 2 or more different optimizers across clients. Last modified 2021/03. 
* `MNIST_DFL_diverseUnbSGD.py`: Same as the above, but you can also control the amount of skewed-ness in the dataset partition for different clients. Last modified 2021/03, **Fully commented**
* `MNIST_DFL_extremeCommun.py`: Adds controllability to the model aggregation step, where you can twik the number of iterations, stopping criteria, etc. to save communication cost. Half-baked; refer to other files below. Last modified 2021/03.
* `MNIST_DFL_fulldata.py`: Trains with each client accessing the full dataset. Very preliminary implementation from 2021/02. 
* `MNIST_DFL_overlapdata.py`: Trains with each client accessing its own dataset, plus some from its neighbors. Refined version of a preliminary implementation from 2021/02. **Fully commented**. Later, the file structured changed, and is reflected in `MNIST_DFL_diverseUnbSGD.py`.
* `MNIST_DFL_reducedSharing.py`: Most up-to-date implementation where we attempt to cut the model into segments, and only aggregate between segments, in order to save computation cost  while maintaining the same level of performance. The file structure is nearly overhauled, when compared to `MNIST_DFL_diverseUnbSGD.py`. Last modified in 2021/04, and is **fully commented**.
* `MNIST_DFL_segmentModel.py`: Repeats Hu et al.'s work when partitioning the dataset with partial sharing in mind. Last modified 2021/02-03.
* `MNIST_DFL_splitData.py`: Generates partitions of data with different sizes etc. Last modified 2021/03.

**.ipynb files**: Visualization files etc. Read explanations of code by opening them. 
* `consensus_playground.ipynb`: This file simulates the consensus protocol / model aggregation step, so that you can see if the implemented method would actually converge or not. 
* `FL_MNIST.ipynb`: Pre-historic file to interact with the training process. I didn't end up using it in PACE, because interactively dealing with PACE sounded weird, although probably feasible. 
* `plot_npys.ipynb`: Contains methods that interpret the recorded data from the python files, and also visualization methods for each type of data.
* `pretty_plotter.ipynb`: Visualization methods for pictures in the report. Takes in organized data and plot out the trends. This file contains more usage of matplotlib than others do, and could be useful for other projects as well.
