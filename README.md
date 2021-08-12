### Introduction
### Code Structure
### Installation
1.  Install Deeppoly

   Enter the folder “env_setup”, then run the following commands

   ```
   sudo ./install_gurobi.sh
   sudo ./install.sh
   ```

2. Configure  Gurobi

   1. Apply a free key from [this site](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) for the further license generation

   2. Generate an academic license with the following command:

      ```
      gurobi902/linux64/bin/grbgetkey your_license_key
      ```

   3. Set an environment variable to point to the license's path

      ```
      echo "export GRB_LICENSE_FILE=[your_path]/gurobi.lic" >> ~/.bashrc
      ```

3. Configure python environment

   1. Create an python3.6 environment named "py36" ( current folder: nn_repair/env_setup/)

      ```
      virtualenv -p python36 py36
      ```

   2. Install the python interface of Gurobi (current folder: nn_repair/env_setup/gurobi902/linux64/):

      ```
      sudo [your virtualenv parent path]/py36/bin/python3.6 setup.py install
      ```

   3. Install other python dependencies (current folder: nn_repair/):

      ```
      source ./py36/bin/activate
      pip install -r requirements.txt
      ```

4. Check the environment. ( current folder: nn_repair/)

   Run the following command and if no errors reported, then you make it!

   ```
   cd ./vanilla_deeppoly/
   python -u . --dataset acasxu --complete True --domain deeppoly --netname ../data/acasxu/onnx/ACASXU_run2a_2_1_batch_2000.onnx --specnumber 2
   ```
### Usage
   

