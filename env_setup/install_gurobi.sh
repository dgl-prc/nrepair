# https://packages.gurobi.com/9.0/gurobi9.0.0_linux64.tar.gz

gurobi_path=./gurobi9.0.2_linux64.tar.gz
if test ! -f "$gurobi_path"; then
    wget -c https://packages.gurobi.com/9.0/gurobi9.0.2_linux64.tar.gz
fi
tar -xvf gurobi9.0.2_linux64.tar.gz
cd gurobi902/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cp ../../lib/libgurobi90.so /usr/lib
cd ../..
python3 setup.py install
cd ../..

export GUROBI_HOME="$(pwd)/gurobi902/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib


if test ! -f "deepg"; then
    git clone git@github.com:eth-sri/deepg.git
fi
cd deepg/code
mkdir build
make shared_object
cp ./build/libgeometric.so /usr/lib
cd ../..

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib

echo "please generate the key of gurobi by run 'grbgetkey' "