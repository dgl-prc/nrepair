#!/bin/bash

set -e

has_cuda=0

while : ; do
    case "$1" in
        "")
            break;;
        -use-cuda|--use-cuda)
         has_cuda=1;;
        *)
            echo "unknown option $1, try -help"
            exit 2;;
    esac
    shift
done

m4_path=./m4-1.4.1.tar.gz
if test ! -f "$m4_path"; then
    wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
fi
tar -xvzf m4-1.4.1.tar.gz
cd m4-1.4.1
./configure
make
make install
cp src/m4 /usr/bin
cd ..
rm m4-1.4.1.tar.gz

gmp_path=./gmp-6.1.2.tar.xz
if test ! -f "$gmp_path"; then
    wget -c https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
fi
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz

mpfr_path=./mpfr-4.0.2.tar.xz
if test ! -f "$mpfr_path"; then
  wget https://www.mpfr.org/mpfr-4.0.2/mpfr-4.0.2.tar.xz
fi
tar -xvf mpfr-4.0.2.tar.xz
cd mpfr-4.0.2
./configure
make
make install
cd ..
rm mpfr-4.0.2.tar.xz

cdd_path=./cddlib-0.94j.tar.gz
if test ! -f "$cdd_path"; then
  wget https://github.com/cddlib/cddlib/releases/download/0.94j/cddlib-0.94j.tar.gz
fi
tar -xvf cddlib-0.94j.tar.gz
cd cddlib-0.94j
./configure
make
make install
cd ..
rm cddlib-0.94j.tar.gz

git clone https://github.com/eth-sri/ELINA.git
cd ELINA
if test "$has_cuda" -eq 1
then
    ./configure -use-cuda -use-deepoly
else
    ./configure -use-deeppoly
fi
make
make install
cd ..

ldconfig