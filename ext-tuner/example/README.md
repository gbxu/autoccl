# Contribution

## docker
```
docker pull nvcr.io/nvidia/pytorch:23.08-py3
```

## Code Style
```sh
# download clang-format 10.0.0
wget https://gh-proxy.com/https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
# uncompress
tar -xf clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
clangtidy=./clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04/bin/clang-tidy
clangformat=./clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04/bin/clang-format 

# use bear to generate compile_commands.json
sudo apt install bear
bear make
find . -name '*.cc' -or -name '*.h' -or -name '*.cu' | xargs $clangtidy -p . -fix-errors
find . -name '*.cc' -or -name '*.h' -or -name '*.cu' | xargs $clangtidy -p . -checks=cppcoreguidelines-init-variables  > tidy_result.txt 2>&1

pip3 install cpplint

# check the format
find . -name '*.cc' -or -name '*.h' -or -name '*.cu' | xargs cpplint > lint_result.txt 2>&1
find . -name '*.cc' -or -name '*.h' -or -name '*.cu' | xargs $clangformat -i
```

## performance debug
```sh
apt update
apt install linux-tools-`uname -r | cut -d- -f1-2`-`uname -r | cut -d- -f3` -y
apt install linux-tools-common
apt install linux-tools-generic -y
make perf
perf report -g -i perf.data
```
* flamegraph
```
git clone https://github.com/brendangregg/FlameGraph.git
perf script -i perf.data | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > out.svg
```
