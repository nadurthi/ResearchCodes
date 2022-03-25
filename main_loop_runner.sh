#!/bin/sh
for ((i=1;i<=200;++i))
do
  randport=$(python -S -c "import random; print(random.randrange(1,2000))")
  echo $randport
  python main_kitti_filter_localize_cpp_async_runs.py --k0 $randport
done
