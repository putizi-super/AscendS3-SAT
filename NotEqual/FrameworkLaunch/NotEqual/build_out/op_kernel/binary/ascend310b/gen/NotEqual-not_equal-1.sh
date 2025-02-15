#!/bin/bash
echo "[Ascend310B1] Generating NotEqual_dc606b12f4fa9412a691f2568bd71cab ..."
opc $1 --main_func=not_equal --input_param=/root/zhanghao/NotEqual/FrameworkLaunch/NotEqual/build_out/op_kernel/binary/ascend310b/gen/NotEqual_dc606b12f4fa9412a691f2568bd71cab_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/NotEqual_dc606b12f4fa9412a691f2568bd71cab.json ; then
  echo "$2/NotEqual_dc606b12f4fa9412a691f2568bd71cab.json not generated!"
  exit 1
fi

if ! test -f $2/NotEqual_dc606b12f4fa9412a691f2568bd71cab.o ; then
  echo "$2/NotEqual_dc606b12f4fa9412a691f2568bd71cab.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating NotEqual_dc606b12f4fa9412a691f2568bd71cab Done"
