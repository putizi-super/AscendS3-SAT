#!/bin/bash
echo "[Ascend310B1] Generating NotEqual_2799cbb2df8b9381fa04afc3705605d3 ..."
opc $1 --main_func=not_equal --input_param=/root/zhanghao/NotEqual/FrameworkLaunch/NotEqual/build_out/op_kernel/binary/ascend310b/gen/NotEqual_2799cbb2df8b9381fa04afc3705605d3_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/NotEqual_2799cbb2df8b9381fa04afc3705605d3.json ; then
  echo "$2/NotEqual_2799cbb2df8b9381fa04afc3705605d3.json not generated!"
  exit 1
fi

if ! test -f $2/NotEqual_2799cbb2df8b9381fa04afc3705605d3.o ; then
  echo "$2/NotEqual_2799cbb2df8b9381fa04afc3705605d3.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating NotEqual_2799cbb2df8b9381fa04afc3705605d3 Done"
