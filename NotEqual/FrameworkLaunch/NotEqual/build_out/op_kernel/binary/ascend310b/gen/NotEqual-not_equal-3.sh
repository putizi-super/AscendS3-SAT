#!/bin/bash
echo "[Ascend310B1] Generating NotEqual_2120fcc04c6b69eacdac7a18462f18d7 ..."
opc $1 --main_func=not_equal --input_param=/root/zhanghao/NotEqual/FrameworkLaunch/NotEqual/build_out/op_kernel/binary/ascend310b/gen/NotEqual_2120fcc04c6b69eacdac7a18462f18d7_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/NotEqual_2120fcc04c6b69eacdac7a18462f18d7.json ; then
  echo "$2/NotEqual_2120fcc04c6b69eacdac7a18462f18d7.json not generated!"
  exit 1
fi

if ! test -f $2/NotEqual_2120fcc04c6b69eacdac7a18462f18d7.o ; then
  echo "$2/NotEqual_2120fcc04c6b69eacdac7a18462f18d7.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating NotEqual_2120fcc04c6b69eacdac7a18462f18d7 Done"
