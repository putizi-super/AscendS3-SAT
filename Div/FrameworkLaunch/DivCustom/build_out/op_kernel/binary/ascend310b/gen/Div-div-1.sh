#!/bin/bash
echo "[Ascend310B1] Generating Div_b358f972ebfc84543602b83749c13265 ..."
opc $1 --main_func=div --input_param=/root/zpt_files/DivCustomSample_NEW/DivCustom/FrameworkLaunch/DivCustom/build_out/op_kernel/binary/ascend310b/gen/Div_b358f972ebfc84543602b83749c13265_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Div_b358f972ebfc84543602b83749c13265.json ; then
  echo "$2/Div_b358f972ebfc84543602b83749c13265.json not generated!"
  exit 1
fi

if ! test -f $2/Div_b358f972ebfc84543602b83749c13265.o ; then
  echo "$2/Div_b358f972ebfc84543602b83749c13265.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Div_b358f972ebfc84543602b83749c13265 Done"
