#!/bin/bash
echo "[Ascend310B1] Generating Div_5116c9ee4d0642fa1e2808937c833487 ..."
opc $1 --main_func=div --input_param=/root/zpt_files/DivCustomSample_NEW/DivCustom/FrameworkLaunch/DivCustom/build_out/op_kernel/binary/ascend310b/gen/Div_5116c9ee4d0642fa1e2808937c833487_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Div_5116c9ee4d0642fa1e2808937c833487.json ; then
  echo "$2/Div_5116c9ee4d0642fa1e2808937c833487.json not generated!"
  exit 1
fi

if ! test -f $2/Div_5116c9ee4d0642fa1e2808937c833487.o ; then
  echo "$2/Div_5116c9ee4d0642fa1e2808937c833487.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Div_5116c9ee4d0642fa1e2808937c833487 Done"
