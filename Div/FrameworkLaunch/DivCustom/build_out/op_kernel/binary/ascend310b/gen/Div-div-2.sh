#!/bin/bash
echo "[Ascend310B1] Generating Div_b1df8ea769386640e60dc802a3814dae ..."
opc $1 --main_func=div --input_param=/root/zpt_files/DivCustomSample_NEW/DivCustom/FrameworkLaunch/DivCustom/build_out/op_kernel/binary/ascend310b/gen/Div_b1df8ea769386640e60dc802a3814dae_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Div_b1df8ea769386640e60dc802a3814dae.json ; then
  echo "$2/Div_b1df8ea769386640e60dc802a3814dae.json not generated!"
  exit 1
fi

if ! test -f $2/Div_b1df8ea769386640e60dc802a3814dae.o ; then
  echo "$2/Div_b1df8ea769386640e60dc802a3814dae.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Div_b1df8ea769386640e60dc802a3814dae Done"
