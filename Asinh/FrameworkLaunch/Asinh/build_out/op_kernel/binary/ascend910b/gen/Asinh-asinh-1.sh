#!/bin/bash
echo "[Ascend910B1] Generating Asinh_d4eba078e7fc69d8a8ecb889ca0e3451 ..."
opc $1 --main_func=asinh --input_param=/root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out/op_kernel/binary/ascend910b/gen/Asinh_d4eba078e7fc69d8a8ecb889ca0e3451_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Asinh_d4eba078e7fc69d8a8ecb889ca0e3451.json ; then
  echo "$2/Asinh_d4eba078e7fc69d8a8ecb889ca0e3451.json not generated!"
  exit 1
fi

if ! test -f $2/Asinh_d4eba078e7fc69d8a8ecb889ca0e3451.o ; then
  echo "$2/Asinh_d4eba078e7fc69d8a8ecb889ca0e3451.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating Asinh_d4eba078e7fc69d8a8ecb889ca0e3451 Done"
