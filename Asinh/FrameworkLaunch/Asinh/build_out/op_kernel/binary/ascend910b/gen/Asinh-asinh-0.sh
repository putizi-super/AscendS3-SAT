#!/bin/bash
echo "[Ascend910B1] Generating Asinh_d25a300a27ae4e77dd3ae19fed835630 ..."
opc $1 --main_func=asinh --input_param=/root/zpt_files/AsinhCustomSample/AsinhSample/FrameworkLaunch/Asinh/build_out/op_kernel/binary/ascend910b/gen/Asinh_d25a300a27ae4e77dd3ae19fed835630_param.json --soc_version=Ascend910B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Asinh_d25a300a27ae4e77dd3ae19fed835630.json ; then
  echo "$2/Asinh_d25a300a27ae4e77dd3ae19fed835630.json not generated!"
  exit 1
fi

if ! test -f $2/Asinh_d25a300a27ae4e77dd3ae19fed835630.o ; then
  echo "$2/Asinh_d25a300a27ae4e77dd3ae19fed835630.o not generated!"
  exit 1
fi
echo "[Ascend910B1] Generating Asinh_d25a300a27ae4e77dd3ae19fed835630 Done"
