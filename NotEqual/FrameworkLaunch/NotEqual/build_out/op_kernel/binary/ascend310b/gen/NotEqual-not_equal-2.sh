#!/bin/bash
echo "[Ascend310B1] Generating NotEqual_2e5c9109bcb0c983b75f7cce0cfede47 ..."
opc $1 --main_func=not_equal --input_param=/root/zhanghao/NotEqual/FrameworkLaunch/NotEqual/build_out/op_kernel/binary/ascend310b/gen/NotEqual_2e5c9109bcb0c983b75f7cce0cfede47_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/NotEqual_2e5c9109bcb0c983b75f7cce0cfede47.json ; then
  echo "$2/NotEqual_2e5c9109bcb0c983b75f7cce0cfede47.json not generated!"
  exit 1
fi

if ! test -f $2/NotEqual_2e5c9109bcb0c983b75f7cce0cfede47.o ; then
  echo "$2/NotEqual_2e5c9109bcb0c983b75f7cce0cfede47.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating NotEqual_2e5c9109bcb0c983b75f7cce0cfede47 Done"
