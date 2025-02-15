#!/bin/bash
echo "[Ascend310B1] Generating ScatterElements_d8a53a6405959e2ed94ad2e0e92ceacf ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=scatter_elements --input_param=/root/zhanghao/ScatterElements/FrameworkLaunch/ScatterElements/build_out/op_kernel/binary/ascend310b/gen/ScatterElements_d8a53a6405959e2ed94ad2e0e92ceacf_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ScatterElements_d8a53a6405959e2ed94ad2e0e92ceacf.json ; then
  echo "$2/ScatterElements_d8a53a6405959e2ed94ad2e0e92ceacf.json not generated!"
  exit 1
fi

if ! test -f $2/ScatterElements_d8a53a6405959e2ed94ad2e0e92ceacf.o ; then
  echo "$2/ScatterElements_d8a53a6405959e2ed94ad2e0e92ceacf.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating ScatterElements_d8a53a6405959e2ed94ad2e0e92ceacf Done"
