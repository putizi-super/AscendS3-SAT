#!/bin/bash
echo "[Ascend310B1] Generating ScatterElements_2c6c0079c0086a215e490a74885547e6 ..."
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
res=$(opc $1 --main_func=scatter_elements --input_param=/root/zhanghao/ScatterElements/FrameworkLaunch/ScatterElements/build_out/op_kernel/binary/ascend310b/gen/ScatterElements_2c6c0079c0086a215e490a74885547e6_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ScatterElements_2c6c0079c0086a215e490a74885547e6.json ; then
  echo "$2/ScatterElements_2c6c0079c0086a215e490a74885547e6.json not generated!"
  exit 1
fi

if ! test -f $2/ScatterElements_2c6c0079c0086a215e490a74885547e6.o ; then
  echo "$2/ScatterElements_2c6c0079c0086a215e490a74885547e6.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating ScatterElements_2c6c0079c0086a215e490a74885547e6 Done"
