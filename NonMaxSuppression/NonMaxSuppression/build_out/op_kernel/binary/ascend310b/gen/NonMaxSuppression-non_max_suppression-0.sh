#!/bin/bash
echo "[Ascend310B1] Generating NonMaxSuppression_e8014c2fcd1a515e6a5c3226d4faf52f ..."
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
res=$(opc $1 --main_func=non_max_suppression --input_param=/root/zhanghao/NonMaxSuppression/NonMaxSuppression/build_out/op_kernel/binary/ascend310b/gen/NonMaxSuppression_e8014c2fcd1a515e6a5c3226d4faf52f_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/NonMaxSuppression_e8014c2fcd1a515e6a5c3226d4faf52f.json ; then
  echo "$2/NonMaxSuppression_e8014c2fcd1a515e6a5c3226d4faf52f.json not generated!"
  exit 1
fi

if ! test -f $2/NonMaxSuppression_e8014c2fcd1a515e6a5c3226d4faf52f.o ; then
  echo "$2/NonMaxSuppression_e8014c2fcd1a515e6a5c3226d4faf52f.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating NonMaxSuppression_e8014c2fcd1a515e6a5c3226d4faf52f Done"
