#!/bin/bash
echo "[Ascend310B1] Generating SoftMax_39aca3f4bd05c4a1799bf6471e0e4fab ..."
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
res=$(opc $1 --main_func=soft_max --input_param=/root/samples/operator_contrib/SoftMaxSample/FrameworkLaunch/SoftMaxCustom/build_out/op_kernel/binary/ascend310b/gen/SoftMax_39aca3f4bd05c4a1799bf6471e0e4fab_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/SoftMax_39aca3f4bd05c4a1799bf6471e0e4fab.json ; then
  echo "$2/SoftMax_39aca3f4bd05c4a1799bf6471e0e4fab.json not generated!"
  exit 1
fi

if ! test -f $2/SoftMax_39aca3f4bd05c4a1799bf6471e0e4fab.o ; then
  echo "$2/SoftMax_39aca3f4bd05c4a1799bf6471e0e4fab.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating SoftMax_39aca3f4bd05c4a1799bf6471e0e4fab Done"
