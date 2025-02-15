#!/bin/bash
echo "[Ascend310B1] Generating ReplicationPad2d_8b40497ddeedfca5a93ef5f82650ab55 ..."
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
res=$(opc $1 --main_func=replication_pad2d --input_param=/root/zhanghao/ReplicationPad2dCopy/ReplicationPad2d/build_out/op_kernel/binary/ascend310b/gen/ReplicationPad2d_8b40497ddeedfca5a93ef5f82650ab55_param.json --soc_version=Ascend310B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ReplicationPad2d_8b40497ddeedfca5a93ef5f82650ab55.json ; then
  echo "$2/ReplicationPad2d_8b40497ddeedfca5a93ef5f82650ab55.json not generated!"
  exit 1
fi

if ! test -f $2/ReplicationPad2d_8b40497ddeedfca5a93ef5f82650ab55.o ; then
  echo "$2/ReplicationPad2d_8b40497ddeedfca5a93ef5f82650ab55.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating ReplicationPad2d_8b40497ddeedfca5a93ef5f82650ab55 Done"
