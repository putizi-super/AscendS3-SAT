cd /root/samples/operator_contrib/LogSumExp/FrameworkLaunch/SoftMaxCustom
./build.sh
cd /root/samples/operator_contrib/LogSumExp/FrameworkLaunch/SoftMaxCustom/build_out
./custom_opp_ubuntu_aarch64.run
cd /root/samples/operator_contrib/LogSumExp/FrameworkLaunch/LogSumExpCase1
rm /root/samples/operator_contrib/LogSumExp/FrameworkLaunch/LogSumExpCase1/input/*
./run.sh