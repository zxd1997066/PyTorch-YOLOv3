set -xe

model_name=$(echo $@ |sed 's/.*--model_name.//;s/ *--.*//')

if [ "${model_name}" == "YoloV3" ];then
    ./launch_benchmark_yolov3.sh $@
elif [ "${model_name}" == "YoloV3-tiny" ];then
    ./launch_benchmark_yolov3tiny.sh $@
fi
