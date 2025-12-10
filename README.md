The official code repository for "PCoKG: Personality-aware Commonsense Reasoning with Debate", accepted by AAAI 2026.

# [Optional] Step 1: Get $(event, relation)$ pairs

* You can obtain $(event, relation)$ pairs from the [Link](https://pan.baidu.com/s/1AhbyNeBrWEgTK6oNWwM6gQ?pwd=tevp).*

If you want to run the code, modify the `config.yaml` configuration to set the LLM's `model`, `api_base`, and `api_key`.

# [Optional] Step2: Build PCoKG

* You can obtain mbti / ocean PCoKG, named `train/valid/test.csv` or `train/valid/test_alpaca.json` from
  the [Link](https://pan.baidu.com/s/1AhbyNeBrWEgTK6oNWwM6gQ?pwd=tevp).*

If you want to run the code, modify the `config.yaml` configuration to set the LLM's `model`, `api_base`, and `api_key`.

# Run PCoKGM

It is recommended to deploy vllm or sglang with docker and load the model save points

```shell
port=8001
device=1
modelPath="modelPath"
# -e CUDA_VISIBLE_DEVICES=0,1 \
docker run -d \
    --gpus "device=$device" \
    -v "$modelPath": "$modelPath"\
    -p $port:$port \
    -e CUDA_VISIBLE_DEVICES=$device \
    --ipc="host" \
    --network="host" \
    --privileged=true \
    --security-opt "seccomp:unconfined" \
   docker.1ms.run/lmsysorg/sglang:v0.4.6.post1-cu121 \
    python3 -m sglang.launch_server \
    --model "$modelPath" \
    --tp 1 \
    --trust-remote-code \
    --port $port \
    --host="0.0.0.0" \
    --served-model-name "pcokgm"
```
