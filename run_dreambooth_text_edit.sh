INSTANCE_DIR=$1
TOKEN="sks"
TRAIN_PROMPT="a photo of $TOKEN ${INSTANCE_DIR//_/ }"
INFERENCE_PROMPT="a photo of $TOKEN ${INSTANCE_DIR//_/ }"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
NUM_IMAGES=4
DATA_DIR="/mnt/default/eval"
OUTPUT_DIR="/mnt/default/text_edit_output"
MODEL_DIR="/mnt/default/text_edit_models"
PRECISION=fp16


if [ ! -d $MODEL_DIR/${INSTANCE_DIR}/dreambooth ]; then
    accelerate launch --multi_gpu --mixed_precision=$PRECISION examples/dreambooth/train_dreambooth.py \
	    --pretrained_model_name_or_path=$MODEL_NAME \
    	--instance_data_dir=$DATA_DIR/$INSTANCE_DIR/reference \
	    --output_dir=$MODEL_DIR/${INSTANCE_DIR}/dreambooth \
    	--instance_prompt="$TRAIN_PROMPT" \
	    --gradient_accumulation_steps=1 \
    	--learning_rate=5e-6 \
	    --lr_scheduler="constant" \
    	--lr_warmup_steps=0 \
	    --max_train_steps=800 \
    	--push_to_hub \
	    --use_8bit_adam \
    	--gradient_checkpointing \
	    --mixed_precision=$PRECISION \
    	--train_batch_size=1
fi

python examples/dreambooth/inference_text_edit.py \
	--model_path=$MODEL_DIR/${INSTANCE_DIR}/dreambooth \
	--save_dir=$OUTPUT_DIR/${INSTANCE_DIR}/dreambooth \
	--num_images=$NUM_IMAGES \
    --token="$TOKEN" \
    --class_name=$INSTANCE_DIR


