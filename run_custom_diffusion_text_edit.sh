INSTANCE_DIR=$1
TOKEN="<new1>"
TRAIN_PROMPT="a photo of $TOKEN ${INSTANCE_DIR//_/ }"
INFERENCE_PROMPT="a cartoon of a $TOKEN ${INSTANCE_DIR//_/ }"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
NUM_IMAGES=4
DATA_DIR="/mnt/default/eval"
OUTPUT_DIR="/mnt/default/text_edit_output"
MODEL_DIR="/mnt/default/text_edit_models"
PRECISION=no  # Currently not supporting mixed precision

if [ ! -d $MODEL_DIR/${INSTANCE_DIR}/custom_diffusion ]; then
    accelerate launch --mixed_precision=$PRECISION examples/custom_diffusion/train_custom_diffusion.py \
    	--pretrained_model_name_or_path=$MODEL_NAME \
	    --instance_data_dir=$DATA_DIR/$INSTANCE_DIR/reference \
    	--output_dir=$MODEL_DIR/$INSTANCE_DIR/custom_diffusion \
	    --instance_prompt="$TRAIN_PROMPT" \
    	--gradient_accumulation_steps=1 \
	    --learning_rate=1e-5 \
    	--lr_scheduler="constant" \
	    --lr_warmup_steps=0 \
    	--max_train_steps=250 \
	    --scale_lr \
    	--hflip \
	    --modifier_token=$TOKEN \
    	--use_8bit_adam \
	    --gradient_checkpointing \
    	--mixed_precision=$PRECISION \
	    --no_safe_serialization \
    	--train_batch_size=2
fi

python examples/custom_diffusion/inference_text_edit.py \
	--pretrained_model_name_or_path=$MODEL_NAME \
	--model_path=$MODEL_DIR/$INSTANCE_DIR/custom_diffusion \
	--token=$TOKEN \
    --class_name=$INSTANCE_DIR \
	--save_dir=$OUTPUT_DIR/$INSTANCE_DIR/custom_diffusion \
	--num_images=$NUM_IMAGES

