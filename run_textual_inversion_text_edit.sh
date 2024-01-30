INSTANCE_DIR="$1"
TOKEN="<${INSTANCE_DIR}>"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
PRECISION=fp16
NUM_IMAGES=4
DATA_DIR="/mnt/default/eval"
OUTPUT_DIR="/mnt/default/text_edit_output"
MODEL_DIR="/mnt/default/text_edit_models"

if [ ! -d $MODEL_DIR/${INSTANCE_DIR}/textual_inversion ]; then

    accelerate launch --multi_gpu --mixed_precision=$PRECISION examples/textual_inversion/textual_inversion.py \
	    --pretrained_model_name_or_path=$MODEL_NAME \
	    --train_data_dir=$DATA_DIR/$INSTANCE_DIR/reference \
	    --learnable_property=$PROPERTY \
	    --placeholder_token=$TOKEN \
    	--initializer_token="${INSTANCE_DIR}" \
    	--train_batch_size=1 \
	    --gradient_accumulation_steps=4 \
    	--max_train_steps=3000 \
	    --learning_rate=5.0e-04 \
    	--scale_lr \
	    --lr_scheduler="constant" \
    	--lr_warmup_steps=0 \
	    --push_to_hub \
        --repeats=1000 \
	    --mixed_precision=$PRECISION \
    	--enable_xformers_memory_efficient_attention \
	    --output_dir=$MODEL_DIR/${INSTANCE_DIR}/textual_inversion
fi

python examples/textual_inversion/inference_text_edit.py \
	--model_path=$MODEL_DIR/${INSTANCE_DIR}/textual_inversion \
	--save_dir=$OUTPUT_DIR/${INSTANCE_DIR}/textual_inversion \
	--num_images=$NUM_IMAGES \
    --token=$TOKEN \
    --class_name=$INSTANCE_DIR
