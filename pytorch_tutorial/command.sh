torch-model-archiver --model-name mnist \
                     --version 1.0 \
                     --serialized-file mnist.pt \
                     --handler no_prepost_processing_handler.py \
                     --extra-files no_prepost_processing_handler.py -f
