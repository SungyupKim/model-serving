torch-model-archiver --model-name sample \
--version 1 \
--serialized-file "tensor.pt" \
--extra-files Handler.py \
--handler handle.py \
--export-path /Users/mzc01-sungyup/workspace/torch-tutorial
